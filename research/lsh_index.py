"""Bounded random-hyperplane candidate search over disk KV-address metadata.

Independent follow-up: no changes to frozen semantic_index or GPU experiments.
Limits bound postings returned and vectors reranked, not B-tree/page-cache cost.
Fixed budgets can lose recall as the archive grows. No full-scan fallback.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


@dataclass(frozen=True)
class LSHConfig:
    dimensions: int = 384
    tables: int = 16
    bits: int = 10
    posting_cap: int = 16
    rerank_cap: int = 512
    seed: int = 251

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == 'seed' else 1):
                raise ValueError('invalid LSH configuration: '+name)
        if self.bits > 30:
            raise ValueError('signature exceeds supported bit width')


def priority(block):
    """Stable bucket sample order, independent of insertion order and age."""
    return hashlib.sha256(str(block).encode('ascii')).digest()[:8]


class DiskLSHIndex:
    def __init__(self, path, *, create=False, config=None, readonly=False):
        if create and readonly:
            raise ValueError('cannot create read-only index')
        if not create and not Path(path).is_file():
            raise FileNotFoundError(path)
        self.db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode='+
                                  ('ro' if readonly else 'rwc'), uri=True)
        self.db.execute('PRAGMA cache_size=-4096')
        self.db.execute('PRAGMA mmap_size=0')
        self.db.execute('PRAGMA temp_store=FILE')
        try:
            if create:
                self.config = config or LSHConfig()
                c = self.config
                planes = np.random.default_rng(c.seed).normal(
                    size=(c.tables, c.bits, c.dimensions)).astype('<f4')
                with self.db:
                    self.db.execute('CREATE TABLE metadata(config TEXT, planes BLOB)')
                    self.db.execute('INSERT INTO metadata VALUES(?,?)',
                                    (json.dumps(asdict(c), sort_keys=True), planes.tobytes()))
                    self.db.execute('CREATE TABLE blocks(id INTEGER PRIMARY KEY, start INTEGER NOT NULL, end INTEGER NOT NULL, vector BLOB NOT NULL)')
                    self.db.execute('CREATE TABLE buckets(table_id INTEGER, signature INTEGER, priority BLOB, block_id INTEGER, PRIMARY KEY(table_id,signature,priority,block_id)) WITHOUT ROWID')
            encoded, raw = self.db.execute('SELECT config,planes FROM metadata').fetchone()
            self.config = LSHConfig(**json.loads(encoded))
            if config is not None and config != self.config:
                raise ValueError('persistent LSH configuration mismatch')
            c = self.config
            self.planes = np.frombuffer(raw, dtype='<f4').reshape(c.tables, c.bits, c.dimensions)
            if not np.isfinite(self.planes).all():
                raise ValueError('nonfinite LSH planes')
            self.last_search = {}
        except BaseException:
            self.db.close()
            raise

    def _normalized(self, vectors):
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.config.dimensions:
            raise ValueError('invalid vector dimensions')
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError('vectors must be finite and nonzero')
        return (values / norms).astype('<f4')

    def _signatures(self, vector):
        # Fixed reduction order avoids a BLAS dispatch change at batch boundaries.
        signs = np.einsum('tbd,d->tb', self.planes, vector, optimize=False) >= 0
        return (signs.astype(np.int64) * (1 << np.arange(self.config.bits))).sum(axis=1)

    def add(self, rows, vectors):
        """Append unique (block_id,start,end) records atomically per caller batch."""
        rows = list(rows)
        values = self._normalized(vectors)
        if len(rows) != len(values):
            raise ValueError('row/vector count mismatch')
        for block, start, end in rows:
            if any(type(v) is not int for v in (block, start, end)) or block < 0 or start < 0 or end <= start:
                raise ValueError('invalid KV address extent')
        with self.db:
            for (block, start, end), vector in zip(rows, values):
                self.db.execute('INSERT INTO blocks VALUES(?,?,?,?)', (block, start, end, vector.tobytes()))
                self.db.executemany('INSERT INTO buckets VALUES(?,?,?,?)',
                    [(table, int(sig), priority(block), block)
                     for table, sig in enumerate(self._signatures(vector))])

    def search(self, vector, k):
        c = self.config
        if type(k) is not int or not 0 < k <= c.rerank_cap:
            raise ValueError('invalid result budget')
        query = self._normalized(np.asarray(vector)[None, :])[0]
        votes = Counter()
        rows_read = truncated = probes = 0
        for table, signature in enumerate(self._signatures(query)):
            # Probe exact signature and Hamming distance one, in fixed order.
            for mask in [0] + [1 << bit for bit in range(c.bits)]:
                found = self.db.execute(
                    'SELECT block_id FROM buckets WHERE table_id=? AND signature=? ORDER BY priority,block_id LIMIT ?',
                    (table, int(signature) ^ mask, c.posting_cap + 1)).fetchall()
                rows_read += len(found)
                truncated += len(found) > c.posting_cap
                probes += 1
                votes.update(block for (block,) in found[:c.posting_cap])
        candidates = sorted(votes, key=lambda block: (-votes[block], priority(block), block))[:c.rerank_cap]
        scored = []
        for block in candidates:
            raw = self.db.execute('SELECT vector FROM blocks WHERE id=?', (block,)).fetchone()[0]
            value = np.frombuffer(raw, dtype='<f4')
            score = float(np.einsum('i,i->', value, query, optimize=False))
            if not np.isfinite(score):
                raise ValueError('nonfinite cosine score')
            scored.append((score, block))
        self.last_search = {'probes': probes, 'posting_rows_read': rows_read,
            'posting_rows_bound': c.tables*(c.bits+1)*(c.posting_cap+1),
            'truncated_buckets': truncated, 'unique_candidates': len(votes),
            'reranked_vectors': len(candidates), 'rerank_cap': c.rerank_cap,
            'vector_bytes_read': len(candidates)*c.dimensions*4,
            'plane_bytes': self.planes.nbytes, 'sqlite_cache_kib': 4096,
            'scan_fallback': False}
        return [block for _, block in sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:k]]

    def close(self):
        self.db.close()

"""Exact-semantic reference for the separate generation diagnostic.

The index returns KV addresses only. Encoder and lookup work happens inside
the timed request; archive/index preparation is separately reported. This is
a linear scan reference, not the proposed bounded deployment solution.
"""
import json
from pathlib import Path
import time

from research.semantic_index import DiskTextIndex, ENCODER_ID, ENCODER_REVISION
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.types import digest


def diagnostic_policies():
    rows = [dict(attention='stock', rounds=1, cap_per_round=128,
                 format='direct', schedule_labels=['direct_vanilla'])]
    for attention in ('stock', 'sparse'):
        for rounds in (1, 2):
            rows.append(dict(attention=attention, rounds=rounds, cap_per_round=128,
                             format='json', schedule_labels=['semantic_diagnostic_128_per_round']))
    for p in rows:
        prefix = 'sparse-semantic' if p['attention'] == 'sparse' else 'stock'
        p['id'] = f"{prefix}-{p['format']}-r{p['rounds']}-cap{p['cap_per_round']}"
    return rows


class SemanticAddressIndex:
    def __init__(self, index, encoder, database_sha256, blocks):
        self.index = index
        self.encoder = encoder
        self.database_sha256 = database_sha256
        self.blocks = blocks
        self.last_search = {}

    def search(self, query, vector, k, method):
        if method != 'semantic_exact' or vector is not None or k != 4:
            raise ValueError('semantic generation reference requires public-text query and four seed blocks')
        start = time.perf_counter()
        encoded = self.encoder.encode([query])[0]
        encode_s = time.perf_counter()-start
        start = time.perf_counter()
        selected = self.index.search(query, encoded, k, 'semantic')
        search_s = time.perf_counter()-start
        self.last_search = {'selector': 'semantic_exact', 'encoder_id': ENCODER_ID,
            'encoder_revision': ENCODER_REVISION, 'query_vector': encoded.tolist(),
            'encoder_s': encode_s, 'scan_s': search_s, 'scanned_blocks': self.blocks,
            'seed_blocks': selected, 'database_sha256': self.database_sha256,
            'linear_scan': True, 'gold_used_for_selection': False}
        return selected

    def close(self):
        self.index.close()


def semantic_metadata_index(tokenizer, example, root, encoder, encoder_weights_sha256):
    cutoff = example['prefix_tokens_count']-2000+1
    if cutoff <= 0:
        raise ValueError('semantic diagnostic needs a nonempty cold archive')
    path = Path(root)/(example['id']+'-semantic-cold.sqlite3')
    marker = path.with_suffix('.complete.json')
    expected = {'fixture_sha256': digest(example), 'cutoff': cutoff, 'padding': 32,
        'encoder_id': ENCODER_ID, 'encoder_revision': ENCODER_REVISION,
        'encoder_weights_sha256': encoder_weights_sha256}
    if path.exists() and not marker.exists():
        raise RuntimeError('incomplete semantic index retained: '+str(path))
    if marker.exists() and not path.exists():
        raise RuntimeError('semantic index payload missing: '+str(path))
    if not path.exists():
        index = DiskTextIndex(path, create=True)
        try:
            for offset in range(0, cutoff, 16*128):
                rows = [(s//128, s, min(s+128, cutoff), tokenizer.decode(example['tokens'][
                    max(0, s-32):min(s+160, cutoff)], skip_special_tokens=False))
                    for s in range(offset, min(offset+16*128, cutoff), 128)]
                index.add(rows, encoder.encode([r[3] for r in rows]))
        finally:
            index.close()
        atomic_json(marker, {**expected, 'database_sha256': file_sha(path)})
    saved = json.loads(marker.read_text())
    if saved != {**expected, 'database_sha256': file_sha(path)}:
        raise ValueError('semantic index identity/checksum differs')
    return SemanticAddressIndex(DiskTextIndex(path), encoder, saved['database_sha256'], (cutoff+127)//128)

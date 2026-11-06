"""Synthetic ANN work/recall screen; not semantic or generator quality."""
from dataclasses import asdict
import heapq
import json
from pathlib import Path
import platform
import sqlite3
import time

import numpy as np

from research.lsh_index import DiskLSHIndex, LSHConfig
from research.resources import HostObservation
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.types import digest


def exact(index, query, k):
    query = index._normalized(np.asarray(query)[None, :])[0]
    best = []
    rows_read = 0
    cursor = index.db.execute('SELECT id,vector FROM blocks ORDER BY id')
    while rows := cursor.fetchmany(64):
        values = np.stack([np.frombuffer(raw, dtype='<f4') for _, raw in rows])
        scores = np.einsum('ij,j->i', values, query, optimize=False)
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite exact reference')
        for (block, _), score in zip(rows, scores):
            heapq.heappush(best, (float(score), -block))
            if len(best) > k:
                heapq.heappop(best)
        rows_read += len(rows)
    return [-block for _, block in sorted(best, reverse=True)], {'vectors_read': rows_read}


def main():
    config = LSHConfig()
    identity = {'config': asdict(config), 'blocks': [32, 256, 2048, 8192, 32768],
        'notional_tokens_per_block': 128, 'trials': 7, 'archive_seed': 263,
        'query_seed': 257, 'query_cosines': [.95, .8, .5, 0.],
        'target_positions': ['first', 'middle', 'last'], 'result_budget': 16,
        'scope': 'isotropic synthetic vectors only; no text encoding, KV payload or model inference',
        'cache': 'one warm lookup then seven trials; OS cache uncontrolled',
        'platform': platform.platform(), 'numpy': np.__version__, 'sqlite': sqlite3.sqlite_version,
        'index_sha256': file_sha('research/lsh_index.py'), 'script_sha256': file_sha(__file__),
        'reference': 'stream all stored vectors in batches of 64; identical normalization and float32 dot product'}
    run = digest(identity)[:12]
    root = Path('artifacts/lsh-scaling')/run
    root.mkdir(parents=True, exist_ok=False)
    output = Path('results/evidence')/('lsh-scaling-'+run+'.json')
    index = DiskLSHIndex(root/'index.sqlite3', create=True, config=config)
    rng = np.random.default_rng(identity['archive_seed'])
    qrng = np.random.default_rng(identity['query_seed'])
    rows = []
    stages = []
    previous = 0
    try:
        for count in identity['blocks']:
            start = time.perf_counter()
            for offset in range(previous, count, 128):
                n = min(128, count-offset)
                index.add([(i, i*128, (i+1)*128) for i in range(offset, offset+n)],
                          rng.normal(size=(n, config.dimensions)))
            stages.append({'blocks': count, 'append_s': time.perf_counter()-start,
                           'database_bytes': (root/'index.sqlite3').stat().st_size})
            previous = count
            for target in (0, count//2, count-1):
                raw = index.db.execute('SELECT vector FROM blocks WHERE id=?', (target,)).fetchone()[0]
                value = np.frombuffer(raw, dtype='<f4').astype(np.float64)
                value /= np.linalg.norm(value)
                for cosine in identity['query_cosines']:
                    orthogonal = qrng.normal(size=config.dimensions)
                    orthogonal -= np.dot(value, orthogonal)*value
                    orthogonal /= np.linalg.norm(orthogonal)
                    query = (cosine*value + np.sqrt(1-cosine**2)*orthogonal).astype(np.float32)
                    reference, _ = exact(index, query, 16)
                    for method in ('exact', 'lsh'):
                        def lookup():
                            if method == 'exact':
                                return exact(index, query, 16)
                            selected = index.search(query, 16)
                            return selected, dict(index.last_search)
                        lookup()
                        times = []
                        with HostObservation(scope='seven warm index queries, same process; no encoder or generator') as host:
                            for _ in range(identity['trials']):
                                started = time.perf_counter()
                                selected, details = lookup()
                                times.append((time.perf_counter()-started)*1000)
                        if method == 'exact' and selected != reference:
                            raise AssertionError('reference not repeatable')
                        if method == 'lsh':
                            assert details['posting_rows_read'] <= details['posting_rows_bound']
                            assert details['reranked_vectors'] <= config.rerank_cap
                        rows.append({'blocks': count, 'target': target, 'query_cosine': cosine,
                            'query': query.tolist(), 'method': method, 'selected': selected,
                            'exact_selected': reference, 'recall_at_4': len(set(selected[:4]) & set(reference[:4]))/4,
                            'recall_at_16': len(set(selected) & set(reference))/16,
                            'target_retrieved': target in selected, 'target_in_exact': target in reference,
                            'latency_ms': times, 'work': details, **host.metrics})
                atomic_json(output, {'identity': identity, 'status': 'running', 'stages': stages, 'rows': rows})
            print(json.dumps({'blocks': count, 'rows': len(rows), 'median_lsh_ms': float(np.median([
                t for r in rows if r['blocks'] == count and r['method'] == 'lsh' for t in r['latency_ms']]))}), flush=True)
    finally:
        index.close()
    atomic_json(output, {'identity': identity, 'status': 'complete', 'stages': stages, 'rows': rows,
                         'database_sha256': file_sha(root/'index.sqlite3')})
    print(output, flush=True)


if __name__ == '__main__':
    main()

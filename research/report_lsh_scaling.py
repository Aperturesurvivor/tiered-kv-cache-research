"""Audit saved synthetic screen against independent float64 cosine rankings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

import numpy as np

from tiered_kv.store import atomic_json, file_sha
from tiered_kv.types import digest


def audit(path):
    data = json.loads(path.read_text())
    identity = data['identity']
    assert data['status'] == 'complete'
    assert file_sha('research/lsh_index.py') == identity['index_sha256']
    assert file_sha('research/lsh_scaling.py') == identity['script_sha256']
    run = digest(identity)[:12]
    database = Path('artifacts/lsh-scaling')/run/'index.sqlite3'
    assert file_sha(database) == data['database_sha256']
    db = sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True)
    records = db.execute('SELECT id,start,end,vector FROM blocks ORDER BY id').fetchall()
    encoded, raw_planes = db.execute('SELECT config,planes FROM metadata').fetchone()
    config = json.loads(encoded)
    assert config == identity['config']
    planes = np.frombuffer(raw_planes, dtype='<f4').reshape(config['tables'], config['bits'], config['dimensions']).astype(np.float64)
    assert [(r[0], r[1], r[2]) for r in records] == [(i, i*128, (i+1)*128) for i in range(max(identity['blocks']))]
    values = np.stack([np.frombuffer(r[3], dtype='<f4') for r in records]).astype(np.float64)
    rng = np.random.default_rng(identity['archive_seed'])
    expected = rng.normal(size=values.shape)
    expected = (expected/np.linalg.norm(expected, axis=1, keepdims=True)).astype('<f4')
    np.testing.assert_array_equal(values, expected)
    keys = [(r['blocks'], r['target'], r['query_cosine'], r['method']) for r in data['rows']]
    wanted = [(n, t, c, m) for n in identity['blocks'] for t in (0, n//2, n-1)
              for c in identity['query_cosines'] for m in ('exact', 'lsh')]
    assert len(keys) == len(wanted) and set(keys) == set(wanted)
    for row in data['rows']:
        query = np.asarray(row['query'], dtype=np.float64)
        assert query.shape == (identity['config']['dimensions'],) and np.isfinite(query).all()
        query /= np.linalg.norm(query)
        scores = np.einsum('ij,j->i', values[:row['blocks']], query, optimize=False)
        assert np.isfinite(scores).all()
        reference = np.argsort(-scores, kind='stable')[:16].tolist()
        assert reference == row['exact_selected']
        assert abs(scores[row['target']] - row['query_cosine']) < 1e-6
        selected = row['selected']
        assert len(selected) == len(set(selected)) <= 16
        assert all(type(i) is int and 0 <= i < row['blocks'] for i in selected)
        assert selected == sorted(selected, key=lambda i: (-scores[i], i))
        assert row['recall_at_4'] == len(set(selected[:4]) & set(reference[:4]))/4
        assert row['recall_at_16'] == len(set(selected) & set(reference))/16
        assert row['target_in_exact'] == (row['target'] in reference)
        assert row['target_retrieved'] == (row['target'] in selected)
        assert len(row['latency_ms']) == identity['trials']
        assert all(np.isfinite(t) and t > 0 for t in row['latency_ms'])
        if row['method'] == 'exact':
            assert selected == reference and row['work']['vectors_read'] == row['blocks']
        else:
            c = identity['config']
            w = row['work']
            assert w['probes'] == c['tables']*(c['bits']+1)
            assert w['posting_rows_bound'] == w['probes']*(c['posting_cap']+1)
            assert w['posting_rows_read'] <= w['posting_rows_bound']
            assert w['unique_candidates'] <= w['probes']*c['posting_cap']
            assert w['reranked_vectors'] == min(w['unique_candidates'], c['rerank_cap'])
            assert w['vector_bytes_read'] == w['reranked_vectors']*c['dimensions']*4
            assert not w['scan_fallback']
    # Post-hoc failure attribution for the final archive, using float64 plane
    # projections and independently reconstructed candidate votes/reranking.
    diagnosis = []
    for row in data['rows']:
        if row['blocks'] != max(identity['blocks']) or row['method'] != 'lsh':
            continue
        query = np.asarray(row['query'], dtype=np.float64)
        query /= np.linalg.norm(query)
        signatures = ((np.einsum('tbd,d->tb', planes, query, optimize=False) >= 0)
                      * (1 << np.arange(config['bits']))).sum(axis=1)
        votes = Counter()
        eligible = survived = read = truncated = 0
        for table, signature in enumerate(signatures):
            for mask in [0]+[1 << bit for bit in range(config['bits'])]:
                bindings = (table, int(signature)^mask)
                eligible += db.execute('SELECT count(*) FROM buckets WHERE table_id=? AND signature=? AND block_id=?', bindings+(row['target'],)).fetchone()[0]
                found = [r[0] for r in db.execute('SELECT block_id FROM buckets WHERE table_id=? AND signature=? ORDER BY priority,block_id LIMIT ?', bindings+(config['posting_cap']+1,))]
                read += len(found)
                truncated += len(found) > config['posting_cap']
                retained = found[:config['posting_cap']]
                survived += row['target'] in retained
                votes.update(retained)
        ranked = sorted(votes, key=lambda b: (-votes[b], hashlib.sha256(str(b).encode('ascii')).digest()[:8], b))
        candidates = ranked[:config['rerank_cap']]
        scores = np.einsum('ij,j->i', values, query, optimize=False)
        assert sorted(candidates, key=lambda b: (-scores[b], b))[:16] == row['selected']
        assert read == row['work']['posting_rows_read']
        assert truncated == row['work']['truncated_buckets']
        assert len(votes) == row['work']['unique_candidates']
        if not row['target_in_exact']:
            continue
        reason = ('retrieved' if row['target_retrieved'] else 'no_probed_bucket' if not eligible
                  else 'posting_truncation' if not survived else 'rerank_cap' if row['target'] not in candidates
                  else 'cosine_topk')
        diagnosis.append({'target': row['target'], 'cosine': row['query_cosine'],
            'eligible_buckets': eligible, 'surviving_votes': survived,
            'candidate_rank': ranked.index(row['target'])+1 if row['target'] in votes else None,
            'outcome': reason})
    db.close()
    aggregate = []
    for count in identity['blocks']:
        rows = [r for r in data['rows'] if r['blocks'] == count]
        aggregate.append({'blocks': count,
            'median_ms': {m: statistics.median(t for r in rows if r['method'] == m for t in r['latency_ms']) for m in ('exact', 'lsh')},
            'mean_recall_at_16': statistics.mean(r['recall_at_16'] for r in rows if r['method'] == 'lsh'),
            'target_hits_by_cosine': {str(c): {
                'hits': sum(r['target_retrieved'] and r['target_in_exact'] for r in rows if r['method'] == 'lsh' and r['query_cosine'] == c),
                'eligible': sum(r['target_in_exact'] for r in rows if r['method'] == 'lsh' and r['query_cosine'] == c)} for c in identity['query_cosines']}})
    return {'status': 'passed', 'rows': len(data['rows']), 'raw_sha256': file_sha(path),
        'database_sha256': file_sha(database), 'auditor_sha256': file_sha(__file__),
        'independent_reference': 'float64 normalized dense cosine; exact top16 and all selection scores checked',
        'limits': 'No independent retiming. Bucket candidate generation replay covers only the final archive; cosine-reference and metric audits cover every size.',
        'final_archive_target_diagnosis': diagnosis,
        'aggregate': aggregate}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    result = audit(args.result)
    atomic_json(args.result.with_suffix('.audit.json'), result)
    print(json.dumps(result, indent=2))

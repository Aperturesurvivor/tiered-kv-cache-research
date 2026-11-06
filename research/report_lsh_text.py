"""Replay text-index selections and independently score exact record coverage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

import numpy as np
import torch
from transformers import AutoTokenizer

from research.reasoning_protocol import make_fixture
from research.semantic_index import DiskTextIndex, Encoder
from tiered_kv.config import MODEL_PATH
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.types import digest


def audit(root):
    manifest = json.loads((root/'manifest.json').read_text())
    fixtures = json.loads((root/'fixtures.json').read_text())
    cases = json.loads((root/'cases.json').read_text())
    rows = json.loads((root/'samples.json').read_text())
    marker = json.loads((root/'complete.json').read_text())
    assert marker['cases'] == len(cases) == len(fixtures) == 16
    assert marker['rows'] == len(rows) == 96
    assert digest(fixtures) == manifest['fixtures_digest']
    assert file_sha(root/'fixtures.json') == manifest['fixtures_sha256']
    for name, sha in manifest['source_hashes'].items():
        assert file_sha(name) == sha
    keys = [(r['example_id'], r['method'], r['k']) for r in rows]
    assert len(set(keys)) == len(keys)
    assert set(keys) == {(e['id'], m, k) for e in fixtures for m in manifest['methods'] for k in manifest['k']}
    assert len({c['example_id'] for c in cases}) == len(cases)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    encoder = Encoder()
    max_encoder_error = 0.
    for ex in fixtures:
        assert digest(make_fixture(tokenizer, ex['budget'], ex['seed'], ex['task'])) == digest(ex)
        case = next(c for c in cases if c['example_id'] == ex['id'])
        assert case['cutoff'] == ex['prefix_tokens_count']-manifest['recent_window']+1
        cutoff = case['cutoff']
        for field in ('metadata', 'lsh'):
            assert file_sha(case[field+'_path']) == case[field+'_sha256']
        with sqlite3.connect(Path(case['metadata_path']).resolve().as_uri()+'?mode=ro', uri=True) as db:
            metadata = db.execute('SELECT id,start,end,text,vector FROM chunks ORDER BY id').fetchall()
        with sqlite3.connect(Path(case['lsh_path']).resolve().as_uri()+'?mode=ro', uri=True) as db:
            blocks = db.execute('SELECT id,start,end,vector FROM blocks ORDER BY id').fetchall()
            encoded, raw = db.execute('SELECT config,planes FROM metadata').fetchone()
            postings = db.execute('SELECT table_id,signature,priority,block_id FROM buckets ORDER BY table_id,signature,priority,block_id').fetchall()
        config = json.loads(encoded)
        assert config == manifest['config']
        assert [b[:3] for b in blocks] == [b[:3] for b in metadata] == [(s//128, s, min(s+128, cutoff)) for s in range(0, cutoff, 128)]
        for block, start, end, text, _ in metadata:
            assert text == tokenizer.decode(ex['tokens'][max(0, start-32):min(start+160, cutoff)], skip_special_tokens=False)
        values = np.stack([np.frombuffer(b[3], dtype='<f4') for b in blocks])
        original = np.stack([np.frombuffer(b[4], dtype='<f4') for b in metadata]).astype(np.float64)
        np.testing.assert_array_equal(values, (original/np.linalg.norm(original, axis=1, keepdims=True)).astype('<f4'))
        query = np.asarray(case['query_vector'], dtype=np.float64)
        checked = encoder.encode([ex['query'], metadata[0][3], metadata[-1][3]])
        expected = np.stack([query, original[0], original[-1]])
        error = float(np.max(np.abs(checked-expected)))
        assert error < 2e-6
        max_encoder_error = max(error, max_encoder_error)
        query = (query/np.linalg.norm(query)).astype('<f4')
        planes = np.frombuffer(raw, dtype='<f4').reshape(config['tables'], config['bits'], config['dimensions'])
        expected_planes = np.random.default_rng(config['seed']).normal(size=planes.shape).astype('<f4')
        np.testing.assert_array_equal(planes, expected_planes)
        def signatures(vector):
            return ((np.einsum('tbd,d->tb', planes.astype(np.float64), vector.astype(np.float64), optimize=False) >= 0)
                    * (1 << np.arange(config['bits']))).sum(axis=1)
        bucket_map = {}
        priority = lambda b: hashlib.sha256(str(b).encode('ascii')).digest()[:8]
        expected_postings = set()
        for b, value in enumerate(values):
            expected_postings.update((t, int(s), priority(b), b) for t, s in enumerate(signatures(value)))
        assert set(postings) == expected_postings and len(postings) == len(expected_postings)
        for table, sig, _, block in postings:
            bucket_map.setdefault((table, sig), []).append(block)
        votes = Counter()
        posting_rows = truncated = 0
        for table, sig in enumerate(signatures(query)):
            for mask in [0]+[1 << bit for bit in range(config['bits'])]:
                found = bucket_map.get((table, int(sig)^mask), [])
                posting_rows += min(len(found), config['posting_cap']+1)
                truncated += len(found) > config['posting_cap']
                votes.update(found[:config['posting_cap']])
        candidates = sorted(votes, key=lambda b: (-votes[b], priority(b), b))[:config['rerank_cap']]
        # Float32 reference reproduces declared tie behavior; float64 comparison
        # independently checks every computed cosine within 1e-6.
        scores = np.einsum('ij,j->i', values, query, optimize=False)
        scores64 = np.einsum('ij,j->i', values.astype(np.float64), query.astype(np.float64), optimize=False)
        np.testing.assert_allclose(scores, scores64, atol=1e-6, rtol=1e-6)
        reference = sorted(range(len(values)), key=lambda b: (-float(scores[b]), b))
        approximate = sorted(candidates, key=lambda b: (-float(scores[b]), b))
        lexical = DiskTextIndex(case['metadata_path'])
        try:
            for row in [r for r in rows if r['example_id'] == ex['id']]:
                k = row['k']
                if row['method'] == 'semantic_exact':
                    selected = reference[:k]
                    assert row['work']['vectors_read'] == len(values)
                elif row['method'] == 'lsh':
                    selected = approximate[:k]
                    assert row['work']['posting_rows_read'] == posting_rows
                    assert row['work']['truncated_buckets'] == truncated
                    assert row['work']['reranked_vectors'] == len(candidates)
                else:
                    selected = lexical.search(ex['query'], None, k, 'lexical_rare')
                    assert row['work'] == lexical.last_search
                assert row['selected'] == selected
                assert row['exact_selected'] == reference[:k]
                assert row['neighbor_recall'] == len(set(selected) & set(reference[:k]))/len(reference[:k])
                assert len(row['lookup_ms']) == 7 and all(np.isfinite(t) and t > 0 for t in row['lookup_ms'])
                assert row['query_encode_s'] == (0. if row['method'] == 'lexical_rare' else case['query_encode_s'])
                available = []
                retrieved = []
                working = []
                for i in ex['required_record_indices']:
                    start, end = ex['record_spans'][i]
                    got = [any(b*128 <= pos < min((b+1)*128, cutoff) for b in selected) for pos in range(start, end)]
                    hot = [pos < 128 or cutoff <= pos < ex['prefix_tokens_count'] for pos in range(start, end)]
                    retrieved.append(sum(got)/(end-start))
                    available.append(sum(a or b for a, b in zip(got, hot))/(end-start))
                    working.append(all(hot))
                assert row['retrieved_record_fractions'] == retrieved
                assert row['available_record_fractions'] == available
                assert row['all_required_records_available'] == all(v == 1 for v in available)
                assert row['all_required_records_already_working'] == all(working)
        finally:
            lexical.close()
    aggregate = []
    for budget in (8192, 32768):
        for k in (4, 16):
            for method in manifest['methods']:
                selected = [r for r in rows if (r['budget'], r['k'], r['method']) == (budget, k, method)]
                aggregate.append({'budget': budget, 'k': k, 'method': method, 'cases': len(selected),
                    'whole_records_available': sum(r['all_required_records_available'] for r in selected),
                    'mean_neighbor_recall': statistics.mean(r['neighbor_recall'] for r in selected),
                    'median_lookup_ms': statistics.median(t for r in selected for t in r['lookup_ms']),
                    'median_encode_ms': 1000*statistics.median(r['query_encode_s'] for r in selected)})
    return {'status': 'passed', 'rows': len(rows), 'fixtures_regenerated': len(fixtures),
        'query_and_edge_block_embeddings_reencoded': 3*len(fixtures), 'max_embedding_error': max_encoder_error,
        'all_block_text_and_address_extents_verified': True, 'all_lsh_postings_and_selections_replayed': True,
        'scope': 'diagnostic index coverage; no generator quality or independent retiming',
        'auditor_sha256': file_sha(__file__), 'raw_hashes': {n: file_sha(root/n) for n in
            ('manifest.json', 'fixtures.json', 'cases.json', 'samples.json', 'complete.json')},
        'aggregate': aggregate}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    atomic_json(args.root/'audit.json', result)
    print(json.dumps(result, indent=2))

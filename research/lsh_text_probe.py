"""Diagnostic text-embedding retrieval on the saved reasoning fixtures.

Gold record spans are used only after address selection. No model generation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from research.lsh_index import DiskLSHIndex, LSHConfig
from research.lsh_scaling import exact
from research.resources import HostObservation
from research.semantic_index import DiskTextIndex, Encoder, ENCODER_ID, ENCODER_REVISION
from tiered_kv.config import MODEL_PATH, MODEL_REVISION
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.types import digest


def record_coverage(example, selected, cutoff):
    """Only exact payload block positions count; embedding padding adds none."""
    prefix = example['prefix_tokens_count']
    if any(type(b) is not int or not 0 <= b < (cutoff+127)//128 for b in selected):
        raise ValueError('invalid retrieved block')
    retrieved = {p for b in selected for p in range(b*128, min((b+1)*128, cutoff))}
    working = set(range(cutoff, prefix)) | set(range(min(128, prefix)))
    required = [set(range(*example['record_spans'][i])) for i in example['required_record_indices']]
    if not required or any(not s or min(s)<0 or max(s)>=prefix for s in required):
        raise ValueError('invalid required record')
    return {'required_records': len(required),
        'retrieved_record_fractions': [len(s & retrieved)/len(s) for s in required],
        'available_record_fractions': [len(s & (retrieved | working))/len(s) for s in required],
        'all_required_records_available': all(s <= (retrieved | working) for s in required),
        'all_required_records_already_working': all(s <= working for s in required)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, default=Path('results/evidence/reasoning-Qwen--Qwen2.5-1.5B-Instruct-8862fb5f28f2/fixtures.json'))
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    assert len(fixtures) == 16
    identity = {'fixtures_sha256': file_sha(args.fixtures), 'fixtures_digest': digest(fixtures),
        'config': asdict(LSHConfig()), 'encoder_id': ENCODER_ID, 'encoder_revision': ENCODER_REVISION,
        'generator_tokenizer_revision': MODEL_REVISION, 'encoder_device': 'cpu', 'encoder_dtype': 'float32',
        'block_size': 128, 'text_padding': 32, 'recent_window': 2000, 'sink_tokens': 128,
        'query_policy': 'original public question only, no gold-assisted rewrite',
        'k': [4, 16], 'trials': 7, 'schedule_seed': 277,
        'methods': ['lexical_rare', 'semantic_exact', 'lsh'],
        'scope': 'post-hoc diagnostic on reused reasoning fixtures; no generated answers',
        'cache': 'one warmup per method/budget, then seven warm trials; OS cache uncontrolled',
        'platform': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__,
        'source_hashes': {p: file_sha(p) for p in ['research/lsh_index.py', 'research/lsh_scaling.py',
            'research/semantic_index.py', 'research/lsh_text_probe.py']}}
    run = digest(identity)[:12]
    root = Path('artifacts/lsh-text')/run
    output = Path('results/evidence')/('lsh-text-'+run)
    root.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'manifest.json', identity)
    atomic_json(output/'fixtures.json', fixtures)
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    encoder = Encoder()
    schedule = random.Random(identity['schedule_seed'])
    rows = []
    cases = []
    for example in fixtures:
        assert digest(example['tokens']) == example['token_hash']
        prefix = example['prefix_tokens_count']
        cutoff = max(0, prefix-identity['recent_window']+1)
        assert cutoff > 0 and prefix <= len(example['tokens'])
        dbpath = root/(example['id']+'.metadata.sqlite3')
        lshpath = root/(example['id']+'.lsh.sqlite3')
        metadata = DiskTextIndex(dbpath, create=True)
        index = DiskLSHIndex(lshpath, create=True)
        try:
            started = time.perf_counter()
            for offset in range(0, cutoff, 16*128):
                blocks = [(s//128, s, min(s+128, cutoff), tok.decode(example['tokens'][
                    max(0, s-32):min(s+128+32, cutoff)], skip_special_tokens=False))
                    for s in range(offset, min(offset+16*128, cutoff), 128)]
                vectors = encoder.encode([b[3] for b in blocks])
                metadata.add(blocks, vectors)
                index.add([b[:3] for b in blocks], vectors)
            build_s = time.perf_counter()-started
            started = time.perf_counter()
            query = encoder.encode([example['query']])[0]
            query_s = time.perf_counter()-started
            reference, _ = exact(index, query, 16)
            case = {'example_id': example['id'], 'cutoff': cutoff, 'build_both_indexes_s': build_s,
                'query_encode_s': query_s, 'query_vector': query.tolist(),
                'metadata_path': str(dbpath), 'lsh_path': str(lshpath),
                'metadata_sha256': file_sha(dbpath), 'lsh_sha256': file_sha(lshpath),
                'metadata_bytes': dbpath.stat().st_size, 'lsh_bytes': lshpath.stat().st_size}
            cases.append(case)
            order = [(method, k) for method in identity['methods'] for k in identity['k']]
            schedule.shuffle(order)
            for method, k in order:
                def lookup():
                    if method == 'semantic_exact':
                        return exact(index, query, k)
                    if method == 'lsh':
                        selected = index.search(query, k)
                        return selected, dict(index.last_search)
                    selected = metadata.search(example['query'], None, k, 'lexical_rare')
                    return selected, dict(metadata.last_search)
                lookup()
                times = []
                with HostObservation(scope='seven warm text-index lookups; encoder/build/generation excluded') as host:
                    for _ in range(identity['trials']):
                        started = time.perf_counter()
                        selected, work = lookup()
                        times.append((time.perf_counter()-started)*1000)
                rows.append({'example_id': example['id'], 'budget': example['budget'], 'task': example['task'],
                    'method': method, 'k': k, 'selected': selected, 'exact_selected': reference[:k],
                    'neighbor_recall': len(set(selected) & set(reference[:k]))/len(reference[:k]),
                    'lookup_ms': times, 'query_encode_s': 0. if method == 'lexical_rare' else query_s,
                    'work': work, **record_coverage(example, selected, cutoff), **host.metrics})
            atomic_json(output/'cases.json', cases)
            atomic_json(output/'samples.json', rows)
            print(json.dumps({'example': example['id'], 'rows': len(rows), 'build_s': build_s}), flush=True)
        finally:
            metadata.close()
            index.close()
    atomic_json(output/'complete.json', {'rows': len(rows), 'cases': len(cases), 'scope': identity['scope']})
    print(output, flush=True)


if __name__ == '__main__':
    main()

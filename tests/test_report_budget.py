"""Reject incomplete or mismatched diagnostic evidence before reporting it."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from research.budget_probe import plan, SOURCE
from research.report_budget import validate_rows, report
from tiered_kv.store import file_sha


@pytest.fixture
def observations():
    source = Path(__file__).resolve().parents[1]/'results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235'
    _, examples, baseline, configurations = plan(source)
    rows = []
    # Synthetic reporting inputs only, not generator observations or evidence.
    for ex in examples:
        for cfg in configurations[ex['id']]:
            row = deepcopy(baseline[ex['id']])
            row.update(config_id=cfg.id, policy=asdict(cfg), post_hoc=True,
                       original_token_ids_reproduced=True)
            rows.append(row)
    for trial in range(3):
        for cfg in [*configurations['niah-32768-0.5-53'], None]:
            rows.append({'config_id': cfg.id if cfg else 'stock_hf',
                         'example_id': f'niah-32768-0.5-53-budget-forced-{trial}',
                         'budget': 32768, 'task': 'latency', 'status': 'ok', 'post_hoc': True,
                         'trial': trial, 'policy': asdict(cfg) if cfg else None,
                         'latency_ms': [1.]*112, 'all_latency_ms': [1.]*128})
    marker = {'rows': 48, 'weights_unchanged': True, 'original_k4_token_sequences_reproduced': 12}
    return examples, baseline, configurations, rows, marker


def test_accepts_exact_declared_matrix(observations):
    result = validate_rows(*observations)
    assert result == {'quality_rows': 36, 'timing_rows': 12, 'original_k4_token_sequences_reproduced': 12}


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'selection', 'reproduction', 'timing', 'score', 'marker'])
def test_rejects_false_complete_or_mismatched_comparisons(observations, mutation):
    examples, baseline, configurations, rows, marker = observations
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1] = deepcopy(rows[-2])
    elif mutation == 'selection':
        rows[1]['policy']['fixed_blocks'] = (0,)
    elif mutation == 'reproduction':
        rows[0]['token_ids'] = [0]
    elif mutation == 'timing':
        rows[-1]['latency_ms'].pop()
    elif mutation == 'score':
        rows[1]['code_found'] = 100-rows[1]['code_found']
    elif mutation == 'marker':
        marker['weights_unchanged'] = False
    with pytest.raises(ValueError):
        validate_rows(examples, baseline, configurations, rows, marker)


def test_report_keeps_original_quality_reference_separate(tmp_path, observations):
    examples, baseline, configurations, rows, marker = observations
    source = Path(__file__).resolve().parents[1]/'results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235'
    original = json.loads((source/'manifest.json').read_text())
    source_validation = json.loads((source/'validation.json').read_text())
    manifest = {'source_revision': SOURCE, 'expected_rows': 48, 'order_seed': 137,
                'implementation_hash': original['implementation_hash'], 'runtime': original['runtime'],
                'source_files_sha256': {name: file_sha(source/name) for name in
                                       ('manifest.json', 'fixtures.json', 'samples.jsonl', 'complete.json', 'validation.json')},
                'selections': {key: [asdict(cfg) for cfg in value] for key, value in configurations.items()}}
    for row in rows:
        if row['task'] == 'latency':
            row['cuda_decode_peak_allocated_bytes'] = 2**30
    for name, value in {'manifest.json': manifest, 'fixtures.json': {'examples': examples},
                        'complete.json': marker, 'validation.json': {
                            'weights_sha256': source_validation['weights_sha256'],
                            'parity': next(p for p in source_validation['parity'] if p['length'] == 512)}}.items():
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary = report(tmp_path, source)
    assert summary['original_full_quality'] == {'cases': 12, 'code_correct': 12}
    assert {r['policy'] for r in summary['quality']} == {'rare_context', 'rare_context_k16', 'rare_context_k32'}
    assert len(summary['factual_cases']) == 36
    assert len(summary['latency_trial_ranges']) == 4
    assert 'not held-out policy validation' in (tmp_path/'report.md').read_text()

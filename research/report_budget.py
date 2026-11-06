"""Audit and report the predeclared post-hoc neighboring-context budget probe."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from research.budget_probe import plan, SOURCE
from research.report_demand import trial_ranges
from tiered_kv.evals import score_needle
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.types import digest


def validate_rows(examples, baseline, configurations, rows, marker):
    expected = {}
    for ex in examples:
        for cfg in configurations[ex['id']]:
            expected[cfg.id, ex['id']] = ('niah', cfg)
    timing = 'niah-32768-0.5-53'
    for trial in range(3):
        for cfg in configurations[timing]:
            expected[cfg.id, timing+f'-budget-forced-{trial}'] = ('latency', cfg)
        expected['stock_hf', timing+f'-budget-forced-{trial}'] = ('latency', None)
    keys = [(r['config_id'], r['example_id']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected) or len(rows) != 48:
        raise ValueError('budget probe lacks its exact 48 distinct declared observations')
    if marker != {'rows': 48, 'weights_unchanged': True,
                  'original_k4_token_sequences_reproduced': 12}:
        raise ValueError('budget probe completion claims differ')
    for row in rows:
        task, cfg = expected[row['config_id'], row['example_id']]
        if (row['task'] != task or row['status'] != 'ok' or row.get('post_hoc') is not True
                or row['budget'] != 32768):
            raise ValueError('failed or mislabeled budget observation')
        if cfg is not None and digest(row['policy']) != digest(asdict(cfg)):
            raise ValueError('observed selection differs from predeclared expansion')
        if task == 'niah' and cfg.k == 4:
            original = baseline[row['example_id']]
            if (row['token_ids'] != original['token_ids']
                    or row.get('original_token_ids_reproduced') is not True):
                raise ValueError('original k4 token sequence did not reproduce')
        if task == 'niah':
            ex = next(ex for ex in examples if ex['id'] == row['example_id'])
            rescored = score_needle(row['prediction'], ex['answer'])
            if any(row[key] != rescored[key] for key in ('em', 'code_found')):
                raise ValueError('factual score differs from saved prediction')
        if task == 'latency':
            trial = int(row['example_id'].rsplit('-', 1)[1])
            if (row.get('trial') != trial or len(row['latency_ms']) != 112
                    or len(row['all_latency_ms']) != 128):
                raise ValueError('timing trial or matched forward-call counts differ')
    return {'quality_rows': 36, 'timing_rows': 12, 'original_k4_token_sequences_reproduced': 12}


def report(run, source_run):
    run, source_run = Path(run), Path(source_run)
    source, examples, baseline, configurations = plan(source_run)
    manifest = json.loads((run/'manifest.json').read_text())
    fixture = json.loads((run/'fixtures.json').read_text())
    marker = json.loads((run/'complete.json').read_text())
    validation = json.loads((run/'validation.json').read_text())
    rows = [json.loads(line) for line in (run/'samples.jsonl').read_text().splitlines()]
    if (manifest['source_revision'] != SOURCE or manifest['expected_rows'] != 48
            or manifest['order_seed'] != 137
            or manifest['implementation_hash'] != source['implementation_hash']
            or manifest['runtime'] != source['runtime'] or fixture != {'examples': examples}):
        raise ValueError('source, runtime, or complete fixture identity differs')
    if set(manifest['source_files_sha256']) != {'manifest.json', 'fixtures.json', 'samples.jsonl',
                                               'complete.json', 'validation.json'}:
        raise ValueError('reference hashes do not cover the declared source files')
    for name, sha in manifest['source_files_sha256'].items():
        if Path(name).name != name or file_sha(source_run/name) != sha:
            raise ValueError('immutable reference file differs: '+name)
    selections = {key: [asdict(cfg) for cfg in value] for key, value in configurations.items()}
    if digest(selections) != digest(manifest['selections']):
        raise ValueError('declared selections differ from original-query seed expansion')
    original_validation = json.loads((source_run/'validation.json').read_text())
    parity = validation['parity']
    if (validation['weights_sha256'] != original_validation['weights_sha256']
            or parity['length'] != 512 or len(parity['max_errors']) != 17
            or len(parity['greedy_agreement']) != 17 or max(parity['max_errors']) > 1e-4
            or not all(parity['greedy_agreement']) or len(parity['same_chunk_stock']) != 20
            or max(r['max_abs'] for r in parity['same_chunk_stock']) > 1e-4):
        raise ValueError('weight identity or matched-schedule parity check failed')
    audit = validate_rows(examples, baseline, configurations, rows, marker)
    references = [json.loads(line) for line in (source_run/'samples.jsonl').read_text().splitlines()]
    full = {r['example_id']: r for r in references if r['config_id'] == 'full_buffered'
            and r['example_id'] in configurations}
    if len(full) != 12:
        raise ValueError('original full-cache quality reference missing')
    factual = []
    for row in rows:
        if row['task'] != 'niah':
            continue
        ex = next(ex for ex in examples if ex['id'] == row['example_id'])
        cfg = row['policy']
        cutoff = max(0, ex['prefix_tokens_count']-cfg['W']+1)
        ids = set(range(cfg['sink_tokens'])) | set(range(cutoff, ex['prefix_tokens_count']))
        for block in cfg['fixed_blocks']:
            ids.update(range(block*cfg['R'], min((block+1)*cfg['R'], cutoff)))
        factual.append({'policy': row['config_id'], 'example_id': row['example_id'],
            'code_found': row['code_found'], 'em': row['em'], 'prediction': row['prediction'],
            'reference_answer': ex['answer'], 'original_full_code_found': full[ex['id']]['code_found'],
            'selected_blocks': cfg['fixed_blocks'],
            'whole_record_position_coverage': set(range(*ex['record_span'])) <= ids})
    quality = []
    for policy in ('rare_context', 'rare_context_k16', 'rare_context_k32'):
        group = [r for r in factual if r['policy'] == policy]
        paired = [r for r in group if r['original_full_code_found'] == 100]
        quality.append({'policy': policy, 'cases': len(group),
            'code_correct': sum(r['code_found'] == 100 for r in group),
            'strict_correct': sum(r['em'] == 100 for r in group),
            'original_full_correct': len(paired),
            'original_full_correct_retained': sum(r['code_found'] == 100 for r in paired)})
    latency = [{'policy': r['config_id'], 'budget': 32768, 'trial': r['trial'],
                'mean_ms': statistics.mean(r['latency_ms']),
                'all_calls_mean_ms': statistics.mean(r['all_latency_ms']),
                'cuda_peak_GiB': r['cuda_decode_peak_allocated_bytes']/2**30,
                'archive_verify_open_s': r.get('archive_verify_open_s'),
                'query_s': r.get('query_s'), 'controller_init_s': r.get('controller_init_s')}
               for r in rows if r['task'] == 'latency']
    ranges = trial_ranges(latency)
    summary = {'status': 'complete_post_hoc_diagnostic', **audit,
        'source_files_sha256': manifest['source_files_sha256'],
        'probe_files_sha256': {name: file_sha(run/name) for name in
                              ('manifest.json', 'fixtures.json', 'samples.jsonl', 'complete.json', 'validation.json')},
        'report_script_sha256': file_sha(__file__), 'quality': quality,
        'factual_cases': factual, 'latency': latency, 'latency_trial_ranges': ranges,
        'original_full_quality': {'cases': 12, 'code_correct': sum(r['code_found'] == 100 for r in full.values())},
        'limits': ['post-hoc policies reuse all twelve 32K Qwen confirmation cases; not held-out confirmation',
                   'quality full-cache reference is the original frozen run; stock timing is fresh and randomized',
                   'larger budgets add nearby context to the original query-selected blocks; no new semantic search',
                   'coverage is inferred from first-query token positions, not a new per-layer payload audit',
                   'only one model, one context length and one forced-token timing fixture; three trial ranges are not confidence intervals',
                   'index preparation/search excluded; mount, query/setup and checkpoint are outside steady decoding']}
    atomic_json(run/'summary.json', summary)
    lines = ['# Post-hoc neighboring-context budget diagnostic', '',
        'All 48 declared observations are present; all twelve original k4 token sequences reproduce.',
        'These policies reuse the confirmation cases and are not held-out policy validation.', '',
        '| Policy | Correct code | Strict output | Original full-cache successes retained |',
        '|---|---:|---:|---:|']
    for q in quality:
        lines.append(f"| {q['policy']} | {q['code_correct']}/12 | {q['strict_correct']}/12 | {q['original_full_correct_retained']}/{q['original_full_correct']} |")
    lines += ['', '| Policy | Mean decode ms (trial range) | Mean paired stock speed ratio | Peak CUDA GiB |',
              '|---|---:|---:|---:|']
    for r in ranges:
        peak = max(t['cuda_peak_GiB'] for t in latency if t['policy'] == r['policy'])
        lines.append(f"| {r['policy']} | {r['mean_of_trial_means_ms']:.2f} ({r['min_trial_mean_ms']:.2f}–{r['max_trial_mean_ms']:.2f}) | {r['mean_paired_throughput_ratio']:.3f} | {peak:.3f} |")
    lines += ['', 'Each timing trial has 128 matched forwards, including 16 warmup forwards; steady means use the remaining 112.',
              'The JSON preserves all-call means, per-case answers and position coverage, source hashes, and setup costs.',
              '', *['- '+limit+'.' for limit in summary['limits']]]
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source-run', type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.source_run)
    print(args.run/'report.md')

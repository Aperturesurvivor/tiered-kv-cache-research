"""Audit the separate exact-semantic generation reference, never the ANN claim."""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import statistics
import subprocess

import numpy as np

from research.report_reasoning import audit_rows, audit_validation, accounted_service_s, digest, need


def declared_policies():
    result = [dict(id='stock-direct-r1-cap128', attention='stock', rounds=1,
                   cap_per_round=128, format='direct', schedule_labels=['direct_vanilla'])]
    for attention in ('stock', 'sparse'):
        for rounds in (1, 2):
            prefix = 'sparse-semantic' if attention == 'sparse' else 'stock'
            result.append(dict(id=f'{prefix}-json-r{rounds}-cap128', attention=attention,
                rounds=rounds, cap_per_round=128, format='json',
                schedule_labels=['semantic_diagnostic_128_per_round']))
    return result


def expected_schedule(fixtures):
    rng = random.Random(173)
    result = []
    for ex in fixtures:
        for trial in range(3 if ex['task'] == 'linked_fact' and ex['seed'] == 157 else 1):
            policies = declared_policies()
            rng.shuffle(policies)
            result.extend(dict(example_id=ex['id'], trial=trial, policy=p,
                               purpose='quality' if trial == 0 else 'repeated_timing') for p in policies)
    return result


def verify_source(manifest, fixtures, repo, revision):
    need(bool(re.fullmatch('[0-9a-f]{40}', revision)), 'full immutable source revision required')
    source = lambda name: subprocess.check_output(['git', 'show', revision+':'+name], cwd=repo)
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision], cwd=repo, text=True).splitlines()
    core = sorted(f for f in files if len(Path(f).parts) == 2 and Path(f).parts[0] in ('tiered_kv', 'tests') and f.endswith('.py'))
    core += ['configs/matrix.yaml', 'configs/eval.yaml', 'requirements.lock']
    need(digest({f: hashlib.sha256(source(f)).hexdigest() for f in core}) == manifest['implementation_hash'],
         'frozen implementation/test/dependency identity differs')
    need(hashlib.sha256(source('research/reasoning_run.py')).hexdigest() == manifest['script_sha256'], 'generator source differs')
    paths = {'research/'+n+'.py' for n in ('reasoning_protocol', 'reasoning_sessions', 'persistent_kv',
             'budget_probe', 'semantic_index', 'resources', 'semantic_reasoning')}
    need(set(manifest['dependencies_sha256']) == paths, 'research dependency identity incomplete')
    for name, sha in manifest['dependencies_sha256'].items():
        need(hashlib.sha256(source(name)).hexdigest() == sha, 'frozen dependency differs: '+name)
    for name in ('reasoning_protocol', 'persistent_kv', 'budget_probe', 'semantic_index'):
        name = 'research/'+name+'.py'
        need((repo/name).read_bytes() == source(name), 'audit helper differs from frozen source: '+name)
    declarations = json.loads(source('research/reasoning-fixture-expectations.json'))
    model = manifest['model']
    pinned = next(m for m in declarations['models'] if m['model_id'] == model['model_id'])
    specs = json.loads(source('research/model-manifest.json'))+json.loads(source('research/long-context-candidates.json'))['models']
    need(model == next(s for s in specs if s['model_id'] == model['model_id']), 'model specification differs')
    need(model['revision'] == pinned['revision'], 'model revision differs')
    need(digest(fixtures) == manifest['fixture_sha256'] == pinned['fixture_sha256'], 'fixture digest differs')
    need(len(fixtures) == 16 and manifest['schedule'] == expected_schedule(fixtures)
         and len(manifest['schedule']) == 100, 'semantic diagnostic schedule differs')
    from research.semantic_index import ENCODER_ID, ENCODER_REVISION
    selector = manifest['selector']
    need(bool(re.fullmatch('[0-9a-f]{64}', selector['encoder_weights_sha256'])), 'encoder weight identity missing')
    need(selector == dict(method='semantic_exact', encoder_id=ENCODER_ID, encoder_revision=ENCODER_REVISION,
        encoder_weights_sha256=selector['encoder_weights_sha256'], cold_cutoff='prefix_tokens_count-2000+1',
        text_padding=32, seed_blocks=4, expanded_block_budget=16, linear_scan=True), 'semantic selector differs')
    need(manifest['runtime']['device'] == 'cuda', 'not CUDA generator evidence')
    return pinned


def check_completion(rows, marker):
    if marker is not None:
        need(marker == dict(rows=100, quality_rows=80, repeated_timing_rows=20,
                            weights_unchanged=True, encoder_weights_unchanged=True)
             and len(rows) == 100, 'semantic completion marker or extent differs')


def check_semantic_round(round, vectors, encoder_vector, database_sha256):
    """Float64 top-k validation allows 1e-6 numerical boundary ties explicitly.

    Retained query/database vectors are the generator's inputs; independently
    re-encoded queries check identity without forcing cross-machine bit equality.
    """
    from research.semantic_index import ENCODER_ID, ENCODER_REVISION
    detail = round['index_details']
    need(set(detail) == {'selector', 'encoder_id', 'encoder_revision', 'query_vector', 'encoder_s',
        'scan_s', 'scanned_blocks', 'seed_blocks', 'database_sha256', 'linear_scan', 'gold_used_for_selection'},
        'semantic query telemetry incomplete or unexpected')
    need(detail['selector'] == 'semantic_exact' and detail['encoder_id'] == ENCODER_ID
         and detail['encoder_revision'] == ENCODER_REVISION and detail['linear_scan'] is True
         and detail['gold_used_for_selection'] is False, 'semantic query protocol differs')
    need(detail['database_sha256'] == database_sha256 and detail['scanned_blocks'] == len(vectors), 'query index identity/extent differs')
    query = np.asarray(detail['query_vector'], dtype=np.float64)
    need(query.shape == (vectors.shape[1],) and np.isfinite(query).all(), 'invalid saved query vector')
    need(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
         and abs(np.linalg.norm(query)-1) < 1e-5, 'unnormalized semantic vectors')
    encoded = np.asarray(encoder_vector, dtype=np.float64)
    need(encoded.shape == query.shape and np.isfinite(encoded).all(), 'invalid independently encoded query')
    error = float(np.max(np.abs(query-encoded)))
    need(error <= 1e-5, 'saved query does not reproduce from the public text')
    scores = np.einsum('ij,j->i', vectors.astype(np.float64), query, optimize=False)
    seeds = detail['seed_blocks']
    need(len(seeds) == min(4, len(vectors)) and len(seeds) == len(set(seeds))
         and all(type(b) is int and 0 <= b < len(vectors) for b in seeds), 'invalid semantic seed budget/address')
    exact = sorted(range(len(scores)), key=lambda b: (-float(scores[b]), b))[:len(seeds)]
    need(all(scores[a]+1e-6 >= scores[b] for a, b in zip(seeds, seeds[1:])), 'semantic seeds not score-ordered')
    threshold = min(scores[b] for b in seeds)
    gap = max((score-threshold for b, score in enumerate(scores) if b not in seeds), default=0.)
    need(gap <= 1e-6, 'semantic seeds omit a materially higher-scoring block')
    costs = [detail['encoder_s'], detail['scan_s']]
    need(math.isfinite(round['lookup_and_selection_s']) and round['lookup_and_selection_s'] >= 0
         and all(math.isfinite(t) and t >= 0 for t in costs)
         and sum(costs) <= round['lookup_and_selection_s']+1e-6, 'semantic work excluded from lookup timing')
    return {'query_embedding_max_abs_error': error, 'float64_order_matches': seeds == exact,
            'max_omitted_score_gap': max(0., float(gap))}


def replay_semantic(root, manifest, fixtures, rows, tokenizer, encoder):
    from research.budget_probe import expand_blocks
    from tiered_kv.store import file_sha
    checked = ties = 0
    max_error = max_edge_error = 0.
    hashes = {}
    query_cache = {}
    for ex in fixtures:
        relevant = [r for r in rows if r['example_id'] == ex['id']]
        if not relevant:
            continue
        path = root/'semantic-indexes'/(ex['id']+'-semantic-cold.sqlite3')
        marker = path.with_suffix('.complete.json')
        sha = file_sha(path)
        hashes[str(path.relative_to(root))] = sha
        hashes[str(marker.relative_to(root))] = file_sha(marker)
        cutoff = ex['prefix_tokens_count']-1999
        expected = dict(fixture_sha256=digest(ex), cutoff=cutoff, padding=32,
            encoder_id=manifest['selector']['encoder_id'], encoder_revision=manifest['selector']['encoder_revision'],
            encoder_weights_sha256=manifest['selector']['encoder_weights_sha256'], database_sha256=sha)
        need(json.loads(marker.read_text()) == expected, 'semantic index marker differs')
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            blocks = db.execute('SELECT id,start,end,text,vector FROM chunks ORDER BY id').fetchall()
        need([b[:3] for b in blocks] == [(s//128, s, min(s+128, cutoff)) for s in range(0, cutoff, 128)],
             'semantic index address extents differ')
        for block, start, end, text, _ in blocks:
            need(text == tokenizer.decode(ex['tokens'][max(0,start-32):min(start+160,cutoff)], skip_special_tokens=False),
                 'semantic index contains different or future text')
        vectors = np.stack([np.frombuffer(b[4], dtype='<f4') for b in blocks])
        need(vectors.shape == (len(blocks), 384), 'semantic vector dimensions differ')
        edges = encoder.encode([blocks[0][3], blocks[-1][3]])
        edge_error = float(np.max(np.abs(edges-vectors[[0,-1]])))
        need(edge_error <= 1e-5, 'edge block embeddings differ from pinned encoder')
        max_edge_error = max(max_edge_error, edge_error)
        for row in relevant:
            need(all(row[k] is None for k in ('memory_before','memory_after','memory_update')), 'undeclared memory harness')
            for round in row['rounds']:
                if row['policy']['attention'] == 'stock':
                    need(round['index_details'] == {}, 'stock control carries semantic retrieval')
                    continue
                query = round['query']
                if query not in query_cache:
                    query_cache[query] = encoder.encode([query])[0]
                numeric = check_semantic_round(round, vectors, query_cache[query], sha)
                max_error = max(max_error, numeric['query_embedding_max_abs_error'])
                ties += not numeric['float64_order_matches']
                selected = list(expand_blocks(tuple(round['index_details']['seed_blocks']),
                                             (ex['prefix_tokens_count']+127)//128, 16))
                need(round['selected_blocks'] == selected, 'semantic neighbor expansion differs')
                checked += 1
    return dict(semantic_rounds_checked=checked, numerical_boundary_tie_rounds=ties,
        query_embedding_max_abs_error=max_error, edge_embedding_max_abs_error=max_edge_error,
        retained_index_hashes=hashes,
        selection_scope='all text/extents, query re-encoding, float64 top-k within 1e-6, exact recorded-seed expansion; edge block embeddings sampled, other vectors checksum-bound')


def summarize(fixtures, rows):
    examples = {e['id']: e for e in fixtures}
    bykey = {(r['example_id'], r['policy']['id'], r['trial']): r for r in rows}
    quality = []
    timing = []
    for budget in (8192, 32768):
        for policy in declared_policies():
            pid = policy['id']
            group = [r for r in rows if r['trial'] == 0 and r['policy']['id'] == pid and examples[r['example_id']]['budget'] == budget]
            direct = [bykey[r['example_id'], 'stock-direct-r1-cap128', 0] for r in group]
            quality.append(dict(budget=budget, policy=pid, cases=len(group), correct=sum(r['exact_correct'] for r in group),
                lost_vs_direct=sum(b['exact_correct'] and not r['exact_correct'] for r,b in zip(group,direct)),
                gained_vs_direct=sum(not b['exact_correct'] and r['exact_correct'] for r,b in zip(group,direct)),
                mean_request_s=statistics.mean(r['request_s'] for r in group),
                mean_accounted_service_s=statistics.mean(accounted_service_s(r) for r in group),
                mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in group),
                mean_model_token_evaluations=statistics.mean(r['model_token_evaluations'] for r in group),
                mean_lookup_s=statistics.mean(sum(t['lookup_and_selection_s'] for t in r['rounds']) for r in group),
                max_active_KV_entries=max(r['max_active_KV_entries'] for r in group),
                mean_shared_prefill_s=statistics.mean(r['prefill_s'] for r in group),
                mean_shared_index_build_or_open_s=statistics.mean(r['index_build_or_open_s'] for r in group)))
            repeated = [r for r in rows if r['example_id'] == f'reasoning-linked_fact-{budget}-157' and r['policy']['id'] == pid]
            need(len(repeated) == 3, 'missing repeated timing trials')
            control = pid.replace('sparse-semantic', 'stock')
            timing.append(dict(budget=budget, policy=pid, trials=3,
                request_mean_s=statistics.mean(r['request_s'] for r in repeated),
                request_range_s=[min(r['request_s'] for r in repeated), max(r['request_s'] for r in repeated)],
                paired_matching_stock_request_ratios=[bykey[r['example_id'],control,r['trial']]['request_s']/r['request_s'] for r in repeated],
                paired_matching_stock_service_ratios=[accounted_service_s(bykey[r['example_id'],control,r['trial']])/accounted_service_s(r) for r in repeated],
                cuda_peak_GiB=max(r['cuda_peak_allocated_bytes'] for r in repeated)/2**30,
                host_sampled_peak_GiB=max(r['host_rss_sampled_peak_bytes'] for r in repeated)/2**30))
    return dict(quality=quality, timing=timing)


def write_report(root, manifest, audit):
    need(audit['status'] == 'complete' and audit['tokenizer_verified'], 'cannot report an incomplete comparison')
    lines = ['# Exact-semantic retrieval generation diagnostic', '',
        f"Model: `{manifest['model']['model_id']}` at `{manifest['model']['revision']}`.",
        'All 100 declared rows passed the independent audit: 80 quality observations and 20 extra timing observations.',
        'These are eight underlying questions repeated at two lengths. Exact semantic lookup scans the full cold index.', '',
        '| Context | Policy | Correct | Lost/gained vs direct | Request s | Accounted service s | Query/selection s | Evaluated tokens |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in audit['quality']:
        lines.append(f"| {r['budget']} | {r['policy']} | {r['correct']}/{r['cases']} | {r['lost_vs_direct']}/{r['gained_vs_direct']} | {r['mean_request_s']:.3f} | {r['mean_accounted_service_s']:.3f} | {r['mean_lookup_s']:.4f} | {r['mean_model_token_evaluations']:.1f} |")
    lines += ['', 'Accounted service sums measured mount, initialization, request and checkpoint. Shared dense prefill,',
        'index build/open and encoder startup are reported separately in summary.json. No best-draft selection is used.',
        'Repeated timing covers one linked-fact question per length with three trials; incorrect answers remain in all costs.', '',
        '| Context | Policy | Request mean (range) s | Matching stock/request ratios | Peak CUDA GiB |',
        '|---:|---|---:|---|---:|']
    for r in audit['timing']:
        lo, hi = r['request_range_s']
        ratios = ', '.join(f'{v:.3f}' for v in r['paired_matching_stock_request_ratios'])
        lines.append(f"| {r['budget']} | {r['policy']} | {r['request_mean_s']:.3f} ({lo:.3f}–{hi:.3f}) | {ratios} | {r['cuda_peak_GiB']:.3f} |")
    lines += ['', 'Ratios above one favor sparse execution. They must be interpreted alongside correctness.', '',
        'Audit scope: '+audit['selection_scope']+'.', '',
        f"Numerical boundary/tie-order differences: {audit['numerical_boundary_tie_rounds']} retrieval rounds.",
        'Their stored-vector float64 top-k boundary error is constrained to 1e-6; no bitwise cross-machine ranking claim is made.', '',
        *['- '+limit for limit in audit['limits']], '']
    (root/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = args.run
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'manifest.json').read_text())
    fixtures = json.loads((root/'fixtures.json').read_text())
    rows = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    pinned = verify_source(manifest, fixtures, repo, args.source_revision)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from research.semantic_index import Encoder
    from tiered_kv.generate import weights_digest
    from tiered_kv.store import atomic_json, file_sha
    import torch
    torch.set_num_threads(1)
    snapshot = snapshot_download(manifest['model']['model_id'], revision=pinned['revision'], local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tok.apply_chat_template = partial(tok.apply_chat_template, enable_thinking=False)
    need(digest(tok.chat_template) == pinned['chat_template_sha256'], 'tokenizer template differs')
    from research.reasoning_protocol import make_fixture
    need(all(digest(make_fixture(tok,e['budget'],e['seed'],e['task'])) == digest(e) for e in fixtures), 'fixture regeneration differs')
    generation_path = Path(snapshot)/'generation_config.json'
    generation = json.loads(generation_path.read_text())
    eos = generation['eos_token_id']; eos = set(eos if isinstance(eos,list) else [eos])
    encoder = Encoder()
    encoder_hash = weights_digest(encoder.model)
    need(encoder_hash == manifest['selector']['encoder_weights_sha256'], 'actual encoder weights differ')
    names = {r['validation_file'] for r in rows}
    weights = set()
    startup = {}
    for name in names:
        need(Path(name).name == name and name.startswith('validation-') and name.endswith('.json'), 'unsafe validation filename')
        validation = json.loads((root/name).read_text())
        audit_validation(validation, manifest['model'])
        weights.add(validation['weights_sha256'])
        need(validation['encoder']['weights_sha256'] == encoder_hash, 'validation encoder identity differs')
        duration = validation['encoder']['load_and_hash_s']
        need(math.isfinite(duration) and duration >= 0, 'invalid encoder startup cost')
        startup[name] = duration
    need(len(weights) == 1, 'missing or inconsistent actual generator weight identity')
    marker = json.loads((root/'complete.json').read_text()) if (root/'complete.json').exists() else None
    need(args.partial or marker is not None, 'matrix incomplete; use --partial for non-conclusive audit')
    check_completion(rows, marker)
    # Reuse the row-level interpreter with its completion check disabled; the
    # distinct 100-row completion declaration is checked immediately above.
    audit = audit_rows(manifest, fixtures, rows, None, tokenizer=tok, eos_ids=eos)
    audit.update(replay_semantic(root,manifest,fixtures,rows,tok,encoder))
    need(weights_digest(encoder.model) == encoder_hash, 'auditor encoder mutated')
    audit.update(status='complete' if marker else 'incomplete', source_revision=args.source_revision,
        encoder_startup_s_by_validation=startup, encoder_weights_sha256=encoder_hash,
        generator_weights_sha256=next(iter(weights)), generation_config_sha256=file_sha(generation_path),
        audit_sources_sha256={p: file_sha(repo/'research'/p) for p in ('report_semantic_reasoning.py','report_reasoning.py')},
        files_sha256={p: file_sha(root/p) for p in ['manifest.json','fixtures.json','samples.jsonl',*sorted(names)]+(['complete.json'] if marker else [])})
    audit['limits'] = ['Post-hoc reused fixtures: eight underlying questions repeated at two lengths.',
        'Exact semantic scan is linear, not ANN or fixed retrieval cost.',
        'Request includes encoding/search and all prompt/generation forwards; shared prefill/index, mount/init/checkpoint and encoder startup are separate.',
        'Accounted service is mount+init+request+checkpoint, not a fresh whole-call timer; stock also reloads the shared archive.',
        'Stock batches prompts; sparse evaluates prompt tokens serially. Page cache uncontrolled; backups may add I/O noise.',
        'Address coverage is not a new per-layer KV-payload audit. No 1M quality or equivalence claim.']
    if marker:
        audit.update(summarize(fixtures, rows))
    atomic_json(root/('summary.json' if marker else 'audit-partial.json'), audit)
    if marker:
        write_report(root, manifest, audit)
    print(json.dumps({k: audit[k] for k in ('status','observed_rows','expected_rows','semantic_rounds_checked')}))


if __name__ == '__main__':
    main()

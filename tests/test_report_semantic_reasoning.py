from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from research.budget_probe import expand_blocks
from research.reasoning_run import trial_plan
from research.report_semantic_reasoning import (
    check_completion, check_semantic_round, expected_schedule, replay_semantic,
    summarize, write_report,
    verify_source,
)
from tiered_kv.types import digest
from research.semantic_index import ENCODER_ID, ENCODER_REVISION
from research.semantic_reasoning import semantic_metadata_index


def sample_round():
    scores = np.array([.95, .8, .6, .4, .2])
    vectors = np.stack([scores, np.sqrt(1-scores**2)], axis=1)
    query = np.array([1., 0.])
    row = dict(lookup_and_selection_s=.02, index_details=dict(
        selector='semantic_exact', encoder_id=ENCODER_ID, encoder_revision=ENCODER_REVISION,
        query_vector=query.tolist(), encoder_s=.01, scan_s=.005, scanned_blocks=5,
        seed_blocks=[0, 1, 2, 3], database_sha256='index-hash', linear_scan=True,
        gold_used_for_selection=False))
    return row, vectors, query


def test_exact_semantic_topk_and_explicit_numeric_boundary_tie():
    row, vectors, query = sample_round()
    assert check_semantic_round(row, vectors, query, 'index-hash')['float64_order_matches']
    vectors[4] = [vectors[3, 0]-5e-7, np.sqrt(1-(vectors[3, 0]-5e-7)**2)]
    row['index_details']['seed_blocks'][-1] = 4
    result = check_semantic_round(row, vectors, query, 'index-hash')
    assert not result['float64_order_matches']
    assert 0 < result['max_omitted_score_gap'] < 1e-6


@pytest.mark.parametrize('mutation', ['query', 'nonfinite', 'unnormalized', 'wrong_topk',
    'reordered', 'duplicate', 'future', 'hidden_work', 'nan_time', 'index', 'extent', 'gold',
    'encoder', 'encoded_shape'])
def test_rejects_corrupt_semantic_retrieval_evidence(mutation):
    row, vectors, query = sample_round()
    detail = row['index_details']
    if mutation == 'query': detail['query_vector'] = [0., 1.]
    elif mutation == 'nonfinite': vectors[0, 0] = np.nan
    elif mutation == 'unnormalized': vectors[0] *= 2
    elif mutation == 'wrong_topk': detail['seed_blocks'] = [0, 1, 2, 4]
    elif mutation == 'reordered': detail['seed_blocks'] = [1, 0, 2, 3]
    elif mutation == 'duplicate': detail['seed_blocks'] = [0, 1, 2, 2]
    elif mutation == 'future': detail['seed_blocks'] = [0, 1, 2, 5]
    elif mutation == 'hidden_work': detail['encoder_s'] = .1
    elif mutation == 'nan_time': row['lookup_and_selection_s'] = np.nan
    elif mutation == 'index': detail['database_sha256'] = 'other-index'
    elif mutation == 'extent': detail['scanned_blocks'] -= 1
    elif mutation == 'gold': detail['gold_used_for_selection'] = True
    elif mutation == 'encoder': detail['encoder_revision'] = 'other-model'
    elif mutation == 'encoded_shape': query = np.ones((2, 1))
    with pytest.raises(ValueError):
        check_semantic_round(row, vectors, query, 'index-hash')


def examples():
    return [dict(id=f'reasoning-{task}-{budget}-{seed}', task=task, budget=budget, seed=seed)
        for budget in (8192, 32768) for seed in (157, 163)
        for task in ('literal', 'paraphrase', 'linked_fact', 'sum_two_facts')]


def test_independent_plan_and_completion_cannot_accept_original_220_row_study():
    fixtures = examples()
    assert expected_schedule(fixtures) == trial_plan(fixtures, 'semantic_exact')
    marker = dict(rows=100, quality_rows=80, repeated_timing_rows=20,
                  weights_unchanged=True, encoder_weights_unchanged=True)
    check_completion(expected_schedule(fixtures), marker)
    for rows, invalid in [(expected_schedule(fixtures)[:-1], marker),
                          (trial_plan(fixtures), marker),
                          (expected_schedule(fixtures), {**marker, 'encoder_weights_unchanged':False})]:
        with pytest.raises(ValueError): check_completion(rows, invalid)


def test_source_audit_accepts_frozen_semantic_identity_and_rejects_tampering():
    repo = Path(__file__).resolve().parents[1]
    revision = 'ddd782e094ac9224bbf55313a35aadebd5608e8e'
    source = lambda name: subprocess.check_output(['git','show',revision+':'+name], cwd=repo)
    root = repo/'results/evidence/reasoning-Qwen--Qwen2.5-1.5B-Instruct-8862fb5f28f2'
    # Colab deployment is a source export, not a Git clone with historical
    # evidence. Source-provenance rejection is tested in the owning checkout.
    if not (root/'fixtures.json').is_file() or subprocess.run(
            ['git','cat-file','-e',revision+'^{commit}'], cwd=repo,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        pytest.skip('requires owning Git history and retained reasoning fixtures')
    fixtures = json.loads((root/'fixtures.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    files = subprocess.check_output(['git','ls-tree','-r','--name-only',revision], cwd=repo, text=True).splitlines()
    core = sorted(f for f in files if len(Path(f).parts)==2 and Path(f).parts[0] in ('tiered_kv','tests') and f.endswith('.py'))
    core += ['configs/matrix.yaml','configs/eval.yaml','requirements.lock']
    manifest['implementation_hash'] = digest({f:hashlib.sha256(source(f)).hexdigest() for f in core})
    manifest['script_sha256'] = hashlib.sha256(source('research/reasoning_run.py')).hexdigest()
    manifest['dependencies_sha256']['research/semantic_reasoning.py'] = ''
    manifest['dependencies_sha256'] = {f:hashlib.sha256(source(f)).hexdigest() for f in manifest['dependencies_sha256']}
    manifest['schedule'] = expected_schedule(fixtures)
    manifest['selector'] = dict(method='semantic_exact', encoder_id=ENCODER_ID,
        encoder_revision=ENCODER_REVISION, encoder_weights_sha256='a'*64,
        cold_cutoff='prefix_tokens_count-2000+1', text_padding=32, seed_blocks=4,
        expanded_block_budget=16, linear_scan=True)
    assert verify_source(manifest, fixtures, repo, revision)['fixture_sha256'] == digest(fixtures)
    for mutation in ('source', 'fixture', 'schedule', 'dependency', 'scan'):
        changed = deepcopy(manifest)
        if mutation == 'source': changed['script_sha256'] = '0'*64
        elif mutation == 'fixture': changed['fixture_sha256'] = '0'*64
        elif mutation == 'schedule': changed['schedule'].reverse()
        elif mutation == 'dependency': changed['dependencies_sha256'].pop('research/semantic_reasoning.py')
        elif mutation == 'scan': changed['selector']['linear_scan'] = False
        with pytest.raises(ValueError): verify_source(changed, fixtures, repo, revision)


def test_disk_replay_checks_text_memory_expansion_and_index_identity(tmp_path):
    class Tokenizer:
        def decode(self, ids, **kwargs): return ''.join(map(chr, ids))
    class Encoder:
        def encode(self, texts):
            result = np.zeros((len(texts), 384), np.float32)
            result[:, 0] = 1
            return result
    tok, encoder = Tokenizer(), Encoder()
    ex = dict(id='fixture', prefix_tokens_count=2700, tokens=[65]*2700)
    root = tmp_path/'semantic-indexes'; root.mkdir()
    index = semantic_metadata_index(tok, ex, root, encoder, 'frozen-encoder')
    seeds = index.search('A', None, 4, 'semantic_exact')
    round = dict(query='A', index_details=deepcopy(index.last_search),
        lookup_and_selection_s=1., selected_blocks=list(expand_blocks(tuple(seeds), 22, 16)))
    index.close()
    manifest = dict(selector=dict(encoder_id=ENCODER_ID, encoder_revision=ENCODER_REVISION,
                                  encoder_weights_sha256='frozen-encoder'))
    row = dict(example_id=ex['id'], policy=dict(attention='sparse'), rounds=[round],
               memory_before=None, memory_after=None, memory_update=None)
    result = replay_semantic(tmp_path, manifest, [ex], [row], tok, encoder)
    assert result['semantic_rounds_checked'] == 1
    for mutation in ('memory', 'expansion', 'future_text', 'index_identity'):
        altered = deepcopy(row); altered_ex = deepcopy(ex)
        if mutation == 'memory': altered['memory_before'] = {'hidden':'fact'}
        elif mutation == 'expansion': altered['rounds'][0]['selected_blocks'] = [20]
        elif mutation == 'future_text': altered_ex['tokens'][0] = 66
        elif mutation == 'index_identity': altered['rounds'][0]['index_details']['database_sha256'] = 'wrong'
        with pytest.raises(ValueError):
            replay_semantic(tmp_path, manifest, [altered_ex], [altered], tok, encoder)


def test_summary_pairs_same_round_controls_and_reports_failed_answers(tmp_path):
    fixtures = examples()
    rows = []
    for planned in expected_schedule(fixtures):
        sparse = planned['policy']['attention'] == 'sparse'
        rows.append({**planned, 'exact_correct':not sparse, 'request_s':4. if sparse else 2.,
            'archive_checksum_mount_s':1., 'controller_init_s':1., 'checkpoint_s':1.,
            'prefill_s':5., 'index_build_or_open_s':3., 'generated_tokens':5,
            'model_token_evaluations':10, 'max_active_KV_entries':100,
            'cuda_peak_allocated_bytes':2**30, 'host_rss_sampled_peak_bytes':2**30,
            'rounds':[dict(lookup_and_selection_s=.1)]})
    report = summarize(fixtures, rows)
    group = next(r for r in report['quality'] if r['policy']=='sparse-semantic-json-r2-cap128')
    assert group['correct'] == 0 and group['lost_vs_direct'] == 8
    assert group['mean_accounted_service_s'] == 7.
    timing = next(r for r in report['timing'] if r['policy']=='sparse-semantic-json-r2-cap128')
    assert timing['paired_matching_stock_request_ratios'] == [.5]*3
    assert timing['paired_matching_stock_service_ratios'] == [5./7]*3
    audit = dict(status='complete', tokenizer_verified=True, selection_scope='test',
                 numerical_boundary_tie_rounds=0, limits=['Synthetic test'], **report)
    manifest = dict(model=dict(model_id='test', revision='test'))
    write_report(tmp_path, manifest, audit)
    assert '0/8' in (tmp_path/'report.md').read_text()
    audit['status'] = 'incomplete'
    with pytest.raises(ValueError): write_report(tmp_path, manifest, audit)

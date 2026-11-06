"""Post-hoc prefix-retention control for the first 32K persistent failure.

Isolated extension of persistent_probe at 1881a71. Preserve the original source
and retrieval selection; compare four initial tokens, the complete 29-token
instruction header, and 128 initial tokens. No performance claim. Run only in
the original 23964ff source root, after archival and with other GPU jobs idle.
"""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path

import numpy as np
import torch

from research.persistent_kv import StockSession
from tiered_kv.adapter import DenseController
from tiered_kv.cache import TieredCache
from tiered_kv.config import Config, MODEL_REVISION
from tiered_kv.evals import score_needle
from tiered_kv.generate import RunEngine, append_jsonl, decode_greedy, load_model, weights_digest
from tiered_kv.store import ColdStore, atomic_json, file_sha
from tiered_kv.telemetry import DEVICE, ResourceGuard, device_empty_cache, runtime_identity
from tiered_kv.types import digest
from tiered_kv.validation import implementation_hash, parity


SOURCE = '23964ff04123671c230249d7b58ebe7403616b0f'


def plan(source_run):
    source_run = Path(source_run)
    manifest = json.loads((source_run/'manifest.json').read_text())
    complete = json.loads((source_run/'complete.json').read_text())
    fixture = json.loads((source_run/'fixtures.json').read_text())
    rows = [json.loads(line) for line in (source_run/'samples.jsonl').read_text().splitlines()]
    if complete != {'status': 'complete', 'weights_unchanged': True} or len(rows) != 128:
        raise ValueError('expected the immutable completed 128-row persistent study')
    if digest(fixture) != manifest['fixture_sha256'] or manifest['model_revision'] != MODEL_REVISION:
        raise ValueError('fixture or model revision differs')
    example = next(ex for ex in fixture['examples'] if ex['budget'] == 32768)
    baseline = {}
    for policy in ('stock_hf', 'rare_context'):
        matching = [r for r in rows if r.get('budget') == 32768 and r.get('turn') == 0
                    and r.get('condition') == 'warm' and r.get('policy') == policy]
        if len(matching) != 1 or matching[0]['status'] != 'ok':
            raise ValueError('expected one successful first-query observation per control')
        baseline[policy] = matching[0]
    sparse = Config(**baseline['rare_context']['policy_config'])
    if (sparse.fixed_blocks != (23, 24, 22, 25) or sparse.C != 0 or sparse.W != 2000
            or baseline['stock_hf']['code_found'] != 100 or baseline['rare_context']['code_found'] != 0):
        raise ValueError('observed failure or policy identity differs')
    policies = (sparse, replace(sparse, id='prefix_29', sink_tokens=29),
                replace(sparse, id='prefix_128', sink_tokens=128))
    return manifest, example, baseline, policies


def compare(actual, expected):
    return {'max_abs': float((actual-expected).abs().max()),
            'allclose_1e4': bool(torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
            'greedy_agreement': bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    manifest, example, baseline, policies = plan(args.source_run)
    if args.plan_only:
        print(json.dumps({'example': example['id'], 'turn': 0, 'policies': [asdict(p) for p in policies]}))
        return
    if DEVICE != 'cuda' or Path('SOURCE_REVISION').read_text().strip() != SOURCE:
        raise ValueError('requires original 23964ff CUDA source root')
    torch.set_num_threads(manifest['runtime']['threads'])
    torch.set_num_interop_threads(manifest['runtime']['interop_threads'])
    if implementation_hash() != manifest['source'] or runtime_identity() != manifest['runtime']:
        raise ValueError('original implementation/runtime mismatch')
    if (file_sha('research/persistent_kv.py') != manifest['script_sha256']
            or file_sha('research/semantic_index.py') != manifest['index_sha256']):
        raise ValueError('original experiment or index source changed')
    prefix = Path('artifacts/scratch')/args.source_run.name/(example['id']+'-prefix')
    metadata = json.loads((prefix/'checkpoint.json').read_text())['metadata']
    if metadata['token_hash'] != example['token_hash'] or metadata['model_revision'] != MODEL_REVISION:
        raise ValueError('source checkpoint identity mismatch')
    identity = {'source_revision': SOURCE, 'implementation_hash': implementation_hash(),
        'script_sha256': file_sha(__file__), 'runtime': runtime_identity(),
        'source_files_sha256': {name: file_sha(args.source_run/name) for name in
                               ('manifest.json', 'fixtures.json', 'samples.jsonl', 'complete.json', 'validation.json')},
        'checkpoint_sha256': file_sha(prefix/'checkpoint.json'), 'example_sha256': digest(example),
        'policies': [asdict(p) for p in policies], 'turn': 0,
        'scope': 'post-hoc first-query prefix-retention control; 4/29/128 initial tokens; original case reused; no performance claim'}
    root = Path('results')/('prefix-probe-'+digest(identity)[:12])
    root.mkdir(exist_ok=False)
    atomic_json(root/'manifest.json', identity)
    scratch = Path('artifacts/scratch')/root.name
    scratch.mkdir(exist_ok=False)
    logit_payloads = {}
    def save_logits(policy, tensor):
        destination = scratch/(policy+'-question-logits.npy')
        np.save(destination, tensor.numpy(), allow_pickle=False)
        logit_payloads[policy] = {'path': str(destination), 'sha256': file_sha(destination),
                                  'shape': list(tensor.shape), 'dtype': 'float32'}
    with ResourceGuard() as guard:
        model, tok = load_model()
        before = weights_digest(model)
        if before != json.loads((args.source_run/'validation.json').read_text())['weights_sha256']:
            raise ValueError('loaded weights differ from original experiment')
        atomic_json(root/'validation.json', {'parity': parity(model, 512), 'weights_sha256': before})
        target = example['queries'][0]
        header = tok.decode(example['tokens'][:29], skip_special_tokens=False)
        expected_header = ('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. '
                           'You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
                           'Read the following archive.\n\n')
        if header != expected_header or target['record_span'][0] < 128:
            raise ValueError('declared instruction header or target separation differs')
        question = tok.encode(target['question']+example['first_suffix'], add_special_tokens=False)
        eos_ids = model.generation_config.eos_token_id
        eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        atomic_json(root/'fixture.json', {'target': target, 'query_token_ids': question,
            'instruction_header_tokens': 29, 'instruction_header_text': header,
            'prefix_tokens_count': example['prefix_tokens_count'], 'original_rows': baseline})
        parent = ColdStore.open(prefix, readonly=True)
        if parent.count != example['prefix_tokens_count']:
            raise ValueError('archive count differs')
        stock = StockSession(model, parent)
        stock_logits = []
        try:
            for token in question:
                guard.check()
                logits = stock.step(token, stock.position)
                stock_logits.append(logits.detach().float().cpu())
            generated, _ = decode_greedy(stock, logits, 64, eos, guard=guard)
            prediction = tok.decode(generated, skip_special_tokens=True)
            reproduced = (prediction == baseline['stock_hf']['prediction']
                          and len(generated) == baseline['stock_hf']['output_tokens'])
            append_jsonl(root/'samples.jsonl', {'policy': 'stock_hf', 'token_ids': generated,
                'prediction': prediction, 'original_prediction_and_length_reproduced': reproduced,
                **score_needle(prediction, target['answer'])})
            if not reproduced:
                raise RuntimeError('original stock output failed to reproduce; stop causal comparison')
            stock_tokens = list(generated)
            stock_logits = torch.stack(stock_logits)
            save_logits('stock_hf', stock_logits)
        finally:
            parent.close()
            del stock, parent
            gc.collect()
            device_empty_cache()
        engine = RunEngine(model, tok, root, guard)
        controls = []
        try:
            for cfg in policies:
                parent = ColdStore.open(prefix, readonly=True)
                store = ColdStore(engine.scratch/cfg.id, engine.layers, engine.h, engine.d, cfg.R,
                                  prefix=parent, staged_tensor_writes=True)
                # Mirror the original empty selection at mount, then select at
                # the complete-token boundary immediately before the question.
                controller = TieredCache(replace(cfg, fixed_blocks=()), store, engine.rotary, DEVICE)
                controller.select_blocks(cfg.fixed_blocks)
                engine.adapter.set_controller(controller)
                payload_checks = []
                def audit(layer, pos, q, view):
                    if pos != parent.count:
                        return
                    archived = view.ids < parent.count
                    ids = view.ids[archived]
                    source = parent.read_range(layer, 0, parent.count)
                    indices = torch.as_tensor(ids, dtype=torch.long)
                    actual_indices = torch.as_tensor(np.flatnonzero(archived), device=DEVICE)
                    same_k = torch.equal(view.k.index_select(2, actual_indices).cpu(), source.k.index_select(2, indices))
                    same_v = torch.equal(view.v.index_select(2, actual_indices).cpu(), source.v.index_select(2, indices))
                    full_coverage = np.array_equal(ids, np.arange(parent.count))
                    record_coverage = set(range(*target['record_span'])) <= set(ids.tolist())
                    prefix_coverage = set(range(cfg.sink_tokens)) <= set(ids.tolist())
                    row = {'layer': layer, 'archived_entries': len(ids), 'archive_K_bit_equal': same_k,
                           'archive_V_bit_equal': same_v, 'full_chronological_coverage': full_coverage,
                           'whole_fact_present': record_coverage, 'whole_declared_prefix_present': prefix_coverage}
                    payload_checks.append(row)
                    if not same_k or not same_v or not record_coverage or not prefix_coverage:
                        atomic_json(root/(cfg.id+'-payload-failure.json'), row)
                        raise RuntimeError('first-token cache payload audit failed')
                controller.trace = audit
                comparisons = []
                actual_logits = []
                try:
                    for offset, token in enumerate(question):
                        guard.check()
                        logits = engine.adapter.step(token, engine.adapter.position)
                        actual = logits.detach().float().cpu()
                        actual_logits.append(actual)
                        comparisons.append(compare(actual, stock_logits[offset]))
                    save_logits(cfg.id, torch.stack(actual_logits))
                    if [r['layer'] for r in payload_checks] != list(range(engine.layers)):
                        raise RuntimeError('payload audit did not visit every model layer exactly once')
                    generated, _ = decode_greedy(engine.adapter, logits, 64, eos, guard=guard)
                    prediction = tok.decode(generated, skip_special_tokens=True)
                    row = {'policy': cfg.id, 'policy_config': asdict(cfg), 'token_ids': generated,
                        'prediction': prediction, 'question_logit_comparisons': comparisons,
                        'question_allclose_1e4': all(c['allclose_1e4'] for c in comparisons),
                        'question_max_abs': max(c['max_abs'] for c in comparisons),
                        'stock_generation_token_ids_equal': generated == stock_tokens,
                        'payload_checks': payload_checks, **score_needle(prediction, target['answer'])}
                    if cfg.id == 'rare_context':
                        row['original_prediction_and_length_reproduced'] = (
                            prediction == baseline['rare_context']['prediction']
                            and len(generated) == baseline['rare_context']['output_tokens'])
                    append_jsonl(root/'samples.jsonl', row)
                    controls.append(row)
                    print(cfg.id, prediction, 'QUESTION_MAX_ABS', row['question_max_abs'], flush=True)
                    if row.get('original_prediction_and_length_reproduced') is False:
                        raise RuntimeError('original sparse output failed to reproduce')
                finally:
                    engine.adapter.set_controller(DenseController(engine.layers))
                    store.close()
                    del controller, store, parent
                    gc.collect()
                    device_empty_cache()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model) != before:
            raise RuntimeError('frozen weights changed')
        if len(controls) != 3:
            raise RuntimeError('prefix-control matrix incomplete')
        atomic_json(root/'complete.json', {'rows': 4, 'weights_unchanged': True,
            'original_stock_and_sparse_outputs_reproduced': True,
            'control': 'prefix retention: 4/29/128 initial tokens',
            'question_logit_payloads': logit_payloads})
    print('PREFIX_PROBE_DONE', root, flush=True)


if __name__ == '__main__':
    main()

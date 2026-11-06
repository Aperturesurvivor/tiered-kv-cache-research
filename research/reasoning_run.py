"""Separate, predeclared 1/2/4-round exact-KV retrieval experiment.

Shared immutable dense archive; every question/review token uses its evaluated
policy. Stock batches prompts. Sparse retrieves only addresses between rounds.
No source text is reinserted, and selections never consult evaluator answers.
"""
import argparse
from dataclasses import asdict
from functools import partial
import gc
import json
from pathlib import Path
import random
import shutil
import time
import uuid

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.budget_probe import expand_blocks
from research.persistent_kv import next_question_tokens
from research.reasoning_protocol import (SEEDS, TASKS, WORKING_KV, make_fixture,
    next_search, parse_response, policies, round_request)
from research.reasoning_sessions import BatchedStockSession, CountedSession
from research.resources import HostObservation, logical_store_reads
from research.semantic_index import metadata_index
from tiered_kv.adapter import DenseController, ModelAdapter
from tiered_kv.cache import TieredCache
from tiered_kv.config import Config
from tiered_kv.generate import RunEngine, append_jsonl, decode_greedy, online_smoke, sync, weights_digest
from tiered_kv.store import ColdStore, atomic_json, file_sha
from tiered_kv.telemetry import DEVICE, ResourceGuard, device_empty_cache, runtime_identity
from tiered_kv.types import digest
from tiered_kv.validation import implementation_hash, parity


def trial_plan(examples, selector='lexical_rare'):
    if selector not in ('lexical_rare', 'semantic_exact'):
        raise ValueError('unknown reasoning selector')
    if selector == 'semantic_exact':
        from research.semantic_reasoning import diagnostic_policies
        selected_policies = diagnostic_policies
    else:
        selected_policies = policies
    rng = random.Random(173)
    result = []
    for ex in examples:
        trials = range(3) if ex['task'] == 'linked_fact' and ex['seed'] == SEEDS[0] else range(1)
        for trial in trials:
            ordered = list(selected_policies())
            rng.shuffle(ordered)
            result.extend({'example_id': ex['id'], 'trial': trial, 'policy': p,
                           'purpose': 'quality' if trial == 0 else 'repeated_timing'} for p in ordered)
    return result


def prepare_fixtures(model_id):
    expected = json.loads(Path('research/reasoning-fixture-expectations.json').read_text())
    expectation = next(m for m in expected['models'] if m['model_id'] == model_id)
    if file_sha('research/reasoning_protocol.py') != expected['protocol_sha256']:
        raise ValueError('fixture protocol differs from predeclared source')
    specs = (json.loads(Path('research/model-manifest.json').read_text())
             + json.loads(Path('research/long-context-candidates.json').read_text())['models'])
    spec = next(s for s in specs if s['model_id'] == model_id)
    if spec['revision'] != expectation['revision']:
        raise ValueError('model revision differs from predeclared fixture')
    snapshot = snapshot_download(model_id, revision=spec['revision'],
        allow_patterns=['config.json','tokenizer.json','tokenizer_config.json',
                        'special_tokens_map.json','merges.txt','vocab.json'])
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tok.apply_chat_template = partial(tok.apply_chat_template, enable_thinking=False)
    examples = [make_fixture(tok, n, s, t) for n in (8192,32768) for s in SEEDS for t in TASKS]
    if digest(examples) != expectation['fixture_sha256'] or digest(tok.chat_template) != expectation['chat_template_sha256']:
        raise ValueError('actual pinned-tokenizer fixture differs from declaration')
    return spec, snapshot, tok, examples


def check_prior_validation(root, rows, weights_sha256):
    """A resumed matrix must use the same actual weights as its saved rows."""
    for name in {row['validation_file'] for row in rows}:
        if Path(name).name != name or not name.startswith('validation-') or not name.endswith('.json'):
            raise ValueError('invalid prior validation reference')
        previous = json.loads((root/name).read_text())
        if previous['weights_sha256'] != weights_sha256:
            raise ValueError('resumed model weights differ from prior validation')


def run_policy(engine, ex, policy, trial, parent_path, index, memory=None, selector_method='lexical_rare'):
    """One complete answer; evaluate only after its retrieval/generation choices."""
    model, tok, guard = engine.model, engine.tokenizer, engine.guard
    attempt = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    parent = ColdStore.open(parent_path, readonly=True)
    mount_s = time.perf_counter() - start
    if parent.count != ex['prefix_tokens_count']:
        parent.close()
        raise ValueError('archive prefix extent differs')
    store = None
    counted = None
    stock = policy['attention'] == 'stock'
    sync()
    start = time.perf_counter()
    try:
        if stock:
            engine.adapter.uninstall()
            adapter = BatchedStockSession(model, parent)
            cfg = None
        else:
            cfg = Config(id=policy['id'], model_revision=engine.model_revision,
                         retrieval='fixed', fixed_blocks=(), buffered_writes=True, **WORKING_KV)
            store = ColdStore(engine.scratch/(ex['id']+'-'+policy['id']+f'-t{trial}-'+attempt),
                              engine.layers, engine.h, engine.d, cfg.R, prefix=parent,
                              staged_tensor_writes=True)
            controller = TieredCache(cfg, store, engine.rotary, DEVICE)
            engine.adapter.set_controller(controller)
            adapter = engine.adapter
        counted = CountedSession(adapter, model,
            active_limit=None if stock else cfg.W + cfg.k*cfg.R + cfg.sink_tokens + 1)
        sync()
        init_s = time.perf_counter() - start
        eos_ids = model.generation_config.eos_token_id
        eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        previous_last = None
        previous_response = ''
        generated_total = 0
        rounds = []
        logical_start = logical_store_reads(store if store is not None else parent)
        with HostObservation(scope='complete multi-round request; excludes archive construction, checksum mount, initialization and final checkpoint') as host:
            sync()
            torch.cuda.reset_peak_memory_stats()
            request_start = time.perf_counter()
            memory_before = memory.snapshot() if memory is not None else None
            for turn in range(policy['rounds']):
                if counted.controller.pending_token is not None:
                    counted.consume_pending()
                query, query_detail = next_search(ex['query'], previous_response, tok)
                selected, lookup_detail = [], {}
                lookup_start = time.perf_counter()
                if not stock:
                    # No answer, gold-span metadata or correctness result is used.
                    seeds = index.search(query, None, 4, selector_method)
                    lookup_detail = dict(index.last_search)
                    eligible = (ex['prefix_tokens_count'] + cfg.R - 1) // cfg.R
                    if memory is not None and turn == 0:
                        remembered = memory.addresses(ex['query'], eligible)
                        lookup_detail['remembered_seed_blocks'] = remembered
                        seeds = list(dict.fromkeys([*seeds, *remembered]))[:cfg.k]
                    selected = list(expand_blocks(tuple(seeds), eligible, cfg.k)) if seeds else []
                    controller.select_blocks(selected)
                lookup_s = time.perf_counter() - lookup_start
                prompt = round_request(ex['query'], turn, policy['rounds'], direct=policy['format']=='direct')
                if memory is not None and turn == 0:
                    memory_prompt = memory.prompt()
                    if memory_prompt:
                        prompt += '\n' + memory_prompt
                tokens = (tok.encode(prompt + ex['first_suffix'], add_special_tokens=False) if turn == 0 else
                          next_question_tokens(tok, prompt, previous_last, eos))
                if counted.position + len(tokens) + policy['cap_per_round'] > min(ex['budget'], model.config.max_position_embeddings):
                    raise RuntimeError('declared task/context budget would be exceeded')
                before = counted.metrics()
                logits = counted.feed_prompt(tokens, guard)
                sync()
                first_token_s = time.perf_counter() - request_start
                generated, decode_ms = decode_greedy(counted, logits, policy['cap_per_round'], eos, guard=guard)
                sync()
                response = tok.decode(generated, skip_special_tokens=True)
                previous_last = generated[-1]
                previous_response = response
                generated_total += len(generated)
                parsed = parse_response(response) if policy['format'] == 'json' else None
                # The latest draft is always selected. Invalid JSON is retained
                # and scored as a protocol failure, never repaired using gold.
                answer = response.strip() if policy['format'] == 'direct' else parsed['answer'] if parsed else None
                elapsed = time.perf_counter() - request_start
                rounds.append({'turn': turn, 'prompt_token_ids': tokens, 'generated_token_ids': generated,
                    'response': response, 'parsed_response': parsed, 'answer': answer,
                    'query': query, 'query_details': query_detail, 'selected_blocks': selected,
                    'index_details': lookup_detail, 'lookup_and_selection_s': lookup_s,
                    'cumulative_first_token_s': first_token_s, 'cumulative_request_s': elapsed,
                    'cumulative_generated_tokens': generated_total, 'decode_ms': decode_ms,
                    'stopped_by_eos': previous_last in eos,
                    'work_before_prompt': before, 'cumulative_work': counted.metrics()})
            memory_update = (memory.observe(ex['query'], previous_response,
                                            rounds[-1]['selected_blocks'])
                             if memory is not None else None)
            memory_after = memory.snapshot() if memory is not None else None
            request_s = time.perf_counter() - request_start
            cuda_peak = torch.cuda.max_memory_allocated()
        logical_reads = logical_store_reads(store if store is not None else parent) - logical_start
        checkpoint_start = time.perf_counter()
        if store:
            controller.checkpoint(pending_token=controller.pending_token, token_hash=digest(ex))
        sync()
        checkpoint_s = time.perf_counter() - checkpoint_start
        # Evaluation begins after all model and retrieval decisions are finished.
        for row in rounds:
            row['exact_correct'] = row['answer'] == ex['answer']
            row['answer_contains_gold_diagnostic_only'] = bool(row['answer'] and ex['answer'] in row['answer'])
        return {'status': 'ok', 'example_id': ex['id'], 'task': ex['task'], 'trial': trial,
            'policy': policy, 'attempt': attempt, 'archive_tokens': parent.count,
            'archive_checksum_mount_s': mount_s, 'controller_init_s': init_s,
            'request_s': request_s, 'checkpoint_s': checkpoint_s,
            'request_plus_checkpoint_s': request_s + checkpoint_s,
            'generated_tokens': generated_total, 'rounds': rounds,
            'memory_before': memory_before, 'memory_after': memory_after,
            'memory_update': memory_update,
            'exact_correct': rounds[-1]['exact_correct'], 'prediction': rounds[-1]['answer'],
            'gold_answer': ex['answer'], 'policy_config': asdict(cfg) if cfg else None,
            'logical_request_read_bytes': logical_reads, 'cuda_peak_allocated_bytes': cuda_peak,
            **counted.metrics(), **host.metrics,
            'limitations': 'shared dense archive; current request from initialized cache; metadata warm, KV OS-page-cache condition uncontrolled; no 1M or asymptotic claim'}
    finally:
        if stock:
            parent.close()
            if not hasattr(model, '_tiered_adapter'):
                engine.adapter = ModelAdapter.install(model, DenseController(engine.layers))
        else:
            engine.adapter.set_controller(DenseController(engine.layers))
            if store:
                store.close()
            else:
                parent.close()
        if counted:
            del counted
        # adapter/controller locals are released on return; outer code performs
        # GC before the next independent allocation/timing measurement.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen3-4B-Instruct-2507'])
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--selector', choices=['lexical_rare','semantic_exact'], default='lexical_rare')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec, snapshot, tok, examples = prepare_fixtures(args.model)
    schedule = trial_plan(examples, args.selector)
    semantic = args.selector == 'semantic_exact'
    expected_rows = 100 if semantic else 220
    quality_rows = sum(r['purpose']=='quality' for r in schedule)
    timing_rows = expected_rows-quality_rows
    assert len(schedule) == expected_rows
    if args.plan_only:
        print(json.dumps({'model_id': args.model, 'fixture_sha256': digest(examples),
                          'quality_rows': quality_rows, 'repeated_timing_rows': timing_rows,
                          'total_rows': len(schedule), 'selector': args.selector}))
        return
    if DEVICE != 'cuda':
        raise RuntimeError('CUDA required; no timed CPU/MPS fallback')
    config = spec['config']
    dim = config.get('head_dim', config['hidden_size']//config['num_attention_heads'])
    kv_bytes = 4 * config['num_hidden_layers'] * config['num_key_value_heads'] * dim
    parameters = spec['parameters']
    # Conservative retained disk bound: all dense prefixes, all sparse overlays
    # up to the reserved 2048 tokens, weights, and a 20 GiB operating reserve.
    required_disk = (sum(e['prefix_tokens_count'] for e in examples)*kv_bytes
        + sum(1 for p in schedule if p['policy']['attention']=='sparse')*2048*kv_bytes
        + parameters*2 + (21 if semantic else 20)*2**30)
    if shutil.disk_usage('.').free < required_disk:
        raise RuntimeError(f'insufficient retained-evidence disk capacity: need conservative {required_disk/2**30:.1f} GiB')
    free_gpu, _ = torch.cuda.mem_get_info()
    required_gpu_lower = parameters*2 + max(e['prefix_tokens_count'] for e in examples)*kv_bytes
    if free_gpu < required_gpu_lower + 2**30:
        raise RuntimeError('dense baseline tensor lower bound plus 1 GiB reserve exceeds free GPU memory')
    identity = {'model': spec, 'fixture_sha256': digest(examples), 'runtime': runtime_identity(),
        'implementation_hash': implementation_hash(), 'script_sha256': file_sha(__file__),
        'dependencies_sha256': {p:file_sha(p) for p in ['research/reasoning_protocol.py',
            'research/reasoning_sessions.py','research/persistent_kv.py','research/budget_probe.py',
            'research/semantic_index.py','research/resources.py']}, 'schedule': schedule,
        'scope': 'separate exploratory 1/2/4-round cohort; greedy; stock batched prompts; shared dense immutable archive'}
    encoder = None
    encoder_before = None
    encoder_init_s = None
    if semantic:
        from research.semantic_index import Encoder, ENCODER_ID, ENCODER_REVISION
        from research.semantic_reasoning import semantic_metadata_index
        encoder_start = time.perf_counter()
        encoder = Encoder()
        encoder_before = weights_digest(encoder.model)
        encoder_init_s = time.perf_counter()-encoder_start
        identity['dependencies_sha256']['research/semantic_reasoning.py'] = file_sha('research/semantic_reasoning.py')
        identity['selector'] = {'method': args.selector, 'encoder_id': ENCODER_ID,
            'encoder_revision': ENCODER_REVISION, 'encoder_weights_sha256': encoder_before,
            'cold_cutoff': 'prefix_tokens_count-2000+1', 'text_padding': 32,
            'seed_blocks': 4, 'expanded_block_budget': 16, 'linear_scan': True}
        identity['scope'] = 'post-hoc exact-semantic 1/2-round generation diagnostic; matched fresh stock controls; reused fixtures; linear scan; shared dense immutable archive'
    root = Path('results')/(('semantic-reasoning-' if semantic else 'reasoning-')+args.model.replace('/','--')+'-'+digest(identity)[:12])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        if json.loads((root/'manifest.json').read_text()) != identity:
            raise ValueError('existing run identity differs')
        if json.loads((root/'fixtures.json').read_text()) != examples:
            raise ValueError('existing fixture payload differs')
    else:
        if any(root.iterdir()):
            raise ValueError('existing run lacks identity; preserve and investigate')
        atomic_json(root/'manifest.json', identity)
        atomic_json(root/'fixtures.json', examples)
    output = root/'samples.jsonl'
    rows = [json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
    keys = lambda row: (row['example_id'], row['policy']['id'], row['trial'])
    done = {keys(r) for r in rows if r['status']=='ok'}
    if len(done) != len(rows) or not done <= {keys(r) for r in schedule}:
        raise ValueError('duplicate, failed or unexpected evidence rows require investigation')
    if (root/'complete.json').exists():
        marker = json.loads((root/'complete.json').read_text())
        assert len(done) == marker['rows'] == expected_rows and marker['weights_unchanged']
        if semantic:
            assert marker['encoder_weights_unchanged']
        print('REASONING_ALREADY_COMPLETE', root)
        return
    snapshot = snapshot_download(args.model, revision=spec['revision'],
        allow_patterns=['*.safetensors','*.json','*.model','merges.txt','vocab.txt'])
    with ResourceGuard() as guard:
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'':DEVICE}, attn_implementation='sdpa').eval().requires_grad_(False)
        model.config._commit_hash = spec['revision']
        before = weights_digest(model)
        check_prior_validation(root, rows, before)
        checks = [parity(model, n) for n in (64,512,1025)]
        mechanics = [Config(id='full_buffered', full=True, W=128,C=0,B=32,R=32,k=2,
                            buffered_writes=True,model_revision=spec['revision']),
                     Config(W=128,C=0,B=32,R=32,k=2,retrieval='fixed',fixed_blocks=(0,2),
                            sink_tokens=4,buffered_writes=True,model_revision=spec['revision'])]
        resume = online_smoke(model,tok,root,mechanics,guard,tokens_count=512,checkpoint_pos=256)
        if len(resume)!=2 or any(r['status']!='ok' for r in resume):
            raise RuntimeError('actual-model checkpoint mechanics failed')
        validation = {'weights_sha256':before, 'parity':checks, 'online':resume,
                      'loaded_parameters':sum(p.numel() for p in model.parameters())}
        if semantic:
            validation['encoder'] = {'weights_sha256': encoder_before, 'load_and_hash_s': encoder_init_s}
        if validation['loaded_parameters'] != parameters:
            raise ValueError('actual parameter count differs from declared capacity')
        validation_path = root/('validation-'+uuid.uuid4().hex[:8]+'.json')
        atomic_json(validation_path, validation)
        engine = RunEngine(model,tok,root,guard)
        try:
            for ex in examples:
                pending = [r for r in schedule if r['example_id']==ex['id'] and keys(r) not in done]
                if not pending:
                    continue
                prefix, prefill_s = engine.prefix(ex)
                index_start = time.perf_counter()
                semantic_root = root/'semantic-indexes'
                if semantic:
                    semantic_root.mkdir(exist_ok=True)
                index = (semantic_metadata_index(tok, ex, semantic_root, encoder, encoder_before)
                         if semantic else metadata_index(tok, ex, engine.scratch, 32))
                index_build_or_open_s = time.perf_counter() - index_start
                try:
                    for item in pending:
                        gc.collect(); device_empty_cache()
                        row = run_policy(engine,ex,item['policy'],item['trial'],prefix,index,
                                         selector_method=args.selector)
                        row.update(prefill_s=prefill_s, index_build_or_open_s=index_build_or_open_s,
                                   purpose=item['purpose'], validation_file=validation_path.name)
                        append_jsonl(output,row)
                        done.add(keys(row))
                        print(ex['id'],item['policy']['id'],item['trial'],row['exact_correct'],round(row['request_s'],3),flush=True)
                finally:
                    index.close()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model) != before:
            raise RuntimeError('frozen weights changed')
        if done != {keys(r) for r in schedule}:
            raise RuntimeError('multi-round matrix incomplete')
        completion = {'rows':len(done),'quality_rows':quality_rows,
                      'repeated_timing_rows':timing_rows,'weights_unchanged':True}
        if semantic:
            if weights_digest(encoder.model) != encoder_before:
                raise RuntimeError('frozen query encoder weights changed')
            completion['encoder_weights_unchanged'] = True
        atomic_json(root/'complete.json', completion)
    print('REASONING_DONE',root,flush=True)


if __name__ == '__main__':
    main()

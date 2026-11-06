"""Post-hoc 128-token prefix check across all original persistent questions.

Run in the unchanged 23964ff implementation/runtime. Three consecutive turns
per archive and policy; fresh stock and original sparse controls must reproduce.
This is a correctness/quality diagnostic, with no performance claim.
"""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path

import numpy as np
import torch

from research.persistent_kv import StockSession, next_question_tokens, metadata_index, with_neighbors
from tiered_kv.adapter import DenseController, ModelAdapter
from tiered_kv.cache import TieredCache
from tiered_kv.config import Config, MODEL_REVISION
from tiered_kv.evals import score_needle
from tiered_kv.generate import RunEngine, append_jsonl, decode_greedy, load_model, weights_digest
from tiered_kv.store import ColdStore, atomic_json, file_sha
from tiered_kv.telemetry import DEVICE, ResourceGuard, device_empty_cache, runtime_identity
from tiered_kv.types import digest
from tiered_kv.validation import implementation_hash, parity

SOURCE = '23964ff04123671c230249d7b58ebe7403616b0f'
POLICIES = ('stock_hf', 'rare_context', 'prefix_128')


def plan(source):
    manifest = json.loads((source/'manifest.json').read_text())
    fixture = json.loads((source/'fixtures.json').read_text())
    complete = json.loads((source/'complete.json').read_text())
    rows = [json.loads(x) for x in (source/'samples.jsonl').read_text().splitlines()]
    if complete != {'status':'complete','weights_unchanged':True} or len(rows) != 128:
        raise ValueError('expected original complete persistent study')
    if digest(fixture) != manifest['fixture_sha256'] or manifest['model_revision'] != MODEL_REVISION:
        raise ValueError('source fixture or model identity differs')
    if [e['budget'] for e in fixture['examples']] != [4096,8192,16384,32768]:
        raise ValueError('original context matrix differs')
    controls = {}
    for ex in fixture['examples']:
        if len(ex['queries']) != 3:
            raise ValueError('expected three consecutive original questions')
        for policy in POLICIES[:2]:
            for turn in range(3):
                matches = [r for r in rows if r.get('budget') == ex['budget']
                           and r.get('policy') == policy and r.get('turn') == turn
                           and r.get('condition') == 'warm']
                if len(matches) != 1 or matches[0]['status'] != 'ok':
                    raise ValueError('missing original control')
                controls[(ex['budget'],policy,turn)] = matches[0]
    return manifest, fixture, controls


def run_session(engine, ex, policy, prefix, index, controls, output):
    model, tok, guard = engine.model, engine.tokenizer, engine.guard
    parent = ColdStore.open(prefix, readonly=True)
    if parent.count != ex['prefix_tokens_count']:
        raise ValueError('archive extent differs')
    stock = policy == 'stock_hf'
    store = None
    if stock:
        engine.adapter.uninstall()
        adapter = StockSession(model,parent)
    else:
        original = controls[(ex['budget'],'rare_context',0)]
        cfg = replace(Config(**original['policy_config']), id=policy,
                      sink_tokens=128 if policy=='prefix_128' else 4, fixed_blocks=())
        store = ColdStore(engine.scratch/(ex['id']+'-'+policy),engine.layers,engine.h,engine.d,
                          cfg.R,prefix=parent,staged_tensor_writes=True)
        controller = TieredCache(cfg,store,engine.rotary,DEVICE)
        engine.adapter.set_controller(controller)
        adapter = engine.adapter
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids,list) else [eos_ids])
    previous_last = None
    try:
        for turn,target in enumerate(ex['queries']):
            if adapter.controller.pending_token is not None:
                adapter.consume_pending()
            selected = []
            audits = []
            if not stock:
                seeds = index.search(target['question'],None,4,'lexical_rare')
                selected = list(with_neighbors(seeds,(ex['prefix_tokens_count']+127)//128))
                if selected != controls[(ex['budget'],'rare_context',turn)]['selected_blocks']:
                    raise ValueError('query-only retrieval differs from original control')
                controller.select_blocks(selected)
                first_pos = adapter.position
                def audit(layer,pos,q,view):
                    if len(view.ids) > cfg.W + cfg.k*cfg.R + cfg.sink_tokens + 1:
                        raise RuntimeError('fixed active memory limit exceeded')
                    if pos != first_pos:
                        return
                    ids = view.ids.tolist()
                    present = set(ids)
                    prefix_ids = list(range(cfg.sink_tokens))
                    if not set(prefix_ids) <= present:
                        raise RuntimeError('declared initial prefix missing')
                    src = parent.read_range(layer,0,cfg.sink_tokens)
                    loc = torch.as_tensor([ids.index(i) for i in prefix_ids],device=DEVICE)
                    same_k = torch.equal(view.k.index_select(2,loc).cpu(),src.k)
                    same_v = torch.equal(view.v.index_select(2,loc).cpu(),src.v)
                    if not same_k or not same_v:
                        raise RuntimeError('initial prefix KV differs from archive')
                    audits.append({'layer':layer,'active_entries':len(ids),
                        'prefix_K_bit_equal':same_k,'prefix_V_bit_equal':same_v,
                        'whole_fact_present':set(range(*target['record_span'])) <= present})
                controller.trace = audit
            tokens = (tok.encode(target['question']+ex['first_suffix'],add_special_tokens=False)
                      if turn==0 else next_question_tokens(tok,target['question'],previous_last,eos))
            if adapter.position+len(tokens)+64 > model.config.max_position_embeddings:
                raise ValueError('native context budget exceeded')
            for token in tokens:
                guard.check()
                logits = adapter.step(token,adapter.position)
            if not stock and [r['layer'] for r in audits] != list(range(engine.layers)):
                raise RuntimeError('first-question-token audit missed layers')
            generated,_ = decode_greedy(adapter,logits,64,eos,guard=guard)
            previous_last = generated[-1]
            prediction = tok.decode(generated,skip_special_tokens=True)
            row = {'policy':policy,'budget':ex['budget'],'turn':turn,'question':target['question'],
                'query_token_ids':tokens,'token_ids':generated,'prediction':prediction,
                'answer':target['answer'],'selected_blocks':selected,'payload_checks':audits,
                'position_after':adapter.position,'policy_config':asdict(controller.config) if store else None,
                'max_active_KV_entries':controller.max_entries if store else adapter.position,
                **score_needle(prediction,target['answer'])}
            if policy != 'prefix_128':
                original = controls[(ex['budget'],policy,turn)]
                row['original_prediction_and_length_reproduced'] = (
                    prediction == original['prediction'] and len(generated)==original['output_tokens'])
            append_jsonl(output,row)
            print(ex['budget'],policy,turn,prediction,row['code_found'],flush=True)
            if row.get('original_prediction_and_length_reproduced') is False:
                raise RuntimeError('original control failed to reproduce; stop comparison')
        if store:
            controller.trace = None
            controller.checkpoint(pending_token=controller.pending_token,token_hash=digest(ex))
    finally:
        if stock:
            parent.close()
            engine.adapter = ModelAdapter.install(model,DenseController(engine.layers))
        else:
            engine.adapter.set_controller(DenseController(engine.layers))
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',required=True,type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args = parser.parse_args()
    manifest,fixture,controls = plan(args.source_run)
    if args.plan_only:
        print(json.dumps({'budgets':[e['budget'] for e in fixture['examples']],
                          'policies':POLICIES,'rows':36,'reproduction_controls':24}))
        return
    if DEVICE != 'cuda' or Path('SOURCE_REVISION').read_text().strip() != SOURCE:
        raise ValueError('requires original CUDA source root')
    torch.set_num_threads(manifest['runtime']['threads'])
    torch.set_num_interop_threads(manifest['runtime']['interop_threads'])
    if implementation_hash()!=manifest['source'] or runtime_identity()!=manifest['runtime']:
        raise ValueError('original implementation/runtime differs')
    if (file_sha('research/persistent_kv.py')!=manifest['script_sha256'] or
        file_sha('research/semantic_index.py')!=manifest['index_sha256']):
        raise ValueError('original experiment/index source differs')
    prefixes = {e['id']:Path('artifacts/scratch')/args.source_run.name/(e['id']+'-prefix')
                for e in fixture['examples']}
    for ex in fixture['examples']:
        meta = json.loads((prefixes[ex['id']]/'checkpoint.json').read_text())['metadata']
        if meta['token_hash']!=ex['token_hash'] or meta['model_revision']!=MODEL_REVISION:
            raise ValueError('original checkpoint differs')
    identity = {'source_revision':SOURCE,'implementation_hash':implementation_hash(),
        'script_sha256':file_sha(__file__),'runtime':runtime_identity(),
        'source_files_sha256':{n:file_sha(args.source_run/n) for n in
            ('manifest.json','fixtures.json','samples.jsonl','complete.json','validation.json')},
        'checkpoint_sha256':{name:file_sha(p/'checkpoint.json') for name,p in prefixes.items()},
        'policies':POLICIES,'scope':'post-hoc all twelve original persistent queries; one known prefix-128 win; no performance claim'}
    root = Path('results')/('prefix-sessions-'+digest(identity)[:12])
    root.mkdir(exist_ok=False)
    atomic_json(root/'manifest.json',identity)
    atomic_json(root/'fixtures.json',fixture)
    with ResourceGuard() as guard:
        model,tok = load_model()
        before = weights_digest(model)
        if before != json.loads((args.source_run/'validation.json').read_text())['weights_sha256']:
            raise ValueError('actual weights differ')
        atomic_json(root/'validation.json',{'weights_sha256':before,'parity':parity(model,512)})
        engine = RunEngine(model,tok,root,guard)
        try:
            for ex in fixture['examples']:
                if any(t['answer'] in tok.decode(ex['tokens'][:128],skip_special_tokens=False)
                       for t in ex['queries']):
                    raise ValueError('answer occurs in enlarged prefix')
                index = metadata_index(tok,ex,engine.scratch,32)
                try:
                    for policy in POLICIES:
                        gc.collect(); device_empty_cache()
                        run_session(engine,ex,policy,prefixes[ex['id']],index,controls,root/'samples.jsonl')
                finally:
                    index.close()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model)!=before:
            raise RuntimeError('frozen weights changed')
        rows = [json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines()]
        if len(rows)!=36:
            raise RuntimeError('incomplete prefix-session matrix')
        atomic_json(root/'complete.json',{'rows':36,'weights_unchanged':True,
                    'original_prediction_and_length_reproductions':24})
    print('PREFIX_SESSIONS_DONE',root,flush=True)


if __name__ == '__main__':
    main()

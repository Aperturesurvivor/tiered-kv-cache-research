"""Paired bounded-memory episodes, using the counted multi-round executor.

Fresh per-question model KV; only the declared bounded RAM memory persists.
This hybrid KV-plus-text screen does not add source-text retrieval to prompts.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import shutil
import time
import uuid

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from research.memory_harness import BoundedMemory, STATE_TOKEN_CAP, PROMPT_TOKEN_CAP
from research.memory_protocol import episode, policies, question_fixture, initialize_memory
from research.reasoning_run import prepare_fixtures, run_policy, check_prior_validation
from research.semantic_index import metadata_index
from tiered_kv.config import Config
from tiered_kv.generate import RunEngine, append_jsonl, online_smoke, weights_digest
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.telemetry import DEVICE, ResourceGuard, device_empty_cache, runtime_identity
from tiered_kv.types import digest
from tiered_kv.validation import implementation_hash, parity


def schedule(episodes):
    rng = random.Random(197)
    result = []
    for ep in episodes:
        ordered = policies()
        rng.shuffle(ordered)
        for p in ordered:
            result.append({'episode_id':ep['id'], 'policy':p, 'queries':len(ep['queries'])})
    return result


def restore_episode(memory, episode, policy, rows):
    """Resume only an unbroken sequence of matching memory state transitions."""
    initialize_memory(memory, episode, policy)
    for i, row in enumerate(rows):
        if (row['status'] != 'ok' or row['query_index'] != i
                or row['episode_id'] != episode['id'] or row['policy'] != policy
                or row['memory_before'] != memory.snapshot()):
            raise ValueError('episode memory or query sequence differs')
        memory.restore(row['memory_after'])
    if len(rows) > len(episode['queries']):
        raise ValueError('extra episode rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True,
                        choices=['Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen3-4B-Instruct-2507'])
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    spec, _, tok, fixtures = prepare_fixtures(args.model)
    episodes = [episode(tok, e) for e in fixtures if e['task']=='linked_fact']
    plan = schedule(episodes)
    expected = json.loads(Path('research/memory-fixture-expectations.json').read_text())
    declaration = next(m for m in expected['models'] if m['model_id']==args.model)
    if (file_sha('research/memory_protocol.py') != expected['protocol_sha256']
            or digest(episodes) != declaration['fixture_sha256']
            or spec['revision'] != declaration['revision']):
        raise ValueError('memory fixtures differ from pre-execution declaration')
    total = sum(item['queries'] for item in plan)
    assert len(episodes)==4 and total==160
    if args.plan_only:
        print(json.dumps({'model':args.model, 'fixture_sha256':digest(episodes),
                          'episodes':4, 'policies':8, 'queries_per_episode':5,
                          'natural_memory_rows':120, 'synthetic_stale_rows':40, 'rows':total}))
        return
    if DEVICE != 'cuda':
        raise RuntimeError('CUDA required for generator measurements')
    c = spec['config']
    d = c.get('head_dim', c['hidden_size']//c['num_attention_heads'])
    kv_bytes = 4*c['num_hidden_layers']*c['num_key_value_heads']*d
    sparse_queries = sum(i['queries'] for i in plan if i['policy']['attention']=='sparse')
    need_disk = (sum(e['prefix_tokens_count'] for e in episodes)*kv_bytes
                 + sparse_queries*2048*kv_bytes + 2*spec['parameters'] + 20*2**30)
    if shutil.disk_usage('.').free < need_disk:
        raise RuntimeError(f'insufficient retained evidence capacity: {need_disk/2**30:.1f} GiB required')
    free_gpu, _ = torch.cuda.mem_get_info()
    if free_gpu < 2*spec['parameters'] + max(e['prefix_tokens_count'] for e in episodes)*kv_bytes + 2**30:
        raise RuntimeError('dense tensor lower bound plus reserve exceeds GPU capacity')
    sources = ['research/memory_run.py','research/memory_protocol.py','research/memory_harness.py',
               'research/reasoning_run.py','research/reasoning_sessions.py','research/reasoning_protocol.py',
               'research/budget_probe.py','research/persistent_kv.py','research/semantic_index.py','research/resources.py']
    identity = {'model':spec, 'fixture_sha256':digest(episodes), 'runtime':runtime_identity(),
                'implementation_hash':implementation_hash(),
                'sources_sha256':{p:file_sha(p) for p in sources}, 'schedule':plan,
                'memory_state_token_cap':STATE_TOKEN_CAP, 'memory_prompt_token_cap':PROMPT_TOKEN_CAP,
                'scope':'correlated exploratory episodes; fresh per-query KV; only bounded declared memory persists; synthetic stale cases labeled separately'}
    root = Path('results')/('memory-'+args.model.replace('/','--')+'-'+digest(identity)[:12])
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        if json.loads((root/'manifest.json').read_text()) != identity or json.loads((root/'fixtures.json').read_text()) != episodes:
            raise ValueError('existing run identity differs')
    else:
        if any(root.iterdir()): raise ValueError('unidentified existing output')
        atomic_json(root/'manifest.json',identity); atomic_json(root/'fixtures.json',episodes)
    output = root/'samples.jsonl'
    rows = [json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
    key = lambda r:(r['episode_id'],r['policy']['id'],r['query_index'])
    planned = {(i['episode_id'],i['policy']['id'],q) for i in plan for q in range(i['queries'])}
    if len({key(r) for r in rows}) != len(rows) or not {key(r) for r in rows} <= planned:
        raise ValueError('duplicate or unexpected memory rows')
    for item in plan:
        ep = next(e for e in episodes if e['id']==item['episode_id'])
        prior = [r for r in rows if key(r)[:2]==(ep['id'],item['policy']['id'])]
        restore_episode(BoundedMemory(tok,item['policy']['memory']),ep,item['policy'],prior)
    if (root/'complete.json').exists():
        marker = json.loads((root/'complete.json').read_text())
        assert len(rows)==marker['rows']==total and marker['weights_unchanged']
        print('MEMORY_ALREADY_COMPLETE',root); return
    snapshot = snapshot_download(args.model,revision=spec['revision'],
        allow_patterns=['*.safetensors','*.json','*.model','merges.txt','vocab.txt'])
    with ResourceGuard() as guard:
        model = AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':DEVICE},attn_implementation='sdpa').eval().requires_grad_(False)
        model.config._commit_hash = spec['revision']
        before = weights_digest(model); check_prior_validation(root,rows,before)
        checks = [parity(model,n) for n in (64,512,1025)]
        mechanics = [Config(id='full_buffered',full=True,W=128,C=0,B=32,R=32,k=2,buffered_writes=True,model_revision=spec['revision']),
                     Config(W=128,C=0,B=32,R=32,k=2,retrieval='fixed',fixed_blocks=(0,2),sink_tokens=4,buffered_writes=True,model_revision=spec['revision'])]
        resume = online_smoke(model,tok,root,mechanics,guard,tokens_count=512,checkpoint_pos=256)
        if len(resume)!=2 or any(r['status']!='ok' for r in resume):
            raise RuntimeError('actual-model resume mechanics failed')
        if sum(p.numel() for p in model.parameters()) != spec['parameters']:
            raise ValueError('actual model geometry differs')
        validation_name = 'validation-'+uuid.uuid4().hex[:8]+'.json'
        atomic_json(root/validation_name,dict(weights_sha256=before,parity=checks,online=resume))
        engine = RunEngine(model,tok,root,guard)
        try:
            for ep in episodes:
                prefix,prefill_s = engine.prefix(ep)
                start = time.perf_counter()
                index = metadata_index(tok,ep,engine.scratch,32)
                index_s = time.perf_counter()-start
                try:
                    for item in [i for i in plan if i['episode_id']==ep['id']]:
                        p = item['policy']
                        prior = [r for r in rows if key(r)[:2]==(ep['id'],p['id'])]
                        memory = BoundedMemory(tok,p['memory'])
                        restore_episode(memory,ep,p,prior)
                        for qi in range(len(prior),len(ep['queries'])):
                            gc.collect(); device_empty_cache()
                            row = run_policy(engine,question_fixture(ep,qi),p,0,prefix,index,memory=memory)
                            row.update(episode_id=ep['id'],query_index=qi,validation_file=validation_name,
                                prefill_s=prefill_s,index_build_or_open_s=index_s,
                                purpose='synthetic_stale_memory_stress' if p['memory_seed']=='synthetic_stale' else 'natural_model_memory',
                                comparison='hybrid KV-plus-model-text-memory' if p['memory'] in ('notes','both') else 'no text-memory prompt')
                            append_jsonl(output,row); rows.append(row)
                            print(ep['id'],p['id'],qi,row['exact_correct'],round(row['request_s'],3),flush=True)
                finally:
                    index.close()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model) != before or {key(r) for r in rows} != planned:
            raise RuntimeError('weights changed or matrix incomplete')
        atomic_json(root/'complete.json',dict(rows=total,weights_unchanged=True,natural_memory_rows=120,synthetic_stale_rows=40))
    print('MEMORY_DONE',root,flush=True)


if __name__=='__main__':
    main()

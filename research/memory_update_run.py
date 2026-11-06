"""Counted memory requests across physically appended, versioned KV archives."""
import argparse
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

from research.memory_harness import BoundedMemory
from research.memory_run import restore_episode
from research.memory_update_archive import append_version, validate_append
from research.memory_update_protocol import BUDGETS, SEEDS, make_episode, policies
from research.reasoning_run import check_prior_validation, run_policy
from research.semantic_index import metadata_index
from tiered_kv.config import Config
from tiered_kv.generate import RunEngine, append_jsonl, online_smoke, weights_digest
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.telemetry import DEVICE, ResourceGuard, device_empty_cache, runtime_identity
from tiered_kv.types import digest
from tiered_kv.validation import implementation_hash, parity


def prepare(model_id):
    declaration = json.loads(Path('research/memory-update-fixture-expectations.json').read_text())
    pin = next(m for m in declaration['models'] if m['model_id'] == model_id)
    specs = (json.loads(Path('research/model-manifest.json').read_text())
             + json.loads(Path('research/long-context-candidates.json').read_text())['models'])
    spec = next(m for m in specs if m['model_id'] == model_id)
    if spec['revision'] != pin['revision']:
        raise ValueError('model revision differs from fixture declaration')
    snapshot = snapshot_download(model_id, revision=pin['revision'],
        allow_patterns=['config.json','tokenizer.json','tokenizer_config.json',
                        'special_tokens_map.json','merges.txt','vocab.json'])
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tok.apply_chat_template = partial(tok.apply_chat_template, enable_thinking=False)
    episodes = [make_episode(tok, n, s) for n in BUDGETS for s in SEEDS]
    if (file_sha('research/memory_update_protocol.py') != declaration['protocol_sha256']
            or digest(episodes) != pin['fixture_sha256']
            or digest(tok.chat_template) != pin['chat_template_sha256']):
        raise ValueError('archive-update fixtures differ from declaration')
    return spec, tok, episodes


def schedule(episodes):
    rng = random.Random(239)
    result = []
    for ep in episodes:
        ordered = policies(); rng.shuffle(ordered)
        result.extend(dict(episode_id=ep['id'], policy=p, queries=5) for p in ordered)
    return result


def question_fixture(ep, qi):
    q = ep['queries'][qi]
    version = ep['versions'][q['version']]
    return {**version, 'id':ep['id']+f'-q{qi}', 'task':q['kind'],
            'query':q['query'], 'answer':q['answer'], 'required_record_indices':[q['record']]}


def audit_index(index, tok, version):
    """Check every metadata chunk against only this version's visible tokens."""
    n = version['prefix_tokens_count']
    rows = list(index.db.execute('SELECT id,start,end,text FROM chunks ORDER BY id'))
    if len(rows) != (n+127)//128:
        raise ValueError('index version has missing or extra blocks')
    for i, (bid, start, end, text) in enumerate(rows):
        a, b = max(0, i*128-32), min((i+1)*128+32, n)
        if ((bid,start,end) != (i,a,b)
                or text != tok.decode(version['tokens'][a:b],skip_special_tokens=False)):
            raise ValueError('index contains changed or future version metadata')
    return dict(blocks=len(rows), visible_positions=n, exact_version_metadata=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True,
        choices=['Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen3-4B-Instruct-2507'])
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    spec, tok, episodes = prepare(args.model)
    plan = schedule(episodes)
    c = spec['config']; d = c.get('head_dim', c['hidden_size']//c['num_attention_heads'])
    kv_bytes = 4*c['num_hidden_layers']*c['num_key_value_heads']*d
    disk = (sum(ep['versions'][-1]['prefix_tokens_count'] for ep in episodes)*kv_bytes
            + 40*2048*kv_bytes + 2*spec['parameters'] + 20*2**30)
    gpu = (2*spec['parameters']
           + max(ep['versions'][-1]['prefix_tokens_count'] for ep in episodes)*kv_bytes + 2**30)
    if args.plan_only:
        print(json.dumps(dict(model=args.model, fixture_sha256=digest(episodes), rows=80,
            required_disk_bytes=disk, required_gpu_lower_plus_reserve_bytes=gpu,
            scope='plan only; dense append and bounded query serving are separate')))
        return
    if DEVICE != 'cuda':
        raise RuntimeError('CUDA required for measured archive-update study')
    if shutil.disk_usage('.').free < disk or torch.cuda.mem_get_info()[0] < gpu:
        raise RuntimeError('insufficient retained-evidence disk or dense-baseline GPU capacity')
    sources = ['memory_update_run','memory_update_archive','memory_update_protocol','memory_run',
               'memory_harness','memory_protocol','reasoning_run','reasoning_sessions',
               'reasoning_protocol','persistent_kv','budget_probe','semantic_index','resources']
    identity = dict(model=spec, fixture_sha256=digest(episodes), schedule=plan,
        runtime=runtime_identity(), implementation_hash=implementation_hash(),
        sources_sha256={f'research/{s}.py':file_sha(f'research/{s}.py') for s in sources},
        scope='four correlated update episodes; shared dense append; fresh query KV; bounded fallible memory; no repeated timing')
    root = Path('results')/('memory-update-'+args.model.replace('/','--')+'-'+digest(identity)[:12])
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        if (json.loads((root/'manifest.json').read_text()) != identity
                or json.loads((root/'fixtures.json').read_text()) != episodes):
            raise ValueError('existing update run identity differs')
    else:
        if any(root.iterdir()): raise ValueError('unidentified output retained')
        atomic_json(root/'manifest.json', identity); atomic_json(root/'fixtures.json', episodes)
    output = root/'samples.jsonl'
    rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    key = lambda r:(r['episode_id'],r['policy']['id'],r['query_index'])
    expected = [(i['episode_id'],i['policy']['id'],qi) for i in plan for qi in range(5)]
    if [key(r) for r in rows] != expected[:len(rows)] or len(rows) > 80:
        raise ValueError('missing, duplicate or reordered update rows')
    for item in plan:
        ep = next(e for e in episodes if e['id'] == item['episode_id'])
        restore_episode(BoundedMemory(tok,item['policy']['memory']),ep,item['policy'],
                        [r for r in rows if key(r)[:2] == (ep['id'],item['policy']['id'])])
    if (root/'complete.json').exists():
        marker = json.loads((root/'complete.json').read_text())
        if len(rows) != marker['rows'] or len(rows) != 80 or not marker['weights_unchanged']:
            raise ValueError('inconsistent completed update run')
        print('MEMORY_UPDATE_ALREADY_COMPLETE',root); return
    snapshot = snapshot_download(args.model,revision=spec['revision'],
        allow_patterns=['*.safetensors','*.json','*.model','merges.txt','vocab.txt'])
    with ResourceGuard() as guard:
        model = AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':DEVICE},attn_implementation='sdpa').eval().requires_grad_(False)
        model.config._commit_hash = spec['revision']
        before = weights_digest(model); check_prior_validation(root,rows,before)
        checks = [parity(model,n) for n in (64,512,1025)]
        configs = [Config(id='full_buffered',full=True,W=128,C=0,B=32,R=32,k=2,buffered_writes=True,model_revision=spec['revision']),
                   Config(W=128,C=0,B=32,R=32,k=2,retrieval='fixed',fixed_blocks=(0,2),sink_tokens=4,buffered_writes=True,model_revision=spec['revision'])]
        resume = online_smoke(model,tok,root,configs,guard,tokens_count=512,checkpoint_pos=256)
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != spec['parameters'] or len(resume) != 2 or any(r['status'] != 'ok' for r in resume):
            raise RuntimeError('actual model geometry or resume mechanics failed')
        validation = 'validation-'+uuid.uuid4().hex[:8]+'.json'
        engine = RunEngine(model,tok,root,guard)
        try:
            append_check = validate_append(engine)
            atomic_json(root/validation,dict(weights_sha256=before,parity=checks,online=resume,
                                             loaded_parameters=parameters,append_control=append_check))
            for ep in episodes:
                indices = []
                try:
                    prepared = []; paths = []
                    for vi, version in enumerate(ep['versions']):
                        started = time.perf_counter()
                        version_started = started
                        if vi == 0:
                            target, prefill_s = engine.prefix(version)
                            receipt = dict(prefill_s=prefill_s,scope='initial shared dense archive')
                        else:
                            target, receipt = append_version(engine,ep['versions'][vi-1],version,paths[-1])
                        prepare_wall_s = time.perf_counter()-started
                        paths.append(target)
                        started = time.perf_counter()
                        index = metadata_index(tok,version,engine.scratch,32); indices.append(index)
                        index_s = time.perf_counter()-started
                        index_audit = audit_index(index,tok,version)
                        prepared.append(dict(version=vi,archive=str(target),receipt=receipt,
                            archive_prepare_or_open_wall_s=prepare_wall_s,
                            index_build_or_open_s=index_s,index_audit=index_audit,
                            version_setup_wall_s=time.perf_counter()-version_started,
                            index_mode='full version rebuild or verified reopen; not incremental'))
                    archive_file = f'archives-{ep["id"]}-{uuid.uuid4().hex[:8]}.json'
                    atomic_json(root/archive_file,prepared)
                    for item in [i for i in plan if i['episode_id'] == ep['id']]:
                        p = item['policy']; memory = BoundedMemory(tok,p['memory'])
                        prior = [r for r in rows if key(r)[:2] == (ep['id'],p['id'])]
                        restore_episode(memory,ep,p,prior)
                        for qi in range(len(prior),5):
                            vi = ep['queries'][qi]['version']
                            gc.collect(); device_empty_cache()
                            started = time.perf_counter()
                            row = run_policy(engine,question_fixture(ep,qi),p,0,paths[vi],indices[vi],memory=memory)
                            row.update(episode_id=ep['id'],query_index=qi,archive_version=vi,
                                validation_file=validation,archive_evidence_file=archive_file,
                                policy_call_wall_s=time.perf_counter()-started,
                                prefill_s=prepared[0]['receipt']['prefill_s'],
                                index_build_or_open_s=prepared[vi]['index_build_or_open_s'],
                                purpose='natural_model_memory_archive_update',
                                comparison='same-version archive; dense append shared; memory fallible')
                            append_jsonl(output,row); rows.append(row)
                            print(ep['id'],p['id'],qi,row['exact_correct'],round(row['request_s'],3),flush=True)
                finally:
                    for index in indices:index.close()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model) != before or [key(r) for r in rows] != expected:
            raise RuntimeError('weights changed or update matrix incomplete')
        atomic_json(root/'complete.json',dict(rows=80,weights_unchanged=True))
    print('MEMORY_UPDATE_DONE',root,flush=True)


if __name__ == '__main__':
    main()

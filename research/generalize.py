"""Pinned cross-model screening. Per-model validation precedes all measurements.

python -m research.generalize --model Qwen/Qwen2.5-0.5B-Instruct
No model quality result is pooled across different tokenizers/context limits.
"""
import argparse
from dataclasses import asdict,replace
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
from tiered_kv.adapter import ModelAdapter,DenseController
from tiered_kv.config import Config,DATA_PATH,DATA_REVISION
from tiered_kv.evals import make_needle,make_latency,render,GOV_PROMPT,score_rouge
from tiered_kv.generate import RunEngine,online_smoke,weights_digest,append_jsonl
from tiered_kv.store import atomic_json,file_sha
from tiered_kv.telemetry import ResourceGuard,disk_preflight,runtime_identity,DEVICE,device_empty_cache
from tiered_kv.types import digest
from tiered_kv.validation import parity,implementation_hash
from research.benchmark_stock import measure as stock_measure
from research.resources import HostObservation


def policy_configs(revision):
    return [Config(id='full',full=True,model_revision=revision),
            Config(model_revision=revision),
            Config(id='k16',k=16,model_revision=revision),
            Config(id='k32',k=32,model_revision=revision),
            Config(id='recent_only',C=0,k=0,model_revision=revision),
            Config(id='retrieval_only',C=0,model_revision=revision),
            Config(id='compression_only',k=0,model_revision=revision)]


def make_fixtures(tok,spec,stage):
    budgets=[n for n in (1024,4096,8192,16384,32768) if n<=spec['config']['max_position_embeddings']]
    if stage=='online':
        budgets=[n for n in budgets if n in (4096,8192)]
    seeds=[11] if stage=='screen' else ([23] if stage=='online' else [23,37])
    trials=[0] if stage=='screen' else [1,2]
    examples=[]
    # Begin with latency at each length so memory/speed failure is visible early.
    for n in budgets:
        if stage!='online':
            examples.extend(make_latency(tok,n,t) for t in trials)
        examples.extend(make_needle(tok,n,d,s) for d in (.1,.5,.9) for s in seeds)
    # Complete, independently selected documents; no truncation to force a fit.
    import zipfile
    with zipfile.ZipFile(DATA_PATH) as z:
        docs=[json.loads(line) for line in z.read('data/gov_report.jsonl').decode().splitlines()]
    candidates=[]
    for doc in docs:
        tokens=tok.encode(render(tok,GOV_PROMPT.format(context=doc['context'])),add_special_tokens=False)
        if 4096<=len(tokens)<=min(budgets[-1]-512,32256):
            source_id=str(doc.get('_id',doc.get('id',digest(doc))))
            candidates.append({'id':'gov-'+source_id,'source_id':source_id,'task':'gov_report',
                               'tokens':tokens,'references':doc['answers'],'max_new_tokens':512,
                               'budget':next(n for n in budgets if n>=len(tokens)+512),
                               'band':'low' if len(tokens)<=7680 else 'high'})
    candidates.sort(key=lambda d:hashlib.sha256((d['source_id']+'generalization-41').encode()).hexdigest())
    start=0 if stage=='screen' else 3
    if stage!='online':
        examples.extend(candidates[start:start+3])
    for ex in examples:
        ex['token_hash']=digest(ex['tokens'])
    return {'budgets':budgets,'seeds':seeds,'trials':trials,'examples':examples,
            'complete_reports_available':len(candidates),'template_sha256':digest(tok.chat_template),
            'chat_template_kwargs':{'enable_thinking':False},'data_revision':DATA_REVISION,
            'data_archive_sha256':file_sha(DATA_PATH)}


def run_model(spec,stage):
    disk_preflight()
    # Estimate only download space here; the per-example store forecast follows.
    needed=2*(spec.get('parameters') or 4_000_000_000)
    if shutil.disk_usage('.').free-needed<5e9:
        raise RuntimeError('insufficient disk reserve for model download')
    snapshot=snapshot_download(spec['model_id'],revision=spec['revision'],
        allow_patterns=['*.safetensors','*.json','*.model','merges.txt','vocab.txt'])
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    # Qwen3's default thinking mode would confound strict code-only output scoring.
    tok.apply_chat_template=partial(tok.apply_chat_template,enable_thinking=False)
    fixture=make_fixtures(tok,spec,stage)
    fixture['hash']=digest(fixture)
    configs=policy_configs(spec['revision'])
    if stage=='online':
        configs=[replace(c,protocol='online_tiered') for c in configs if c.id in ('full','default','k32')]
    identity={'model':spec,'stage':stage,'runtime':runtime_identity(),
              'implementation_hash':implementation_hash(),'research_script_sha256':file_sha(__file__),
              'stock_script_sha256':file_sha('research/benchmark_stock.py'),
              'host_script_sha256':file_sha('research/resources.py'),
              'fixture_hash':fixture['hash'],'configs':[asdict(c) for c in configs]}
    slug=spec['model_id'].replace('/','--')
    root=Path('results')/('generalize-'+slug+'-'+stage+'-'+digest(identity)[:10])
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'manifest.json',identity)
    atomic_json(root/'fixtures.json',fixture)
    output=root/'samples.jsonl'
    done={(s['config_id'],s['example_id']) for line in output.read_text().splitlines()
          if (s:=json.loads(line))['status']=='ok'} if output.exists() else set()
    print('MODEL_START',spec['model_id'],stage,str(root),flush=True)
    with ResourceGuard() as guard:
        model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':DEVICE},attn_implementation='sdpa').eval().requires_grad_(False)
        if any(p.device.type!=DEVICE or p.dtype!=torch.float16 for p in model.parameters()):
            raise RuntimeError('unexpected model placement/precision')
        before=weights_digest(model)
        evidence={'model_revision':spec['revision'],'weights_sha256':before,
                  'parameters':sum(p.numel() for p in model.parameters()),'measurements':[]}
        for n in (64,512,1025):
            evidence['measurements'].append(parity(model,n))
            guard.check()
        natural=tok.encode('Read the archive and identify the relevant evidence. Cities, rivers and public reports are described in these records. '*200)
        evidence['measurements'].append(parity(model,1025,natural[:1025],probe='natural_text'))
        online_configs=[Config(id=name,full=name=='full',W=128,C=4,B=32,R=32,k=2,model_revision=spec['revision']) for name in ('full','default')]
        online=online_smoke(model,tok,root,online_configs,guard,tokens_count=512,checkpoint_pos=256)
        if any(r['status']!='ok' for r in online):
            atomic_json(root/'validation.json',{**evidence,'online':online,'status':'failed'})
            raise RuntimeError('actual-model tier transition/resume validation failed')
        atomic_json(root/'validation.json',{**evidence,'online':online,'status':'passed'})
        engine=RunEngine(model,tok,root,guard)
        rng=np.random.default_rng(41)
        try:
            for ex in fixture['examples']:
                pending=[configs[int(i)] for i in rng.permutation(len(configs)) if (configs[int(i)].id,ex['id']) not in done]
                if not pending and (ex['task']!='latency' or ('stock_hf',ex['id']) in done):
                    continue
                bytes_per_token=2*engine.layers*engine.h*engine.d*2
                estimate=len(ex['tokens'])*bytes_per_token+len(pending)*(ex['max_new_tokens']*bytes_per_token+64*1024**2)
                if stage=='online':
                    estimate+=len(pending)*len(ex['tokens'])*bytes_per_token
                if shutil.disk_usage('.').free-estimate<5e9:
                    raise RuntimeError('insufficient disk reserve for next example')
                prefix=engine.prefix(ex) if pending and stage!='online' else None
                for cfg in pending:
                    with HostObservation() as host:
                        row=engine.run(cfg,ex,prefix_info=prefix,protocol=cfg.protocol)
                    row.update(host.metrics)
                    row.update(model_id=spec['model_id'],model_revision=spec['revision'],stage=stage,
                               runtime=identity['runtime'],fixture_hash=fixture['hash'])
                    append_jsonl(output,row)
                    print(json.dumps({k:row.get(k) for k in ('model_id','config_id','example_id','status','em','rougeL','failure_reason')}),flush=True)
                    if row['status']!='ok':
                        raise RuntimeError(row['failure_reason'])
                    done.add((cfg.id,ex['id']))
                if ex['task']=='latency' and ('stock_hf',ex['id']) not in done:
                    engine.adapter.uninstall()
                    try:
                        with HostObservation() as host:
                            row=stock_measure(model,ex,guard)
                        row.update(host.metrics)
                    finally:
                        engine.adapter=ModelAdapter.install(model,DenseController(engine.layers))
                    row.update(config_id='stock_hf',task='latency',model_id=spec['model_id'],model_revision=spec['revision'],stage=stage)
                    append_jsonl(output,row)
                    print('STOCK',ex['id'],float(np.median(row['latency_ms'])),flush=True)
                    done.add(('stock_hf',ex['id']))
                gc.collect()
                device_empty_cache()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model)!=before:
            raise RuntimeError('model weights changed')
        atomic_json(root/'complete.json',{'weights_unchanged':True,'rows':len(done),
                    'model_id':spec['model_id'],'fixture_hash':fixture['hash']})
    print('MODEL_DONE',spec['model_id'],stage,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--stage',choices=['screen','confirm','online'],default='screen')
    args=parser.parse_args()
    specs=json.loads(Path('research/model-manifest.json').read_text())
    run_model(next(s for s in specs if s['model_id']==args.model),args.stage)


if __name__=='__main__':
    main()

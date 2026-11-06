"""Post-hoc neighboring-context budget sweep on completed 32K Qwen cases.

Run only in the frozen 4d17602 root, with other GPU jobs idle. The original
query-selected blocks seed a deterministic distance-based expansion; answer
spans never influence selection. This is not held-out policy confirmation.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from research.benchmark_stock import measure as stock_measure
from research.resources import HostObservation
from tiered_kv.adapter import ModelAdapter, DenseController
from tiered_kv.config import Config, MODEL_REVISION
from tiered_kv.generate import RunEngine, append_jsonl, load_model, weights_digest
from tiered_kv.store import atomic_json, file_sha
from tiered_kv.telemetry import DEVICE, ResourceGuard, device_empty_cache, runtime_identity
from tiered_kv.types import digest
from tiered_kv.validation import implementation_hash, parity

SOURCE='4d17602a0339735630922e012938532202135dd1'
BUDGETS=(4,16,32)
ORDER_SEED=137


def expand_blocks(seeds, eligible, capacity):
    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(b<0 or b>=eligible for b in seeds):
        raise ValueError('expected distinct eligible query-selected seed blocks')
    if capacity<len(seeds):
        raise ValueError('budget cannot remove original selected blocks')
    # Preserve the original underfilled k4 condition exactly for reproduction.
    if capacity==4:
        return seeds
    candidates=sorted((b for b in range(eligible) if b not in seeds),
                      key=lambda b:(min(abs(b-s) for s in seeds),b))
    return seeds+tuple(candidates[:capacity-len(seeds)])


def plan(source_run):
    source_run=Path(source_run)
    manifest=json.loads((source_run/'manifest.json').read_text())
    marker=json.loads((source_run/'complete.json').read_text())
    fixture=json.loads((source_run/'fixtures.json').read_text())
    rows=[json.loads(line) for line in (source_run/'samples.jsonl').read_text().splitlines()]
    if (manifest['stage']!='confirm' or manifest['model']['model_id']!='Qwen/Qwen2.5-1.5B-Instruct'
            or manifest['model']['revision']!=MODEL_REVISION or len(rows)!=264
            or marker!={'rows':264,'weights_unchanged':True} or digest(fixture)!=manifest['fixture_hash']):
        raise ValueError('requires immutable completed Qwen1.5 confirmation')
    examples=[ex for ex in fixture['examples'] if ex['budget']==32768]
    if len(examples)!=12 or any(ex['task']!='niah' for ex in examples):
        raise ValueError('expected all twelve 32K factual cases')
    baseline={}
    configurations={}
    for ex in examples:
        selected=[r for r in rows if r['example_id']==ex['id'] and r['config_id']=='rare_context']
        if len(selected)!=1 or selected[0]['status']!='ok':
            raise ValueError('original sparse control missing')
        baseline[ex['id']]=selected[0]
        cfg=Config(**selected[0]['policy'])
        if cfg.C!=0 or cfg.W!=2000 or cfg.k!=4 or cfg.retrieval!='fixed':
            raise ValueError('source policy differs')
        eligible=(max(0,ex['prefix_tokens_count']-cfg.W+1)+cfg.R-1)//cfg.R
        configurations[ex['id']]=[
            replace(cfg,id=cfg.id if k==4 else f'rare_context_k{k}',k=k,
                    fixed_blocks=expand_blocks(cfg.fixed_blocks,eligible,k)) for k in BUDGETS]
    return manifest,examples,baseline,configurations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    source,examples,baseline,configurations=plan(args.source_run)
    selections={key:[asdict(cfg) for cfg in value] for key,value in configurations.items()}
    if args.plan_only:
        print(json.dumps({'examples':len(examples),'quality_rows':36,'timing_rows':12,
                          'budgets':BUDGETS,'selections':selections}))
        return
    if DEVICE!='cuda' or Path('SOURCE_REVISION').read_text().strip()!=SOURCE:
        raise ValueError('requires original 4d17602 CUDA root')
    torch.set_num_threads(source['runtime']['threads'])
    torch.set_num_interop_threads(source['runtime']['interop_threads'])
    if implementation_hash()!=source['implementation_hash'] or runtime_identity()!=source['runtime']:
        raise ValueError('frozen implementation/runtime differs')
    for key,name in {'script_sha256':'demand_kv.py','index_sha256':'semantic_index.py',
                     'stock_sha256':'benchmark_stock.py','host_sha256':'resources.py'}.items():
        if file_sha(Path('research')/name)!=source[key]:
            raise ValueError('frozen research dependency differs: '+name)
    prefixes={}
    for ex in examples:
        prefix=Path('artifacts/scratch')/args.source_run.name/(ex['id']+'-prefix')
        metadata=json.loads((prefix/'checkpoint.json').read_text())['metadata']
        if metadata['token_hash']!=ex['token_hash'] or metadata['model_revision']!=MODEL_REVISION:
            raise ValueError('source archive identity differs')
        prefixes[ex['id']]=(prefix,metadata['prefill_s'])
    identity={'source_revision':SOURCE,'implementation_hash':implementation_hash(),
        'script_sha256':file_sha(__file__),'runtime':runtime_identity(),
        'source_files_sha256':{name:file_sha(args.source_run/name) for name in
            ('manifest.json','fixtures.json','samples.jsonl','complete.json','validation.json')},
        'checkpoint_sha256':{key:file_sha(value[0]/'checkpoint.json') for key,value in prefixes.items()},
        'selections':selections,'order_seed':ORDER_SEED,'expected_rows':48,
        'scope':'post-hoc nearby-context budget sweep; original cases reused, not held-out confirmation; no index-search timing'}
    root=Path('results')/('budget-probe-'+digest(identity)[:12])
    root.mkdir(exist_ok=False)
    atomic_json(root/'manifest.json',identity)
    atomic_json(root/'fixtures.json',{'examples':examples})
    output=root/'samples.jsonl'
    count=0
    reproduced=0
    with ResourceGuard() as guard:
        model,tok=load_model()
        before=weights_digest(model)
        if before!=json.loads((args.source_run/'validation.json').read_text())['weights_sha256']:
            raise ValueError('loaded weights differ')
        atomic_json(root/'validation.json',{'parity':parity(model,512),'weights_sha256':before})
        engine=RunEngine(model,tok,root,guard)
        rng=np.random.default_rng(ORDER_SEED)
        try:
            for ex in examples:
                for index in rng.permutation(3):
                    cfg=configurations[ex['id']][int(index)]
                    with HostObservation() as host:
                        row=engine.run(cfg,ex,protocol='query_replay',prefix_info=prefixes[ex['id']])
                    row.update(host.metrics,policy=asdict(cfg),probe='neighbor_budget',post_hoc=True)
                    if cfg.k==4 and row['status']=='ok':
                        row['original_token_ids_reproduced']=row['token_ids']==baseline[ex['id']]['token_ids']
                        reproduced+=int(row['original_token_ids_reproduced'])
                    append_jsonl(output,row);count+=1
                    if row['status']!='ok' or row.get('original_token_ids_reproduced') is False:
                        raise RuntimeError('source reproduction or quality execution failed; preserve evidence')
                    print('BUDGET_QUALITY',ex['id'],cfg.k,row['code_found'],flush=True)
                    device_empty_cache()
            ex=next(ex for ex in examples if ex['id']=='niah-32768-0.5-53')
            continuation=tok.encode(' The archive contains numbered cabinets and blue folders.'*40,
                                    add_special_tokens=False)[:min(144,model.config.max_position_embeddings-len(ex['tokens']))]
            for trial in range(3):
                example={**ex,'id':ex['id']+f'-budget-forced-{trial}','task':'latency',
                         'trial':trial,'continuation':continuation}
                for index in rng.permutation(4):
                    index=int(index)
                    if index==3:
                        engine.adapter.uninstall()
                        try:
                            with HostObservation() as host:
                                row=stock_measure(model,example,guard)
                            row.update(host.metrics,config_id='stock_hf',task='latency',post_hoc=True)
                        finally:
                            engine.adapter=ModelAdapter.install(model,DenseController(engine.layers))
                    else:
                        cfg=configurations[ex['id']][index]
                        with HostObservation() as host:
                            row=engine.run(cfg,example,protocol='query_replay',prefix_info=prefixes[ex['id']])
                        row.update(host.metrics,policy=asdict(cfg),post_hoc=True)
                    append_jsonl(output,row);count+=1
                    if row['status']!='ok':
                        raise RuntimeError('timing execution failed')
                    print('BUDGET_TIMING',trial,row['config_id'],flush=True)
                    device_empty_cache()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model)!=before or reproduced!=12 or count!=48:
            raise RuntimeError('completion, original-output reproduction or frozen-weight check failed')
        atomic_json(root/'complete.json',{'rows':count,'weights_unchanged':True,'original_k4_token_sequences_reproduced':reproduced})
    print('BUDGET_PROBE_DONE',root,flush=True)


if __name__=='__main__':
    main()

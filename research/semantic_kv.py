"""Test text-indexed exact disk KV; retrieved text never enters the prompt.

The entire question is replayed through the tested cache, so a dense query KV
cannot carry an answer into the sparse control before retrieval is enabled.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
from tiered_kv.adapter import DenseController,ModelAdapter
from tiered_kv.config import Config
from tiered_kv.generate import RunEngine,load_model,weights_digest,online_smoke,append_jsonl
from tiered_kv.store import atomic_json,file_sha
from tiered_kv.telemetry import ResourceGuard,runtime_identity,device_empty_cache
from tiered_kv.types import digest
from tiered_kv.validation import parity,implementation_hash
from research.resources import HostObservation
from research.benchmark_stock import measure as stock_measure


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--index-results',type=Path,required=True)
    args=parser.parse_args()
    fixture=json.loads((args.index_results/'fixtures.json').read_text())
    coverage=json.loads((args.index_results/'coverage.json').read_text())
    index_manifest=json.loads((args.index_results/'manifest.json').read_text())
    lookup={(r['example_id'],r['method'],r['k']):r for r in coverage}
    identity={'implementation_hash':implementation_hash(),'runtime':runtime_identity(),
        'script_sha256':file_sha(__file__),'stock_sha256':file_sha('research/benchmark_stock.py'),
        'host_sha256':file_sha('research/resources.py'),'index_manifest':index_manifest,
        'coverage_sha256':file_sha(args.index_results/'coverage.json'),
        'fixture_hash':digest(fixture),'protocol':'query_replay',
        'scope':'shared exact archive; complete question uses tested policy; fixed retrieved KV stays resident through answer',
        'timing_boundary':'initial archive prefill, controller initialization, index construction, query embedding/search, question replay and steady decode are separate'}
    root=Path('results')/('semantic-kv-'+digest(identity)[:12])
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'manifest.json',identity)
    shutil.copy2(args.index_results/'fixtures.json',root/'fixtures.json')
    output=root/'samples.jsonl'
    done={(r['config_id'],r['example_id']) for line in output.read_text().splitlines()
          if (r:=json.loads(line))['status']=='ok'} if output.exists() else set()
    with ResourceGuard() as guard:
        model,tok=load_model()
        before=weights_digest(model)
        checks=[parity(model,n) for n in (64,512,1025)]
        mechanics=[Config(id='full',full=True,W=128,C=4,B=32,R=32,k=2),
                   Config(W=128,C=4,B=32,R=32,k=2,retrieval='fixed',fixed_blocks=(0,2))]
        online=online_smoke(model,tok,root,mechanics,guard,tokens_count=512,checkpoint_pos=256)
        if any(r['status']!='ok' for r in online):
            raise RuntimeError('fixed-address actual-model resume validation failed')
        atomic_json(root/'validation.json',{'status':'passed','parity':checks,'online':online,'weights_sha256':before})
        engine=RunEngine(model,tok,root,guard)
        rng=np.random.default_rng(41)
        try:
            for original in fixture['examples']:
                if original['task']!='niah':
                    continue
                policies=[Config(id='full',full=True,protocol='query_replay'),
                          Config(protocol='query_replay'),
                          Config(id='k32',k=32,protocol='query_replay')]
                for method in ('lexical','semantic','hybrid_bounded'):
                    row=lookup[original['id'],method,4]
                    policies.append(Config(id=method+'_k4',protocol='query_replay',retrieval='fixed',fixed_blocks=tuple(row['selected_blocks'])))
                policies.append(Config(id='oracle_k4',protocol='query_replay',retrieval='fixed',fixed_blocks=tuple(original['gold_blocks'])))
                examples=[original]
                if original['depth']==.5 and original['seed']==23 and original['id'].startswith('niah-'):
                    examples.append({**original,'id':original['id']+'-forced','task':'latency','trial':0,
                        'continuation':tok.encode(' The archive contains numbered cabinets and blue folders.'*40,add_special_tokens=False)[:min(144,model.config.max_position_embeddings-len(original['tokens']))],
                        'max_new_tokens':144})
                prefix=None
                for ex in examples:
                    pending=[policies[int(i)] for i in rng.permutation(len(policies)) if (policies[int(i)].id,ex['id']) not in done]
                    if pending and prefix is None:
                        prefix=engine.prefix(original)
                    for cfg in pending:
                        with HostObservation() as host:
                            row=engine.run(cfg,ex,protocol='query_replay',prefix_info=prefix)
                        row.update(host.metrics,policy=asdict(cfg),oracle=cfg.id=='oracle_k4',external_index_memory_included=False)
                        if cfg.retrieval=='fixed' and cfg.id!='oracle_k4':
                            selection=lookup[original['id'],cfg.id[:-3],4]
                            row['index_query_evidence']=selection
                        append_jsonl(output,row)
                        print(json.dumps({key:row.get(key) for key in ('example_id','config_id','status','em','failure_reason')}),flush=True)
                        if row['status']!='ok':
                            raise RuntimeError(row['failure_reason'])
                        done.add((cfg.id,ex['id']))
                    if ex['task']=='latency' and ('stock_hf',ex['id']) not in done:
                        engine.adapter.uninstall()
                        try:
                            with HostObservation() as host:
                                row=stock_measure(model,ex,guard)
                            row.update(host.metrics,config_id='stock_hf',task='latency',protocol='query_replay')
                            append_jsonl(output,row)
                            done.add(('stock_hf',ex['id']))
                        finally:
                            engine.adapter=ModelAdapter.install(model,DenseController(engine.layers))
                    device_empty_cache()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model)!=before:
            raise RuntimeError('frozen model weights changed')
        atomic_json(root/'complete.json',{'rows':len(done),'weights_unchanged':True})
    print('SEMANTIC_KV_DONE',root,flush=True)


if __name__=='__main__':
    main()

"""Paired actual-model check of the unnecessary eviction CPU round trip."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import numpy as np
from tiered_kv.config import Config
from tiered_kv.generate import RunEngine,load_model,weights_digest,append_jsonl
from tiered_kv.telemetry import DEVICE,ResourceGuard,runtime_identity,device_empty_cache
from tiered_kv.tiers import RecentTier
from tiered_kv.types import digest
from tiered_kv.store import atomic_json,file_sha
from tiered_kv.validation import parity,implementation_hash
from research.resources import HostObservation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--semantic-root',type=Path,required=True)
    args=p.parse_args()
    if DEVICE!='cuda':
        raise RuntimeError('eviction timing requires actual CUDA')
    runs=list((args.semantic_root/'results').glob('semantic-kv-*/complete.json'))
    indices=list((args.semantic_root/'results/semantic-index-screen').glob('*/complete.json'))
    if len(runs)!=1 or len(indices)!=1:
        raise ValueError('expected one completed source study and index')
    fixture=json.loads((runs[0].parent/'fixtures.json').read_text())
    ex=next(e for e in fixture['examples'] if e['id']=='niah-32768-0.5-23')
    coverage=json.loads((indices[0].parent/'coverage.json').read_text())
    blocks=next(r['selected_blocks'] for r in coverage if r['example_id']==ex['id'] and r['method']=='lexical' and r['k']==4)
    prefix=args.semantic_root/'artifacts/scratch'/runs[0].parent.name/(ex['id']+'-prefix')
    metadata=json.loads((prefix/'checkpoint.json').read_text())['metadata']
    if metadata['token_hash']!=ex['token_hash']:
        raise ValueError('source prefix/fixture mismatch')
    identity={'implementation_hash':implementation_hash(),'script_sha256':file_sha(__file__),
              'fixture_hash':digest(ex),'source_manifest_sha256':file_sha(runs[0].parent/'manifest.json'),
              'runtime':runtime_identity(),'scope':'same selected exact KV; paired CPU/device eviction handling',
              'trials':3,'order_seed':101,'selected_blocks':blocks}
    root=Path('results')/('eviction-'+digest(identity)[:12])
    root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():
        print('EVICTION_ALREADY_COMPLETE',root,flush=True)
        return
    if (root/'samples.jsonl').exists():
        raise RuntimeError('partial paired probe retained; use fresh execution identity before retry')
    atomic_json(root/'manifest.json',identity)
    output=root/'samples.jsonl'
    original=RecentTier.prepare
    def cpu_roundtrip(self,layer,pos):
        old,recent=original(self,layer,pos)
        return old.cpu(),recent
    with ResourceGuard() as guard:
        model,tok=load_model()
        before=weights_digest(model)
        atomic_json(root/'validation.json',{'parity':parity(model,512),'weights_sha256':before})
        engine=RunEngine(model,tok,root,guard)
        rng=np.random.default_rng(101)
        pairs=[]
        try:
            for trial in range(3):
                example={**ex,'id':ex['id']+f'-paired-{trial}','task':'latency','trial':trial,
                    'continuation':tok.encode(' The archive contains numbered cabinets and blue folders.'*40,add_special_tokens=False)[:128]}
                measured={}
                for mode in rng.permutation(['cpu_evictions','device_evictions']):
                    mode=str(mode)
                    RecentTier.prepare=cpu_roundtrip if mode=='cpu_evictions' else original
                    config=Config(id=mode,C=0,retrieval='fixed',fixed_blocks=tuple(blocks),sink_tokens=4,
                                  buffered_writes=True,protocol='query_replay')
                    with HostObservation() as host:
                        row=engine.run(config,example,protocol='query_replay',prefix_info=(prefix,metadata['prefill_s']))
                    row.update(host.metrics,policy=asdict(config))
                    append_jsonl(output,row)
                    if row['status']!='ok':
                        raise RuntimeError(row['failure_reason'])
                    measured[mode]=row
                    device_empty_cache()
                if measured['cpu_evictions']['selected_hash']!=measured['device_evictions']['selected_hash']:
                    raise RuntimeError('paired forced-token predictions differ')
                pair={'trial':trial,'predictions_identical':True,**{
                    mode:statistics.mean(row['latency_ms']) for mode,row in measured.items()}}
                pairs.append(pair)
                print('EVICTION_PAIR',json.dumps(pair),flush=True)
        finally:
            RecentTier.prepare=original
            engine.adapter.uninstall()
        if weights_digest(model)!=before:
            raise RuntimeError('frozen weights changed')
        atomic_json(root/'complete.json',{'rows':6,'weights_unchanged':True,'pairs':pairs})
    print('EVICTION_DONE',root,flush=True)


if __name__=='__main__':
    main()

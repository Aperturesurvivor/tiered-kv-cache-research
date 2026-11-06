"""Check online bound scorer against all saved sparse-oracle screen probes."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import psutil
import torch
from research.attention_trajectory_diagnostic import ROOT, file_sha
from research.attention_quest_reference import QuestBoundIndex
from research.attention_representations import _normalization_positions
from tiered_kv.types import KVBatch

EVIDENCE=ROOT/'results/evidence/attention-representation-screen-1'
HASHES={'trace.json':'e61c77db51e43f09a74191f6fe2d9a70bb170fdd3cb74dccb794bef74f80be65',
        'trace.npz':'4d96eeab4683b76305901bc208b29cbb191c965224204c910c471c1511b65bd8',
        'screen.jsonl':'cd125a3b90fe144ed955420d6b7186b66f16caee7462dc8b9963f042bf5977c0',
        'complete.json':'8844e0302617afefabace4be6713c65b05325d5b7744c9ffe65c08c9e85f6918'}
CHECKPOINT='8e5885af3dedbcfe85f9975d39bdee8372e42f772a313a7df28ffb0b6b24ed5a'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prefix',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    assert all(file_sha(EVIDENCE/name)==value for name,value in HASHES.items())
    assert file_sha(args.prefix/'checkpoint.json')==CHECKPOINT
    rows=[json.loads(x) for x in (EVIDENCE/'screen.jsonl').read_text().splitlines()];assert len(rows)==168
    if not args.execute:
        print(json.dumps({'status':'plan','rows':len(rows),'checkpoint_sha256':CHECKPOINT}));return
    args.output.mkdir(parents=True,exist_ok=False)
    from tiered_kv.store import ColdStore,atomic_json
    torch.set_num_threads(1);soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    cap=psutil.Process().memory_info().vms+512*2**20
    if hard!=resource.RLIM_INFINITY:cap=min(cap,hard)
    if soft!=resource.RLIM_INFINITY:cap=min(cap,soft)
    resource.setrlimit(resource.RLIMIT_AS,(cap,hard))
    def timeout(*_):raise TimeoutError('screen comparison exceeded 300 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300);start=time.monotonic();parent=None
    try:
        parent=ColdStore.open(args.prefix,readonly=True)
        assert (parent.layers,parent.h,parent.d,parent.count)==(28,2,128,30698)
        done=0
        with np.load(EVIDENCE/'trace.npz',allow_pickle=False) as data:
            for layer in range(28):
                prefix=parent.read_range(layer,0,parent.count).k[0].numpy();delta=data[f'l{layer}-delta']
                for row in (x for x in rows if x['layer']==layer):
                    pos=row['position'];past=np.concatenate((prefix,delta[:,:pos-parent.count,:]),axis=1)
                    q=data[f'p{pos}-l{layer}-q'];current=data[f'p{pos}-l{layer}-current']
                    assert past.shape==(2,pos,128) and q.shape==(12,128) and current.shape==(2,1,128)
                    cutoff=pos-2000+1;index=QuestBoundIndex(1,2,128,128,128)
                    values=torch.from_numpy(past[:,:cutoff,:].copy())[None]
                    index.absorb(0,KVBatch(np.arange(cutoff),values,torch.zeros_like(values)))
                    full=np.concatenate((past,current),axis=1)
                    fixed=full[:,_normalization_positions(pos+1,cutoff,128,128),:]
                    index.stage(0,torch.from_numpy(fixed.copy()))
                    actual,_=index.search(0,torch.from_numpy(q)[None,:,None,:],15)
                    expected=row['results']['quest_bound']['pages']
                    if actual.tolist()!=expected:
                        atomic_json(args.output/'mismatch.json',dict(status='failed',layer=layer,position=pos,
                            expected=expected,actual=actual.tolist(),compared=done))
                        raise ValueError('online scorer differs from recorded screened ranking')
                    done+=1
                print(json.dumps({'phase':'layer_checked','layer':layer,'compared':done}),flush=True)
        assert done==168
        atomic_json(args.output/'complete.json',dict(status='complete',compared=done,
            screen_sha256=HASHES['screen.jsonl'],trace_sha256=HASHES['trace.npz'],checkpoint_sha256=CHECKPOINT,
            quest_source_sha256=file_sha(ROOT/'research/attention_quest_reference.py'),
            checker_source_sha256=file_sha(Path(__file__)),elapsed_s=time.monotonic()-start,
            scope='offline recorded sparse Q/K scorer equivalence; does not establish free-generation quality or latency'))
    finally:
        signal.alarm(0)
        if parent:parent.close()
        resource.setrlimit(resource.RLIMIT_AS,(soft,hard))


if __name__=='__main__':main()

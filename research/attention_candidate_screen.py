"""Offline 16/32/64 Quest-pool exact-QK reranking on 168 saved probes."""
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
from research.attention_trajectory_diagnostic import ROOT, WEIGHTS, load_plan, validate_source, file_sha
from research.attention_representation_trace import baseline,probe_plan,CASE,CHECKPOINT_SHA,REPORT_SHA
from research.attention_candidate_rerank import score_candidates,POOLS
from research.attention_quest_screen_check import EVIDENCE,HASHES

WORKSPACE=512*2**20
CPU_SECONDS=300
MAX_ROWS_BYTES=12*2**20


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    plan=load_plan(CASE);source=validate_source();rounds=baseline()
    assert all(file_sha(EVIDENCE/name)==digest for name,digest in HASHES.items())
    assert file_sha(args.prefix/'checkpoint.json')==CHECKPOINT_SHA
    metadata=json.loads((EVIDENCE/'trace.json').read_text())
    assert metadata['status']=='complete' and metadata['case']==CASE and metadata['weights_sha256']==WEIGHTS
    assert metadata['source_hashes']==source and metadata['prefix_checkpoint_sha256']==CHECKPOINT_SHA
    assert metadata['baseline_report_sha256']==REPORT_SHA and metadata['trace_sha256']==HASHES['trace.npz']
    positions={int(p):label for p,label in metadata['positions'].items()};assert positions==probe_plan(rounds)
    screen=[json.loads(x) for x in (EVIDENCE/'screen.jsonl').read_text().splitlines()];assert len(screen)==168
    expected={(r['position'],r['layer']):r['results'] for r in screen};assert len(expected)==168
    if not args.execute:
        print(json.dumps(dict(status='plan',case=CASE,probes=len(positions),layers=28,pools=POOLS,final_pages=15,
                              workspace_limit_bytes=WORKSPACE,deadline_s=CPU_SECONDS)));return
    args.output.mkdir(parents=True,exist_ok=False)
    from tiered_kv.store import ColdStore,atomic_json
    torch.set_num_threads(1);soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    cap=psutil.Process().memory_info().vms+WORKSPACE
    if hard!=resource.RLIM_INFINITY:cap=min(cap,hard)
    if soft!=resource.RLIM_INFINITY:cap=min(cap,soft)
    resource.setrlimit(resource.RLIMIT_AS,(cap,hard))
    def timeout(*_):raise TimeoutError('candidate screen exceeded 300 seconds; partial rows retained')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(CPU_SECONDS);started=time.monotonic();parent=None
    try:
        parent=ColdStore.open(args.prefix,readonly=True)
        assert (parent.layers,parent.h,parent.d,parent.count)==(28,2,128,metadata['prefix_count'])
        required={i//128 for j in plan['fixture']['required_record_indices'] for i in range(*plan['fixture']['record_spans'][j])}
        done=0
        with np.load(EVIDENCE/'trace.npz',allow_pickle=False) as trace:
            for layer in range(28):
                prefix=parent.read_range(layer,0,parent.count).k[0].numpy()
                delta=trace[f'l{layer}-delta'];assert delta.shape==(2,metadata['continuation_count'],128) and delta.dtype==np.float16
                for pos,label in sorted(positions.items()):
                    assert time.monotonic()-started<CPU_SECONDS
                    past=np.concatenate((prefix,delta[:,:pos-parent.count,:]),axis=1)
                    q=trace[f'p{pos}-l{layer}-q'];current=trace[f'p{pos}-l{layer}-current']
                    assert past.shape==(2,pos,128) and q.shape==(12,128) and current.shape==(2,1,128)
                    result=score_candidates(q,past,current,pos)
                    assert result['quest_top15']==expected[(pos,layer)]['quest_bound']['pages'], 'Quest ranking differs from saved screen'
                    assert result['oracle_pages']==expected[(pos,layer)]['oracle']['pages'], 'oracle ranking differs from saved screen'
                    # Ground truth is used only after all six page selections.
                    for entry in result['results']:
                        entry['candidate_required_pages_covered']=len(set(entry['candidate_pages'])&required)
                        for arm in entry['arms'].values():
                            arm['required_pages_covered']=len(set(arm['pages'])&required)
                            arm['required_pages_total']=len(required)
                            assert arm['active_tokens']==4048
                    row=dict(status='partial',position=pos,probe=label,layer=layer,**result)
                    encoded=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
                    rowfile=args.output/'screen.jsonl'
                    assert (rowfile.stat().st_size if rowfile.exists() else 0)+len(encoded)<=MAX_ROWS_BYTES
                    with rowfile.open('ab') as out:out.write(encoded);out.flush();os.fsync(out.fileno())
                    done+=1
                print(json.dumps({'phase':'layer_complete','layer':layer,'probes':done}),flush=True)
        assert done==168
        atomic_json(args.output/'complete.json',dict(status='complete',case=CASE,probes=done,settings_per_probe=6,
            candidate_ceilings=list(POOLS),final_pages=15,active_tokens=4048,
            trace_sha256=HASHES['trace.npz'],prior_screen_sha256=HASHES['screen.jsonl'],
            checkpoint_sha256=CHECKPOINT_SHA,rows_sha256=file_sha(args.output/'screen.jsonl'),source_hashes=source,
            candidate_source_sha256=file_sha(ROOT/'research/attention_candidate_rerank.py'),
            runner_source_sha256=file_sha(Path(__file__)),workspace_limit_bytes=WORKSPACE,address_space_limit_bytes=cap,
            cpu_elapsed_s=time.monotonic()-started,
            limitations=['full Quest metadata scan and full-key evaluation are diagnostic references',
                'candidate QK and fixed QK are recomputed separately and compared to full logits; only the control uses full-key normalizers',
                'one replayed sparse-oracle trajectory is not free-generation quality or deployable speed evidence',
                'candidate_kv_payload_bytes is hypothetical uncompressed K/V payload, not measured physical I/O',
                'required-page coverage is computed only after selection']))
        print(json.dumps({'status':'complete','probes':done}),flush=True)
    finally:
        signal.alarm(0)
        if parent:parent.close()
        resource.setrlimit(resource.RLIMIT_AS,(soft,hard))


if __name__=='__main__':main()

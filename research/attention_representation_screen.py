"""Offline full-scan representation screen from a verified oracle trace.

One probe/layer at a time; no index candidates or online answer information.
"""
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
from research.attention_representation_trace import baseline, probe_plan, CASE, CHECKPOINT_SHA, REPORT_SHA, MAX_ARTIFACT
from research.attention_representations import compare_representations

WORKSPACE=512*2**20
CPU_SECONDS=300


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=Path,required=True);ap.add_argument('--trace',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();plan=load_plan(CASE);source=validate_source();rounds=baseline()
    assert file_sha(args.prefix/'checkpoint.json')==CHECKPOINT_SHA
    metadata=json.loads((args.trace/'trace.json').read_text())
    assert metadata['status']=='complete' and metadata['case']==CASE
    assert metadata['weights_sha256']==WEIGHTS and metadata['source_hashes']==source
    assert metadata['prefix_checkpoint_sha256']==CHECKPOINT_SHA and metadata['baseline_report_sha256']==REPORT_SHA
    assert metadata['trace_source_sha256']==file_sha(ROOT/'research/attention_representation_trace.py')
    assert metadata['oracle_source_sha256']==file_sha(ROOT/'research/attention_fullkey_oracle.py')
    trace_file=args.trace/'trace.npz';assert trace_file.stat().st_size<=MAX_ARTIFACT and file_sha(trace_file)==metadata['trace_sha256']
    positions={int(p):label for p,label in metadata['positions'].items()}
    assert positions==probe_plan(rounds) and 0<len(positions)<=6
    assert metadata['layers']==28 and metadata['continuation_count']<=768
    assert metadata['trace_bytes']==trace_file.stat().st_size
    assert metadata['routing_events_checked']==sum(len(r['selected_events']) for r in rounds)
    assert metadata['generated_argmax_checks']==sum(len(r['generated_token_ids']) for r in rounds)
    if not args.execute:
        print(json.dumps(dict(case=CASE,probes=len(positions),layers=28,workspace_limit_bytes=WORKSPACE,cpu_deadline_s=CPU_SECONDS)));return
    args.output.mkdir(parents=True,exist_ok=False)
    from tiered_kv.store import ColdStore,atomic_json
    torch.set_num_threads(1)
    # Enforce a 512 MiB additional virtual-address allocation budget over the
    # initialized process. This is not a cap on process baseline RSS.
    baseline_vms=psutil.Process().memory_info().vms
    soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    cap=baseline_vms+WORKSPACE
    if hard!=resource.RLIM_INFINITY:cap=min(cap,hard)
    if soft!=resource.RLIM_INFINITY:cap=min(cap,soft)
    resource.setrlimit(resource.RLIMIT_AS,(cap,hard))
    def timeout(*_):raise TimeoutError('representation CPU deadline exceeded; partial rows retained')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(CPU_SECONDS)
    started=time.monotonic();parent=None
    try:
        parent=ColdStore.open(args.prefix,readonly=True)
        assert parent.count==metadata['prefix_count'] and parent.count==plan['fixture']['prefix_tokens_count']
        assert parent.layers==28 and parent.h==2 and parent.d==128
        required={i//128 for j in plan['fixture']['required_record_indices'] for i in range(*plan['fixture']['record_spans'][j])}
        expected={(e['position'],e['layer']):e['pages'] for r in rounds for e in r['selected_events'] if e['position'] in positions}
        assert len(expected)==len(positions)*28
        rows=0
        with np.load(trace_file,allow_pickle=False) as trace:
            for layer in range(28):
                prefix=parent.read_range(layer,0,parent.count).k[0].numpy()
                delta=trace[f'l{layer}-delta'];assert delta.shape==(2,metadata['continuation_count'],128) and delta.dtype==np.float16
                for pos,label in sorted(positions.items()):
                    assert time.monotonic()-started<CPU_SECONDS
                    past=np.concatenate((prefix,delta[:,:pos-parent.count,:]),axis=1)
                    q=trace[f'p{pos}-l{layer}-q'];current=trace[f'p{pos}-l{layer}-current']
                    assert past.shape==(2,pos,128) and q.shape==(12,128) and current.shape==(2,1,128)
                    result=compare_representations(q,past,current,pos)
                    assert result['oracle']['pages']==expected[(pos,layer)], 'float32 CPU oracle ordering differs from recorded GPU trajectory'
                    # Gold enters only after every selection is finalized.
                    for item in result.values():
                        item['required_pages_covered']=len(set(item['pages'])&required)
                        item['required_pages_total']=len(required)
                    row=dict(status='partial',position=pos,probe=label,layer=layer,results=result)
                    encoded=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
                    rowfile=args.output/'screen.jsonl'
                    assert (rowfile.stat().st_size if rowfile.exists() else 0)+len(encoded)<=8*2**20
                    with rowfile.open('ab') as out:
                        out.write(encoded);out.flush();os.fsync(out.fileno())
                    rows+=1
                print(json.dumps({'phase':'layer_complete','layer':layer,'rows':rows}),flush=True)
        assert rows==len(positions)*28
        atomic_json(args.output/'complete.json',dict(status='complete',case=CASE,rows=rows,trace_sha256=metadata['trace_sha256'],
            screen_sha256=file_sha(args.output/'screen.jsonl'),source_hashes=source,
            screen_source_sha256=file_sha(Path(__file__)),representation_source_sha256=file_sha(ROOT/'research/attention_representations.py'),
            cpu_elapsed_s=time.monotonic()-started,workspace_limit_bytes=WORKSPACE,address_space_limit_bytes=cap,
            normalizer='approximate complete cold-page mass plus exact sink/recent/current and unselectable partial cold boundary; no true complete-cold oracle normalization',
            limitations=['all representations are full scans; no ANN candidate misses measured',
                'one replayed sparse-oracle trajectory, not free-generation quality evidence',
                'CPU float32 oracle page IDs are required to match saved GPU output; backend differences fail closed',
                'required-record coverage is retrospective and never selects pages']))
        print(json.dumps({'status':'complete','rows':rows}),flush=True)
    finally:
        signal.alarm(0)
        if parent:parent.close()
        resource.setrlimit(resource.RLIMIT_AS,(soft,hard))


if __name__=='__main__':main()

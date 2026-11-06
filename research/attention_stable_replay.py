"""Evaluation-only stable mass replay for unchanged candidate-screen selections."""
import argparse
import json
import math
import os
import platform
from pathlib import Path
import resource
import signal
import statistics
import time
import numpy as np
import psutil
import torch
from research.attention_trajectory_diagnostic import ROOT,WEIGHTS,load_plan,validate_source,file_sha
from research.attention_representation_trace import baseline,probe_plan,CASE,CHECKPOINT_SHA,REPORT_SHA
from research.attention_quest_screen_check import EVIDENCE,HASHES
from research.attention_stable_metrics import stable_metrics

CAND=ROOT/'results/evidence/attention-candidate-rerank-1'
ROWS_SHA='054f57a77bd6c4cd3f4e8ad9cc0d5bc49ebf9c71b2279b4eb74a74e07c42281f'
MARKER_SHA='47c815e7409f1bcc5f87cc9d955caf42e5e3702b9e32880923a9ebc6bd0775ab'
WORKSPACE=512*2**20
CPU_SECONDS=300


def selection_choices(row,prior):
    """Verify frozen selections/structure; never recalculate candidate choices."""
    pos=row['position'];cutoff=pos-2000+1
    assert row['quest_top15']==prior['quest_bound']['pages'] and row['oracle_pages']==prior['oracle']['pages']
    assert [e['candidate_ceiling'] for e in row['results']]==[16,32,64]
    choices={'quest':row['quest_top15'],'oracle':row['oracle_pages']};previous=[]
    for entry in row['results']:
        n=entry['candidate_ceiling'];pool=entry['candidate_pages']
        assert len(pool)==len(set(pool))==entry['candidate_count']==n and pool[:15]==row['quest_top15']
        assert pool[:len(previous)]==previous;previous=pool
        assert all(type(b) is int and b>=1 and (b+1)*128<=cutoff for b in pool)
        assert entry['candidate_key_positions']==n*128 and entry['candidate_kv_payload_bytes']==n*128*2*2*128*2
        assert set(entry['arms'])=={'candidate_only','full_key_normalizer_control'}
        for name,arm in entry['arms'].items():
            pages=arm['pages'];assert len(pages)==len(set(pages))==arm['selected_count']==15
            assert set(pages)<=set(pool) and arm['active_tokens']==4048
            assert arm['oracle_overlap']==len(set(pages)&set(row['oracle_pages']))
            choices[f'{n}-{name}']=pages
    return choices


def summarize(rows,positions):
    def mean_mass(items,name):return statistics.mean(v for row in items for v in row['metrics'][name]['retained_mass'])
    q=mean_mass(rows,'quest');o=mean_mass(rows,'oracle');gap=o-q;assert gap>0
    eligibility=[]
    for n in (16,32,64):
        name=f'{n}-candidate_only';mean=mean_mass(rows,name)
        worst=min(mean_mass([r for r in rows if r['position']==pos],name)-
                  mean_mass([r for r in rows if r['position']==pos],'quest') for pos in positions)
        closure=(mean-q)/gap
        eligibility.append(dict(candidate_ceiling=n,mean=mean,closure=closure,worst_probe_delta=worst,
                                qualifies=closure>=0.5 and worst>=-2e-6))
    return dict(quest_mean=q,oracle_mean=o,eligibility=eligibility,
                selected=next((x['candidate_ceiling'] for x in eligibility if x['qualifies']),None),
                rule='candidate-only >=50% mean full-mass gap closure and no probe worse than Quest by >2e-6; smallest qualifying pool')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prefix',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    plan=load_plan(CASE);source=validate_source();rounds=baseline()
    assert all(file_sha(EVIDENCE/name)==value for name,value in HASHES.items())
    assert file_sha(CAND/'screen.jsonl')==ROWS_SHA and file_sha(CAND/'complete.json')==MARKER_SHA
    assert file_sha(args.prefix/'checkpoint.json')==CHECKPOINT_SHA
    marker=json.loads((CAND/'complete.json').read_text())
    assert marker['status']=='complete' and marker['probes']==168 and marker['settings_per_probe']==6
    assert marker['rows_sha256']==ROWS_SHA and marker['source_hashes']==source
    assert marker['candidate_source_sha256']==file_sha(ROOT/'research/attention_candidate_rerank.py')
    assert marker['runner_source_sha256']==file_sha(ROOT/'research/attention_candidate_screen.py')
    trace_meta=json.loads((EVIDENCE/'trace.json').read_text())
    assert trace_meta['weights_sha256']==WEIGHTS and trace_meta['prefix_checkpoint_sha256']==CHECKPOINT_SHA
    assert trace_meta['baseline_report_sha256']==REPORT_SHA and trace_meta['trace_sha256']==HASHES['trace.npz']
    positions={int(p):label for p,label in trace_meta['positions'].items()};assert positions==probe_plan(rounds)
    original=[json.loads(x) for x in (CAND/'screen.jsonl').read_text().splitlines()];assert len(original)==168
    prior={(r['position'],r['layer']):r['results'] for r in map(json.loads,(EVIDENCE/'screen.jsonl').read_text().splitlines())};assert len(prior)==168
    seen=set()
    for row in original:
        key=(row['position'],row['layer']);assert key in prior and key not in seen;seen.add(key)
        assert row['status']=='partial' and row['probe']==positions[row['position']]
        selection_choices(row,prior[key])
    assert len(seen)==168
    if not args.execute:
        print(json.dumps(dict(status='plan',probes=len(positions),layers=28,unchanged_selections=8,workspace_limit_bytes=WORKSPACE,deadline_s=CPU_SECONDS)));return
    args.output.mkdir(parents=True,exist_ok=False)
    from tiered_kv.store import ColdStore,atomic_json
    torch.set_num_threads(1);soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    cap=psutil.Process().memory_info().vms+WORKSPACE
    if hard!=resource.RLIM_INFINITY:cap=min(cap,hard)
    if soft!=resource.RLIM_INFINITY:cap=min(cap,soft)
    resource.setrlimit(resource.RLIMIT_AS,(cap,hard))
    def timeout(*_):raise TimeoutError('stable evaluation exceeded 300 seconds; partial rows retained')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(CPU_SECONDS);start=time.monotonic();parent=None
    try:
        parent=ColdStore.open(args.prefix,readonly=True)
        assert (parent.layers,parent.h,parent.d,parent.count)==(28,2,128,trace_meta['prefix_count'])
        output=[]
        with np.load(EVIDENCE/'trace.npz',allow_pickle=False) as trace:
            for layer in range(28):
                prefix=parent.read_range(layer,0,parent.count).k[0].numpy()
                delta=trace[f'l{layer}-delta'];assert delta.shape==(2,trace_meta['continuation_count'],128) and delta.dtype==np.float16
                for row in (r for r in original if r['layer']==layer):
                    pos=row['position'];past=np.concatenate((prefix,delta[:,:pos-parent.count,:]),axis=1)
                    q=trace[f'p{pos}-l{layer}-q'];current=trace[f'p{pos}-l{layer}-current']
                    assert past.shape==(2,pos,128) and q.shape==(12,128) and current.shape==(2,1,128)
                    keys=torch.from_numpy(np.concatenate((past,current),axis=1)).float()
                    logits=torch.bmm(torch.from_numpy(q).float().reshape(2,6,128),keys.transpose(1,2)).reshape(12,pos+1).numpy()/np.float32(math.sqrt(128))
                    assert logits.dtype==np.float32
                    result=stable_metrics(logits,pos,selection_choices(row,prior[(pos,layer)]))
                    record=dict(status='partial',position=pos,probe=row['probe'],layer=layer,**result)
                    encoded=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode();target=args.output/'stable.jsonl'
                    assert (target.stat().st_size if target.exists() else 0)+len(encoded)<=12*2**20
                    with target.open('ab') as out:out.write(encoded);out.flush();os.fsync(out.fileno())
                    output.append(record)
                print(json.dumps({'phase':'layer_complete','layer':layer,'probes':len(output)}),flush=True)
        assert len(output)==168
        summary=summarize(output,positions)
        atomic_json(args.output/'complete.json',dict(status='complete',case=CASE,probes=len(output),
            input_rows_sha256=ROWS_SHA,prior_screen_sha256=HASHES['screen.jsonl'],trace_sha256=HASHES['trace.npz'],
            checkpoint_sha256=CHECKPOINT_SHA,rows_sha256=file_sha(args.output/'stable.jsonl'),source_hashes=source,
            evaluator_sha256=file_sha(ROOT/'research/attention_stable_metrics.py'),runner_sha256=file_sha(Path(__file__)),
            cpu_elapsed_s=time.monotonic()-start,workspace_limit_bytes=WORKSPACE,address_space_limit_bytes=cap,
            runtime={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'torch_threads':torch.get_num_threads()},
            max_normalization_abs_error=max(abs(v-1) for row in output for v in row['normalization_sums']),selection_rule=summary,
            limitations=['float32 QK retained; float64 max-shift-and-divide softmax for evaluation only',
                'no selection, candidate, budget, input or original result was changed',
                'same 168 correlated layer/probes from one oracle trajectory; not generation or latency evidence']))
        print(json.dumps({'status':'complete','probes':len(output),'selected':summary['selected']}),flush=True)
    finally:
        signal.alarm(0)
        if parent:parent.close()
        resource.setrlimit(resource.RLIMIT_AS,(soft,hard))


if __name__=='__main__':main()

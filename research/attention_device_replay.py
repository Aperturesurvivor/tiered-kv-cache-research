"""Compare bounded device summaries against the frozen CPU selector on saved probes.

No model inference, answer data, or selector mutation. Completion records all
168 comparisons; equivalence additionally requires declared numerical and ordered
ranking checks. GPU arithmetic may disagree even when the replay completes.
"""
import argparse, json, math, os, signal, subprocess, time
from pathlib import Path
import numpy as np
import torch
from research.attention_device_summaries import score_summaries
from research.attention_rerank_screen_check import EVIDENCE, HASHES, CHECKPOINT
from research.attention_stable_replay import CAND, ROWS_SHA, MARKER_SHA
from research.attention_trajectory_diagnostic import ROOT, file_sha, validate_source
from research.attention_quest_reference import QuestBoundIndex
from research.attention_representations import _lse, _normalization_positions
from tiered_kv.types import KVBatch
from tiered_kv.store import ColdStore, atomic_json

ATOL=2e-5
RTOL=2e-5
PAGE=128
POOL=32
FINAL=15
SOURCE_HASHES={
    'research/attention_device_summaries.py':'69a98c16452cf62e1bd555ebed758ea402ec9ead52bbbef0526f5f2b76c6e757',
    'research/attention_quest_reference.py':'a97ecdfcc3fb5652b02649ac00b10899e7a2f851a111e4c93342ff66f49c6ce1',
    'research/attention_representations.py':'2a9765aa4ea39a2ee46ccb99974027c444172b0453a57d7bf66f5856ca7eab85',
    'research/attention_rerank_screen_check.py':'a7b5111c1d576aa115d27628ca9c141d7a2868d364d036471a763265b82f20b4',
    'research/attention_stable_replay.py':'8faefe65c079b08299b7f3f9d839f1a3a7e08a0fe1f2573ad99e5bdfca76a338'}


def rank(logmass, fixed, ids, cap):
    """Frozen shared-head formula; expose every exact tie before ID tie-break."""
    if not len(ids):return [], [], []
    denom=np.logaddexp(fixed,_lse(logmass,axis=1))
    scores=np.exp(logmass-denom[:,None]).sum(axis=0)
    ordered=np.lexsort((ids,-scores)); ties=[]
    for a,b in zip(ordered[:-1],ordered[1:]):
        if scores[a]==scores[b]:ties.append([int(ids[a]),int(ids[b])])
    return ids[ordered[:cap]].tolist(),scores.tolist(),ties


def compare_arrays(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid numerical comparison')
    mask=~np.isclose(b,a,atol=ATOL,rtol=RTOL)
    mismatches=[{'index':idx.tolist(),'cpu':float(a[tuple(idx)]),'cuda':float(b[tuple(idx)])} for idx in np.argwhere(mask)]
    return {'max_abs':float(np.max(np.abs(a-b))) if a.size else 0.0,
            'allclose':not bool(mismatches),'mismatches':mismatches}


def aggregate(rows, expected=168):
    if len(rows)!=expected or len({(r['layer'],r['position']) for r in rows})!=expected:
        raise ValueError('incomplete or duplicate probe matrix')
    return {'status':'complete','compared':len(rows),
        'equivalent':all(r['candidate_match'] and r['final_match'] and all(x['allclose'] for x in r['numerics'].values()) for r in rows),
        'candidate_disagreements':sum(not r['candidate_match'] for r in rows),
        'final_disagreements':sum(not r['final_match'] for r in rows),
        'numeric_disagreements':sum(not all(x['allclose'] for x in r['numerics'].values()) for r in rows),
        'ties_present':sum(any(r['ties'].values()) for r in rows)}


def project(row):
    return {'layer':row['layer'],'position':row['position'],'candidate_pages':row['results'][1]['candidate_pages'],
            'final_pages':row['results'][1]['arms']['candidate_only']['pages']}


def summaries(q,keys,fixed,device):
    qt=torch.from_numpy(q.copy()).to(device); kt=torch.from_numpy(keys.copy()).to(device); ft=torch.from_numpy(fixed.copy()).to(device)
    page,normal,meta=score_summaries(qt,kt,ft)
    return page.cpu().numpy(),normal.cpu().numpy(),meta


def cpu_reference(q,keys,fixed):
    grouped=torch.from_numpy(q.copy()).float().reshape(2,6,128)
    f=torch.from_numpy(fixed.copy()).float()
    fixed_logits=torch.bmm(grouped,f.transpose(1,2)).reshape(12,-1).numpy()/np.float32(math.sqrt(128))
    if keys.shape[1]:
        k=torch.from_numpy(keys.copy()).float().reshape(2,-1,128)
        logits=torch.bmm(grouped,k.transpose(1,2)).reshape(12,keys.shape[1],PAGE).numpy()/np.float32(math.sqrt(128))
        masses=_lse(logits,axis=2)
    else:masses=np.empty((12,0),np.float32)
    return masses,_lse(fixed_logits,axis=1)


def route_pair(bound,ids,cpu_fixed,gpu_fixed,get_final_scores,pool=POOL,final=FINAL):
    """Route each path independently; injected scoring enables CPU-only tests."""
    cpu_ids,cpu_scores,cpu_ties=rank(bound,cpu_fixed,ids,pool)
    gpu_ids,gpu_scores,gpu_ties=rank(bound,gpu_fixed,ids,pool)
    cpu_blocks,cpu_mass,cpu_norm=get_final_scores('cpu',cpu_ids)
    gpu_blocks,gpu_mass,gpu_norm=get_final_scores('cuda',gpu_ids)
    cpu_final,cpu_final_scores,cpu_final_ties=rank(cpu_mass,cpu_norm,cpu_blocks,final)
    gpu_final,gpu_final_scores,gpu_final_ties=rank(gpu_mass,gpu_norm,gpu_blocks,final)
    return locals()


def replay_one(layer,pos,past,q,current,expected):
    cutoff=pos-2000+1
    assert 2000 < pos <= 32768 and past.shape==(2,pos,128) and q.shape==(12,128) and current.shape==(2,1,128)
    index=QuestBoundIndex(1,2,128,PAGE,128)
    values=torch.from_numpy(past[:,:cutoff,:].copy())[None]
    index.absorb(0,KVBatch(np.arange(cutoff),values,torch.zeros_like(values)))
    fixed=np.concatenate((past,current),axis=1)[:,_normalization_positions(pos+1,cutoff,128,PAGE),:]
    ids=np.flatnonzero((index.counts[0]==PAGE)&(np.arange(len(index.counts[0]))*PAGE>=128))
    grouped=q.astype(np.float32).reshape(2,6,128); lo=index.sums[0][ids];hi=index.maxima[0][ids]
    upper=(np.einsum('kgd,kpd->kgp',np.maximum(grouped,0),hi.transpose(1,0,2))+np.einsum('kgd,kpd->kgp',np.minimum(grouped,0),lo.transpose(1,0,2))).reshape(12,len(ids))/np.float32(math.sqrt(128))
    bound=upper+np.float32(math.log(PAGE))
    empty=np.empty((2,0,PAGE,128),dtype=past.dtype)
    _,cpu_fixed=cpu_reference(q,empty,fixed);_,gpu_fixed,meta=summaries(q,empty,fixed,'cuda')
    # The CPU candidate ordering is independently checked against the actual frozen implementation.
    index.stage(0,torch.from_numpy(fixed.copy())); reference,_=index.search(0,torch.from_numpy(q)[None,:,None,:],POOL)
    def candidate_keys(selected):
        blocks=np.asarray(sorted(selected),dtype=np.int64)
        positions=(blocks[:,None]*PAGE+np.arange(PAGE)[None,:]).reshape(-1)
        return blocks,past[:,positions,:].reshape(2,len(blocks),PAGE,128)
    call_accounting={'candidate_fixed_only':meta}
    def final_scores(device,selected):
        blocks,keys=candidate_keys(selected)
        if device=='cpu':mass,norm=cpu_reference(q,keys,fixed)
        else:
            mass,norm,call_accounting['final_cuda_pool']=summaries(q,keys,fixed,'cuda')
        return blocks,mass,norm
    routed=route_pair(bound,ids,cpu_fixed,gpu_fixed,final_scores)
    cpu_ids=routed['cpu_ids'];gpu_ids=routed['gpu_ids']
    if cpu_ids!=reference.tolist() or cpu_ids!=expected['candidate_pages']:
        raise ValueError(f'CPU candidate reference mismatch at layer {layer}, position {pos}: computed={cpu_ids}, frozen={reference.tolist()}, saved={expected["candidate_pages"]}')
    cpu_blocks,cpu_keys=candidate_keys(cpu_ids);gpu_blocks,gpu_keys=candidate_keys(gpu_ids)
    cpu_mass=routed['cpu_mass'];gpu_mass=routed['gpu_mass']
    # Compare the same candidate keys numerically even when candidate rankings differ.
    gpu_same,gpu_same_fixed,call_accounting['audit_cpu_pool']=summaries(q,cpu_keys,fixed,'cuda')
    cpu_gpu_pool,_=cpu_reference(q,gpu_keys,fixed)
    cpu_final=routed['cpu_final'];gpu_final=routed['gpu_final']
    if cpu_final!=expected['final_pages']:
        raise ValueError(f'CPU final reference mismatch at layer {layer}, position {pos}: computed={cpu_final}, saved={expected["final_pages"]}')
    return {'layer':layer,'position':pos,'candidate_cpu':cpu_ids,'candidate_cuda':gpu_ids,
        'final_cpu':cpu_final,'final_cuda':gpu_final,'candidate_match':cpu_ids==gpu_ids,'final_match':cpu_final==gpu_final,
        'numerics':{'candidate_fixed_actual':compare_arrays(cpu_fixed,gpu_fixed),'final_fixed_actual':compare_arrays(routed['cpu_norm'],routed['gpu_norm']),
                    'fixed_repeated':compare_arrays(cpu_fixed,gpu_same_fixed),'page_logmass_cpu_pool':compare_arrays(cpu_mass,gpu_same),
                    'page_logmass_cuda_pool':compare_arrays(cpu_gpu_pool,gpu_mass)},
        'ties':{'candidate_cpu':routed['cpu_ties'],'candidate_cuda':routed['gpu_ties'],'final_cpu':routed['cpu_final_ties'],'final_cuda':routed['gpu_final_ties']},
        'candidate_score_ids':ids.tolist(),'candidate_scores_cpu':routed['cpu_scores'],'candidate_scores_cuda':routed['gpu_scores'],
        'final_score_ids_cpu':cpu_blocks.tolist(),'final_scores_cpu':routed['cpu_final_scores'],
        'final_score_ids_cuda':gpu_blocks.tolist(),'final_scores_cuda':routed['gpu_final_scores'],
        'accounting':{'device_summary_calls':call_accounting,'scope':'logical helper inputs and outputs per call, including repeated audit; not measured transfers, traffic or peak workspace'}}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prefix',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    assert all(file_sha(EVIDENCE/n)==h for n,h in HASHES.items()) and file_sha(a.prefix/'checkpoint.json')==CHECKPOINT
    assert file_sha(CAND/'screen.jsonl')==ROWS_SHA and file_sha(CAND/'complete.json')==MARKER_SHA
    assert all(file_sha(ROOT/name)==value for name,value in SOURCE_HASHES.items())
    source=validate_source()
    # The hash-bound source rows include retrospective gold fields. Only this
    # answer-free projection enters replay_one or either scoring path.
    raw=[json.loads(x) for x in (CAND/'screen.jsonl').read_text().splitlines()]
    rows=[project(r) for r in raw]
    del raw
    assert len(rows)==168 and len({(x['layer'],x['position']) for x in rows})==168
    hashes={'trace':HASHES['trace.npz'],'checkpoint':CHECKPOINT,'candidate_rows':ROWS_SHA,'candidate_marker':MARKER_SHA,'sources':SOURCE_HASHES,'runner':file_sha(Path(__file__)),'frozen_helpers':source}
    if not a.execute:print(json.dumps({'status':'plan','probes':len(rows),'hashes':hashes}));return
    assert torch.cuda.is_available() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert torch.cuda.device_count()==1 and torch.cuda.current_device()==0 and torch.cuda.get_device_name(0)=='Tesla T4'
    free,total=torch.cuda.mem_get_info();assert free>=2*2**30
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.monotonic();results=[]
    torch.cuda.reset_peak_memory_stats();parent=None;current_probe=None;unsaved_result=None
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('replay exceeded 600 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        parent=ColdStore.open(a.prefix,readonly=True)
        assert (parent.layers,parent.h,parent.d,parent.count)==(28,2,128,30698)
        with np.load(EVIDENCE/'trace.npz',allow_pickle=False) as data:
            for layer in range(28):
                prefix=parent.read_range(layer,0,parent.count).k[0].numpy();delta=data[f'l{layer}-delta']
                for row in (x for x in rows if x['layer']==layer):
                    pos=row['position'];current_probe={'layer':layer,'position':pos};past=np.concatenate((prefix,delta[:,:pos-parent.count,:]),axis=1)
                    result=replay_one(layer,pos,past,data[f'p{pos}-l{layer}-q'],data[f'p{pos}-l{layer}-current'],row)
                    unsaved_result=result
                    data_line=json.dumps(result,allow_nan=False,sort_keys=True)+'\n'
                    target=a.output/'partial.jsonl'
                    assert (target.stat().st_size if target.exists() else 0)+len(data_line.encode())<=64*2**20
                    with target.open('a') as f:f.write(data_line);f.flush();os.fsync(f.fileno())
                    results.append(result);unsaved_result=None
                    assert torch.cuda.max_memory_allocated()<=512*2**20
                print(json.dumps({'layer':layer,'compared':len(results)}),flush=True)
        summary=aggregate(results);summary.update(hashes=hashes,atol=ATOL,rtol=RTOL,elapsed_s=time.monotonic()-start,
            torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(0),cuda_device_count=torch.cuda.device_count(),cuda_current_device=torch.cuda.current_device(),
            matmul_precision=torch.get_float32_matmul_precision(),tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
            gpu_free_before=free,gpu_total=total,gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),
            rows_sha256=file_sha(a.output/'partial.jsonl'),scope='numerical replay only; no generation, quality, or latency claim')
        atomic_json(a.output/'complete.json',summary)
    except Exception as exc:
        atomic_json(a.output/'failure.json',{'status':'failed','probe':current_probe,'completed_rows':len(results),'error':repr(exc),'unsaved_probe_result':unsaved_result,'hashes':hashes})
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        if parent:parent.close()

if __name__=='__main__':main()

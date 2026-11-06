"""Full-scan page representations, evaluated on captured Q/K only.

All outputs are float32 diagnostics, not candidate-index or online policies.
"""
import math
import numpy as np
import torch
from tiered_kv.retrieval import BlockIndex
from tiered_kv.types import KVBatch
from research.attention_fullkey_oracle import fullkey_rank


def _lse(x,axis):
    maximum=np.max(x,axis=axis,keepdims=True)
    return np.squeeze(maximum+np.log(np.exp(x-maximum).sum(axis=axis,keepdims=True)),axis=axis)


def _normalization_positions(length,cutoff,sink,page):
    # The last cold page may be partial and unselectable. Its exact logits
    # contribute to the proxy denominator but not the active attention set.
    fixed=np.r_[np.arange(min(sink,length)),np.arange(cutoff,length)]
    boundary=np.arange((cutoff//page)*page,cutoff)
    return np.unique(np.r_[fixed,boundary])


def _shared_rank(logmass,fixed_logits,candidates,k):
    # Normalizer uses exact sink/recent/current and partial-boundary logits plus predicted cold
    # page masses only. It never uses true full-cold oracle normalizers.
    fixed=_lse(fixed_logits,axis=1)
    cold=_lse(logmass,axis=1)
    denominator=np.logaddexp(fixed,cold)
    scores=np.exp(logmass-denominator[:,None]).sum(axis=0)
    order=np.lexsort((candidates,-scores))[:k]
    return candidates[order].tolist()


def compare_representations(q,past,current,position,*,sink=128,window=2000,page=128,k=15):
    """Return same-budget IDs and true retained mass; no gold or answer input."""
    q=np.asarray(q);past=np.asarray(past);current=np.asarray(current)
    kv,n,d=past.shape;heads,qd=q.shape
    assert n==position and current.shape==(kv,1,d) and qd==d and heads%kv==0
    assert 0<n<=32768 and heads<=128 and d<=256 and page%4==0 and 0<k<=15
    cutoff=max(0,position-window+1);blocks=np.array([b for b in range(cutoff//page) if b*page>=sink],dtype=np.int64)
    assert len(blocks)>=k and np.array_equal(blocks,np.arange(blocks[0],blocks[-1]+1))
    # Exact oracle uses the same float32 torch scoring as the GPU run. A caller
    # must check recorded IDs; numerical backend differences may fail the gate.
    tq=torch.from_numpy(q);tk=torch.from_numpy(past);tc=torch.from_numpy(current)
    oracle,_=fullkey_rank(tq,tk,tc,position,sink=sink,window=window,page=page,limit=k)
    full=torch.cat((tk,tc),1).float(); group=tq.float().reshape(kv,heads//kv,d)
    logits=torch.bmm(group,full.transpose(1,2)).reshape(heads,n+1).numpy()/np.float32(math.sqrt(d))
    probs=np.exp(logits-logits.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
    eligible=torch.from_numpy(past[:,:cutoff,:].copy())[None]
    idx=BlockIndex(1,kv,d,page)
    idx.absorb(0,KVBatch(np.arange(cutoff),eligible,torch.zeros_like(eligible)))
    ranked,_=idx.search(0,torch.from_numpy(q)[None,:,None,:],max(1,math.ceil(cutoff/page)))
    centroid=[int(b) for b in ranked if b in set(blocks.tolist())][:k]
    assert len(centroid)==k
    start=int(blocks[0])*page;end=(int(blocks[-1])+1)*page
    cold=past[:,start:end,:].astype(np.float32).reshape(kv,len(blocks),page,d)
    grouped=q.astype(np.float32).reshape(kv,heads//kv,d)
    lower=cold.min(axis=2);upper=cold.max(axis=2)
    quest=(np.einsum('kgd,kpd->kgp',np.maximum(grouped,0),upper)+np.einsum('kgd,kpd->kgp',np.minimum(grouped,0),lower)).reshape(heads,len(blocks))/np.float32(math.sqrt(d))
    sub=cold.reshape(kv,len(blocks),4,page//4,d)
    centroids=sub.mean(axis=3)
    representatives=sub[:,:,:,page//8,:]
    def approximate(vectors):
        raw=np.einsum('kgd,kpmd->kgpm',grouped,vectors).reshape(heads,len(blocks),4)/np.float32(math.sqrt(d))
        return _lse(raw+np.float32(math.log(page//4)),axis=2)
    fixed=np.r_[np.arange(min(sink,n+1)),np.arange(cutoff,n+1)]
    fixed_logits=logits[:,_normalization_positions(n+1,cutoff,sink,page)]
    choices={'oracle':oracle.tolist(),'centroid':centroid,
             'quest_bound':_shared_rank(quest+np.float32(math.log(page)),fixed_logits,blocks,k),
             'subblock_centroids':_shared_rank(approximate(centroids),fixed_logits,blocks,k),
             'actual_representatives':_shared_rank(approximate(representatives),fixed_logits,blocks,k)}
    fixed_set=set(map(int,fixed));cold_indices=np.concatenate([np.arange(b*page,(b+1)*page) for b in blocks])
    cold_total=probs[:,cold_indices].sum(axis=1)
    metrics={}
    for name,chosen in choices.items():
        selected=sorted(fixed_set|{i for b in chosen for i in range(b*page,(b+1)*page)})
        chosen_indices=np.concatenate([np.arange(b*page,(b+1)*page) for b in chosen])
        cold_selected=probs[:,chosen_indices].sum(axis=1)
        metrics[name]=dict(pages=chosen,oracle_overlap=len(set(chosen)&set(choices['oracle'])),
            retained_mass=probs[:,selected].sum(axis=1).tolist(),
            common_cold_fraction=[float(a/b) if b else None for a,b in zip(cold_selected,cold_total)],
            active_tokens=len(selected))
    assert len({item['active_tokens'] for item in metrics.values()})==1
    return metrics

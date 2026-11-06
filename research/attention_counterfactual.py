"""Bounded CPU-side, matched-Q/K/V attention counterfactual primitives.

All positions are absolute. Select one shared cold-page set per layer; query
heads retain their actual GQA mapping. Dense scans here are diagnostic only.
"""
import math
import numpy as np

from tiered_kv.retrieval import BlockIndex
from tiered_kv.types import KVBatch
import torch

PAGE = 128
MAX_HEADS = 128
MAX_DIM = 256
MAX_POSITIONS = 32768 + 512
MAX_PAGES = 16


def eligible_pages(length, position, *, sink=128, window=2000, page=PAGE):
    """Only complete pages disjoint from fixed sink and recent/current slots."""
    assert 0 < length <= MAX_POSITIONS and position == length-1
    cutoff = max(0, position-window+1)
    return [b for b in range(length//page)
            if b*page >= sink and (b+1)*page <= cutoff]


def original_centroid_ranking(q, reader, cutoff, *, page=PAGE):
    """Exact original float32 centroid accumulation/scoring, not QK logits.

    Rank all original eligible blocks first; callers may then filter to a
    shared comparison universe. The filter changes availability, not scoring.
    """
    q = np.asarray(q, dtype=np.float32)
    assert q.ndim == 2 and q.shape[0] <= MAX_HEADS and q.shape[1] <= MAX_DIM
    first_k, _ = reader(0, min(page, cutoff))
    kv_heads, _, dim = first_k.shape
    assert dim == q.shape[1] and q.shape[0] % kv_heads == 0
    index = BlockIndex(1, kv_heads, dim, page)
    for start in range(0, cutoff, page):
        end = min(start+page, cutoff)
        k, v = reader(start, end)
        batch = KVBatch(np.arange(start,end), torch.from_numpy(np.asarray(k))[None],
                        torch.from_numpy(np.asarray(v))[None])
        index.absorb(0,batch)
    ranking, scores = index.search(0,torch.from_numpy(q)[None,:,None,:],max(1,math.ceil(cutoff/page)))
    return ranking.tolist(), scores.tolist()


def _logsumexp(x):
    maximum=np.max(x,axis=1)
    return maximum + np.log(np.exp(x-maximum[:,None]).sum(axis=1))


def _score(q, k):
    q=np.asarray(q,dtype=np.float64);k=np.asarray(k,dtype=np.float64)
    assert q.ndim == 2 and k.ndim == 3 and q.shape[1] == k.shape[2]
    assert q.shape[0] % k.shape[0] == 0
    heads=np.arange(q.shape[0])//(q.shape[0]//k.shape[0])
    return np.einsum('hd,hnd->hn',q,k[heads])/math.sqrt(q.shape[1])


def oracle_pages(q, reader, length, candidates, count, *, page=PAGE):
    """Maximum summed per-query-head dense mass for one shared page set.

    Equal-size disjoint cold pages make ranking by sum of head-normalized page
    mass exact for this diagnostic objective, not output-error optimal.
    """
    assert 0 <= count <= min(MAX_PAGES,len(candidates)) and len(set(candidates)) == len(candidates)
    assert length <= MAX_POSITIONS and all(0 <= b and (b+1)*page <= length for b in candidates)
    all_lse=np.full(len(q),-np.inf,dtype=np.float64); page_lse={}
    for start in range(0,length,page):
        end=min(start+page,length);k,_=reader(start,end)
        logits=_score(q,k); lse=_logsumexp(logits)
        all_lse=np.logaddexp(all_lse,lse)
        if start//page in candidates: page_lse[start//page]=lse
    ranked=sorted(candidates,key=lambda b:(-float(np.exp(page_lse[b]-all_lse).sum()),b))
    return ranked[:count]


def streamed_metrics(q, reader, length, fixed_positions, selections, *, page=PAGE):
    """Return retained mass and selected/full value-output error for each head.

    `reader(start,end)` supplies original K/V [KV-head, token, dim] on CPU;
    at most one 128-token score tile is live. All candidate sets are shared by
    heads and layers are processed independently by the caller.
    """
    q=np.asarray(q,dtype=np.float64)
    assert q.ndim==2 and 0<q.shape[0]<=MAX_HEADS and 0<q.shape[1]<=MAX_DIM
    assert 0<length<=MAX_POSITIONS and all(0<=i<length for i in fixed_positions)
    positions=set(fixed_positions); normalized={}
    for name,blocks in selections.items():
        assert len(blocks)<=MAX_PAGES and len(set(blocks))==len(blocks)
        assert all(type(b) is int and 0<=b and (b+1)*page<=length for b in blocks)
        normalized[name]=set(blocks)
    heads,dim=q.shape
    accum={name:dict(m=np.full(heads,-np.inf),z=np.zeros(heads),u=np.zeros((heads,dim)))
           for name in ('dense',*normalized)}
    for start in range(0,length,page):
        end=min(start+page,length);k,v=reader(start,end)
        k=np.asarray(k);v=np.asarray(v);assert k.shape==v.shape and k.shape[1]==end-start and k.shape[2]==dim
        logits=_score(q,k); mapping=np.arange(heads)//(heads//k.shape[0]); values=np.asarray(v,dtype=np.float64)[mapping]
        local=np.arange(start,end)
        for name,state in accum.items():
            mask=np.ones(len(local),dtype=bool) if name=='dense' else np.array([int(i) in positions or int(i)//page in normalized[name] for i in local])
            if not mask.any():continue
            x=logits[:,mask];value=values[:,mask,:]; tile_max=x.max(axis=1);m=np.maximum(state['m'],tile_max)
            old=np.exp(state['m']-m);w=np.exp(x-m[:,None]);state['u']=state['u']*old[:,None]+np.einsum('hn,hnd->hd',w,value)
            state['z']=state['z']*old+w.sum(axis=1);state['m']=m
    dense=accum.pop('dense');assert np.all(dense['z']>0)
    dense_lse=dense['m']+np.log(dense['z']); dense_out=dense['u']/dense['z'][:,None]
    result={}
    for name,state in accum.items():
        assert np.all(state['z']>0)
        mass=np.exp(state['m']+np.log(state['z'])-dense_lse)
        error=np.linalg.norm(state['u']/state['z'][:,None]-dense_out,axis=1)
        scale=np.linalg.norm(dense_out,axis=1)
        selected=positions | {i for b in normalized[name] for i in range(b*page,(b+1)*page)}
        result[name]=dict(retained_mass=mass.tolist(),output_error_l2=error.tolist(),
            dense_output_l2=scale.tolist(),output_error_relative=(error/np.maximum(scale,1e-12)).tolist(),
            active_tokens=len(selected),pages=sorted(normalized[name]))
    return result


def retained_common_cold_fraction(q, reader, candidates, selections, *, page=PAGE):
    """Per-head selected/common-cold mass; None if common cold mass is zero.

    Ratios are computed in log space; no candidate pages means no denominator.
    """
    assert len(set(candidates)) == len(candidates)
    if not candidates:
        return {name: [None] * len(q) for name in selections}
    sums={name:np.full(len(q),-np.inf) for name in ('common',*selections)}
    for block in candidates:
        k,_=reader(block*page,(block+1)*page)
        lse=_logsumexp(_score(q,k))
        sums['common']=np.logaddexp(sums['common'],lse)
        for name,selected in selections.items():
            if block in selected: sums[name]=np.logaddexp(sums[name],lse)
    return {name:np.exp(value-sums['common']).tolist() for name,value in sums.items() if name!='common'}

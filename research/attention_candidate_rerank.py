"""Post-hoc candidate-pool exact-QK reranking; full scans are references only."""
import math
import numpy as np
import torch
from tiered_kv.types import KVBatch
from research.attention_quest_reference import QuestBoundIndex
from research.attention_representations import _lse,_shared_rank,_normalization_positions
from research.attention_fullkey_oracle import fullkey_rank

POOLS=(16,32,64)


def rerank_exact(page_logmass,fixed_logits,full_lse,ids,k):
    """Return practical bounded normalization and labeled full-key control."""
    practical=_shared_rank(page_logmass,fixed_logits,ids,k)
    scores=np.exp(page_logmass-full_lse[:,None]).sum(axis=0)
    control=ids[np.lexsort((ids,-scores))[:k]].tolist()
    return practical,control


def score_candidates(q,past,current,position,*,sink=128,window=2000,page=128,k=15,pools=POOLS):
    """Use bounded Quest candidates and exact candidate logits, never answer data."""
    q=np.asarray(q);past=np.asarray(past);current=np.asarray(current)
    kv,n,d=past.shape;heads,qd=q.shape
    assert n==position and current.shape==(kv,1,d) and qd==d and heads%kv==0
    assert 0<n<=32768 and 0<heads<=128 and 0<d<=256 and 0<k<=15
    assert tuple(pools)==tuple(sorted(set(pools))) and all(k<=p<=64 for p in pools)
    cutoff=max(0,position-window+1);assert cutoff>=sink
    candidates=np.array([b for b in range(cutoff//page) if b*page>=sink],dtype=np.int64)
    assert len(candidates)>=max(pools)
    tq=torch.from_numpy(q);tk=torch.from_numpy(past);tc=torch.from_numpy(current)
    full=torch.cat((tk,tc),dim=1).float()
    logits=torch.bmm(tq.float().reshape(kv,heads//kv,d),full.transpose(1,2)).reshape(heads,n+1).numpy()/np.float32(math.sqrt(d))
    all_lse=_lse(logits,axis=1)
    probs=np.exp(logits-all_lse[:,None])
    exact_fixed=np.r_[np.arange(sink),np.arange(cutoff,n+1)]
    normalization=_normalization_positions(n+1,cutoff,sink,page)
    fixed_keys=torch.cat((tk,tc),dim=1)[:,normalization,:].float()
    fixed_logits=torch.bmm(tq.float().reshape(kv,heads//kv,d),fixed_keys.transpose(1,2)).reshape(heads,-1).numpy()/np.float32(math.sqrt(d))
    fixed_error=float(np.max(np.abs(fixed_logits-logits[:,normalization])))
    assert np.allclose(fixed_logits,logits[:,normalization],atol=1e-5,rtol=1e-5)
    cutoff_keys=torch.from_numpy(past[:,:cutoff,:].copy())[None]
    index=QuestBoundIndex(1,kv,d,page,sink)
    index.absorb(0,KVBatch(np.arange(cutoff),cutoff_keys,torch.zeros_like(cutoff_keys)))
    index.stage(0,torch.from_numpy(np.concatenate((past,current),axis=1)[:,normalization,:].copy()))
    quest,_=index.search(0,tq[None,:,None,:],max(pools));quest=quest.tolist()
    oracle,_=fullkey_rank(tq,tk,tc,position,sink=sink,window=window,page=page,limit=k);oracle=oracle.tolist()
    common=np.concatenate([np.arange(b*page,(b+1)*page) for b in candidates])
    common_mass=probs[:,common].sum(axis=1)
    results=[]
    for pool in pools:
        selected_pool=quest[:pool];assert len(selected_pool)==pool
        # Independently recompute candidate QK; full logits are used only for
        # the labeled control and evaluation, never practical normalization.
        candidate_positions=np.concatenate([np.arange(b*page,(b+1)*page) for b in selected_pool])
        candidate_keys=tk[:,candidate_positions,:].float()
        exact_candidate=torch.bmm(tq.float().reshape(kv,heads//kv,d),candidate_keys.transpose(1,2)).reshape(heads,-1).numpy()/np.float32(math.sqrt(d))
        reference_candidate=logits[:,candidate_positions]
        candidate_error=float(np.max(np.abs(exact_candidate-reference_candidate)))
        assert np.allclose(exact_candidate,reference_candidate,atol=1e-5,rtol=1e-5)
        page_logmass=_lse(exact_candidate.reshape(heads,pool,page),axis=2)
        ids=np.asarray(selected_pool,dtype=np.int64)
        practical,control=rerank_exact(page_logmass,fixed_logits,all_lse,ids,k)
        arms={}
        for name,pages in [('candidate_only',practical),('full_key_normalizer_control',control)]:
            chosen=np.concatenate([np.arange(b*page,(b+1)*page) for b in pages])
            active=np.unique(np.r_[exact_fixed,chosen])
            cold_selected=probs[:,chosen].sum(axis=1)
            arms[name]=dict(pages=pages,selected_count=len(pages),active_tokens=len(active),oracle_overlap=len(set(pages)&set(oracle)),
                retained_mass=probs[:,active].sum(axis=1).tolist(),
                common_cold_fraction=[float(a/b) if b else None for a,b in zip(cold_selected,common_mass)])
        assert all(x['active_tokens']==len(exact_fixed)+k*page for x in arms.values())
        results.append(dict(candidate_ceiling=pool,candidate_pages=selected_pool,candidate_count=len(selected_pool),
            candidate_oracle_overlap=len(set(selected_pool)&set(oracle)),
            candidate_key_positions=pool*page,candidate_vs_full_max_abs=candidate_error,
            fixed_vs_full_max_abs=fixed_error,
            candidate_kv_payload_bytes=pool*page*2*kv*d*2,
            arms=arms))
    return dict(quest_top15=quest[:k],oracle_pages=oracle,results=results)

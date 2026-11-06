"""Evaluation-only float64 softmax for fixed, previously recorded page IDs."""
import numpy as np


def stable_metrics(logits,position,choices,*,sink=128,window=2000,page=128):
    """Input logits must be the original float32 QK; no choices are made here."""
    logits=np.asarray(logits)
    assert logits.dtype==np.float32 and logits.ndim==2 and logits.shape[1]==position+1
    assert np.isfinite(logits).all() and position>=0
    cutoff=max(0,position-window+1);assert cutoff>=sink
    candidates=[b for b in range(cutoff//page) if b*page>=sink]
    assert candidates
    causal=logits.astype(np.float64)
    weights=np.exp(causal-causal.max(axis=1,keepdims=True))
    probabilities=weights/weights.sum(axis=1,keepdims=True)
    sums=probabilities.sum(axis=1)
    assert np.isfinite(probabilities).all() and np.all(np.abs(sums-1)<1e-12)
    common=np.concatenate([np.arange(b*page,(b+1)*page) for b in candidates])
    cold_total=probabilities[:,common].sum(axis=1)
    fixed=set(range(sink))|set(range(cutoff,position+1));result={}
    for name,pages in choices.items():
        assert len(pages)==15 and len(set(pages))==15 and all(b in candidates for b in pages)
        cold=np.concatenate([np.arange(b*page,(b+1)*page) for b in pages])
        active=sorted(fixed|set(map(int,cold)));assert len(active)==len(fixed)+15*page
        cold_selected=probabilities[:,cold].sum(axis=1)
        result[name]={'pages':list(pages),'active_tokens':len(active),
            'retained_mass':probabilities[:,active].sum(axis=1).tolist(),
            'common_cold_fraction':[float(a/b) if b else None for a,b in zip(cold_selected,cold_total)]}
    return {'normalization_sums':sums.tolist(),'metrics':result}

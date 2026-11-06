import math
import numpy as np
import pytest
from research.attention_candidate_rerank import score_candidates


def test_candidate_qk_and_normalizers_match_independent_float64():
    rng=np.random.default_rng(173)
    past=rng.normal(size=(2,64,4)).astype('float16')
    current=rng.normal(size=(2,1,4)).astype('float16')
    q=rng.normal(size=(4,4)).astype('float16')
    result=score_candidates(q,past,current,64,sink=4,window=4,page=4,k=2,pools=(2,4,6))
    full=np.concatenate((past,current),axis=1).astype('float64')
    logits=np.array([full[h//2]@q[h].astype('float64')/math.sqrt(4) for h in range(4)])
    exp=np.exp(logits-logits.max(axis=1,keepdims=True));probs=exp/exp.sum(axis=1,keepdims=True)
    cutoff=61;fixed=set(range(4))|set(range(cutoff,65));boundary=set(range(60,61))
    for entry in result['results']:
        pool=entry['candidate_pages'];assert len(pool)==entry['candidate_ceiling']
        by_page=np.stack([exp[:,b*4:(b+1)*4].sum(axis=1) for b in pool],axis=1)
        normalizer=exp[:,sorted(fixed|boundary)].sum(axis=1)+by_page.sum(axis=1)
        practical=(by_page/normalizer[:,None]).sum(axis=0)
        control=np.array([probs[:,b*4:(b+1)*4].sum() for b in pool])
        pick=lambda scores:[int(pool[i]) for i in np.lexsort((np.array(pool),-scores))[:2]]
        assert entry['arms']['candidate_only']['pages']==pick(practical)
        assert entry['arms']['full_key_normalizer_control']['pages']==pick(control)
        assert entry['candidate_kv_payload_bytes']==len(pool)*4*2*2*4*2
        assert all(v['active_tokens']==4+4+2*4 for v in entry['arms'].values())
        assert entry['candidate_vs_full_max_abs']<1e-5


def test_invalid_candidate_budgets_fail_closed():
    q=np.ones((2,4),np.float16);past=np.ones((1,64,4),np.float16);current=np.ones((1,1,4),np.float16)
    with pytest.raises(AssertionError):score_candidates(q,past,current,64,sink=4,window=4,page=4,k=2,pools=(4,2))
    with pytest.raises(AssertionError):score_candidates(q,past,current,64,sink=4,window=4,page=4,k=2,pools=(65,))


def test_candidate_only_and_full_normalizers_can_choose_differently():
    from research.attention_candidate_rerank import rerank_exact
    # An omitted noncandidate dominates head 0 only. Candidate-only weighting
    # sees a tie, whereas the full normalizer discounts head 0 accordingly.
    masses=np.array([[2.,0.],[0.,2.]],np.float32)
    fixed=np.zeros((2,1),np.float32)
    full_lse=np.array([20.,3.],np.float32)
    practical,control=rerank_exact(masses,fixed,full_lse,np.array([1,2]),1)
    assert practical==[1] and control==[2]

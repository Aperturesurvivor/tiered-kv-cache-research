import numpy as np
import pytest

from research.attention_counterfactual import eligible_pages, oracle_pages, original_centroid_ranking, streamed_metrics


def reader(k,v):
    return lambda start,end:(k[:,start:end,:],v[:,start:end,:])


def test_streamed_matches_independent_full_float64_reference():
    rng=np.random.default_rng(314)
    q=rng.normal(size=(4,8)); k=rng.normal(size=(2,384,8)); v=rng.normal(size=(2,384,8))
    fixed=set(range(4))|set(range(368,384)); selections={'text':[1], 'random':[0], 'empty':[]}
    observed=streamed_metrics(q,reader(k,v),384,fixed,selections)
    mapping=np.array([0,0,1,1]); logits=np.einsum('hd,hnd->hn',q,k[mapping])/np.sqrt(8)
    prob=np.exp(logits-logits.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
    full=np.einsum('hn,hnd->hd',prob,v[mapping])
    for name,pages in selections.items():
        ids=sorted(fixed|{i for p in pages for i in range(128*p,128*(p+1))})
        selected=prob[:,ids];mass=selected.sum(axis=1)
        estimate=np.einsum('hn,hnd->hd',selected/mass[:,None],v[mapping][:,ids,:])
        np.testing.assert_allclose(observed[name]['retained_mass'],mass,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(observed[name]['output_error_l2'],np.linalg.norm(estimate-full,axis=1),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(observed[name]['output_error_relative'],np.linalg.norm(estimate-full,axis=1)/np.maximum(np.linalg.norm(full,axis=1),1e-12),rtol=1e-12,atol=1e-12)
        assert observed[name]['active_tokens']==len(ids)


def test_gqa_query_averaging_can_cancel_individual_head_preferences():
    q=np.array([[1.,0.],[-1.,0.]],dtype=np.float32)
    k=np.zeros((1,256,2),np.float32);k[:,0:128,0]=1;k[:,128:,0]=-1
    v=np.ones_like(k)
    rank,scores=original_centroid_ranking(q,reader(k,v),256)
    assert rank==[0,1] and scores==[0.,0.]
    logits=np.einsum('hd,hnd->hn',q,k[np.zeros(2,dtype=int)])
    assert logits[0,:128].mean()>logits[0,128:].mean()
    assert logits[1,:128].mean()<logits[1,128:].mean()


def test_key_centroid_cancellation_and_cosine_magnitude_miss_mass():
    q=np.array([[1.,0.]],dtype=np.float32)
    k=np.zeros((1,256,2),np.float32);k[0,:64,0]=10;k[0,64:128,0]=-10;k[0,128:,0]=1
    v=np.ones_like(k);rank,_=original_centroid_ranking(q,reader(k,v),256)
    assert rank[0]==1
    assert oracle_pages(q,reader(k,v),256,[0,1],1)==[0]
    k[0,:128,0]=1;k[0,128:,0]=8
    rank,scores=original_centroid_ranking(q,reader(k,v),256)
    assert scores[0]==scores[1] and rank[0]==0
    assert oracle_pages(q,reader(k,v),256,[0,1],1)==[1]


def test_denominator_only_does_not_recover_omitted_value_numerator():
    q=np.array([[1.]])
    k=np.zeros((1,256,1));v=np.zeros_like(k);v[:,128:,:]=10
    result=streamed_metrics(q,reader(k,v),256,{0},{'selected':[0]})['selected']
    assert result['retained_mass'][0]==pytest.approx(.5)
    assert result['output_error_l2'][0]==pytest.approx(5.)
    assert 0 != pytest.approx(5.)  # correcting the denominator alone still lacks omitted values


def test_eligibility_excludes_sink_recent_and_partial_pages():
    pages=eligible_pages(8193,8192)
    assert 0 not in pages and all((p+1)*128<=8192-2000+1 for p in pages)
    assert len(pages)>16


def test_shared_oracle_is_exact_best_aggregate_mass_not_per_head_union():
    q=np.array([[1.],[-1.]])
    k=np.zeros((1,384,1)); k[:,0:128]=4;k[:,128:256]=-4;k[:,256:]=0
    v=np.ones_like(k)
    selected=oracle_pages(q,reader(k,v),384,[0,1],1)
    assert len(selected)==1
    result=streamed_metrics(q,reader(k,v),384,set(range(256,384)),{'oracle':selected})
    assert len(result['oracle']['pages'])==1
    scores=[sum(streamed_metrics(q,reader(k,v),384,set(range(256,384)),{'x':[p]})['x']['retained_mass']) for p in (0,1)]
    assert sum(result['oracle']['retained_mass'])==pytest.approx(max(scores))


def test_common_cold_mass_ratio_when_sink_dominates():
    from research.attention_counterfactual import retained_common_cold_fraction
    q=np.array([[1.0],[2.0]])
    keys=np.array([[[20.0],[0.0],[1.0],[2.0],[3.0]]])
    values=np.zeros_like(keys)
    def reader(start,end): return keys[:,start:end,:], values[:,start:end,:]
    got=retained_common_cold_fraction(q,reader,[1,2,3,4],{'chosen':[1,3]},page=1)
    for h in range(2):
        weights=np.exp(q[h,0]*keys[0,1:,0] - q[h,0]*keys[0,1:,0].max())
        assert np.isclose(got['chosen'][h],(weights[0]+weights[2])/weights.sum())
    assert retained_common_cold_fraction(q,reader,[],{'chosen':[]},page=1)=={'chosen':[None,None]}

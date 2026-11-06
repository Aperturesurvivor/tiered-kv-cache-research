import numpy as np
from research.attention_representations import compare_representations, _shared_rank


def test_same_budget_and_true_mass_for_known_case():
    rng=np.random.default_rng(42)
    past=rng.normal(size=(2,56,4)).astype('float16')
    current=rng.normal(size=(2,1,4)).astype('float16')
    q=rng.normal(size=(4,4)).astype('float16')
    result=compare_representations(q,past,current,56,sink=8,window=8,page=8,k=3)
    assert set(result)=={'oracle','centroid','quest_bound','subblock_centroids','actual_representatives'}
    assert result['oracle']['oracle_overlap']==3
    assert len({v['active_tokens'] for v in result.values()})==1
    assert all(len(v['pages'])==3 and all(1<=b<=5 for b in v['pages']) for v in result.values())
    # Independent full softmax in float64 for the selected oracle set.
    full=np.concatenate((past,current),axis=1).astype('float64')
    logits=np.array([full[h//2]@q[h].astype('float64')/2 for h in range(4)])
    probs=np.exp(logits-logits.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
    ids=set(range(8))|set(range(49,57))|{i for b in result['oracle']['pages'] for i in range(b*8,b*8+8)}
    np.testing.assert_allclose(result['oracle']['retained_mass'],probs[:,sorted(ids)].sum(axis=1),rtol=1e-6,atol=1e-6)


def test_approximate_normalizer_includes_fixed_context():
    # Head 0 has huge fixed mass; it must not outweigh head 1's cold evidence.
    page_logs=np.array([[4.,0.],[0.,3.]],np.float32)
    fixed=np.array([[20.],[0.]],np.float32)
    assert _shared_rank(page_logs,fixed,np.array([1,2]),1)==[2]


def test_partial_cold_boundary_normalizes_but_is_not_active():
    from research.attention_representations import _normalization_positions
    positions=_normalization_positions(length=60,cutoff=51,sink=8,page=8)
    assert set(range(48,51)).issubset(positions)
    assert not set(range(8,48)) & set(positions)
    page_logs=np.array([[4.,0.],[0.,3.]],np.float32)
    fixed_plus_boundary=np.array([[0.,20.],[0.,0.]],np.float32)
    assert _shared_rank(page_logs,fixed_plus_boundary,np.array([1,2]),1)==[2]

import numpy as np
import pytest
from research.attention_device_replay import aggregate,compare_arrays,rank,ATOL,RTOL


def row(i, candidate=True, final=True, close=True):
    return dict(layer=i,position=30000,candidate_match=candidate,final_match=final,
                numerics={'fixed':{'allclose':close}},ties={'candidate_cpu':[]})


def test_completed_is_distinct_from_equivalent():
    rows=[row(0),row(1,candidate=False),row(2,final=False),row(3,close=False)]
    doc=aggregate(rows,4)
    assert doc['status']=='complete' and not doc['equivalent']
    assert (doc['candidate_disagreements'],doc['final_disagreements'],doc['numeric_disagreements'])==(1,1,1)
    with pytest.raises(ValueError):aggregate(rows,5)
    with pytest.raises(ValueError):aggregate([row(0),row(0)],2)


def test_stable_id_ties_and_numerical_failures():
    ids=np.array([9,3]);scores=np.zeros((2,2),np.float32);fixed=np.zeros(2,np.float32)
    selected,_,ties=rank(scores,fixed,ids,2)
    assert selected==[3,9] and ties==[[3,9]]
    assert compare_arrays([0.],[ATOL/2])['allclose']
    assert not compare_arrays([0.],[ATOL*2])['allclose']
    with pytest.raises(ValueError):compare_arrays([np.nan],[0])


def test_paths_route_their_own_candidate_pools_and_near_ties():
    from research.attention_device_replay import route_pair
    ids=np.array([10,20]);bound=np.array([[3.,0.],[0.,3.]],np.float32)
    called=[]
    def score(device,selected):
        called.append((device,selected))
        return np.array(selected),np.zeros((2,len(selected)),np.float32),np.zeros(2,np.float32)
    result=route_pair(bound,ids,np.array([10.,0.]),np.array([0.,10.]),score,pool=1,final=1)
    assert result['cpu_ids']!=result['gpu_ids']
    assert called==[('cpu',result['cpu_ids']),('cuda',result['gpu_ids'])]
    assert result['cpu_final']==result['cpu_ids'] and result['gpu_final']==result['gpu_ids']
    a=np.array([[1.,1.000001]],np.float32);b=np.array([[1.000001,1.]],np.float32)
    assert compare_arrays(a,b)['allclose']
    assert rank(a,np.array([0.]),ids,1)[0]!=rank(b,np.array([0.]),ids,1)[0]


def test_projection_excludes_retrospective_fields():
    from research.attention_device_replay import project
    row={'layer':1,'position':30000,'answer':'hidden','required_pages':[7],
         'results':[{}, {'candidate_pages':[3,4], 'required_pages_covered':[7],
                           'arms':{'candidate_only':{'pages':[4], 'gold':True}}}]}
    assert project(row)=={'layer':1,'position':30000,'candidate_pages':[3,4],'final_pages':[4]}

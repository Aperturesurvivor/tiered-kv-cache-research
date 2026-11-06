import math
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from research.attention_fullkey_oracle import FullKeyOracleIndex, fullkey_rank


def test_oracle_matches_independent_gqa_float64_reference():
    rng=np.random.default_rng(19)
    keys=torch.tensor(rng.normal(size=(2,9,3)).astype('float16'))
    current=torch.tensor(rng.normal(size=(2,1,3)).astype('float16'))
    q=torch.tensor(rng.normal(size=(4,3)).astype('float32'))
    ids,scores=fullkey_rank(q,keys,current,9,sink=2,window=2,page=2,limit=2)
    full=np.concatenate([keys.numpy(),current.numpy()],axis=1).astype('float64')
    logits=np.array([full[h//2]@q.numpy()[h].astype('float64')/math.sqrt(3) for h in range(4)])
    prob=np.exp(logits-logits.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
    expected={b:prob[:,b*2:(b+1)*2].sum() for b in (1,2,3)}
    rank=sorted(expected,key=lambda b:(-expected[b],b))[:2]
    assert ids.tolist()==rank
    np.testing.assert_allclose(scores,[expected[b] for b in rank],rtol=2e-6,atol=2e-6)
    # Vary current K by head: shared head weights and ranking may change.
    altered=current.clone();altered[0]*=100
    _,different=fullkey_rank(q,keys,altered,9,sink=2,window=2,page=2,limit=2)
    assert not np.allclose(scores,different)


def test_eligibility_and_deterministic_ties():
    q=torch.zeros((4,2));keys=torch.zeros((2,9,2),dtype=torch.float16)
    current=torch.zeros((2,1,2),dtype=torch.float16)
    ids,_=fullkey_rank(q,keys,current,9,sink=2,window=2,page=2,limit=3)
    assert ids.tolist()==[1,2,3] # excludes sink 0, partial/recent page 4


def test_history_is_sparse_trajectory_and_current_is_staged_once():
    class Store:
        layers=1;h=1;d=2;count=4
        def read_range(self,layer,start,end):
            return SimpleNamespace(k=torch.zeros((1,1,end-start,2),dtype=torch.float16))
    index=FullKeyOracleIndex(Store(),'cpu',6,sink=0,window=1,page=2)
    current=torch.tensor([[[[1.0,2.0]]]],dtype=torch.float16)
    index.stage_current(0,current,4)
    with pytest.raises(ValueError):index.stage_current(0,current,4)
    index.commit_current(0)
    assert index.history_counts==[5] and torch.equal(index.keys[0][:,4,:],current[0,:,0,:])
    with pytest.raises(ValueError):index.stage_current(0,current,4)
    index.stage_current(0,current*3,5);index.commit_current(0)
    assert torch.equal(index.keys[0][:,5,:],(current*3)[0,:,0,:])
    with pytest.raises(ValueError):index.stage_current(0,current,6)


def test_tiny_model_commits_sparse_keys_after_archive(tmp_path):
    from test_attention import tiny
    from research.reasoning_sessions import CountedSession
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    from research.attention_fullkey_oracle import FullKeyOracleCache
    model=tiny();prefix=ColdStore(tmp_path/'prefix',2,2,16,4)
    adapter=ModelAdapter.install(model,DenseController(2,prefix))
    adapter.forward_tokens([3,4,5,6]*5,0);adapter.uninstall();prefix.checkpoint({});prefix.close()
    parent=ColdStore.open(tmp_path/'prefix',readonly=True)
    child=ColdStore(tmp_path/'child',2,2,16,4,prefix=parent,staged_tensor_writes=True)
    cfg=Config(W=3,C=0,B=4,R=4,k=2,sink_tokens=4,retrieval='centroid')
    rotary=Rotary(model.model.rotary_emb.inv_freq.detach().numpy())
    controller=FullKeyOracleCache(cfg,child,rotary,'cpu',dtype=torch.float32,initialize=False)
    controller.index=FullKeyOracleIndex(parent,'cpu',24,sink=4,window=3,page=4)
    controller._initialize();adapter=ModelAdapter.install(model,controller)
    try:
        session=CountedSession(adapter,model,active_limit=16)
        session.feed_prompt([7,8,9])
        assert controller.position==23 and controller.index.history_counts==[23,23]
        assert not controller.index.pending
        for layer in range(2):
            expected=child.read_range(layer,20,23).k[0]
            assert torch.equal(controller.index.keys[layer][:,20:23].cpu(),expected)
    finally:
        controller.trace=None;adapter.uninstall();child.prefix=None;child.close();parent.close()


def test_committed_sparse_history_changes_later_selection():
    class Store:
        layers=1;h=1;d=1;count=4
        def read_range(self,layer,start,end):
            return SimpleNamespace(k=torch.zeros((1,1,end-start,1),dtype=torch.float16))
    rankings=[]
    for value in (0.0,8.0):
        index=FullKeyOracleIndex(Store(),'cpu',7,sink=0,window=1,page=2)
        for pos,key in [(4,value),(5,0.0)]:
            index.stage_current(0,torch.tensor([[[[key]]]],dtype=torch.float16),pos)
            index.commit_current(0)
        index.stage_current(0,torch.zeros((1,1,1,1),dtype=torch.float16),6)
        ranked,_=index.search(0,torch.ones((1,1,1,1)),1)
        rankings.append(ranked.tolist())
    assert rankings==[[0],[2]]

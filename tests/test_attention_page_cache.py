import numpy as np
import pytest
import torch
from research.attention_page_cache import CompletePageCacheStore
from tiered_kv.store import ColdStore
from tiered_kv.types import KVBatch


def make_parent(tmp_path):
    parent=ColdStore(tmp_path/'parent',1,1,2,4)
    values=torch.arange(40,dtype=torch.float16).reshape(1,1,20,2)
    parent.append(0,KVBatch(np.arange(20),values,values+1))
    parent.checkpoint({});parent.close()
    return ColdStore.open(tmp_path/'parent',readonly=True)


def test_complete_page_intersection_reuse_and_validation(tmp_path):
    parent=make_parent(tmp_path)
    cached=CompletePageCacheStore(tmp_path/'cached',1,1,2,4,prefix=parent,cache_device='cpu',cache_pages=2,sink_tokens=4)
    raw=ColdStore(tmp_path/'raw',1,1,2,4,prefix=parent)
    try:
        for selected in ([1,2],[2,1],[2,3],[],[3]):
            reference=raw.read(0,selected,20);actual=cached.read(0,selected,20)
            assert np.array_equal(actual.ids,reference.ids)
            assert torch.equal(actual.k,reference.k) and torch.equal(actual.v,reference.v)
            assert len(cached.pages[0])<=2
        assert cached.stats[0]['unchanged_set']==1 and cached.stats[0]['reordered']==1
        assert cached.stats[0]['cache_hits']==3 and cached.stats[0]['cache_misses']==4
        assert cached.stats[0]['peak_resident_pages']==2
        assert cached.read_bytes==0 # immutable parent owns the read counters
        for bad,end in [([0],20),([4],19),([1,2,3],20),([1,1],20)]:
            with pytest.raises(ValueError):cached.read(0,bad,end)
    finally:
        raw.prefix=None;raw.close();cached.prefix=None;cached.close();parent.close()
    assert not cached.pages[0]


def test_real_dynamic_retrieval_logits_match_uncached(tmp_path):
    from test_attention import tiny
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.retrieval import BlockIndex
    from tiered_kv.types import Rotary
    model=tiny(); store=ColdStore(tmp_path/'parent',2,2,16,4)
    adapter=ModelAdapter.install(model,DenseController(2,store))
    adapter.forward_tokens([3,4,5,6]*5,0)
    adapter.uninstall();store.checkpoint({});store.close()
    parent=ColdStore.open(tmp_path/'parent',readonly=True)
    cfg=Config(W=3,C=0,B=4,R=4,k=2,sink_tokens=4,retrieval='centroid')
    rotary=Rotary(model.model.rotary_emb.inv_freq.detach().numpy())
    sequence=[[1,2],[2,1],[2,3],[],[3]]
    class ScriptedIndex(BlockIndex):
        def search(self,layer,q,k):
            selected=sequence[self.searches[layer]];self.searches[layer]+=1
            return np.array(selected,dtype=np.int64),np.zeros(len(selected),dtype=np.float32)
    outputs=[];read_totals=[]
    for cached in (False,True):
        kwargs=dict(cache_device='cpu',cache_pages=2,sink_tokens=4) if cached else {}
        cls=CompletePageCacheStore if cached else ColdStore
        child=cls(tmp_path/str(cached),2,2,16,4,prefix=parent,**kwargs)
        controller=TieredCache(cfg,child,rotary,'cpu',dtype=torch.float32,initialize=False)
        controller.index=ScriptedIndex(2,2,16,4);controller._initialize()
        adapter=ModelAdapter.install(model,controller);start=parent.read_bytes
        try:
            outputs.append([adapter.step(t,adapter.position).detach().clone() for t in [7,8,9,10,11]])
            read_totals.append(parent.read_bytes-start)
        finally:
            adapter.uninstall();child.prefix=None;child.close()
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
    assert read_totals[1]<read_totals[0]
    parent.close()


def test_read_failure_clears_partial_cache(tmp_path,monkeypatch):
    parent=make_parent(tmp_path)
    cached=CompletePageCacheStore(tmp_path/'cached',1,1,2,4,prefix=parent,cache_device='cpu',cache_pages=2,sink_tokens=4)
    cached.read(0,[1,2],20)
    original=ColdStore.read
    def fail_read(store,layer,blocks,before):
        if store is cached:raise OSError('simulated read failure')
        return original(store,layer,blocks,before)
    monkeypatch.setattr(ColdStore,'read',fail_read)
    with pytest.raises(OSError):cached.read(0,[2,3],20)
    assert not cached.pages[0] and cached.previous_order[0] is None
    cached.prefix=None;cached.close();parent.close()

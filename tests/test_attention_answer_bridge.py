import numpy as np
import torch
from research.attention_answer_bridge import CompleteColdCentroid, filtered_text_blocks, PAGE
from tiered_kv.retrieval import BlockIndex
from tiered_kv.types import KVBatch


def test_centroid_rank_then_filter_matches_original_scores():
    rng=np.random.default_rng(13)
    keys=rng.normal(size=(1,2,PAGE*4+17,4)).astype('float32')
    batch=KVBatch(np.arange(keys.shape[2]),torch.from_numpy(keys),torch.zeros_like(torch.from_numpy(keys)))
    original=BlockIndex(1,2,4,PAGE)
    filtered=CompleteColdCentroid(1,2,4,PAGE)
    original.absorb(0,batch);filtered.absorb(0,batch)
    q=torch.tensor(rng.normal(size=(1,4,1,4)).astype('float32'))
    rank,scores=original.search(0,q,99)
    expected=[(int(b),float(s)) for b,s in zip(rank,scores) if b>=1 and original.counts[0][b]==PAGE][:2]
    result,actual=filtered.search(0,q,2)
    assert result.tolist()==[b for b,_ in expected]
    np.testing.assert_allclose(actual,[s for _,s in expected])
    assert 0 not in result and 4 not in result


def test_text_filter_retains_underfill_without_padding():
    assert filtered_text_blocks([0,1,2,3,4],position=2300)==[1]
    assert filtered_text_blocks([0,1,2,3],position=4000)==[1,2,3]
    assert filtered_text_blocks(list(range(1,25)),position=10000)==list(range(1,16))


def test_detach_breaks_trace_cycle_and_preserves_parent():
    import gc
    import weakref
    from types import SimpleNamespace
    from research.attention_answer_bridge import detach_session
    class Adapter:
        def uninstall(self): self.uninstalled=True
    class Child:
        def close(self): self.closed=True
    controller=SimpleNamespace(trace=None)
    session=SimpleNamespace(controller=controller)
    # Match a bound callback retaining its owner.
    class Callback:
        def __init__(self, owner): self.owner=owner
        def __call__(self,*args): pass
    callback=Callback(session);controller.trace=callback
    adapter=Adapter();child=Child();parent=object();child.prefix=parent
    ref=weakref.ref(callback)
    detach_session(session,child,adapter)
    assert controller.trace is None and child.prefix is None and child.closed and adapter.uninstalled
    callback=None;session=None;controller=None;gc.collect()
    assert ref() is None


def test_real_tiny_model_fixed_and_centroid_sessions(tmp_path, monkeypatch):
    from test_attention import tiny
    from research import attention_answer_bridge as bridge
    from research.reasoning_sessions import CountedSession
    from tiered_kv.adapter import DenseController, ModelAdapter
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.generate import decode_greedy
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary

    monkeypatch.setattr(bridge,'PAGE',4)
    monkeypatch.setattr(bridge,'SINK',1)
    model=tiny()
    model.config._commit_hash='0'*40
    store=ColdStore(tmp_path/'parent',2,2,16,4)
    adapter=ModelAdapter.install(model,DenseController(2,store))
    adapter.forward_tokens([3,4,5,6]*3,0)
    adapter.uninstall();store.checkpoint({'model_revision':'0'*40});store.close()
    parent=ColdStore.open(tmp_path/'parent',readonly=True)
    rotary=Rotary(model.model.rotary_emb.inv_freq.detach().numpy())
    for arm in ('fixed','centroid'):
        cfg=Config(id=arm,W=3,C=0,B=4,R=4,k=1,sink_tokens=1,
                   retrieval='fixed' if arm=='fixed' else 'centroid',model_revision='0'*40)
        child=ColdStore(tmp_path/arm,2,2,16,4,prefix=parent,staged_tensor_writes=True)
        controller=TieredCache(cfg,child,rotary,'cpu',dtype=torch.float32,initialize=False)
        if arm=='centroid':controller.index=bridge.CompleteColdCentroid(2,2,16,4)
        controller._initialize()
        adapter=ModelAdapter.install(model,controller)
        session=CountedSession(adapter,model,active_limit=9)
        try:
            if arm=='fixed':controller.select_blocks([1])
            logits=session.feed_prompt([7,8])
            generated,_=decode_greedy(session,logits,2,set(),synchronize=lambda:None)
            session.consume_pending()
            if arm=='fixed':controller.select_blocks([2])
            session.feed_prompt([9])
            assert len(generated)==2 and session.position==17
            assert session.metrics()['max_active_KV_entries']<=8
            assert sum(controller.index.searches)>0
        finally:
            bridge.detach_session(session,child,adapter)
    assert parent.count==12
    parent.close()


def test_partial_rows_are_durable_before_completion(tmp_path):
    import json
    from research.attention_answer_bridge import persist_partial_row, persist_stage
    validation=tmp_path/'validation.json'
    persist_stage(validation,{'status':'partial','gate':'passed'})
    rows=tmp_path/'samples.jsonl'
    persist_partial_row(rows,{'arm':'stock','exact_correct':True})
    persist_partial_row(rows,{'arm':'fixedtext','exact_correct':False})
    records=[json.loads(line) for line in rows.read_text().splitlines()]
    assert [record['row']['arm'] for record in records]==['stock','fixedtext']
    assert all(record['status']=='partial' for record in records)
    assert json.loads(validation.read_text())['gate']=='passed'
    assert not (tmp_path/'complete.json').exists()

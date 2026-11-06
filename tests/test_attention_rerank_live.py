import numpy as np
import torch
from research.attention_rerank_live import CandidatePageStore,CandidateRerankIndex
from research.attention_representations import _normalization_positions
from research.attention_candidate_rerank import score_candidates
from tiered_kv.types import KVBatch
from tiered_kv.store import ColdStore


def test_live_candidate_selection_matches_screen(tmp_path):
    rng=np.random.default_rng(13);page=4;pos=96;cutoff=pos-8+1
    past=rng.normal(size=(2,pos,4)).astype('float16');current=rng.normal(size=(2,1,4)).astype('float16');q=rng.normal(size=(4,4)).astype('float16')
    expected=score_candidates(q,past,current,pos,sink=4,window=8,page=page,k=3,pools=(16,))['results'][0]
    parent=ColdStore(tmp_path/'parent',1,2,4,page); t=torch.from_numpy(past)[None]
    parent.append(0,np.arange(pos),t,t);parent.checkpoint({});parent.close()
    parent=ColdStore.open(tmp_path/'parent',readonly=True)
    store=CandidatePageStore(tmp_path/'child',1,2,4,page,prefix=parent,cache_device='cpu',cache_pages=16,sink_tokens=4)
    index=CandidateRerankIndex(1,2,4,page,4,store,pool=16)
    old=torch.from_numpy(past[:,:cutoff].copy())[None];index.absorb(0,KVBatch(np.arange(cutoff),old,old))
    fixed=np.concatenate((past,current),1)[:,_normalization_positions(pos+1,cutoff,4,page),:]
    index.stage(0,torch.from_numpy(fixed.copy()))
    chosen,_=index.search(0,torch.from_numpy(q)[None,:,None,:],3)
    assert index.last_candidates[0]==expected['candidate_pages']
    assert chosen.tolist()==expected['arms']['candidate_only']['pages']
    before=store.read_bytes; candidate_set=set(store.pages[0]);result=store.read(0,chosen,cutoff)
    assert store.read_bytes==before and set(store.pages[0])==candidate_set
    positions=np.concatenate([np.arange(b*page,(b+1)*page) for b in sorted(chosen)])
    np.testing.assert_array_equal(result.k.numpy()[0],past[:,positions,:]);store.close();parent.close()


def test_live_model_advances_and_reuses_candidate_pages(tmp_path):
    from test_attention import tiny
    from research.attention_quest_reference import QuestBoundCache
    from research.reasoning_sessions import CountedSession
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.config import Config
    from tiered_kv.types import Rotary
    model=tiny();prefix=ColdStore(tmp_path/'prefix',2,2,16,4)
    adapter=ModelAdapter.install(model,DenseController(2,prefix));adapter.forward_tokens([3,4,5,6]*20,0)
    adapter.uninstall();prefix.checkpoint({});prefix.close();parent=ColdStore.open(tmp_path/'prefix',readonly=True)
    store=CandidatePageStore(tmp_path/'child',2,2,16,4,prefix=parent,staged_tensor_writes=True,cache_device='cpu',cache_pages=8,sink_tokens=4)
    cfg=Config(W=3,C=0,B=4,R=4,k=2,sink_tokens=4,retrieval='centroid')
    controller=QuestBoundCache(cfg,store,Rotary(model.model.rotary_emb.inv_freq.detach().numpy()),'cpu',dtype=torch.float32,initialize=False)
    controller.index=CandidateRerankIndex(2,2,16,4,4,store,pool=8);controller._initialize();adapter=ModelAdapter.install(model,controller)
    try:
        session=CountedSession(adapter,model,active_limit=16)
        assert torch.isfinite(session.feed_prompt([7,8,9])).all()
        assert controller.position==83 and not controller.index.pending
        assert all(len(p)==8 for p in store.pages)
        assert all(s['calls']==3 for s in controller.index.rerank_stats)
    finally:
        controller.trace=None;adapter.uninstall();store.prefix=None;store.close();parent.close()

import numpy as np
import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from research.attention_device_rerank import DeviceQuestCache,DeviceRerankIndex
from research.attention_quest_reference import QuestBoundCache
from research.attention_rerank_live import CandidatePageStore,CandidateRerankIndex
from tiered_kv.adapter import ModelAdapter
from tiered_kv.config import Config
from tiered_kv.store import ColdStore
from tiered_kv.types import Rotary


def tiny(kv):
    torch.manual_seed(417)
    config=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=96,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=kv,max_position_embeddings=32768,rope_theta=10000.,attention_dropout=0.)
    config._attn_implementation='sdpa'
    return Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA route comparison requires CUDA'))])
@pytest.mark.parametrize('kv,pos',[(2,2127),(4,2128),(2,2304),(4,2305),(2,6300),(4,6301)])
def test_actual_model_device_summary_matches_cpu_routing_and_exact_kv(tmp_path,kv,pos,device):
    model=tiny(kv).to(device=device,dtype=torch.float16 if device=='cuda' else torch.float32);rng=np.random.default_rng(22+kv+pos)
    parent=ColdStore(tmp_path/'parent',2,kv,16,128)
    originals=[]
    for layer in range(2):
        keys=torch.from_numpy(rng.normal(size=(1,kv,pos,16)).astype('float16'))
        values=torch.from_numpy(rng.normal(size=(1,kv,pos,16)).astype('float16'))
        originals.append((keys,values));parent.append(layer,np.arange(pos),keys,values)
    parent.checkpoint({});parent.close();parent=ColdStore.open(tmp_path/'parent',readonly=True)
    cfg=Config(W=2000,C=0,B=128,R=128,k=15,sink_tokens=128,retrieval='centroid')
    traces=[];outputs=[]
    for label,cache_class,index_class in [('cpu',QuestBoundCache,CandidateRerankIndex),('device',DeviceQuestCache,DeviceRerankIndex)]:
        store=CandidatePageStore(tmp_path/label,2,kv,16,128,prefix=parent,cache_device=device,cache_pages=32,sink_tokens=128)
        cache=cache_class(cfg,store,Rotary(model.model.rotary_emb.inv_freq.detach().cpu().numpy()),device,dtype=torch.float16 if device=='cuda' else torch.float32,initialize=False)
        cache.index=index_class(2,kv,16,128,128,store,pool=32);cache._initialize();events=[]
        def capture(layer,position,q,view):
            events.append((layer,position,view.ids.copy(),view.k.detach().clone(),view.v.detach().clone(),list(view.selected_blocks),list(cache.index.last_candidates[layer])))
        cache.trace=capture;adapter=ModelAdapter.install(model,cache)
        try:
            with torch.inference_mode():outputs.append([adapter.step(t,adapter.position).detach().clone() for t in (7,8)])
            traces.append(events)
            assert not cache.index.pending
            if label=='device':
                report=cache.device_accounting()
                assert report['logical_store_read_bytes']>0
                assert (report['cached_kv_persistent_bytes']>0)==(pos>=2304)
                assert report['sink_persistent_bytes']==2*kv*128*16*2
                assert all(x['helper_calls']==(4 if pos>=2304 else 0) for x in report['index'])
                assert all(x['boundary_reads']==(1 if pos==2127 else 2) for x in report['layers'])
                assert report['cached_kv_page_capacity']==64
                if device=='cpu':
                    assert report['recent_initialization_h2d_source_bytes']==0
                    assert all(x['query_d2h_copies']==x['summary_d2h_copies']==x['validation_scalar_syncs']==0 for x in report['index'])
                else:
                    assert report['recent_initialization_h2d_source_bytes']==2*2000*store.record_bytes
                    assert all(x['query_d2h_copies']==(2 if pos>=2304 else 0) and x['summary_d2h_copies']==(6 if pos>=2304 else 0) and x['validation_scalar_syncs']==(4 if pos>=2304 else 0) for x in report['index'])
        finally:
            cache.trace=None;adapter.uninstall();store.prefix=None;store.close()
    parent.close()
    assert len(traces[0])==len(traces[1])==4
    for a,b in zip(*traces):
        assert a[:2]==b[:2] and a[5:]==b[5:]
        np.testing.assert_array_equal(a[2],b[2]);torch.testing.assert_close(a[3],b[3],rtol=0,atol=0);torch.testing.assert_close(a[4],b[4],rtol=0,atol=0)
        assert len(np.unique(a[2]))==len(a[2]) and max(a[2])==a[1]
        assert all(block>=1 and (block+1)*128<=a[1]-1999 for block in a[5])
    # Predeclared matched-schedule gate: 1e-4 absolute, no relative tolerance.
    for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=0,atol=1e-4 if device=='cuda' else 0)
    if pos<5000:assert len(traces[1][0][6])<32 and len(traces[1][0][5])<15
    else:assert len(traces[1][0][6])==32 and len(traces[1][0][5])==15


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA accounting requires CUDA')
def test_cuda_transfer_counters_cover_staging_and_both_helper_calls(tmp_path):
    model=tiny(2).to(device='cuda',dtype=torch.float16);pos=2304;parent=ColdStore(tmp_path/'parent',2,2,16,128)
    rng=np.random.default_rng(81)
    for layer in range(2):
        keys=torch.from_numpy(rng.normal(size=(1,2,pos,16)).astype('float16'))
        values=torch.from_numpy(rng.normal(size=(1,2,pos,16)).astype('float16'))
        parent.append(layer,np.arange(pos),keys,values)
    parent.checkpoint({});parent.close();parent=ColdStore.open(tmp_path/'parent',readonly=True)
    child=CandidatePageStore(tmp_path/'child',2,2,16,128,prefix=parent,cache_device='cuda',cache_pages=32,sink_tokens=128)
    cfg=Config(W=2000,C=0,B=128,R=128,k=15,sink_tokens=128,retrieval='centroid')
    cache=DeviceQuestCache(cfg,child,Rotary(model.model.rotary_emb.inv_freq.detach().cpu().numpy()),'cuda',initialize=False)
    cache.index=DeviceRerankIndex(2,2,16,128,128,child);cache._initialize();adapter=ModelAdapter.install(model,cache)
    try:
        with torch.inference_mode():assert torch.isfinite(adapter.step(7,pos)).all()
        report=cache.device_accounting()
        assert report['recent_initialization_h2d_source_bytes']==2*2000*child.record_bytes
        assert report['candidate_kv_h2d_bytes']>0 and report['cached_kv_persistent_bytes']>0
        assert all(x['sink_h2d_bytes']>0 and x['boundary_h2d_bytes']>0 and x['attention_sink_kv_h2d_bytes']>0 for x in report['layers'])
        assert all(x['helper_calls']==2 and x['query_d2h_copies']==1 and x['summary_d2h_copies']==3 and x['validation_scalar_syncs']==2 for x in report['index'])
    finally:
        adapter.uninstall();child.prefix=None;child.close();parent.close()

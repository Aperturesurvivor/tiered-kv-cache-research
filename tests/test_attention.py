import os
import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def tiny():
    torch.manual_seed(41)
    cfg=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=96,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32768,rope_theta=10000.,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    return Qwen2ForCausalLM(cfg).eval()


@pytest.mark.parametrize('length',[64,512])
def test_dense_adapter_stock_and_chunked_parity(length):
    from tiered_kv.adapter import ModelAdapter, DenseController
    model=tiny()
    ids=(torch.arange(length)%127)[None]
    with torch.inference_mode():
        expected=model(ids,use_cache=True,logits_to_keep=1)
        expected_logits=[expected.logits[0,-1]]
        past=expected.past_key_values
        for i in range(16):
            token=int(expected_logits[-1].argmax())
            out=model(torch.tensor([[token]]),past_key_values=past,use_cache=True,logits_to_keep=1)
            past=out.past_key_values
            expected_logits.append(out.logits[0,-1])
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    adapter=ModelAdapter.install(model,DenseController(2))
    got=adapter.forward_tokens(ids[0].tolist(),0)
    torch.testing.assert_close(got,expected_logits[0],atol=1e-4,rtol=1e-4)
    for want in expected_logits[1:]:
        got=adapter.step(int(got.argmax()),adapter.position)
        torch.testing.assert_close(got,want,atol=1e-4,rtol=1e-4)
    adapter.set_controller(DenseController(2))
    for start in range(0,length,31):
        got=adapter.forward_tokens(ids[0,start:start+31].tolist(),start)
    torch.testing.assert_close(got,expected_logits[0],atol=1e-4,rtol=1e-4)
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
    adapter.uninstall()


@pytest.mark.parametrize('device,tol',[('cpu',1e-5),pytest.param('accelerator',1e-2,marks=pytest.mark.accelerator)])
def test_sparse_gather_absolute_mask_permutation(device,tol):
    from tiered_kv.adapter import attend
    if device=='accelerator':
        device=os.environ.get('TIERED_DEVICE','').strip().lower() or ('cuda' if torch.cuda.is_available() else 'mps')
        if device not in ('cuda','mps'):
            pytest.fail('TIERED_DEVICE must select cuda or mps')
        available=torch.cuda.is_available() if device=='cuda' else torch.backends.mps.is_available()
        if not available:
            pytest.skip(f'actual {device} unavailable')
    torch.manual_seed(1)
    dtype=torch.float32 if device=='cpu' else torch.float16
    q=torch.randn(1,4,1,16,device=device,dtype=dtype)
    k=torch.randn(1,2,32002,16,device=device,dtype=dtype)
    v=torch.randn_like(k)
    ids=torch.tensor([0,255,4096,30000,32001],device=device)
    valid=torch.arange(32002,device=device)[None,None,None,:]==ids[:,None,None,None]
    mask=valid.any(0,keepdim=True)&(torch.arange(32002,device=device)[None,None,None,:]<=32000)
    dense=attend(q,k,v,mask)
    got=attend(q,k[:,:,ids],v[:,:,ids],(ids<=32000)[None,None,None])
    torch.testing.assert_close(got,dense,atol=tol,rtol=tol)
    perm=torch.tensor([3,1,4,0,2],device=device)
    shuffled=attend(q,k[:,:,ids[perm]],v[:,:,ids[perm]],(ids[perm]<=32000)[None,None,None])
    torch.testing.assert_close(got,shuffled,atol=tol,rtol=tol)
    with pytest.raises(ValueError,match='all-masked'):
        attend(q,k[:,:,:1],v[:,:,:1],torch.zeros(1,1,1,1,dtype=torch.bool,device=device))


def test_tiered_no_eviction_matches_dense_and_queries_change(tmp_path):
    from tiered_kv.adapter import ModelAdapter, DenseController
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    model=tiny()
    tokens=list(range(19))
    adapter=ModelAdapter.install(model,DenseController(2))
    dense=[]
    for token in tokens:
        dense.append(adapter.step(token,adapter.position).clone())
    cfg=Config(W=32,C=2,B=4,R=4,k=1)
    store=ColdStore(tmp_path/'kv',2,2,16,4)
    cache=TieredCache(cfg,store,Rotary(model.model.rotary_emb.inv_freq.numpy()),'cpu',torch.float32)
    observed=[]
    cache.trace=lambda layer,pos,q,view:observed.append((layer,pos,q.clone()))
    adapter.set_controller(cache)
    for token,want in zip(tokens,dense):
        got=adapter.step(token,adapter.position)
        torch.testing.assert_close(got,want,atol=1e-5,rtol=1e-5)
    assert cache.index.searches==[19,19]
    assert len(observed)==38 and not torch.equal(observed[0][2],observed[2][2])
    store.close()
    adapter.uninstall()

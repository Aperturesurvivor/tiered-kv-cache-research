import numpy as np
import torch
import pytest


def test_diagnostic_mass_and_gap_are_finite():
    from tiered_kv.generate import AttentionDiagnostic
    from tiered_kv.types import KVView, Rotary
    torch.manual_seed(3)
    q=torch.randn(1,4,1,16)
    k=torch.randn(1,2,4,16)
    v=torch.randn_like(k)
    ids=np.array([0,255,4096,32000])
    rot=Rotary(1/(10000**(np.arange(0,16,2,dtype=np.float32)/16)))
    view=KVView(ids,k,v,np.stack((ids,ids),1),['compressed','retrieved','recent','recent'],{},1,[1])
    diag=AttentionDiagnostic({'tokens':[0]*32001,'answer_span':[255,256]},rot)
    diag(0,32000,q,view)
    row=diag.rows[0]
    assert abs(sum(row['tier_mass'].values())-1)<1e-5
    assert 0<=row['gold_mass']<=1 and np.isfinite(row['gap_relative_error'])
    assert row['gap_size']==31998


def test_generation_pending_and_eos_never_replayed(tmp_path):
    from tiered_kv.generate import decode_greedy
    from tiered_kv.adapter import ModelAdapter
    from tiered_kv.cache import TieredCache
    from tiered_kv.store import ColdStore
    from tiered_kv.config import Config
    from tiered_kv.types import Rotary
    from test_attention import tiny
    model=tiny()
    rot=Rotary(model.model.rotary_emb.inv_freq.numpy())
    cfg=Config(W=3,C=2,B=4,R=4,k=1)
    cache=TieredCache(cfg,ColdStore(tmp_path/'kv',2,2,16,4),rot,'cpu',dtype=torch.float32)
    adapter=ModelAdapter.install(model,cache)
    logits=adapter.step(4,0)
    first=int(logits.argmax())
    emitted,times=decode_greedy(adapter,logits,3,{first},synchronize=lambda:None)
    assert emitted==[first] and adapter.position==1 and cache.pending_token==first
    cache.checkpoint(pending_token=first,token_hash='test')
    cache.store.close()
    reopened=TieredCache.reopen(tmp_path/'kv',cfg,rot,'cpu',torch.float32)
    adapter.set_controller(reopened)
    adapter.consume_pending()
    assert adapter.position==2 and reopened.store.count==2 and reopened.pending_token is None
    reopened.store.close()
    adapter.uninstall()


@pytest.mark.parametrize('buffered',[False,True])
def test_qwen3_explicit_head_dimension_through_disk_and_tiered_decode(tmp_path,monkeypatch,buffered):
    from research import benchmark_stock
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from tiered_kv import generate
    from tiered_kv.config import Config
    from tiered_kv.types import digest
    from tiered_kv.store import ColdStore
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate,'DEVICE','cpu')
    monkeypatch.setattr(generate,'sync',lambda:None)
    monkeypatch.setattr(generate,'empty_cache',lambda:None)
    monkeypatch.setattr(benchmark_stock,'DEVICE','cpu')
    monkeypatch.setattr(benchmark_stock,'device_synchronize',lambda:None)
    torch.manual_seed(41)
    cfg=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=96,num_hidden_layers=2,
                   num_attention_heads=4,num_key_value_heads=2,head_dim=32,max_position_embeddings=512)
    cfg._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(cfg).half().eval()
    class Guard:
        def check(self): pass
        def metrics(self): return {}
    root=tmp_path/'results'/'head-dimension'
    root.mkdir(parents=True)
    engine=generate.RunEngine(model,None,root,Guard())
    tokens=list(range(16))
    ex={'id':'tiny-latency','task':'latency','tokens':tokens,'continuation':[4]*144,'prefix_tokens_count':8,
        'max_new_tokens':144,'token_hash':digest(tokens),'budget':512,'trial':0}
    try:
        prefix=engine.prefix(ex)
        full=engine.run(Config(id='full',full=True,protocol='query_replay',buffered_writes=buffered),ex,prefix_info=prefix,protocol='query_replay')
        tiered=engine.run(Config(W=256,protocol='query_replay',buffered_writes=buffered),ex,prefix_info=prefix,protocol='query_replay')
        assert full['status']==tiered['status']=='ok'
        assert full['selected_hash']==tiered['selected_hash']
        assert full['archive_tokens']==8 and full['query_tokens']==8
        with_store=ColdStore.open(tiered['overlay'])
        try:
            assert with_store.d==32
            assert with_store.count==160
        finally:
            with_store.close()
    finally:
        engine.adapter.uninstall()
    stock=benchmark_stock.measure(model,ex,Guard())
    assert stock['selected_hash']==full['selected_hash']
    assert stock['kv_disk_writes']==0 and len(stock['latency_ms'])==128


@pytest.mark.parametrize('names,buffered',[(('full','default'),False),(('full_buffered','custom_fixed'),True)])
def test_short_online_validation_exercises_eviction_and_resume(tmp_path,monkeypatch,names,buffered):
    from test_attention import tiny
    from tiered_kv import generate,evals
    from tiered_kv.config import Config
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate,'DEVICE','cpu')
    monkeypatch.setattr(generate,'sync',lambda:None)
    monkeypatch.setattr(generate,'empty_cache',lambda:None)
    monkeypatch.setattr(evals,'exact_prompt',lambda tok,n,*args,**kwargs:('',{'input_ids':[i%127 for i in range(n)]}))
    class Guard:
        def check(self): pass
        def metrics(self): return {}
    configs=[Config(id=name,full=name.startswith('full'),W=16,C=2,B=8,R=8,k=2,buffered_writes=buffered) for name in names]
    rows=generate.online_smoke(tiny().half(),None,tmp_path,configs,Guard(),tokens_count=96,checkpoint_pos=48)
    assert [r['config_id'] for r in rows]==list(names)
    assert all(r['status']=='ok' for r in rows),rows
    assert all(r['processed_tokens']==96 and r['resume_forward_calls']==48 and r['kv_bit_identical'] for r in rows)
    assert rows[1]['index_bytes']>0


def test_online_quality_starts_empty_and_stores_all_ingested_tokens(tmp_path,monkeypatch):
    from test_attention import tiny
    from tiered_kv import generate
    from tiered_kv.config import Config
    from tiered_kv.types import digest
    from tiered_kv.store import ColdStore
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate,'DEVICE','cpu')
    monkeypatch.setattr(generate,'sync',lambda:None)
    monkeypatch.setattr(generate,'empty_cache',lambda:None)
    class Guard:
        def check(self): pass
        def metrics(self): return {}
    engine=generate.RunEngine(tiny().half(),None,tmp_path/'online-quality',Guard())
    monkeypatch.setattr(engine,'prefix',lambda *args: (_ for _ in ()).throw(AssertionError('dense prefix reused')))
    tokens=list(range(32))
    ex={'id':'online-quality','task':'latency','tokens':tokens,'continuation':[4]*144,
        'max_new_tokens':144,'token_hash':digest(tokens),'budget':512,'trial':0}
    try:
        full=engine.run(Config(id='full',full=True,protocol='online_tiered'),ex,protocol='online_tiered')
        sparse=engine.run(Config(W=256,protocol='online_tiered'),ex,protocol='online_tiered')
        evicted=engine.run(Config(W=16,C=2,B=8,R=8,k=2,protocol='online_tiered'),ex,protocol='online_tiered')
        assert full['status']==sparse['status']==evicted['status']=='ok',evicted
        assert full['selected_hash']==sparse['selected_hash']
        assert evicted['index_bytes']>0
        assert evicted['shared_prefix_logical_bytes']==0
        assert evicted['logical_write_bytes']==176*2*2*16*4
        store=ColdStore.open(evicted['overlay'])
        try:
            assert store.prefix is None and store.count==176
        finally:
            store.close()
    finally:
        engine.adapter.uninstall()


def test_online_fixed_lookup_activates_only_when_question_arrives(tmp_path,monkeypatch):
    from test_attention import tiny
    from tiered_kv import generate
    from tiered_kv.config import Config
    from tiered_kv.cache import TieredCache
    from tiered_kv.types import digest
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate,'DEVICE','cpu')
    monkeypatch.setattr(generate,'sync',lambda:None)
    monkeypatch.setattr(generate,'empty_cache',lambda:None)
    class Guard:
        def check(self): pass
        def metrics(self): return {}
    selections=[]
    present=TieredCache.present
    def observe(self,layer,q,k,v,pos):
        view=present(self,layer,q,k,v,pos)
        selections.append((pos,view.selected_blocks))
        return view
    monkeypatch.setattr(TieredCache,'present',observe)
    engine=generate.RunEngine(tiny().half(),None,tmp_path/'delayed',Guard())
    tokens=list(range(32))
    ex={'id':'delayed','task':'latency','tokens':tokens,'continuation':[4]*20,
        'prefix_tokens_count':24,'max_new_tokens':20,'token_hash':digest(tokens),'budget':512,'trial':0}
    try:
        cfg=Config(W=4,C=0,R=4,k=1,retrieval='fixed',fixed_blocks=(0,),buffered_writes=True,sink_tokens=2,protocol='online_tiered')
        row=engine.run(cfg,ex,protocol='online_tiered')
        assert row['status']=='ok',row
        assert row['query_tokens']==8 and row['retrieval_activation_position']==24
        assert all(not blocks for pos,blocks in selections if pos<24)
        assert all(blocks==[0] for pos,blocks in selections if pos>=24)
    finally:
        engine.adapter.uninstall()


def test_persistent_stock_session_loads_exact_prefix_and_consumes_pending_once(tmp_path,monkeypatch):
    from test_attention import tiny
    from research import persistent_kv
    from tiered_kv.adapter import ModelAdapter,DenseController
    from tiered_kv.store import ColdStore
    monkeypatch.setattr(persistent_kv,'DEVICE','cpu')
    model=tiny().half()
    store=ColdStore(tmp_path/'stock-session',2,2,16,4)
    adapter=ModelAdapter.install(model,DenseController(2,store))
    for token in range(16):
        adapter.step(token,adapter.position)
    store.checkpoint({})
    expected=adapter.step(23,16)
    expected_next=adapter.step(24,17)
    adapter.uninstall()
    store.close()
    parent=ColdStore.open(tmp_path/'stock-session',readonly=True)
    session=persistent_kv.StockSession(model,parent)
    torch.testing.assert_close(session.step(23,16),expected,atol=1e-4,rtol=1e-4)
    session.pending_token=24
    torch.testing.assert_close(session.consume_pending(),expected_next,atol=1e-4,rtol=1e-4)
    assert session.position==18 and session.pending_token is None
    parent.close()

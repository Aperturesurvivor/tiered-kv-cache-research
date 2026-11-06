"""Independent small fixtures catch lossy storage, wrong eviction and retrieval leaks."""
import numpy as np
import pytest
import torch


def batch(start, n, h=2, d=4):
    from tiered_kv.types import KVBatch
    x = torch.arange(start*h*d, (start+n)*h*d, dtype=torch.float32).reshape(n,h,d).transpose(0,1)[None].half()
    return KVBatch(np.arange(start,start+n), x, -x)


def test_config_rejects_unknown_and_expands_matrix(tmp_path):
    from tiered_kv.config import load_matrix
    rows = load_matrix('configs/matrix.yaml')
    assert len(rows) == 15 and len({c.digest for c in rows}) == 15
    bad = tmp_path / 'bad.yaml'
    bad.write_text('defaults: {windwo: 3}\nconfigs: [{id: a, overrides: {}}]\n')
    with pytest.raises(ValueError, match='unknown'):
        load_matrix(bad)


def test_recent_w1_evicts_before_current():
    from tiered_kv.tiers import RecentTier
    r = RecentTier(1,1,2,4,'cpu')
    r.commit(0,batch(0,1))
    old, view = r.prepare(0,1)
    assert old.ids.tolist() == [0] and len(view.ids) == 0
    r.commit(0,batch(1,1))
    assert r.view(0).ids.tolist() == [1]


def test_recent_bulk_restore_matches_single_records_across_wrap():
    from tiered_kv.tiers import RecentTier
    bulk=RecentTier(1,8,2,4,'cpu')
    singles=RecentTier(1,8,2,4,'cpu')
    bulk.commit(0,batch(0,6))
    for i in range(6):
        singles.commit(0,batch(i,1))
    bulk.prepare(0,8)
    singles.prepare(0,8)
    bulk.commit(0,batch(6,3))
    for i in range(6,9):
        singles.commit(0,batch(i,1))
    got,want=bulk.view(0),singles.view(0)
    assert got.ids.tolist()==want.ids.tolist()==list(range(1,9))
    assert torch.equal(got.k,want.k) and torch.equal(got.v,want.v)


@pytest.mark.parametrize('device',['cpu','accelerator'])
def test_evicted_recent_records_keep_device_and_survive_ring_overwrite(device):
    from tiered_kv.tiers import RecentTier
    from tiered_kv.telemetry import DEVICE
    if device=='accelerator':
        device=DEVICE
        if device=='cpu':
            pytest.skip('actual accelerator unavailable')
    ring=RecentTier(1,2,2,4,device)
    for pos in (0,1):
        ring.commit(0,batch(pos,1))
    old,_=ring.prepare(0,2)
    assert old.k.device.type==device and old.v.device.type==device
    ring.commit(0,batch(2,1))
    torch.testing.assert_close(old.k.cpu(),batch(0,1).k,atol=0,rtol=0)
    torch.testing.assert_close(old.v.cpu(),batch(0,1).v,atol=0,rtol=0)


@pytest.mark.parametrize('policy', ['mean_rot','first','mean_pre'])
def test_compression_conserves_counts_means_and_intervals(policy):
    from tiered_kv.tiers import CompressedTier
    from tiered_kv.types import Rotary
    rot = Rotary(np.array([1.,.01],np.float32))
    c = CompressedTier(1,2,3,policy,rot)
    x = batch(0,29)
    for i in range(29):
        c.absorb(0,x.slice(i,i+1))
    entries = c.entries(0)
    assert len(c.sealed[0]) <= 2 and sum(s.count for s in entries) == 29
    assert [(s.start,s.end) for s in entries][0][0] == 0
    assert entries[-1].end == 28
    for a,b in zip(entries,entries[1:]):
        assert a.end+1 == b.start
    for s in entries:
        k,v = c.materialize(s)
        source = x.k[0,:,s.start:s.end+1].float().numpy().transpose(1,0,2)
        if policy == 'first':
            np.testing.assert_array_equal(k,source[0])
            assert s.rep == s.start
        elif policy == 'mean_rot':
            np.testing.assert_allclose(k,source.mean(0),rtol=1e-6)
        else:
            expected = rot.apply(rot.apply(source,np.arange(s.start,s.end+1),inverse=True).mean(0)[None],np.array([s.end]))[0]
            np.testing.assert_allclose(k,expected,rtol=1e-5,atol=1e-4)
        np.testing.assert_allclose(v,-source.mean(0) if policy!='first' else -source[0])


@pytest.mark.parametrize('staged',[False,True])
def test_store_prefix_overlay_buffer_and_corruption(tmp_path,staged):
    from tiered_kv.store import ColdStore
    prefix = ColdStore(tmp_path/'prefix',1,2,4,block_size=3,staged_tensor_writes=staged)
    prefix.append(0,batch(0,5))
    got = prefix.read(0,[0,1],5)
    assert got.ids.tolist()==list(range(5))
    assert torch.equal(got.k,batch(0,5).k)
    prefix.checkpoint({'tag':'fixture'})
    prefix.close()
    p = ColdStore.open(tmp_path/'prefix',readonly=True)
    child = ColdStore(tmp_path/'child',1,2,4,block_size=3,prefix=p,staged_tensor_writes=staged)
    child.append(0,batch(5,2))
    assert torch.equal(child.read(0,[1,2],7).k,batch(3,4).k)
    child.checkpoint({'pending_token':42})
    child.close()
    restored = ColdStore.open(tmp_path/'child')
    assert restored.metadata['pending_token']==42
    assert restored.staged_tensor_writes==staged
    assert torch.equal(restored.read_range(0,0,7).v,batch(0,7).v)
    with pytest.raises(ValueError,match='consecutive'):
        restored.append(0,batch(6,1))
    restored.close()
    f = tmp_path/'child'/'layer-00.kv'
    with f.open('r+b') as out:
        out.write(b'bad')
    with pytest.raises(ValueError,match='checksum'):
        ColdStore.open(tmp_path/'child')


def test_index_partial_updates_ties_and_grouped_queries():
    from tiered_kv.retrieval import BlockIndex
    from tiered_kv.types import KVBatch
    idx=BlockIndex(1,2,4,3)
    zeros=torch.zeros(1,2,7,4).half()
    idx.absorb(0,KVBatch(np.arange(7),zeros,zeros))
    ids,scores=idx.search(0,torch.zeros(1,12,1,4),9)
    assert ids.tolist()==[0,1,2] and scores.tolist()==[0,0,0]
    assert idx.counts[0][:3].tolist()==[3,3,1]
    with pytest.raises(ValueError,match='order'):
        idx.absorb(0,KVBatch(np.array([6]),zeros[:,:,:1],zeros[:,:,:1]))


def test_rotary_inverse_roundtrip():
    from tiered_kv.types import Rotary
    rng=np.random.default_rng(7)
    x=rng.normal(size=(4,2,16)).astype('float32')
    rot=Rotary(1/(10000**(np.arange(0,16,2,dtype=np.float32)/16)))
    ids=np.array([0,255,4096,30000])
    np.testing.assert_allclose(rot.apply(rot.apply(x,ids),ids,True),x,atol=1e-5)


def test_cache_bounds_partial_blocks_disabled_and_rebuild(tmp_path):
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    for W,C,k in [(1,2,2),(3,0,2),(3,2,0),(3,0,0)]:
        cfg=Config(id=f'x{W}{C}{k}',W=W,C=C,B=3,R=4,k=k)
        store=ColdStore(tmp_path/cfg.id,2,2,4,block_size=4)
        cache=TieredCache(cfg,store,Rotary(np.array([1.,.01],np.float32)),'cpu')
        for pos in range(19):
            for layer in range(2):
                record=batch(pos,1)
                view=cache.present(layer,torch.ones(1,12,1,4),record.k,record.v,pos)
                exact=view.ids[np.array(view.tiers)!='compressed']
                assert len(exact)==len(set(exact)) and max(exact)==pos
                assert len(view.ids)<=W+k*4+C+1
                assert view.intervals[:,1].max()<=pos
                cache.commit_layer(layer,record)
            cache.finish_token()
        assert cache.position==19
        assert cache.index.searches==([19,19] if k else [0,0])
        assert not cache.compressed.entries(0) if C==0 else True
        cache.checkpoint(pending_token=42,token_hash='fixed')
        sums=cache.index.sums[0].copy()
        store.close()
        restored=TieredCache.reopen(tmp_path/cfg.id,cfg,cache.rotary,'cpu')
        assert restored.position==19 and restored.pending_token==42
        np.testing.assert_array_equal(restored.index.sums[0],sums)
        restored.verify_rebuild()
        restored.store.close()


@pytest.mark.parametrize('retrieval',['centroid','fixed'])
@pytest.mark.parametrize('staged',[False,True])
def test_online_tiny_resume_consumes_pending_once(tmp_path,retrieval,staged):
    from tiered_kv.adapter import ModelAdapter
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    from test_attention import tiny
    model=tiny()
    cfg=Config(W=3,C=2,B=4,R=4,k=1,retrieval=retrieval,fixed_blocks=(0,) if retrieval=='fixed' else (),sink_tokens=4,buffered_writes=staged)
    rot=Rotary(model.model.rotary_emb.inv_freq.numpy())
    cache=TieredCache(cfg,ColdStore(tmp_path/'online',2,2,16,4,staged_tensor_writes=staged),rot,'cpu',dtype=torch.float32)
    adapter=ModelAdapter.install(model,cache)
    tokens=list(range(23))
    for token in tokens[:11]:
        logits=adapter.step(token,adapter.position)
    cache.checkpoint(pending_token=tokens[11],token_hash='fixture')
    for token in tokens[11:]:
        logits=adapter.step(token,adapter.position)
    expected=logits.clone()
    cache.store.close()
    cache2=TieredCache.reopen(tmp_path/'online',cfg,rot,'cpu',dtype=torch.float32)
    adapter.set_controller(cache2)
    assert cache2.pending_token==11
    adapter.consume_pending()
    for token in tokens[12:]:
        logits=adapter.step(token,adapter.position)
    # Resumed recent values are restored from fp16 cold storage, unlike FP32 fixture.
    torch.testing.assert_close(logits,expected,atol=1e-4,rtol=1e-4)
    assert cache2.position==23 and cache2.pending_token is None
    cache2.store.close()
    adapter.uninstall()


def test_fixed_disk_addresses_preserve_exact_kv_and_bound_resident_blocks(tmp_path):
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    cfg=Config(W=3,C=2,B=4,R=4,k=2,retrieval='fixed',fixed_blocks=(0,3))
    store=ColdStore(tmp_path/'fixed',1,2,4,4)
    cache=TieredCache(cfg,store,Rotary(np.array([1.,.01],np.float32)),'cpu')
    for pos in range(24):
        before=store.read_bytes
        rec=batch(pos,1)
        view=cache.present(0,None,rec.k,rec.v,pos)
        retrieved=np.array(view.tiers)=='retrieved'
        expected=[i for b in (0,3) for i in range(b*4,min((b+1)*4,max(0,pos-2)))]
        assert view.ids[retrieved].tolist()==expected
        if expected:
            want=batch(0,pos).k[:,:,expected]
            assert torch.equal(view.k[:,:,retrieved],want)
        if pos>=19:
            assert store.read_bytes==before # Selected old blocks stay resident.
        cache.commit_layer(0,rec)
        cache.finish_token()
    assert cache.index.nbytes==16
    assert cache.index.sums[0].size==0
    assert len(cache.retrieved_cache[0][1].ids)==8
    cache.checkpoint(pending_token=42,token_hash='fixed-addresses')
    store.close()
    restored=TieredCache.reopen(tmp_path/'fixed',cfg,cache.rotary,'cpu')
    assert restored.index.blocks.tolist()==[0,3]
    assert restored.verify_rebuild()
    restored.store.close()


def test_store_rejects_short_write_short_read_and_truncated_manifest_extent(tmp_path,monkeypatch):
    from tiered_kv.store import ColdStore
    store=ColdStore(tmp_path/'short',1,2,4,2)
    class ShortWriter:
        def write(self,data):
            return len(data)-1
    original=store.files[0]
    store.files[0]=ShortWriter()
    with pytest.raises(OSError,match='short write'):
        store.append(0,batch(0,2))
    store.files[0]=original
    store.close()
    other=ColdStore(tmp_path/'truncated',1,2,4,2)
    other.append(0,batch(0,4))
    other.checkpoint({})
    other.close()
    with (tmp_path/'truncated'/'layer-00.kv').open('r+b') as out:
        out.truncate(1)
    with pytest.raises(ValueError,match='short file'):
        ColdStore.open(tmp_path/'truncated')


def test_disabled_index_allocates_no_statistics(tmp_path):
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    store=ColdStore(tmp_path/'off',1,2,4,4)
    cache=TieredCache(Config(C=0,k=0),store,Rotary(np.array([1.,.01],np.float32)),'cpu')
    assert cache.index.nbytes==0
    store.close()


@pytest.mark.parametrize('selected',[(0,),(2,)])
def test_fixed_initialization_reads_only_recent_and_sinks_deduplicate(tmp_path,selected):
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    store=ColdStore(tmp_path/'fixed-sinks',1,2,4,4,staged_tensor_writes=True)
    store.append(0,batch(0,21))
    cfg=Config(W=3,C=0,R=4,k=1,retrieval='fixed',fixed_blocks=selected,sink_tokens=4,buffered_writes=True)
    cache=TieredCache(cfg,store,Rotary(np.array([1.,.01],np.float32)),'cpu')
    assert store.read_bytes==3*store.record_bytes
    assert cache.index.next_pos==[18]
    rec=batch(21,1)
    view=cache.present(0,None,rec.k,rec.v,21)
    assert len(view.ids)==len(set(view.ids))
    assert set(range(4))<=set(view.ids)
    assert len(view.ids)<=cfg.W+cfg.k*cfg.R+cfg.sink_tokens
    assert torch.equal(view.k,batch(0,22).k[:,:,view.ids])
    cache.commit_layer(0,rec)
    cache.finish_token()
    cache.checkpoint(pending_token=42,token_hash='sinks')
    store.close()
    restored=TieredCache.reopen(tmp_path/'fixed-sinks',cfg,cache.rotary,'cpu')
    rec=batch(22,1)
    resumed=restored.present(0,None,rec.k,rec.v,22)
    assert set(range(4))<=set(resumed.ids)
    assert torch.equal(resumed.k,batch(0,23).k[:,:,resumed.ids])
    restored.store.close()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_staged_store_matches_unbuffered_bytes_and_reads_pending_extent(tmp_path,device):
    from tiered_kv.store import ColdStore
    from tiered_kv.types import KVBatch
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA validation runs on Colab')
    stores=[ColdStore(tmp_path/str(staged),1,2,4,4,staged_tensor_writes=staged) for staged in (False,True)]
    for start,n in ((0,3),(3,6),(9,2)):
        source=batch(start,n)
        source=KVBatch(source.ids,source.k.to(device),source.v.to(device))
        for store in stores:
            store.append(0,source)
            got=store.read_range(0,1,start+n)
            assert torch.equal(got.k,batch(1,start+n-1).k)
    assert stores[1].tensor_buffers[0].numel()==4*2*2*4
    for store in stores:
        store.checkpoint({'test':'exact'})
        store.close()
    assert (tmp_path/'False/layer-00.kv').read_bytes()==(tmp_path/'True/layer-00.kv').read_bytes()


def test_persistent_selection_switch_releases_old_blocks_and_survives_resume(tmp_path):
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    store=ColdStore(tmp_path/'switch',1,2,4,4,staged_tensor_writes=True)
    store.append(0,batch(0,24))
    cfg=Config(W=3,C=0,R=4,k=1,retrieval='fixed',fixed_blocks=(0,),buffered_writes=True)
    cache=TieredCache(cfg,store,Rotary(np.array([1.,.01],np.float32)),'cpu')
    for pos,chosen in enumerate(((0,),(2,),(0,)),24):
        cache.select_blocks(chosen)
        rec=batch(pos,1)
        view=cache.present(0,None,rec.k,rec.v,pos)
        assert view.ids[np.array(view.tiers)=='retrieved'].tolist()==list(range(chosen[0]*4,chosen[0]*4+4))
        assert len(cache.retrieved_cache[0][1].ids)==4
        cache.commit_layer(0,rec)
        with pytest.raises(ValueError,match='complete-token'):
            cache.select_blocks((1,))
        cache.finish_token()
    cache.select_blocks((3,))
    cache.checkpoint(pending_token=42,token_hash='switch')
    store.close()
    reopened=TieredCache.reopen(tmp_path/'switch',cache.config,cache.rotary,'cpu')
    assert reopened.index.blocks.tolist()==[3]
    assert reopened.config.fixed_blocks==(3,)
    reopened.store.close()


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_partial_selected_block_extends_without_repeated_disk_reads(tmp_path,dtype):
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary,KVBatch
    store=ColdStore(tmp_path/'partial',1,2,4,4,staged_tensor_writes=True)
    cfg=Config(W=3,C=0,R=4,k=2,retrieval='fixed',fixed_blocks=(0,3),buffered_writes=True)
    cache=TieredCache(cfg,store,Rotary(np.array([1.,.01],np.float32)),'cpu',dtype)
    for pos in range(24):
        x=batch(pos,1)
        rec=KVBatch(x.ids,(x.k.float()+.3333).to(dtype),x.v.to(dtype))
        before=store.read_bytes
        view=cache.present(0,None,rec.k,rec.v,pos)
        if pos in (4,5,6,16,17,18) or pos>=19:
            assert store.read_bytes==before
        retrieved=np.array(view.tiers)=='retrieved'
        expected=store.read(0,view.selected_blocks,max(0,pos-cfg.W+1))
        assert view.ids[retrieved].tolist()==expected.ids.tolist()
        assert torch.equal(view.k[:,:,retrieved],expected.k.to(dtype))
        assert torch.equal(view.v[:,:,retrieved],expected.v.to(dtype))
        cache.commit_layer(0,rec)
        cache.finish_token()
    store.close()


def test_oracle_replaces_two_gold_blocks_without_losing_first(tmp_path):
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    store=ColdStore(tmp_path/'oracle',1,2,4,4)
    cfg=Config(W=1,C=0,k=2,R=4)
    cache=TieredCache(cfg,store,Rotary(np.array([1.,.01],np.float32)),'cpu')
    for pos in range(25):
        rec=batch(pos,1)
        cache.present(0,torch.ones(1,12,1,4),rec.k,rec.v,pos)
        cache.commit_layer(0,rec)
        cache.finish_token()
    cache.oracle_blocks=[4,5]
    rec=batch(25,1)
    view=cache.present(0,torch.ones(1,12,1,4),rec.k,rec.v,25)
    assert set(view.selected_blocks)=={4,5}
    assert set(range(16,24))<=set(view.ids)
    store.close()


@pytest.mark.parametrize('staged',[False,True])
def test_2k_cpu_online_checkpoint_1024_exact_continuation(tmp_path,staged):
    from tiered_kv.adapter import ModelAdapter
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    from test_attention import tiny
    model=tiny().half()
    cfg=Config(W=13,C=2,B=16,R=16,k=1,buffered_writes=staged)
    rot=Rotary(model.model.rotary_emb.inv_freq.numpy())
    store=ColdStore(tmp_path/'2k',2,2,16,16,staged_tensor_writes=staged)
    cache=TieredCache(cfg,store,rot,'cpu',torch.float16)
    adapter=ModelAdapter.install(model,cache)
    tokens=[i%127 for i in range(2048)]
    for pos,token in enumerate(tokens):
        logits=adapter.step(token,pos)
        if pos==1023:
            cache.checkpoint(pending_token=tokens[1024],token_hash='2k-fixture')
    expected=logits.clone()
    kv=store.read_range(0,0,2048)
    store.close()
    restored=TieredCache.reopen(tmp_path/'2k',cfg,rot,'cpu',torch.float16)
    restored.verify_rebuild()
    adapter.set_controller(restored)
    adapter.consume_pending()
    for pos in range(1025,2048):
        logits=adapter.step(tokens[pos],pos)
    torch.testing.assert_close(logits,expected,atol=1e-5,rtol=1e-5)
    assert restored.position==2048
    assert torch.equal(restored.store.read_range(0,0,2048).k,kv.k)
    assert restored.present_calls==[1024,1024]
    restored.store.close()
    adapter.uninstall()

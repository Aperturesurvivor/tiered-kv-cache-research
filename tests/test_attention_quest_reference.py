import numpy as np
import torch
import pytest
from research.attention_quest_reference import QuestBoundIndex,QuestBoundCache
from research.attention_representations import compare_representations,_normalization_positions
from tiered_kv.types import KVBatch


@pytest.mark.parametrize("seed",[7,19,43,73,97])
def test_online_bounds_match_full_screen_ranking(seed):
    rng=np.random.default_rng(seed);past=rng.normal(size=(2,56,4)).astype('float16')
    current=rng.normal(size=(2,1,4)).astype('float16');q=rng.normal(size=(4,4)).astype('float16')
    expected=compare_representations(q,past,current,56,sink=8,window=8,page=8,k=3)['quest_bound']['pages']
    index=QuestBoundIndex(1,2,4,8,8);cutoff=49
    keys=torch.from_numpy(past[:,:cutoff])[None];index.absorb(0,KVBatch(np.arange(cutoff),keys,torch.zeros_like(keys)))
    full=np.concatenate((past,current),axis=1);fixed=full[:,_normalization_positions(57,cutoff,8,8),:]
    index.stage(0,torch.from_numpy(fixed.copy()))
    got,_=index.search(0,torch.from_numpy(q)[None,:,None,:],3)
    assert got.tolist()==expected
    assert index.next_pos==[49] and index.counts[0][6]==1
    before=index.nbytes
    extra=torch.from_numpy(past[:,49:56])[None]
    index.absorb(0,KVBatch(np.arange(49,56),extra,torch.zeros_like(extra)))
    assert index.counts[0][6]==8 and index.next_pos==[56] and index.nbytes>=before
    np.testing.assert_array_equal(index.sums[0][6],past[:,48:56].astype('float32').min(axis=1))
    np.testing.assert_array_equal(index.maxima[0][6],past[:,48:56].astype('float32').max(axis=1))


def test_tiny_actual_model_stages_current_and_boundary(tmp_path):
    from test_attention import tiny
    from research.reasoning_sessions import CountedSession
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    model=tiny();prefix=ColdStore(tmp_path/'prefix',2,2,16,4)
    adapter=ModelAdapter.install(model,DenseController(2,prefix));adapter.forward_tokens([3,4,5,6]*5,0)
    adapter.uninstall();prefix.checkpoint({});prefix.close()
    parent=ColdStore.open(tmp_path/'prefix',readonly=True)
    child=ColdStore(tmp_path/'child',2,2,16,4,prefix=parent,staged_tensor_writes=True)
    cfg=Config(W=3,C=0,B=4,R=4,k=2,sink_tokens=4,retrieval='centroid')
    cache=QuestBoundCache(cfg,child,Rotary(model.model.rotary_emb.inv_freq.detach().numpy()),'cpu',dtype=torch.float32,initialize=False)
    cache.index=QuestBoundIndex(2,2,16,4,4);cache._initialize();adapter=ModelAdapter.install(model,cache)
    try:
        session=CountedSession(adapter,model,active_limit=16)
        logits=session.feed_prompt([7,8,9]);assert torch.isfinite(logits).all()
        assert cache.position==23 and not cache.index.pending
        assert all(s['boundary_calls']>0 and s['boundary_logical_bytes']>0 for s in cache.index.stats)
        assert all(n>0 for n in cache.index.searches)
    finally:
        cache.trace=None;adapter.uninstall();child.prefix=None;child.close();parent.close()


def test_generation_gate_binds_recorded_probe_check(tmp_path):
    import json
    import pytest
    from pathlib import Path
    from research.attention_quest_32k import verify_screen_check,CHECKPOINT_SHA
    from research.attention_trajectory_diagnostic import ROOT,file_sha
    marker=dict(status='complete',compared=168,checkpoint_sha256=CHECKPOINT_SHA,
        screen_sha256='cd125a3b90fe144ed955420d6b7186b66f16caee7462dc8b9963f042bf5977c0',
        trace_sha256='4d96eeab4683b76305901bc208b29cbb191c965224204c910c471c1511b65bd8',
        quest_source_sha256=file_sha(ROOT/'research/attention_quest_reference.py'),
        checker_source_sha256=file_sha(ROOT/'research/attention_quest_screen_check.py'))
    target=tmp_path/'complete.json';target.write_text(json.dumps(marker))
    assert verify_screen_check(target)==file_sha(target)
    marker['compared']=167;target.write_text(json.dumps(marker))
    with pytest.raises(AssertionError):verify_screen_check(target)

import numpy as np
import torch
from research.quest_index_scaling import build_index,all_layers,LENGTHS,TRIALS,POOL,SEED


def test_deterministic_complete_partial_and_candidate_underfill():
    a,geometry=build_index(512,layers=2,h=2,d=8,page=16,window=64,sink=16)
    b,other=build_index(512,layers=2,h=2,d=8,page=16,window=64,sink=16)
    assert geometry==other and geometry['cutoff']==449 and geometry['complete_pages']==28 and geometry['partial_tokens']==1
    assert geometry['eligible_complete_pages']==27
    for i in range(2):
        assert np.array_equal(a.sums[i],b.sums[i]) and np.array_equal(a.maxima[i],b.maxima[i])
        assert np.isfinite(a.sums[i]).all() and np.all(a.sums[i]<=a.maxima[i])
        assert a.counts[i][27]==16 and a.counts[i][28]==1 and a.counts[i][29]==0
    q=[torch.ones((1,4,1,8)) for _ in range(2)];fixed=[torch.ones((2,25,8)) for _ in range(2)]
    chosen=all_layers(a,q,fixed)
    assert all(len(x)==27 and 0 not in x and 28 not in x for x in chosen)
    assert chosen==all_layers(a,q,fixed)


def test_predeclared_lengths_and_4k_underfill():
    assert LENGTHS==(4096,8192,32768,131072,524288,1048576) and TRIALS==7 and POOL==32 and SEED==317
    _,g=build_index(4096,layers=1,h=2,d=8)
    assert g['cutoff']==2097 and g['partial_tokens']==49 and g['eligible_complete_pages']==15


def test_initial_headroom_and_runtime_floor_are_distinct(tmp_path,monkeypatch):
    import pytest
    from types import SimpleNamespace
    import research.quest_index_scaling as scaling
    monkeypatch.setattr(scaling.psutil,'virtual_memory',lambda:SimpleNamespace(available=5*2**29))
    monkeypatch.setattr(scaling.psutil,'Process',lambda:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=1)))
    result=scaling.check_resources(tmp_path/'result.json')
    assert result['available_bytes']==5*2**29
    with pytest.raises(RuntimeError,match='memory below'):
        scaling.check_resources(tmp_path/'result.json',initial=True)

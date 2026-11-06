import numpy as np
import torch
from types import SimpleNamespace

from research.attention_trajectory_diagnostic import CASES, load_plan, score_layer


def test_recorded_cases_match_expected_json_trajectories():
    for case in CASES:
        plan=load_plan(case)
        assert plan['source_outcomes']=={'stock':True, 'sparse':'literal' in case}
        assert len(plan['text_cold_pages']) <= len(plan['original_text_pages'])
        assert 0 < len(plan['prompt']) <= 160


def test_sink_tradeoff_preserves_budget_and_cold_universe():
    length=4608
    rng=np.random.default_rng(9)
    k=torch.from_numpy(rng.normal(size=(1,1,length,2)).astype('float32'))
    v=torch.from_numpy(rng.normal(size=(1,1,length,2)).astype('float32'))
    view=SimpleNamespace(ids=np.arange(length),k=k,v=v)
    q=torch.tensor([[[[0.2,0.4]], [[-0.5,0.3]]]])
    result=score_layer(q,view,length-1,[1,2], 'synthetic',0)
    sink=result['sink_ablation']
    assert set(sink['arms'])=={'0','4','16','128'}
    assert len({arm['active_tokens'] for arm in sink['arms'].values()})==1
    assert all(1<=b and (b+1)*128<=length-2128 for b in sink['cold_pages'])
    assert len(sink['dense_prefix_mass']['128'])==2
    assert all('common_cold_retained_fraction' in x for x in result['metrics'].values())

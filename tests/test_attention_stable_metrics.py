import numpy as np
from research.attention_stable_metrics import stable_metrics


def test_high_logits_are_stable_and_fixed_selections_are_preserved():
    # 15 selectable pages, nonaligned cutoff and dominant sink/recent keys.
    n=136;logits=np.full((3,n+1),10000,dtype=np.float32)
    logits[0,0]=10010;logits[1,2:10]=9990;logits[2,-1]=10008
    pages=list(range(1,16));choices={'original':pages,'reordered':list(reversed(pages))}
    out=stable_metrics(logits,n,choices,sink=8,window=8,page=8)
    assert out['metrics']['original']['pages']==pages
    assert out['metrics']['reordered']['pages']==list(reversed(pages))
    assert all(abs(x-1)<1e-12 for x in out['normalization_sums'])
    assert all(0<=x<=1 for x in out['metrics']['original']['retained_mass'])
    for field in ('retained_mass','common_cold_fraction'):
        np.testing.assert_allclose(out['metrics']['original'][field],out['metrics']['reordered'][field],atol=1e-14)


def test_saved_choice_structure_rejects_tampering():
    import copy,json,pytest
    from research.attention_stable_replay import CAND,EVIDENCE,selection_choices
    row=json.loads((CAND/'screen.jsonl').read_text().splitlines()[0])
    prior=json.loads((EVIDENCE/'screen.jsonl').read_text().splitlines()[0])['results']
    choices=selection_choices(row,prior)
    assert len(choices)==8 and choices['quest']==row['quest_top15']
    changed=copy.deepcopy(row);pages=changed['results'][1]['arms']['candidate_only']['pages'];pages[0]=pages[1]
    with pytest.raises(AssertionError):selection_choices(changed,prior)


def test_rule_uses_shared_stable_metrics_and_predeclared_threshold():
    from research.attention_stable_replay import summarize
    def row(pos):
        masses={'quest':.5,'oracle':.7,'16-candidate_only':.59,'32-candidate_only':.61,'64-candidate_only':.65}
        return {'position':pos,'metrics':{name:{'retained_mass':[value]*12} for name,value in masses.items()}}
    result=summarize([row(1),row(2)],[1,2])
    assert result['selected']==32 and [e['qualifies'] for e in result['eligibility']]==[False,True,True]

import pytest
import torch
from research.attention_observer_ablation import schedule, compare_final, check_equivalence


def test_schedule_and_paired_balancing():
    rows=schedule();assert len(rows)==12 and len(set(rows))==12
    for budget in (8192,32768):
        for rep in range(3):assert {arm for b,r,arm in rows if (b,r)==(budget,rep)}=={'on','off'}
    assert [rows[i][2] for i in range(0,12,2)]==['on','off','on','off','on','off']


def test_final_logits_and_validation_fail_closed():
    a=torch.tensor([1.,2.]);assert compare_final(a,a)==0
    with pytest.raises(ValueError):compare_final(a,torch.tensor([1.,2.001]))
    with pytest.raises(ValueError):compare_final(a,torch.tensor([1.,float('nan')]))
    with pytest.raises(ValueError):compare_final(a,torch.tensor([1.]))
    ref={'events':[[0,1]],'kv_sha256':['abc'],'metrics':{'calls':1},'logits':[a]}
    assert check_equivalence(ref,ref)==0
    for changed in ({**ref,'events':[]},{**ref,'kv_sha256':['changed']},{**ref,'logits':[torch.tensor([1.,2.001])]}):
        with pytest.raises(ValueError):check_equivalence(ref,changed)


def test_observer_off_retains_count_guard_and_on_wraps_it():
    from types import SimpleNamespace
    from research.attention_observer_ablation import select_observer
    calls=[];events=[];hashes=[]
    def count(layer,pos,q,view):
        calls.append(len(view.ids))
        if len(view.ids)>2:raise RuntimeError('active KV limit')
    controller=SimpleNamespace(index=SimpleNamespace(last_candidates=[[1]]))
    view=SimpleNamespace(ids=[0,1],selected_blocks=[1],k=torch.tensor([1.]),v=torch.tensor([2.]))
    off=select_observer('off',controller,count,events,hashes,False);assert off is count
    off(0,1,None,view);assert calls==[2] and not events
    on=select_observer('on',controller,count,events,hashes,False);on(0,1,None,view)
    assert calls==[2,2] and len(events)==1
    view.ids=[0,1,2]
    with pytest.raises(RuntimeError):off(0,2,None,view)
    with pytest.raises(RuntimeError):on(0,2,None,view)
    with pytest.raises(ValueError):select_observer('off',controller,count,events,hashes,True)

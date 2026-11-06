import pytest
import torch
from research.attention_device_cost import cases,check_equivalence,schedule

def test_cases_and_balanced_schedule():
    p={'fixtures':[{'seed':283,'task':'linked_fact','budget':n,'prefix_tokens_count':100,'tokens':list(range(110)),'id':str(n)} for n in (8192,32768)]}
    assert [tokens for _,tokens in cases(p)]==[list(range(36,100))]*2
    assert len(schedule())==18
    for n in (8192,32768):
        orders=[[x[2] for x in schedule() if x[:2]==(n,r)] for r in range(3)]
        assert all(sorted(x)==['cpu','device','stock'] for x in orders)
        assert all(len({x[i] for x in orders})==3 for i in range(3))
    p['fixtures'].append(p['fixtures'][0])
    with pytest.raises(ValueError):cases(p)

def test_equivalence_rejects_routes_kv_and_logit_differences():
    a={'events':[[0,1,[1],[1],2256]],'kv_sha256':['abc'],'metrics':{'calls':1},'logits':[torch.tensor([1.,2.])]}
    assert check_equivalence(a,a)==0
    for b in ({**a,'events':[]},{**a,'kv_sha256':['other']},{**a,'logits':[torch.tensor([1.,2.001])]}):
        with pytest.raises(ValueError):check_equivalence(a,b)
    assert check_equivalence(a,{**a,'logits':[torch.tensor([1.,2.00005])]})<=1e-4

def test_observer_preserves_routes_and_only_hashes_kv_during_validation():
    from types import SimpleNamespace
    from research.attention_device_cost import make_observer
    controller=SimpleNamespace(index=SimpleNamespace(last_candidates=[[2,3]]))
    view=SimpleNamespace(ids=[0,1,17],selected_blocks=[2],k=torch.tensor([1.,2.]),v=torch.tensor([3.,4.]))
    events=[]; hashes=[]; calls=[]
    observe=make_observer(controller,lambda *args:calls.append(args[0]),events,hashes,True)
    observe(0,17,None,view)
    assert events[0][:5]==[0,17,[2,3],[2],3] and len(events[0][5])==64 and len(hashes)==1 and calls==[0]
    observe=make_observer(controller,lambda *args:None,events,hashes,False)
    view.ids=[0,1,18]
    observe(0,18,None,view)
    assert len(hashes)==1 and events[-1][1]==18

def test_observer_rejects_duplicate_and_future_positions():
    from types import SimpleNamespace
    from research.attention_device_cost import make_observer
    controller=SimpleNamespace(index=SimpleNamespace(last_candidates=[[2]]))
    view=SimpleNamespace(ids=[0,1,1],selected_blocks=[2],k=torch.tensor([1.]),v=torch.tensor([2.]))
    observe=make_observer(controller,lambda *args:None,[],[],False)
    with pytest.raises(ValueError):observe(0,1,None,view)
    view.ids=[0,2]
    with pytest.raises(ValueError):observe(0,1,None,view)

def test_gpu_idle_check_precedes_cuda_initialization():
    import inspect
    from research.attention_device_cost import main
    source=inspect.getsource(main)
    assert source.index("['nvidia-smi','--query-compute-apps=pid'") < source.index('torch.cuda.is_available()') < source.index('torch.cuda.get_device_name(0)')

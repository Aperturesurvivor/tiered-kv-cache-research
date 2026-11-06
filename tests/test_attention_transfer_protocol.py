import pytest
from research.attention_transfer_protocol import BUDGETS,SEEDS,TASKS,POLICY,make_fixture
from research.reasoning_protocol import _task,make_fixture as old_fixture

class Tokenizer:
    chat_template='test'
    def apply_chat_template(self,messages,**kw):return messages[0]['content']+'\n<assistant>'
    def encode(self,text,**kw):return list(map(ord,text))
    def __call__(self,text,**kw):return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def decode(self,ids,**kw):return ''.join(map(chr,ids))

@pytest.mark.parametrize('kind',TASKS)
def test_transfer_orientation_and_required_spans(kind):
    tok=Tokenizer()
    for seed in SEEDS:
        ex=make_fixture(tok,8192,seed,kind);records,query,answer,required=_task(seed,kind)
        if seed==283:records.reverse();required=[1-i for i in required]
        assert ex['records']==records and ex['required_record_indices']==required
        assert ex['query']==query and ex['answer']==answer
        assert all(tok.decode(ex['tokens'][a:b])==record for record,(a,b) in zip(records,ex['record_spans']))
        assert all(b<ex['prefix_tokens_count']-POLICY['W'] for a,b in ex['record_spans'])
    # First seed retains exact original fixture construction semantics apart from its ID/metadata.
    new=make_fixture(tok,8192,281,kind);old=old_fixture(tok,8192,281,kind)
    for key in ('tokens','query','answer','record_spans','required_record_indices','first_suffix','token_hash'):
        assert new[key]==old[key]

def test_declared_cohort_and_fixed_policy():
    assert sum(len(x)*len(SEEDS)*len(TASKS) for x in BUDGETS.values())==18
    assert POLICY=={'W':2000,'C':0,'R':128,'sink_tokens':128,'candidate_ceiling':32,'final_ceiling':15,'rounds':2,'cap_per_round':128}
    with pytest.raises(ValueError):make_fixture(Tokenizer(),8192,211,'literal')

@pytest.mark.parametrize('family,kv_heads',[('qwen2',2),('llama',4)])
@pytest.mark.parametrize('prefix_repeats',[3,5])
def test_tiny_actual_model_rerank_geometry_and_underfilled_pool(tmp_path,family,kv_heads,prefix_repeats):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM,LlamaConfig,LlamaForCausalLM
    from research.attention_rerank_live import CandidatePageStore,CandidateRerankIndex
    from research.attention_quest_reference import QuestBoundCache
    from research.reasoning_sessions import CountedSession
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    cfg_type,model_type=(Qwen2Config,Qwen2ForCausalLM) if family=='qwen2' else (LlamaConfig,LlamaForCausalLM)
    cfg=cfg_type(vocab_size=128,hidden_size=64,intermediate_size=96,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=kv_heads,max_position_embeddings=8192,attention_dropout=0.)
    cfg._attn_implementation='sdpa';model=model_type(cfg).eval();parent=ColdStore(tmp_path/'prefix',2,kv_heads,16,4)
    adapter=ModelAdapter.install(model,DenseController(2,parent));adapter.forward_tokens([3,4,5,6]*prefix_repeats,0);adapter.uninstall();parent.checkpoint({});parent.close()
    parent=ColdStore.open(tmp_path/'prefix',readonly=True)
    child=CandidatePageStore(tmp_path/'child',2,kv_heads,16,4,prefix=parent,staged_tensor_writes=True,cache_device='cpu',cache_pages=8,sink_tokens=4)
    policy=Config(W=3,C=0,B=4,R=4,k=2,sink_tokens=4,retrieval='centroid')
    controller=QuestBoundCache(policy,child,Rotary(model.model.rotary_emb.inv_freq.detach().numpy()),'cpu',dtype=torch.float32,initialize=False)
    controller.index=CandidateRerankIndex(2,kv_heads,16,4,4,child,pool=8);controller._initialize();adapter=ModelAdapter.install(model,controller)
    try:
        session=CountedSession(adapter,model,active_limit=16)
        assert torch.isfinite(session.feed_prompt([7,8,9])).all()
        assert all(len(p)<8 for p in child.pages)
        assert all(sum(stats['candidate_counts'].values())==3 for stats in controller.index.rerank_stats)
        assert all(sum(stats['final_counts'].values())==3 for stats in controller.index.rerank_stats)
        assert all(all(int(k)<=2 for k in stats['final_counts']) for stats in controller.index.rerank_stats)
        if prefix_repeats==3:assert all('1' in stats['final_counts'] for stats in controller.index.rerank_stats)
    finally:
        controller.trace=None;adapter.uninstall();child.prefix=None;child.close();parent.close()

@pytest.mark.parametrize('fail',[False,True])
def test_rerank_session_releases_controller_without_cyclic_gc(tmp_path,fail):
    import gc,weakref,torch
    from test_attention import tiny
    from research.attention_rerank_live import CandidatePageStore,CandidateRerankIndex
    from research.attention_quest_reference import QuestBoundCache
    from research.reasoning_sessions import CountedSession
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    model=tiny();prefix=ColdStore(tmp_path/'prefix',2,2,16,4)
    a=ModelAdapter.install(model,DenseController(2,prefix));a.forward_tokens([3,4,5,6]*5,0);a.uninstall();prefix.checkpoint({});prefix.close()
    parent=ColdStore.open(tmp_path/'prefix',readonly=True);gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        def one():
            child=CandidatePageStore(tmp_path/'child',2,2,16,4,prefix=parent,staged_tensor_writes=True,cache_device='cpu',cache_pages=8,sink_tokens=4)
            cfg=Config(W=3,C=0,B=4,R=4,k=2,sink_tokens=4,retrieval='centroid')
            controller=QuestBoundCache(cfg,child,Rotary(model.model.rotary_emb.inv_freq.detach().numpy()),'cpu',dtype=torch.float32,initialize=False)
            controller.index=CandidateRerankIndex(2,2,16,4,4,child,pool=8);controller._initialize()
            adapter=ModelAdapter.install(model,controller);session=CountedSession(adapter,model,active_limit=16)
            trace=session.controller.trace
            def observe(layer,pos,q,view):trace(layer,pos,q,view);assert session is not None
            controller.trace=observe
            ref=weakref.ref(controller)
            try:
                session.feed_prompt([7,8,9])
                if fail:raise RuntimeError('injected')
            except RuntimeError:
                if not fail:raise
            finally:
                session.controller.trace=None;adapter.uninstall();child.prefix=None;child.close()
                session=child=adapter=controller=trace=observe=None
            return ref
        ref=one();assert ref() is None
    finally:
        parent.close()
        if enabled:gc.enable()
        gc.collect()


def test_bounded_round_telemetry_preserves_max_events(tmp_path):
    import json
    from research.attention_transfer_run import persist_round
    events=[dict(layer=i%28,position=32000+i//28,pages=list(range(200,215)),candidates=list(range(200,232)),active=4048) for i in range(28*768)]
    target=tmp_path/'round.json';n=persist_round(target,{'status':'partial','selected_events':events})
    assert n==target.stat().st_size<8*2**20 and json.loads(target.read_bytes())['selected_events']==events
    with pytest.raises(ValueError,match='byte limit'):persist_round(tmp_path/'too-big.json',{'x':'a'*100},limit=32)
    assert not (tmp_path/'too-big.json').exists()

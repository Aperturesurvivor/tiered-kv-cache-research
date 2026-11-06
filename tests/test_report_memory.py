"""Synthetic trace integrity tests; no model quality or timing evidence."""
from copy import deepcopy
import json

import pytest

from research.memory_harness import BoundedMemory
from research.memory_protocol import initialize_memory
from research.reasoning_protocol import round_request, next_search, parse_response
from research.persistent_kv import next_question_tokens
from research.budget_probe import expand_blocks
from research.report_memory import declared_policies, expected_schedule, replay_memory, MemoryReplay, summarize, write_report, question_coverage


class Tokenizer:
    def encode(self,text,**kwargs):return [ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=False):
        return ''.join(chr(i) for i in ids if i or not skip_special_tokens)
    def apply_chat_template(self,messages,**kwargs):
        return ''.join('<'+m['role']+'>'+m['content']+('\0' if m['role']=='assistant' else '</user>')
                       for m in messages)+'<assistant>'
    def __call__(self,text,**kwargs):
        return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])


class Index:
    last_search={}
    def search(self,query,vector,k,method):
        self.last_search={'candidates':1};return [1]
    def close(self):pass


def observations(attention='sparse',mode='both',seed='empty'):
    tok=Tokenizer()
    p=next(p for p in declared_policies() if p['attention']==attention and p['memory']==mode and p['memory_seed']==seed)
    ep=dict(id='episode',budget=8192,prefix_tokens_count=6000,first_suffix='<assistant>',
            queries=[dict(kind='test',query='PROJECT_A',answer='42') for _ in range(5)])
    model=dict(revision='revision',config=dict(num_hidden_layers=2,num_attention_heads=4,vocab_size=512,max_position_embeddings=8192))
    manifest=dict(model=model,schedule=[dict(episode_id=ep['id'],policy=p,queries=5)])
    memory=BoundedMemory(tok,mode);initialize_memory(memory,ep,p)
    rows=[];stock=attention=='stock'
    for qi,q in enumerate(ep['queries']):
        before=memory.snapshot();extra=memory.prompt();rounds=[]
        evaluated=prompts=calls=generated=0;previous='';last=None
        def work():
            positions=2*(6000*evaluated+evaluated*(evaluated+1)//2) if stock else 2*4176*evaluated
            return dict(model_token_evaluations=evaluated,prompt_token_evaluations=prompts,
                transformer_forward_calls=calls,layer_summed_valid_KV_positions=positions,
                logical_attention_head_pairs=positions*4,
                max_active_KV_entries=(6000+evaluated if stock else 4176) if evaluated else 0)
        for turn in range(2):
            if turn:evaluated+=1;calls+=1
            work_before=work();query,details=next_search(q['query'],previous,tok)
            selected=[];index_details={}
            if not stock:
                seeds=[1];index_details={'candidates':1}
                if turn==0:
                    remembered=memory.addresses(q['query'],47)
                    index_details['remembered_seed_blocks']=remembered
                    seeds=list(dict.fromkeys([*seeds,*remembered]))[:16]
                selected=list(expand_blocks(tuple(seeds),47,16))
            prompt=round_request(q['query'],turn,2)
            if turn==0 and extra:prompt+='\n'+extra
            ids=tok.encode(prompt+ep['first_suffix']) if turn==0 else next_question_tokens(tok,prompt,last,{0})
            response=json.dumps(dict(answer='wrong' if qi==3 and turn==1 else '42',search='',notes='fallible'))
            if qi==2 and turn==1:response='invalid JSON'
            tokens=tok.encode(response)+[0];assert len(tokens)<=128
            parsed=parse_response(response);answer=parsed['answer'] if parsed else None
            evaluated+=len(ids)+len(tokens)-1;prompts+=len(ids);generated+=len(tokens)
            calls+=(1 if stock else len(ids))+len(tokens)-1
            rounds.append(dict(turn=turn,prompt_token_ids=ids,generated_token_ids=tokens,
                response=response,parsed_response=parsed,answer=answer,exact_correct=answer==q['answer'],
                answer_contains_gold_diagnostic_only=bool(answer and q['answer'] in answer),
                query=query,query_details=details,selected_blocks=selected,index_details=index_details,
                decode_ms=[.1]*(len(tokens)-1),stopped_by_eos=True,lookup_and_selection_s=.01,
                cumulative_first_token_s=turn+0.1,cumulative_request_s=turn+1.,
                cumulative_generated_tokens=generated,work_before_prompt=work_before,cumulative_work=work()))
            previous=response;last=0
        update=memory.observe(q['query'],previous,rounds[-1]['selected_blocks'])
        cfg=None if stock else dict(W=2000,C=0,R=128,k=16,sink_tokens=128,retrieval='fixed',
            full=False,buffered_writes=True,fixed_blocks=[],model_revision='revision',id=p['id'])
        rows.append(dict(status='ok',episode_id=ep['id'],query_index=qi,example_id=ep['id']+f'-memory-q{qi}',
            task=q['kind'],policy=p,trial=0,archive_tokens=6000,gold_answer=q['answer'],rounds=rounds,
            purpose='synthetic_stale_memory_stress' if seed=='synthetic_stale' else 'natural_model_memory',
            comparison='hybrid KV-plus-model-text-memory' if mode in ('notes','both') else 'no text-memory prompt',
            memory_before=before,memory_after=memory.snapshot(),memory_update=update,policy_config=cfg,
            prediction=rounds[-1]['answer'],exact_correct=rounds[-1]['exact_correct'],generated_tokens=generated,
            request_s=2.1,checkpoint_s=.1,request_plus_checkpoint_s=2.2,controller_init_s=.1,
            archive_checksum_mount_s=.1,prefill_s=.1,index_build_or_open_s=.1,**work()))
    return manifest,[ep],rows,tok


@pytest.mark.parametrize('attention,mode,seed',[
    ('stock','notes','empty'),('sparse','none','empty'),('sparse','addresses','empty'),
    ('sparse','both','empty'),('sparse','both','synthetic_stale')])
def test_audit_replays_fallible_model_memory_and_counts_its_prompts(attention,mode,seed):
    args=observations(attention,mode,seed)
    audit,trace,replayed=replay_memory(*args,eos_ids={0},index_factory=lambda ep,root:Index())
    assert audit['status']=='incomplete' and audit['tokenizer_verified']
    assert replayed==(10 if attention=='sparse' else 0)
    assert trace[2]['saved'] is False  # invalid JSON is not repaired
    assert trace[3]['saved_incorrect_final_answer']==(mode!='none')
    assert all(r['whole_required_records'] is None for r in audit['round_record_coverage'])
    assert all(r['prompt_text_tokens']<=256 and r['stored_text_tokens']<=256 for r in trace)
    if mode in ('none','addresses'):assert all(r['prompt_text_tokens']==0 for r in trace)


@pytest.mark.parametrize('mutation',['gold_repair','before_chain','eviction','hidden_prompt',
    'remembered_addresses','duplicate','query_order','best_round','budget'])
def test_rejects_memory_trace_corruption(mutation):
    manifest,episodes,rows,tok=observations()
    if mutation=='gold_repair':rows[3]['memory_after']['entries'][-1]['answer']='42'
    elif mutation=='before_chain':rows[1]['memory_before']=rows[0]['memory_before']
    elif mutation=='eviction':rows[4]['memory_update']['evicted']+=1
    elif mutation=='hidden_prompt':rows[1]['rounds'][0]['prompt_token_ids']+=[65]
    elif mutation=='remembered_addresses':rows[1]['rounds'][0]['index_details']['remembered_seed_blocks']=[25]
    elif mutation=='duplicate':rows.insert(1,deepcopy(rows[0]))
    elif mutation=='query_order':rows[1]['query_index']=3
    elif mutation=='best_round':rows[3]['prediction']='42';rows[3]['exact_correct']=True
    elif mutation=='budget':rows[1]['memory_after']['stored_text_tokens']=999
    with pytest.raises(ValueError):replay_memory(manifest,episodes,rows,tok,eos_ids={0},index_factory=lambda ep,root:Index())


def test_independent_state_replay_bounds_fifo_and_preserves_wrong_notes():
    tok=Tokenizer();memory=MemoryReplay(tok,'both')
    for i in range(20):
        memory.update('PROJECT_'+str(i),json.dumps(dict(answer='wrong'*20,search='s'*100,notes='n'*100)),[1,1,2,3,4,5])
        assert len(memory.entries)<=4 and memory.size()<=256
        assert len(tok.encode(memory.prompt()))<=256 and memory.snapshot()['stored_addresses']<=16
    assert memory.entries[-1]['answer']=='wrongwrongwrongw'
    assert memory.addresses('PROJECT_19',3)==[1,2]
    assert memory.addresses('UNRELATED_PROJECT',100)==[]


def test_schedule_separates_fault_injection_and_natural_episodes():
    episodes=[dict(id=str(i)) for i in range(4)]
    plan=expected_schedule(episodes)
    assert len(plan)==32 and sum(r['queries'] for r in plan)==160
    assert sum(r['queries'] for r in plan if r['policy']['memory_seed']=='empty')==120
    assert sum(r['queries'] for r in plan if r['policy']['memory_seed']=='synthetic_stale')==40


def test_summary_pairs_same_memory_stock_and_keeps_fault_rows_separate(tmp_path):
    episodes=[dict(id=f'{n}-{s}',budget=n) for n in (8192,32768) for s in (157,163)]
    rows=[];records=[]
    for ep in episodes:
        for p in declared_policies():
            for qi in range(5):
                rows.append(dict(episode_id=ep['id'],query_index=qi,policy=p,task='test',
                    exact_correct=qi==0 and p['memory']=='both',request_s=2.,generated_tokens=40,
                    model_token_evaluations=50,cuda_peak_allocated_bytes=2**30,
                    host_rss_sampled_peak_bytes=2**30,max_active_KV_entries=4177,prediction='test'))
                records.append(dict(stored_text_tokens=100,prompt_text_tokens=120,stored_addresses=4))
    result=summarize(episodes,rows,records)
    assert len(result['quality'])==16 and len(result['episode_totals'])==32
    natural=next(r for r in result['quality'] if r['budget']==8192 and r['policy']=='sparse-memory-both-r2-cap128')
    assert natural['matching_stock']=='stock-memory-notes-r2-cap128'
    assert natural['gained_vs_matching_stock']==natural['gained_vs_bare']==2
    fault=next(r for r in result['quality'] if r['budget']==8192 and r['policy']=='sparse-memory-both-stale-r2-cap128')
    assert fault['matching_stock']=='stock-memory-notes-stale-r2-cap128'
    assert all(r['paired_episode_speed_ratio']==1 for r in result['episode_totals'])
    manifest=dict(model=dict(model_id='test-only',revision='synthetic'))
    with pytest.raises(ValueError):write_report(tmp_path,manifest,dict(status='incomplete'))
    write_report(tmp_path,manifest,dict(status='complete',limits=['Synthetic unit test'],**result))
    assert 'Synthetic stale-note stress' in (tmp_path/'report.md').read_text()


def test_coverage_uses_each_question_not_the_original_linked_fact_spans():
    tok=Tokenizer();records=['The designated station for PROJECT_A is STATION_B.',
                           'The access code for STATION_B is 12345678.']
    cabinet='The access code for CABINET_000010 is 87654321.'
    text='\n'.join([*records,cabinet]);tokens=tok.encode(text)
    spans=[[text.index(r),text.index(r)+len(r)] for r in records]
    kinds=['bridge_learning','new_entity','linked_reuse','paraphrased_reuse','bridge_reuse']
    ep=dict(id='coverage',records=records,record_spans=spans,tokens=tokens,prefix_tokens_count=len(tokens),
        queries=[dict(kind=k,query='What is the code for CABINET_000010?',answer='87654321') for k in kinds])
    coverage=question_coverage([ep],tok)
    assert coverage['coverage-memory-q0']==coverage['coverage-memory-q4']==[spans[0]]
    assert coverage['coverage-memory-q2']==coverage['coverage-memory-q3']==spans
    assert coverage['coverage-memory-q1']==[[text.index(cabinet),len(text)]]
    ep['record_spans'][0][0]+=1
    with pytest.raises(ValueError,match='span metadata'):question_coverage([ep],tok)

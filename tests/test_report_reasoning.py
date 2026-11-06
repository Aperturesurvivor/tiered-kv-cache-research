"""Reject false completeness, hidden replay cost and altered saved answers."""
from copy import deepcopy
import json

import pytest

from research.report_reasoning import audit_rows, check_work, expected_schedule, summarize


def observations(attention='stock'):
    policy=dict(id=f'{attention}-json-r2-cap128',attention=attention,rounds=2,cap_per_round=128,
                format='json',schedule_labels=['128_per_round'])
    ex=dict(id='tiny',query='Public question',answer='42',task='literal',budget=8192,
            seed=157,prefix_tokens_count=6000,record_spans=[[500,502]],required_record_indices=[0])
    model=dict(revision='revision',config=dict(num_hidden_layers=2,num_attention_heads=4,vocab_size=100,max_position_embeddings=8192))
    plan=dict(example_id='tiny',trial=0,policy=policy,purpose='quality')
    manifest=dict(model=model,schedule=[plan])
    def work(evaluated,prompts,calls):
        positions=2*(6000*evaluated+evaluated*(evaluated+1)//2) if attention=='stock' else 40*evaluated
        return dict(model_token_evaluations=evaluated,prompt_token_evaluations=prompts,
                    transformer_forward_calls=calls,layer_summed_valid_KV_positions=positions,
                    logical_attention_head_pairs=positions*4,
                    max_active_KV_entries=(6000+evaluated if attention=='stock' else 20) if evaluated else 0)
    rounds=[]
    for i,(before,after,prompts) in enumerate([(0,3,2),(4,7,4)]):
        parsed=dict(answer='42' if i==0 else '17',search='',notes='A model draft')
        rounds.append(dict(turn=i,generated_token_ids=[11,0],prompt_token_ids=[21,22],
            decode_ms=[1.],response=json.dumps(parsed),parsed_response=parsed,answer=parsed['answer'],
            exact_correct=i==0,answer_contains_gold_diagnostic_only=i==0,
            selected_blocks=[] if attention=='stock' else [0],query='Public question',index_details={},
            work_before_prompt=work(before,prompts-2,(0 if i==0 else 3) if attention=='stock' else before),
            cumulative_work=work(after,prompts,(2 if i==0 else 5) if attention=='stock' else after),
            cumulative_generated_tokens=(i+1)*2,cumulative_first_token_s=i+.3,
            cumulative_request_s=i+1.,lookup_and_selection_s=.1,stopped_by_eos=True))
    cfg=None if attention=='stock' else dict(W=2000,C=0,R=128,k=16,sink_tokens=128,retrieval='fixed',
        full=False,buffered_writes=True,fixed_blocks=[],model_revision='revision',id=policy['id'])
    row={**plan,'status':'ok','task':'literal','archive_tokens':6000,'gold_answer':'42','rounds':rounds,
         'policy_config':cfg,'generated_tokens':4,'prediction':'17','exact_correct':False,
         'request_s':2.1,'checkpoint_s':.2,'request_plus_checkpoint_s':2.3,'controller_init_s':.5,
         'archive_checksum_mount_s':.5,'prefill_s':1.,'index_build_or_open_s':.2,
         **work(7,4,5 if attention=='stock' else 7)}
    return manifest,[ex],[row]


@pytest.mark.parametrize('attention',['stock','sparse'])
def test_pending_token_and_prompt_work_are_audited_without_best_draft_selection(attention):
    args=observations(attention)
    audit=audit_rows(*args,eos_ids={0})
    assert audit['status']=='incomplete' and audit['observed_rows']==1
    assert audit['round_record_coverage'][0]['exact_correct']
    assert not audit['round_record_coverage'][1]['exact_correct']
    with pytest.raises(ValueError,match='completed token-verified'):
        summarize(*args,audit)


@pytest.mark.parametrize('mutation',['duplicate','pending','prompt','heads','best_draft','cap','early_stop','eos','timing','coverage_address','completion'])
def test_rejects_invalid_evidence(mutation):
    manifest,fixtures,rows=observations('sparse')
    row=rows[0];marker=None
    if mutation=='duplicate':rows.append(deepcopy(row))
    elif mutation=='pending':row['rounds'][1]['work_before_prompt']['model_token_evaluations']-=1
    elif mutation=='prompt':row['prompt_token_evaluations']-=1
    elif mutation=='heads':row['logical_attention_head_pairs']-=1
    elif mutation=='best_draft':row.update(prediction='42',exact_correct=True)
    elif mutation=='cap':row['rounds'][0]['generated_token_ids']=[11]*129
    elif mutation=='early_stop':
        row['rounds'][0]['generated_token_ids'][-1]=12;row['rounds'][0]['stopped_by_eos']=False
    elif mutation=='eos':row['rounds'][0]['generated_token_ids'][0]=0
    elif mutation=='timing':row['rounds'][1]['cumulative_request_s']=.5
    elif mutation=='coverage_address':row['rounds'][0]['selected_blocks']=[999]
    elif mutation=='completion':marker=dict(rows=220,quality_rows=176,repeated_timing_rows=44,weights_unchanged=True)
    with pytest.raises(ValueError):audit_rows(manifest,fixtures,rows,marker,eos_ids={0})


def test_work_counters_distinguish_model_tokens_from_generated_tokens():
    _,_,rows=observations()
    row=rows[0]
    assert row['generated_tokens']==4 and row['model_token_evaluations']==7
    wrong=deepcopy(row);wrong['layer_summed_valid_KV_positions']=0
    with pytest.raises(ValueError):check_work(wrong,True,6000,7,4,5,2,4)


def test_schedule_separates_quality_and_repeated_timing():
    fixtures=[dict(id=f'{n}-{s}-{t}',task=t,seed=s) for n in (8192,32768)
              for s in (157,163) for t in ('literal','paraphrase','linked_fact','sum_two_facts')]
    schedule=expected_schedule(fixtures)
    assert len(schedule)==220
    assert sum(r['purpose']=='quality' for r in schedule)==176
    assert sum(r['purpose']=='repeated_timing' for r in schedule)==44


def test_summary_uses_paired_controls_and_latest_affordable_draft():
    fixtures=[dict(id=f'reasoning-{task}-{budget}-{seed}',task=task,seed=seed,budget=budget)
              for budget in (8192,32768) for seed in (157,163)
              for task in ('literal','paraphrase','linked_fact','sum_two_facts')]
    rows=[]
    for planned in expected_schedule(fixtures):
        p=planned['policy'];sparse=p['attention']=='sparse'
        rows.append({**planned,'exact_correct':not sparse,
            'request_s':5. if sparse else .5 if p['format']=='direct' else 3.,
            'archive_checksum_mount_s':10. if sparse else 1.,'controller_init_s':1.,
            'checkpoint_s':1.,'prefill_s':20.,'index_build_or_open_s':2.,
            'generated_tokens':4 if sparse else 3,'model_token_evaluations':10,
            'logical_attention_head_pairs':100 if sparse else 1000,
            'max_active_KV_entries':100,'cuda_peak_allocated_bytes':2**30,
            'host_rss_sampled_peak_bytes':2**30,
            'rounds':[dict(turn=i,cumulative_request_s=i+1.,cumulative_generated_tokens=i+1,
                           exact_correct=i==0) for i in range(p['rounds'])]})
    result=summarize({},fixtures,rows,dict(status='complete',tokenizer_verified=True))
    target=lambda r:r['example_id']=='reasoning-linked_fact-8192-157' and r['policy']=='sparse-json-r4-cap128'
    pair=next(r for r in result['paired_cases'] if target(r))
    assert pair['matching_stock_speed_ratio']==.6 and pair['direct_stock_speed_ratio']==.1
    assert pair['matching_stock_accounted_service_ratio']==6./17.
    assert pair['direct_stock_accounted_service_ratio']==3.5/17.
    quality=next(r for r in result['quality'] if r['budget']==8192 and r['policy']=='sparse-json-r4-cap128')
    assert quality['mean_accounted_service_s']==17.
    assert quality['mean_shared_prefill_s']==20. and quality['mean_shared_index_build_or_open_s']==2.
    curves=[r for r in result['retrospective_budgets'] if target(r)]
    same=[r for r in curves if r['control']=='matching_stock']
    assert len(same)==2 and all(r['completed_round']==3 and not r['exact_correct'] for r in same)
    direct_time=next(r for r in curves if r['control']=='direct_stock' and r['budget_metric']=='elapsed_time')
    assert direct_time['completed_round'] is None and not direct_time['exact_correct']

import pytest
from research.attention_smol_format_protocol import SEEDS,TASKS,POLICY,round_request,score_response


def test_short_prompt_and_declared_policy():
    assert SEEDS==(293,307) and TASKS==('literal','paraphrase','linked_fact')
    assert POLICY['candidate_ceiling']==32 and POLICY['final_ceiling']==15 and POLICY['rounds']==2 and POLICY['cap_per_round']==32
    assert round_request('Q',0)=='Q\nRound 1 of 2. Give only the eight-digit answer; no explanation. Another round will follow.'
    assert round_request('Q',1)=='Q\nRound 2 of 2. Review your preceding answer. Give only the eight-digit answer; no explanation.'
    with pytest.raises(ValueError):round_request('Q',2)


def test_distinct_standalone_code_scoring():
    answer='12345678'
    assert score_response('12345678 then 12345678',answer)=={'distinct_codes':[answer],'valid':True,'answer':answer,'exact_correct':True}
    assert not score_response('12345678 or 87654321',answer)['valid']
    assert not score_response('none',answer)['valid']
    assert not score_response('9123456780',answer)['valid']
    assert not score_response('87654321',answer)['exact_correct']


def test_frozen_fixture_orientation_and_native_limit():
    import json
    from pathlib import Path
    from research.attention_smol_format_protocol import make_fixture
    from research.reasoning_protocol import _task
    plan=json.loads(Path('results/evidence/attention-smol-format-preparation/smol-plan.json').read_text())
    assert len(plan['fixtures'])==6 and len(plan['schedule'])==12
    for ex in plan['fixtures']:
        records,query,answer,required=_task(ex['seed'],ex['task'])
        if ex['seed']==307:records.reverse();required=[1-i for i in required]
        assert ex['records']==records and ex['query']==query and ex['answer']==answer and ex['required_record_indices']==required
        assert ex['budget']==8192 and ex['prefix_tokens_count']<6144 and all(128<=a<b<ex['prefix_tokens_count']-2000 for a,b in ex['record_spans'])
    with pytest.raises(ValueError):make_fixture(None,32768,293,'literal')


def test_auditor_rejects_unbound_deployment_and_bad_routes():
    from research import report_smol_format as audit
    with pytest.raises(ValueError):audit.controller_provenance({'controller_sha256':'unreviewed','bundle_sha256':'unknown'},required=True,label='smol17')
    good=[{'position':2304,'layer':0,'candidates':[1],'pages':[1],'active':2256}]
    assert audit.audit_events(good,2304,1,1)['candidate_final_counts']=={'1/1':1}
    bad=[{**good[0],'pages':[2]}]
    with pytest.raises(ValueError):audit.audit_events(bad,2304,1,1)


def test_partial_auditor_replays_rounds_and_rejects_rescored_tamper(tmp_path,monkeypatch):
    import json
    from research import report_smol_format as audit
    class Tok:
        def encode(self,*args,**kwargs):return [1]
        def decode(self,ids,**kwargs):return '12345678'
    model='test/model';ex={'id':'fixture','query':'q','first_suffix':'','answer':'12345678','prefix_tokens_count':3000,'budget':8192,'task':'literal','seed':293}
    plan={'model':{'model_id':model,'config':{'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'hidden_size':8,'vocab_size':10,'max_position_embeddings':8192}},'weights_sha256':'w','source_identity':{},'fixtures':[ex],'schedule':[{'fixture_id':'fixture','arm':'stock'}]}
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps(plan));ph=audit.sha(planfile)
    monkeypatch.setattr(audit,'PLAN_HASHES',{'smol17':ph});monkeypatch.setattr(audit,'MODELS',{'smol17':model});monkeypatch.setattr(audit,'WEIGHTS',{model:'w'})
    monkeypatch.setattr(audit,'source_identity',lambda:{});monkeypatch.setattr(audit,'make_plan',lambda _:plan);monkeypatch.setattr(audit,'next_question_tokens',lambda *a:[1])
    (tmp_path/'validation.json').write_text(json.dumps({'weights_sha256':'w','plan_sha256':ph,'parity':[{'length':n,'greedy_agreement':[True]*17,'max_errors':[0]*17} for n in (64,512,1025)]}))
    (tmp_path/'parity-fixture.json').write_text(json.dumps({'fixture_id':'fixture','status':'passed','all_kv_exact':True,'weights_sha256':'w','plan_sha256':ph,'prompt_max_abs':0}))
    rounds=[];hashes={}
    for turn in range(2):
        parsed=score_response('12345678','12345678')
        rr={'turn':turn,'prompt_token_ids':[1],'generated_token_ids':[9],'response':'12345678','parsed_response':parsed,'query':'q','query_details':{'source':'original_question','uses_gold':False},'answer':'12345678','exact_correct':True,'prompt_s':1,'generation_s':1,'decode_ms':[]}
        rounds.append(rr);f=tmp_path/f'fixture-stock-round-{turn}.json';f.write_text(json.dumps({'status':'partial','fixture_id':'fixture','arm':'stock','round':rr,'selected_events':[]}));hashes[str(turn)]=audit.sha(f)
    row={'status':'partial','fixture_id':'fixture','arm':'stock','rounds':rounds,'round_file_sha256':hashes,'prefill_s':1,'mount_s':1,'init_s':1,'request_s':1,'checkpoint_s':1,'logical_read_bytes':0,'peak_allocated':1,'peak_reserved':1,'host':{'host_observation_scope':'sampled','host_rss_samples':1},'metrics':{'model_token_evaluations':3,'prompt_token_evaluations':2,'transformer_forward_calls':3,'layer_summed_valid_KV_positions':18012,'logical_attention_head_pairs':36024,'max_active_KV_entries':3003},'candidate_stats':None,'page_cache_stats':None,'normalizer_stats':None}
    samples=tmp_path/'samples.jsonl';samples.write_text(json.dumps(row)+'\n')
    assert audit.audit_model('smol17',planfile,tmp_path,Tok(),{9},True)['rows']==1
    row['rounds'][1]['exact_correct']=False;f=tmp_path/'fixture-stock-round-1.json';saved=json.loads(f.read_text());saved['round']=row['rounds'][1];f.write_text(json.dumps(saved));row['round_file_sha256']['1']=audit.sha(f);samples.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='scoring mismatch'):audit.audit_model('smol17',planfile,tmp_path,Tok(),{9},True)


def test_final_round_pairs_required_without_partial_aggregation():
    from research.report_smol_format import paired_final
    records=[{'fixture_id':str(i),'task':'literal','seed':293,'turn':turn,'arm':arm,'correct':bool(i%2)} for i in range(6) for turn in (0,1) for arm in ('stock','reranker')]
    assert len(paired_final({'records':records}))==6
    with pytest.raises(ValueError):paired_final({'records':records[:-1]})


def test_auditor_binds_observed_predecessor_and_imports():
    import copy,importlib.util,json
    from pathlib import Path
    from unittest.mock import patch
    from research import report_smol_format as audit
    p=Path('results/evidence/attention-smol-format-preparation');spec=importlib.util.spec_from_file_location('format_bound_controller',p/'bound-supervisor.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.PREDECESSOR_STATE_PATH=p/'observer-state.json'
    with patch.object(m.psutil,'pid_exists',return_value=False):
        predecessor=json.loads(json.dumps(m.predecessor_terminal(p/'observer-terminal-observation.json')))
    from research.report_attention_transfer import FRESH_PREDECESSORS
    prior={**FRESH_PREDECESSORS,'/redacted/runtime/tiered-kv-results/transfer-smol17-a712857-fresh-1/state.json':'b888bf57128045e51022cd44243f57835e021a2b4dcb961684f6ee4ae41690d6'}
    state={'prior_receipts':prior,'controller_sha256':audit.FORMAT_CONTROLLER_SHA,'bundle_sha256':audit.FORMAT_BUNDLE_SHA,'backend_boot_id':m.NEW_BOOT,'predecessor':predecessor,'extra_helper_sha256':m.MISSING_SHA,'import_smoke':{'status':'passed','modules':['research.attention_smol_format_run','research.attention_fullkey_oracle']}}
    assert audit.controller_provenance(state,True,'smol17')['predecessor']==predecessor
    for key,value in [('predecessor',{}),('prior_receipts',{}),('backend_boot_id','other'),('extra_helper_sha256','bad'),('import_smoke',{})]:
        bad=copy.deepcopy(state);bad[key]=value
        with pytest.raises(ValueError):audit.controller_provenance(bad,True,'smol17')

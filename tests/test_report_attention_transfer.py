import pytest
from research.report_attention_transfer import audit_events


def event(layer,pos,candidates,pages):
    return dict(layer=layer,position=pos,candidates=candidates,pages=pages,active=2128+128*len(pages))


def test_route_audit_counts_underfill_and_causality():
    events=[event(layer,pos,[1,2],[1,2]) for pos in (2384,2385) for layer in range(2)]
    result=audit_events(events,2384,2,2)
    assert result['candidate_final_counts']=={'2/2':4} and result['active_counts']=={2384:4}
    broken=[dict(x) for x in events];broken[0]['candidates']=[2,30]
    with pytest.raises(ValueError,match='subset|ineligible'):audit_events(broken,2384,2,2)
    broken=[dict(x) for x in events];broken[0]['pages']=[5]
    with pytest.raises(ValueError,match='subset'):audit_events(broken,2384,2,2)
    broken=[dict(x) for x in events];broken[0]['position']+=1
    with pytest.raises(ValueError,match='reordered'):audit_events(broken,2384,2,2)


def test_partial_and_complete_token_replay_and_tamper(tmp_path,monkeypatch):
    import json
    import research.report_attention_transfer as audit
    class Tok:
        def encode(self,*a,**k):return [1]
        def decode(self,ids,**k):return 'x'
    model='test/model';ex={'id':'fixture','query':'q','first_suffix':'','answer':'secret','prefix_tokens_count':3000,'budget':8192,'task':'literal','seed':281}
    plan={'model':{'model_id':model,'config':{'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'hidden_size':8,'vocab_size':10,'max_position_embeddings':8192}},'weights_sha256':'w','source_identity':{},'fixtures':[ex],'schedule':[{'fixture_id':'fixture','arm':'stock'}]}
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps(plan));ph=audit.sha(planfile)
    monkeypatch.setattr(audit,'PLAN_HASHES',{'qwen15':ph});monkeypatch.setattr(audit,'MODELS',{'qwen15':model});monkeypatch.setattr(audit,'WEIGHTS',{model:'w'})
    monkeypatch.setattr(audit,'source_identity',lambda:{}) ;monkeypatch.setattr(audit,'make_plan',lambda _:plan)
    monkeypatch.setattr(audit,'next_question_tokens',lambda *a:[1]);monkeypatch.setattr(audit,'next_search',lambda *a:('q',{}))
    (tmp_path/'validation.json').write_text(json.dumps({'weights_sha256':'w','plan_sha256':ph,'parity':[{'length':n,'greedy_agreement':[True]*17,'max_errors':[0]*17} for n in (64,512,1025)]}))
    parity=tmp_path/'parity-fixture.json';parity.write_text(json.dumps({'fixture_id':'fixture','status':'passed','all_kv_exact':True,'weights_sha256':'w','plan_sha256':ph,'prompt_max_abs':0}))
    rounds=[];hashes={}
    for t in range(2):
        rr={'turn':t,'prompt_token_ids':[1],'generated_token_ids':[9],'response':'x','parsed_response':None,'query':'q','query_details':{},'answer':None,'exact_correct':False,'prompt_s':1,'generation_s':1,'decode_ms':[]}
        rounds.append(rr);f=tmp_path/f'fixture-stock-round-{t}.json';f.write_text(json.dumps({'status':'partial','fixture_id':'fixture','arm':'stock','round':rr,'selected_events':[]}));hashes[str(t)]=audit.sha(f)
    row={'status':'partial','fixture_id':'fixture','arm':'stock','rounds':rounds,'round_file_sha256':hashes,'prefill_s':1,'mount_s':1,'init_s':1,'request_s':1,'checkpoint_s':1,'logical_read_bytes':0,'peak_allocated':1,'peak_reserved':1,'host':{'host_observation_scope':'sampled','host_rss_samples':1},'metrics':{'model_token_evaluations':3,'prompt_token_evaluations':2,'transformer_forward_calls':3,'layer_summed_valid_KV_positions':18012,'logical_attention_head_pairs':36024,'max_active_KV_entries':3003},'candidate_stats':None,'page_cache_stats':None,'normalizer_stats':None}
    samples=tmp_path/'samples.jsonl';samples.write_text(json.dumps(row)+'\n')
    assert audit.audit_model('qwen15',planfile,tmp_path,Tok(),{9},True)['status']=='partial'
    with pytest.raises(ValueError,match='no verified controller completion'):audit.audit_model('qwen15',planfile,tmp_path,Tok(),{9},False)
    integrity=tmp_path/'integrity.json';integrity.write_text(json.dumps({'plan_sha256':ph,'samples_sha256':audit.sha(samples),'weights_sha256_before':'w','weights_sha256_after':'w','source_identity':{},'fixture_parity_sha256':{'fixture':audit.sha(parity)}}))
    (tmp_path/'complete.json').write_text(json.dumps({'status':'complete','rows':1,'weights_unchanged':True,'plan_sha256':ph,'integrity_sha256':audit.sha(integrity),'samples_sha256':audit.sha(samples)}))
    state={'status':'complete','worker_returncode':0,'plan_sha256':ph,'model':model,'bundle_sha256':audit.BUNDLE_SHA,'controller_sha256':audit.CONTROLLER_SHA,'result':{'rows':1,'files_sha256':{p.name:audit.sha(p) for p in tmp_path.iterdir() if p.is_file()}}}
    assert audit.audit_model('qwen15',planfile,tmp_path,Tok(),{9},controller=state)['status']=='complete'
    repaired={**state,'bundle_sha256':audit.REPAIRED_BUNDLE_SHA,'controller_sha256':audit.REPAIRED_CONTROLLER_SHA,
              'extra_helper_sha256':audit.REPAIRED_HELPER_SHA,'import_smoke':{'status':'passed','modules':['research.attention_fullkey_oracle']}}
    complete=audit.audit_model('qwen15',planfile,tmp_path,Tok(),{9},controller=repaired)
    assert complete['status']=='complete' and complete['controller_provenance']['package']=='repaired_70_member_export'
    assert audit.audit_model('qwen15',planfile,tmp_path,Tok(),{9},partial=True,controller={'status':'error'})['status']=='partial'
    (tmp_path/'fixture-stock-round-0.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):audit.audit_model('qwen15',planfile,tmp_path,Tok(),{9},controller=state)
    # A complete per-layer sparse trace, then tampering with a recomputed hash.
    (tmp_path/'complete.json').unlink();(tmp_path/'integrity.json').unlink()
    plan['schedule']=[{'fixture_id':'fixture','arm':'reranker'}];planfile.write_text(json.dumps(plan));ph=audit.sha(planfile)
    monkeypatch.setattr(audit,'PLAN_HASHES',{'qwen15':ph})
    (tmp_path/'validation.json').write_text(json.dumps({'weights_sha256':'w','plan_sha256':ph,'parity':[{'length':n,'greedy_agreement':[True]*17,'max_errors':[0]*17} for n in (64,512,1025)]}))
    parity.write_text(json.dumps({'fixture_id':'fixture','status':'passed','all_kv_exact':True,'weights_sha256':'w','plan_sha256':ph,'prompt_max_abs':0}))
    row.update(arm='reranker',candidate_stats=[{'calls':3,'candidate_counts':{'6':3},'final_counts':{'6':3}} for _ in range(2)],
               page_cache_stats=[{'calls':3,'peak_resident_pages':6,'cache_hits':12,'cache_misses':6} for _ in range(2)],normalizer_stats=[{},{}],persistent_cache_bound_bytes=131072)
    row['metrics'].update(transformer_forward_calls=3,layer_summed_valid_KV_positions=17376,logical_attention_head_pairs=34752,max_active_KV_entries=2896)
    for turn in range(2):
        positions=[3000] if turn==0 else [3001,3002]
        events=[event(layer,pos,list(range(1,7)),list(range(1,7))) for pos in positions for layer in range(2)]
        f=tmp_path/f'fixture-reranker-round-{turn}.json';f.write_text(json.dumps({'status':'partial','fixture_id':'fixture','arm':'reranker','round':rounds[turn],'selected_events':events}));row['round_file_sha256'][str(turn)]=audit.sha(f)
    samples.write_text(json.dumps(row)+'\n')
    assert audit.audit_model('qwen15',planfile,tmp_path,Tok(),{9},True)['rows']==1
    f=tmp_path/'fixture-reranker-round-0.json';saved=json.loads(f.read_text());saved['selected_events'][0]['candidates']=[1,2,3,4,5,30];f.write_text(json.dumps(saved));row['round_file_sha256']['0']=audit.sha(f);samples.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='subset|ineligible'):audit.audit_model('qwen15',planfile,tmp_path,Tok(),{9},True)


def test_partial_main_accepts_stage_not_started(tmp_path,monkeypatch):
    import json,sys
    import research.report_attention_transfer as audit
    snap=tmp_path/'snapshot';snap.mkdir();(snap/'generation_config.json').write_text('{"eos_token_id":9}')
    q=tmp_path/'q';q.mkdir();qstate=tmp_path/'q-state.json';qstate.write_text('{"status":"running"}')
    monkeypatch.setattr(audit,'tokenizer_and_snapshot',lambda model:(object(),snap))
    def fake(label,plan,output,tok,eos,partial,controller):
        assert partial
        return {'model':label,'status':'partial','rows':0,'expected_rows':1,'records':[]}
    monkeypatch.setattr(audit,'audit_model',fake)
    report=tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['audit','--qwen-output',str(q),'--smol-output',str(tmp_path/'absent'), '--qwen-state',str(qstate),'--smol-state',str(tmp_path/'absent-state'),'--partial','--output',str(report)])
    audit.main();doc=json.loads(report.read_text())
    assert doc['status']=='partial' and doc['paired_groups']==[] and doc['models'][1]['controller_status']=='not_started'


def test_repaired_controller_provenance_rejects_mixed_or_missing_evidence():
    from research import report_attention_transfer as audit
    original={'bundle_sha256':audit.BUNDLE_SHA,'controller_sha256':audit.CONTROLLER_SHA}
    assert audit.controller_provenance(original,required=True)['package']=='original'
    repaired={'bundle_sha256':audit.REPAIRED_BUNDLE_SHA,'controller_sha256':audit.REPAIRED_CONTROLLER_SHA,
              'extra_helper_sha256':audit.REPAIRED_HELPER_SHA,
              'import_smoke':{'status':'passed','modules':['research.attention_fullkey_oracle']}}
    result=audit.controller_provenance(repaired,required=True)
    assert result['package']=='repaired_70_member_export' and result['prior_failed_state_sha256']==audit.PRIOR_FAILURE_STATE_SHA
    for bad in ({**repaired,'controller_sha256':audit.CONTROLLER_SHA},
                {**repaired,'extra_helper_sha256':'bad'},
                {**repaired,'import_smoke':{'status':'passed','modules':[]}}):
        with pytest.raises(ValueError):audit.controller_provenance(bad,required=True)


def test_fresh_smol_controller_requires_exact_bindings():
    import copy
    from research import report_attention_transfer as audit
    fresh={'bundle_sha256':audit.REPAIRED_BUNDLE_SHA,'controller_sha256':audit.FRESH_SMOL_CONTROLLER_SHA,
           'model':audit.MODELS['smol17'],'backend_boot_id':audit.FRESH_BOOT,
           'extra_helper_sha256':audit.REPAIRED_HELPER_SHA,'prior_receipts':copy.deepcopy(audit.FRESH_PREDECESSORS),
           'cost_terminal':copy.deepcopy(audit.FRESH_COST_TERMINAL),
           'import_smoke':{'status':'passed','modules':['research.attention_fullkey_oracle']}}
    assert audit.controller_provenance(fresh,required=True,label='smol17')['package']=='fresh_smol_70_member_export'
    with pytest.raises(ValueError):audit.controller_provenance(fresh,required=True,label='qwen15')
    for key,value in [('bundle_sha256','bad'),('controller_sha256','bad'),('model',audit.MODELS['qwen15']),('backend_boot_id','bad'),('extra_helper_sha256','bad'),('prior_receipts',{}),('cost_terminal',{}),('import_smoke',{'status':'passed','modules':[]})]:
        bad=copy.deepcopy(fresh);bad[key]=value
        with pytest.raises(ValueError):audit.controller_provenance(bad,required=True,label='smol17')
    for key in audit.FRESH_COST_TERMINAL:
        bad=copy.deepcopy(fresh);bad['cost_terminal'][key]='wrong'
        with pytest.raises(ValueError):audit.controller_provenance(bad,required=True,label='smol17')

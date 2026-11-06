import json
import numpy as np
import pytest


def test_strict_em_whole_number_and_rouge():
    from tiered_kv.evals import score_needle, score_rouge
    assert score_needle(' 12345678\n','12345678')=={'em':100.,'code_found':100.}
    assert score_needle('Code: 12345678','12345678')=={'em':0.,'code_found':100.}
    assert score_needle('9123456789','12345678')=={'em':0.,'code_found':0.}
    assert score_rouge('', ['a b c'])==(0.,None)
    assert score_rouge('a b c',['different words','a b c'])[0]>99.99


def test_fixed_needle_token_spans_and_continuation():
    from transformers import AutoTokenizer
    from tiered_kv.config import MODEL_PATH
    from tiered_kv.evals import make_needle, make_latency
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,trust_remote_code=False)
    for depth in [.1,.5,.9]:
        ex=make_needle(tok,8192,depth,11)
        assert len(ex['tokens'])==8064
        decoded=tok.decode(ex['tokens'])
        assert decoded.count(ex['answer'])==1
        assert decoded.count(ex['item'])==2
        a,b=ex['answer_span']
        assert ex['answer'] in tok.decode(ex['tokens'][a:b])
        assert set(ex['gold_blocks'])==set(range(a//128,(b-1)//128+1))
        assert abs(ex['actual_depth']-depth)<.02
    lat=make_latency(tok,8192,0)
    assert len(lat['tokens'])==8048 and len(lat['continuation'])==144
    assert lat==make_latency(tok,8192,0)


def test_many_competing_codes_keep_target_unique_and_query_out_of_archive():
    from transformers import AutoTokenizer
    from tiered_kv.config import MODEL_PATH
    from tiered_kv.evals import make_needle,char_span
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    ex=make_needle(tok,4096,.5,37,distractors=True)
    text=tok.decode(ex['tokens'],skip_special_tokens=False)
    assert text.count('The access code for ITEM_')>20
    assert text.count(ex['answer'])==1 and text.count(ex['item'])==2
    question=f"What is the access code for {ex['item']}? Return only the code."
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ex['tokens']
    start=text.rindex(question)
    boundary=char_span(encoded['offset_mapping'],start,start+len(question))[0]
    assert 'What is the access code' not in tok.decode(ex['tokens'][:boundary])
    assert ex['answer_span'][1]<boundary


def test_report_incomplete_unranked_and_equal_score_weights(tmp_path):
    from tiered_kv.report import rank_rows
    rows=[]
    for name,values,lat,complete in [('full',[100,100,20,20],20,True),('default',[100,100,19,19],5,True),('w1',[100,100,100,100],1,False)]:
        rows.append(dict(config_id=name,status='complete' if complete else 'incomplete',niah_em_8k=values[0],niah_em_32k=values[1],rougeL_low=values[2],rougeL_high=values[3],median_ms_32k=lat,active_KV_bytes=100))
    ranked=rank_rows(rows)
    assert [r['config_id'] for r in ranked]==['full','default','w1']
    assert ranked[0]['quality_score']==60 and ranked[2]['rank']==''
    assert ranked[1]['practical_candidate'] is True


def test_hit_attribution_requires_answer_tokens_and_excludes_recent():
    import torch
    from tiered_kv.evals import HitTracker
    from tiered_kv.types import KVView
    ex={'answer_span':[127,130],'gold_blocks':[0,1],'tokens':list(range(3000))}
    tracker=HitTracker(ex,W=10,k=1)
    def view(ids,tiers):
        x=torch.zeros(1,2,len(ids),4)
        return KVView(np.array(ids),x,x,np.array([[i,i] for i in ids]),tiers,{},25,[0])
    tracker(0,2999,None,view([3,4],['retrieved']*2))
    tracker(1,2999,None,view([127,128,129],['retrieved']*3))
    result=tracker.result()
    assert result['query_hit']==.5 and result['query_opportunities']==2
    assert result['full_span_coverage']==.5 and result['gold_fraction']==.5
    recent=HitTracker({'answer_span':[2997,2999],'tokens':list(range(3000)),'gold_blocks':[23]},W=10,k=1)
    recent(0,2999,None,view([2997,2998],['recent']*2))
    assert recent.result()['query_hit'] is None and recent.result()['recent_excluded']==1


def test_guard_pressure_three_samples_and_swap_abort():
    from tiered_kv.telemetry import evaluate_limits, ResourceAbort
    baseline={'swap_used':0}
    sample={'swap_used':0,'rss':1,'disk_free':10**10,'pressure':2}
    assert evaluate_limits(sample,baseline,0)==1
    assert evaluate_limits(sample,baseline,1)==2
    with pytest.raises(ResourceAbort,match='pressure'):
        evaluate_limits(sample,baseline,2)
    sample['pressure']=1
    sample['swap_used']=512*1024**2+1
    with pytest.raises(ResourceAbort,match='swap'):
        evaluate_limits(sample,baseline,0)


def test_report_cuda_policy_peaks_are_not_cumulative_run_peaks(tmp_path,monkeypatch):
    from dataclasses import asdict
    from tiered_kv.config import Config
    from tiered_kv import report
    configs=[Config(id='full',full=True),Config()]
    examples=[{'id':f'{budget}-{trial}','task':'latency','budget':budget}
              for budget in (8192,32768) for trial in range(3)]
    manifest={'run_id':'memory-regression','profile':'pilot',
              'configs':[asdict(c) for c in configs],'examples':examples}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    samples=[]
    for cfg in configs:
        for ex in examples:
            samples.append({'config_id':cfg.id,'config_hash':cfg.digest,
                'example_id':ex['id'],'task':'latency','budget':ex['budget'],
                'trial':int(ex['id'].split('-')[1]),'status':'ok',
                'prompt_length':ex['budget']-144,'latency_ms':[10.,11.],
                'peak_mps_allocated':999999,'swap_delta':0,'pressure_max':0,
                'cuda_decode_peak_allocated_bytes':ex['budget'] if cfg.full else 2048})
    (tmp_path/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    monkeypatch.setattr(report,'_plots',lambda *args:None)
    rows={r['config_id']:r for r in report.build_report(tmp_path)}
    assert rows['full']['cuda_decode_peak_allocated_bytes_32k']==32768
    assert rows['default']['cuda_decode_peak_allocated_bytes_32k']==2048
    assert rows['default']['cuda_decode_peak_allocated_bytes_8k']==2048
    assert rows['default']['peak_mps_allocated']==999999
    assert 'not per-policy memory' in (tmp_path/'report.md').read_text()

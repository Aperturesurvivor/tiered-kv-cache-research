"""Reporting preserves full capped routing outside the bounded final report."""
import hashlib
import json
import pytest
from research.attention_rerank_32k import persist_bounded_report, report_rows_without_events, persist_stage, CASE


def test_max_budget_routing_is_hash_bound_and_report_size_is_actual(tmp_path):
    # Maximum 28*768 events, each at a 32-page candidate and 15-page final cap.
    events=[dict(layer=i%28,position=32000+i//28,pages=list(range(200,215)),
                 candidate_pages=list(range(200,232))) for i in range(28*768)]
    row={'arm':'quest','rounds':[{'turn':0,'answer':'12345678','selected_events':events[:len(events)//2]},
                                  {'turn':1,'answer':'12345678','selected_events':events[len(events)//2:]}]}
    samples=tmp_path/'samples.jsonl';samples.write_text(json.dumps({'status':'partial','case':CASE,'row':row})+'\n')
    digest=hashlib.sha256(samples.read_bytes()).hexdigest()
    compact=report_rows_without_events([row],samples)
    report={'rows':compact,'weights_sha256':'f'*64}
    target=tmp_path/'bridge.json';size=persist_bounded_report(target,report)
    assert size==target.stat().st_size==len(target.read_bytes()) and size<8*2**20
    saved=json.loads(target.read_bytes())
    original=json.loads(samples.read_text())['row']
    for actual,source in zip(saved['rows'][0]['rounds'],original['rounds']):
        assert 'selected_events' not in actual
        assert actual['selected_events_ref']=={'file':'samples.jsonl','sha256':digest,'arm':'quest','turn':source['turn'],'count':len(source['selected_events'])}
        assert actual['answer']==source['answer']
    assert sum(x['selected_events_ref']['count'] for x in saved['rows'][0]['rounds'])==28*768


def test_report_limit_fails_without_erasing_integrity(tmp_path):
    integrity=tmp_path/'integrity.json';persist_stage(integrity,{'status':'partial','weights_unchanged':True})
    target=tmp_path/'bridge.json'
    with pytest.raises(ValueError,match='exceeding'):
        persist_bounded_report(target,{'text':'x'*100},limit=32)
    assert not target.exists() and json.loads(integrity.read_text())['weights_unchanged'] is True


def test_mismatched_saved_events_are_rejected(tmp_path):
    row={'arm':'quest','rounds':[{'turn':0,'selected_events':[{'pages':[1]}]}]}
    samples=tmp_path/'samples.jsonl'
    samples.write_text(json.dumps({'status':'partial','case':CASE,'row':row})+'\n')
    row['rounds'][0]['selected_events'][0]['pages']=[2]
    with pytest.raises(ValueError,match='saved partial rows differ'):
        report_rows_without_events([row],samples)

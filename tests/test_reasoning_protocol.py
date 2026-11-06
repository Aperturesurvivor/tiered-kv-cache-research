import json
import pytest

from research.reasoning_protocol import (
    latest_completed_at, next_search, parse_response, policies, round_request,
    zero_loss_upper_bound,
)


class CharTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def test_model_search_changes_query_without_evaluator_input():
    response = json.dumps({'answer': 'unknown', 'search': 'STATION_NEW', 'notes': 'Follow the station link.'})
    query, detail = next_search('PROJECT_ORIGINAL', response, CharTokenizer())
    assert query == 'STATION_NEW' and detail['used_model_search']
    assert parse_response(response)['answer'] == 'unknown'


@pytest.mark.parametrize('response', ['not JSON', '{}', '[]',
    '{"answer":123,"search":"x","notes":"y"}',
    '{"answer":"x","search":"x","notes":"y","extra":"z"}'])
def test_invalid_protocol_falls_back_without_inventing_answer(response):
    assert parse_response(response) is None
    assert next_search('original', response, CharTokenizer())[0] == 'original'


def test_empty_search_fallback_and_query_budget():
    empty = '{"answer":"draft","search":"","notes":"review"}'
    assert next_search('original', empty, CharTokenizer())[0] == 'original'
    long = json.dumps({'answer': 'draft', 'search': 'x' * 100, 'notes': ''})
    query, detail = next_search('original', long, CharTokenizer(), cap=64)
    assert query == 'x' * 64 and detail['query_truncated']
    assert detail['original_query_tokens'] == 100 and detail['query_tokens'] == 64


def test_fair_schedule_keeps_both_baselines_and_deduplicates_four_round_case():
    plan = policies()
    assert len(plan) == len({p['id'] for p in plan}) == 11
    for attention in ('stock', 'sparse'):
        subset = [p for p in plan if p['attention'] == attention and p['format'] == 'json']
        assert {(p['rounds'], p['cap_per_round']) for p in subset} == {(1,128),(2,128),(4,128),(1,512),(2,256)}
        assert all(p['rounds'] * p['cap_per_round'] == 512 for p in subset if '512_total' in p['schedule_labels'])


def test_budget_curve_selects_latest_available_not_best_correctness():
    rows = [{'cumulative_request_s': 2., 'cumulative_generated_tokens': 10, 'answer': 'correct'},
            {'cumulative_request_s': 5., 'cumulative_generated_tokens': 20, 'answer': 'wrong'}]
    assert latest_completed_at(rows, seconds=1.) is None
    assert latest_completed_at(rows, seconds=3.) is rows[0]
    assert latest_completed_at(rows, seconds=6.) is rows[1]
    assert latest_completed_at(rows, seconds=6., generated_tokens=10) is rows[0]


def test_final_round_has_no_further_lookup_request():
    assert 'Another round will follow' in round_request('question', 0, 4)
    assert 'final round' in round_request('question', 3, 4)
    with pytest.raises(ValueError):
        round_request('question', 4, 4)


def test_small_perfect_sample_does_not_establish_two_percent_loss_limit():
    assert zero_loss_upper_bound(12) > .2
    assert zero_loss_upper_bound(148) > .02
    assert zero_loss_upper_bound(149) < .02

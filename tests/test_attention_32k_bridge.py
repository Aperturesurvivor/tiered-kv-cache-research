from research.attention_32k_bridge import CASE, compare_historical_round, filtered_text_blocks
from research.attention_trajectory_diagnostic import load_plan


def test_32k_fixture_and_saved_text_pages():
    plan=load_plan(CASE)
    assert CASE=='reasoning-paraphrase-32768-163'
    assert len(plan['text_cold_pages'])==15
    assert filtered_text_blocks(plan['original_text_pages'],plan['fixture']['prefix_tokens_count'])==plan['text_cold_pages']


def test_historical_comparison_does_not_claim_unmatched_prompts():
    old=dict(prompt_token_ids=[1,2],generated_token_ids=[3],answer='a')
    assert compare_historical_round(old,old)==dict(prompt_tokens_match=True,generated_token_ids_match=True,answer_match=True)
    changed=dict(prompt_token_ids=[1,4],generated_token_ids=[3],answer='a')
    assert compare_historical_round(changed,old)==dict(prompt_tokens_match=False,generated_token_ids_match=None,answer_match=None)


def test_comparison_reports_historical_sink_overlap_and_pages():
    prior=dict(prompt_token_ids=[1],generated_token_ids=[2],answer='x',selected_blocks=[0,1,3])
    observed=dict(prompt_token_ids=[1],generated_token_ids=[2],answer='x',selected_text_pages=[1,3])
    value=compare_historical_round(observed,prior,fixed_text=True)
    assert value['historical_selected_pages']==[0,1,3]
    assert value['current_selected_pages']==[1,3]
    assert value['historical_page_zero_already_fixed_sink'] is True

import pytest

from research.lsh_text_probe import record_coverage


def test_whole_record_coverage_needs_both_payload_blocks():
    ex = {'prefix_tokens_count': 5000, 'record_spans': [[250, 270], [4500, 4520]],
          'required_record_indices': [0, 1]}
    partial = record_coverage(ex, [1], 3001)
    assert partial['available_record_fractions'] == [.3, 1.]
    assert not partial['all_required_records_available']
    full = record_coverage(ex, [1, 2], 3001)
    assert full['all_required_records_available']
    assert not full['all_required_records_already_working']


def test_prefix_and_recent_are_explicit_free_coverage_but_future_blocks_rejected():
    ex = {'prefix_tokens_count': 5000, 'record_spans': [[0, 20], [4500, 4520]],
          'required_record_indices': [0, 1]}
    result = record_coverage(ex, [], 3001)
    assert result['all_required_records_already_working']
    assert result['retrieved_record_fractions'] == [0., 0.]
    with pytest.raises(ValueError):
        record_coverage(ex, [24], 3001)

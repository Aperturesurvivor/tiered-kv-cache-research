import pytest
from research.budget_probe import expand_blocks


def test_keeps_original_underfilled_control_and_expands_both_sides():
    assert expand_blocks((5,6,4),20,4)==(5,6,4)
    assert expand_blocks((5,6,4),20,8)==(5,6,4,3,7,2,8,1)


def test_boundary_clipping_and_nested_budgets():
    small=expand_blocks((0,1),40,16)
    large=expand_blocks((0,1),40,32)
    assert small==tuple(range(16))
    assert large==tuple(range(32))
    assert expand_blocks((4,),5,16)==(4,3,2,1,0)


@pytest.mark.parametrize('seeds,eligible,capacity',[([],10,16),([1,1],10,16),([-1],10,16),([10],10,16),([1,2,3],10,2)])
def test_rejects_invalid_or_missing_query_seeds(seeds,eligible,capacity):
    with pytest.raises(ValueError):
        expand_blocks(seeds,eligible,capacity)

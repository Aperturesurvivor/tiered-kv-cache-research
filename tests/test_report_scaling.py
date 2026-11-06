import pytest
from research.report_demand import context_scaling


def test_distinguishes_constant_latency_from_linear_growth():
    rows=[]
    for budget in (4096,8192,16384,32768):
        for policy,value in [('constant',10.),('linear',budget/100.)]:
            rows.append({'policy':policy,'trial':0,'budget':budget,'prompt_tokens':budget,
                         'mean_ms':value,'median_ms':value})
    result={r['policy']:r for r in context_scaling(rows)}
    assert result['constant']['median_32K_over_8K']==1
    assert result['constant']['median_log_log_slope']==pytest.approx(0,abs=1e-12)
    assert result['constant']['median_ratio_at_or_below_1_25']
    assert result['linear']['median_32K_over_8K']==4
    assert result['linear']['median_log_log_slope']==pytest.approx(1)
    assert not result['linear']['median_ratio_at_or_below_1_25']


def test_short_native_context_has_no_extrapolated_ratio():
    rows=[{'policy':'short','trial':0,'budget':n,'prompt_tokens':n-128,'mean_ms':20.,'median_ms':20.}
          for n in (4096,8192)]
    assert context_scaling(rows)[0]['median_32K_over_8K'] is None
    with pytest.raises(ValueError):
        context_scaling(rows+[rows[0]])

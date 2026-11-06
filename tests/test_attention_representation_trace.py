from research.attention_representation_trace import baseline, probe_plan


def test_predeclared_probe_positions_with_pending_token():
    probes=probe_plan(baseline())
    assert probes=={30796:'round0-final-prompt',30797:'round0-decode-1',30828:'round0-decode-32',
                    30965:'round1-final-prompt',30966:'round1-decode-1',30997:'round1-decode-32'}

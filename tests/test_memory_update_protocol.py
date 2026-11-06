from copy import deepcopy

import pytest

from research.memory_update_protocol import policies, validate_episode


def episode():
    tokens = list(range(6200))
    records = ['old revision', 'new revision', 'new entity']
    spans = [[300, 320], [3000, 3020], [3020, 3040]]
    versions = [dict(tokens=tokens[:n] + [99999], prefix_tokens_count=n,
                     records=records[:1] if i == 0 else records,
                     record_spans=spans[:1] if i == 0 else spans)
                for i, n in enumerate((3000, 3060, 6100))]
    return dict(budget=8192, versions=versions,
                queries=[dict(version=i) for i in (0, 1, 2, 2, 2)])


def test_update_retains_absolute_addresses_and_excludes_future_facts():
    ep = episode()
    validate_episode(ep)
    changed = deepcopy(ep)
    changed['versions'][1]['tokens'][500] = -1
    with pytest.raises(ValueError, match='historical token addresses'):
        validate_episode(changed)
    leaked = deepcopy(ep)
    leaked['versions'][1]['record_spans'][1] = [2000, 2020]
    with pytest.raises(ValueError, match='leaked'):
        validate_episode(leaked)


def test_aged_update_cannot_pass_by_remaining_in_recent_window():
    ep = episode()
    ep['versions'][2]['prefix_tokens_count'] = 4000
    with pytest.raises(ValueError, match='still in recent'):
        validate_episode(ep)


def test_update_reserves_reasoning_space_and_matching_memory_baselines():
    ep = episode()
    ep['versions'][2]['prefix_tokens_count'] = 6200
    ep['versions'][2]['tokens'] = list(range(6201))
    with pytest.raises(ValueError, match='room for counted reasoning'):
        validate_episode(ep)
    plan = policies()
    assert {(p['attention'], p['memory']) for p in plan} == {
        ('stock', 'none'), ('stock', 'notes'), ('sparse', 'none'), ('sparse', 'both')}
    assert all(p['rounds'] == 2 and p['cap_per_round'] == 128 for p in plan)

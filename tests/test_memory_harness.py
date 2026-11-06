import json

import pytest

from research.memory_harness import BoundedMemory, STATE_TOKEN_CAP, PROMPT_TOKEN_CAP


class CharTokenizer:
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def decode(self, ids, **kwargs): return ''.join(chr(c) for c in ids)


def response(answer='wrong', search='STATION_Z', notes='A fallible model claim'):
    return json.dumps(dict(answer=answer, search=search, notes=notes))


def test_memory_remains_bounded_across_a_long_session_and_eviction():
    tok = CharTokenizer()
    memory = BoundedMemory(tok, 'both')
    evicted = 0
    for i in range(100):
        event = memory.observe('PROJECT_'+str(i)+'?'*100,
            response('7'*100,'s'*100,'n'*100), list(range(i,i+100)))
        evicted += event['evicted']
        assert len(memory.entries) <= 4
        assert memory.token_count() <= STATE_TOKEN_CAP
        assert len(tok.encode(memory.prompt())) <= PROMPT_TOKEN_CAP
        assert memory.snapshot()['stored_addresses'] <= 16
    assert evicted > 0
    assert all('PROJECT_0?' not in e.question for e in memory.entries)


def test_wrong_notes_are_retained_as_fallible_and_invalid_output_is_not_repaired():
    memory = BoundedMemory(CharTokenizer(), 'both')
    assert not memory.observe('PROJECT_A', 'not JSON', [4])['saved']
    memory.observe('PROJECT_A', response(), [4,4,8,10,12,15])
    assert memory.entries[0].answer == 'wrong'
    assert memory.addresses('PROJECT_A', 9) == [4,8]
    assert 'Unverified' in memory.prompt()
    # Unknown entities cannot reuse another entity's cached addresses.
    assert memory.addresses('PROJECT_B', 99) == []


@pytest.mark.parametrize('mode', ['none','notes','addresses','both'])
def test_memory_ablation_keeps_prompt_and_address_effects_separate(mode):
    memory = BoundedMemory(CharTokenizer(), mode)
    memory.observe('PROJECT_A', response(), [2,3])
    assert bool(memory.prompt()) == (mode in ('notes','both'))
    assert bool(memory.addresses('PROJECT_A', 20)) == (mode in ('addresses','both'))
    assert bool(memory.entries) == (mode != 'none')


def test_latest_matching_memory_wins_without_evaluator_answer_selection():
    memory = BoundedMemory(CharTokenizer(), 'both')
    memory.observe('PROJECT_A', response('old','OLD','old claim'), [2])
    memory.observe('PROJECT_A', response('new','NEW','new claim'), [3])
    assert memory.addresses('PROJECT_A', 20) == [3]
    assert memory.relevant('PROJECT_A').answer == 'new'

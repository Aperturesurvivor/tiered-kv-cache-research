from copy import deepcopy
import json

import pytest

from research.memory_harness import BoundedMemory
from research.memory_protocol import policies, initialize_memory
from research.memory_run import schedule, restore_episode
from test_memory_harness import CharTokenizer, response


def test_screen_keeps_four_episodes_and_fault_injection_separate():
    episodes = [{'id':str(n), 'queries':[{}]*5} for n in range(4)]
    plan = schedule(episodes)
    assert plan == schedule(episodes)
    assert len(plan)==32 and sum(i['queries'] for i in plan)==160
    assert sum(i['queries'] for i in plan if i['policy']['memory_seed']=='empty')==120
    assert sum(i['queries'] for i in plan if i['policy']['memory_seed']=='synthetic_stale')==40
    assert len({(r['episode_id'],r['policy']['id']) for r in plan})==32


def test_resume_requires_exact_memory_chain_and_keeps_wrong_model_notes():
    tok=CharTokenizer()
    policy=next(p for p in policies() if p['attention']=='sparse' and p['memory']=='both' and p['memory_seed']=='empty')
    ep={'id':'episode','queries':[{'query':'PROJECT_A'}]*5}
    memory=BoundedMemory(tok,'both')
    before=memory.snapshot()
    memory.observe('PROJECT_A',response(answer='incorrect'),[3])
    row=dict(status='ok',query_index=0,episode_id=ep['id'],policy=policy,memory_before=before,memory_after=memory.snapshot())
    replay=BoundedMemory(tok,'both')
    restore_episode(replay,ep,policy,json.loads(json.dumps([row])))
    assert replay.snapshot()==memory.snapshot()
    assert replay.entries[-1].answer=='incorrect'
    for field, value in [('query_index',1),('memory_before',memory.snapshot()),('episode_id','other')]:
        bad=deepcopy(row);bad[field]=value
        with pytest.raises(ValueError): restore_episode(BoundedMemory(tok,'both'),ep,policy,[bad])


def test_fault_seed_is_fixed_without_reading_the_evaluator_answer():
    policy=next(p for p in policies() if p['memory_seed']=='synthetic_stale')
    memory=BoundedMemory(CharTokenizer(),'notes')
    ep={'queries':[{'query':'PROJECT_A','answer':'secret_gold'}]}
    initialize_memory(memory,ep,policy)
    assert 'secret_gold' not in json.dumps(memory.snapshot())
    assert memory.entries[0].answer=='STATION_OLD_MEMO'  # char-token test cap=16
    saved=memory.snapshot()
    bad=deepcopy(saved);bad['stored_text_tokens']+=1
    with pytest.raises(ValueError):memory.restore(bad)
    assert memory.snapshot()==saved

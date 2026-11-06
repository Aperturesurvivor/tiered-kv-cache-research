import json
import pytest
import torch

from research import reasoning_run
from research.reasoning_protocol import TASKS, SEEDS
from tiered_kv import generate
from tiered_kv.adapter import DenseController
from tiered_kv.store import ColdStore
from tiered_kv.types import digest
from test_attention import tiny


def test_trial_plan_has_independent_quality_rows_and_labeled_repeated_timings():
    examples = [{'id': f'{n}-{s}-{t}', 'task': t, 'seed': s}
                for n in (8192,32768) for s in SEEDS for t in TASKS]
    plan = reasoning_run.trial_plan(examples)
    assert len(plan) == 220
    assert sum(x['purpose']=='quality' for x in plan) == 176
    assert sum(x['purpose']=='repeated_timing' for x in plan) == 44
    assert len({(x['example_id'],x['policy']['id'],x['trial']) for x in plan}) == 220
    assert reasoning_run.trial_plan(examples) == plan


def test_resumption_rejects_changed_actual_model_weights(tmp_path):
    (tmp_path/'validation-first.json').write_text(json.dumps({'weights_sha256':'first'}))
    rows = [{'validation_file':'validation-first.json'}]
    reasoning_run.check_prior_validation(tmp_path,rows,'first')
    with pytest.raises(ValueError, match='weights differ'):
        reasoning_run.check_prior_validation(tmp_path,rows,'second')
    with pytest.raises(ValueError, match='invalid prior validation'):
        reasoning_run.check_prior_validation(tmp_path,[{'validation_file':'../validation-first.json'}],'first')


@pytest.mark.parametrize('attention', ['stock','sparse'])
@pytest.mark.parametrize('rounds', [1,2,4])
@pytest.mark.parametrize('with_memory', [False,True])
def test_multiround_session_accounts_every_replay_and_generated_token(tmp_path, monkeypatch, attention, rounds, with_memory):
    """Tiny CPU mechanics test, not trained-model quality or timing evidence."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate, 'DEVICE', 'cpu')
    monkeypatch.setattr(generate, 'sync', lambda: None)
    monkeypatch.setattr(generate, 'empty_cache', lambda: None)
    monkeypatch.setattr(reasoning_run, 'DEVICE', 'cpu')
    monkeypatch.setattr(reasoning_run, 'sync', lambda: None)
    monkeypatch.setattr(reasoning_run, 'WORKING_KV', {'W':3,'C':0,'R':4,'k':1,'sink_tokens':1})
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    # decode_greedy has a bound synchronize default; select its explicit CPU
    # test callback without changing generation or pending-token semantics.
    original_decode = reasoning_run.decode_greedy
    monkeypatch.setattr(reasoning_run, 'decode_greedy',
        lambda *args, **kwargs: original_decode(*args, **kwargs, synchronize=lambda: None))

    class Guard:
        def check(self): pass

    class Tokenizer:
        last = ''
        def encode(self, text, **kwargs):
            self.last = text
            return [5,6,7] * (2 if 'Unverified memory' in text else 1)
        def decode(self, ids, **kwargs):
            if list(ids) in ([5,6,7], [5,6,7]*2):
                return self.last
            return json.dumps({'answer':'11111111','search':'STATION_NEXT','notes':'test draft'})
        def apply_chat_template(self, messages, **kwargs):
            return ''.join('<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)+'<assistant>'

    class Index:
        calls = None
        def __init__(self):
            self.calls = []
            self.last_search = {}
        def search(self, query, vector, k, method):
            self.calls.append(query)
            return [0] if len(self.calls)==1 else [1]

    model = tiny().half()
    model.generation_config.eos_token_id = -1
    root = tmp_path/'results'/'test'
    root.mkdir(parents=True)
    ex = {'id':'tiny-case','task':'linked_fact','query':'original question',
          'answer':'88888888','first_suffix':'<assistant>', 'budget':128,
          'tokens':list(range(20)), 'prefix_tokens_count':16,
          'token_hash':digest(list(range(20)))}
    engine = generate.RunEngine(model,Tokenizer(),root,Guard())
    try:
        # This miniature test uses four-token blocks; production prefixes and
        # policies both use 128. Preserve matching geometry in the fixture.
        prefix = engine.scratch/'tiny-prefix'
        archive = ColdStore(prefix,engine.layers,engine.h,engine.d,4)
        engine.adapter.set_controller(DenseController(engine.layers,archive))
        engine.adapter.forward_tokens(ex['tokens'][:16],0)
        archive.checkpoint({'token_hash':ex['token_hash'],'model_revision':engine.model_revision})
        archive.close()
        engine.adapter.set_controller(DenseController(engine.layers))
        index = Index()
        policy = {'id':f'{attention}-{rounds}', 'attention':attention,
                  'rounds':rounds, 'cap_per_round':2, 'format':'json'}
        from research.memory_harness import BoundedMemory
        memory = BoundedMemory(engine.tokenizer,'both') if with_memory else None
        if memory:
            memory.observe('original question',json.dumps({'answer':'wrong prior','search':'','notes':'old note'}),[0])
        row = reasoning_run.run_policy(engine,ex,policy,0,prefix,index,memory=memory)
        assert len(row['rounds']) == rounds
        assert row['generated_tokens'] == 2*rounds
        prompt_count = 3*rounds + (3 if with_memory else 0)
        assert row['prompt_token_evaluations'] == prompt_count
        # Each output is generated, but the very last token is still pending.
        assert row['model_token_evaluations'] == prompt_count + 2*rounds - 1
        assert row['prediction'] == '11111111' and not row['exact_correct']
        assert all(not r['exact_correct'] for r in row['rounds'])
        if attention == 'sparse':
            assert index.calls == ['original question'] + ['STATION_NEXT']*(rounds-1)
            assert row['rounds'][-1]['selected_blocks'] == ([0] if rounds==1 else [1])
            assert row['max_active_KV_entries'] <= 8
        else:
            assert index.calls == []
            assert row['max_active_KV_entries'] == 16 + row['model_token_evaluations']
        assert row['rounds'][-1]['cumulative_generated_tokens'] == row['generated_tokens']
        assert row['rounds'][-1]['cumulative_request_s'] <= row['request_s']
        if with_memory:
            assert row['memory_update']['saved']
            assert row['memory_after'] == memory.snapshot()
            assert all(ex['answer'] not in entry['answer'] for entry in row['memory_after']['entries'])
    finally:
        engine.adapter.uninstall()

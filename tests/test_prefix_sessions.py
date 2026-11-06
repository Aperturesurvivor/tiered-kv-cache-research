"""Small real-cache mechanics checks; no trained-model quality claims."""
from dataclasses import asdict
import json
import pytest

from research import prefix_sessions, persistent_kv
from tiered_kv import generate
from tiered_kv.adapter import DenseController
from tiered_kv.config import Config
from tiered_kv.store import ColdStore
from tiered_kv.types import digest
from test_attention import tiny


@pytest.mark.parametrize('policy',['stock_hf','rare_context','prefix_128'])
def test_three_questions_preserve_session_and_audit_original_prefix(tmp_path,monkeypatch,policy):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate,'DEVICE','cpu')
    monkeypatch.setattr(persistent_kv,'DEVICE','cpu')
    monkeypatch.setattr(prefix_sessions,'DEVICE','cpu')
    original_decode = prefix_sessions.decode_greedy
    monkeypatch.setattr(prefix_sessions,'decode_greedy',
        lambda adapter,logits,cap,eos,guard: original_decode(adapter,logits,2,eos,guard=guard,synchronize=lambda:None))
    class Guard:
        def check(self): pass
    class Tokenizer:
        def encode(self,*a,**k): return [5,6]
        def decode(self,*a,**k): return 'test prediction'
        def apply_chat_template(self,messages,**kwargs):
            return ''.join('<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)+'<assistant>'
    class Index:
        def search(self,*a): return [0]
    model = tiny().half()
    model.generation_config.eos_token_id = -1
    root = tmp_path/'results'/'test'
    root.mkdir(parents=True)
    engine = generate.RunEngine(model,Tokenizer(),root,Guard())
    prefix = engine.scratch/'prefix'
    parent = ColdStore(prefix,engine.layers,engine.h,engine.d,128)
    engine.adapter.set_controller(DenseController(engine.layers,parent))
    engine.adapter.forward_tokens([i%128 for i in range(256)],0)
    parent.checkpoint({'token_hash':digest('test')})
    parent.close()
    engine.adapter.set_controller(DenseController(engine.layers))
    ex = {'id':'test','budget':4096,'prefix_tokens_count':256,'first_suffix':'suffix',
          'queries':[{'question':str(i),'answer':'wrong','record_span':[140,144]} for i in range(3)]}
    cfg = Config(W=3,C=0,k=4,R=128,retrieval='fixed',sink_tokens=4,buffered_writes=True)
    controls = {(4096,p,t):{'prediction':'test prediction','output_tokens':2,
                           'policy_config':asdict(cfg),'selected_blocks':[0,1]}
                for p in ('stock_hf','rare_context') for t in range(3)}
    output = root/'samples.jsonl'
    try:
        prefix_sessions.run_session(engine,ex,policy,prefix,Index(),controls,output)
        rows = [json.loads(x) for x in output.read_text().splitlines()]
        assert len(rows)==3
        # First turn leaves its last output pending; subsequent turns consume it.
        assert [r['position_after'] for r in rows] == [259,263,267]
        assert all(r['code_found']==0 for r in rows)
        if policy=='stock_hf':
            assert all(r['payload_checks']==[] for r in rows)
        else:
            assert all(len(r['payload_checks'])==2 for r in rows)
            assert all(c['prefix_K_bit_equal'] and c['prefix_V_bit_equal']
                       and c['whole_fact_present'] for r in rows for c in r['payload_checks'])
            assert all(r['policy_config']['sink_tokens']==(128 if policy=='prefix_128' else 4) for r in rows)
        if policy!='prefix_128':
            assert all(r['original_prediction_and_length_reproduced'] for r in rows)
    finally:
        engine.adapter.uninstall()

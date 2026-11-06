import json
import pytest


def test_parity_checks_each_stock_chunk_and_decode():
    from test_attention import tiny
    from tiered_kv.validation import parity
    result=parity(tiny(),257)
    assert len(result['same_chunk_stock'])==3+16
    assert result['same_chunk_stock'][2]['label']=='prefill-257'
    assert max(x['max_abs'] for x in result['same_chunk_stock'])<1e-4
    assert result['atol']==1e-4
    assert 'stock_max_abs' in result['cross_shape']


@pytest.mark.parametrize('family',['llama','qwen3'])
def test_other_family_parity_preserves_norms_and_head_dimensions(family):
    import torch
    from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
    from tiered_kv.validation import parity
    torch.manual_seed(41)
    cls,model_cls=(LlamaConfig,LlamaForCausalLM) if family=='llama' else (Qwen3Config,Qwen3ForCausalLM)
    cfg=cls(vocab_size=128,hidden_size=64,intermediate_size=96,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=32,max_position_embeddings=2048)
    cfg._attn_implementation='sdpa'
    model=model_cls(cfg).eval()
    result=parity(model,257)
    assert max(result['max_errors'])<1e-4
    assert max(x['max_abs'] for x in result['same_chunk_stock'])<1e-4


def test_gate_evidence_cannot_cross_runtime_or_omit_identity(tmp_path,monkeypatch):
    from tiered_kv import validation
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(validation,'implementation_hash',lambda:'same-code')
    monkeypatch.setattr(validation,'runtime_identity',lambda:{'device':'cuda','gpu':'T4'})
    root=tmp_path/'results'/'validation'
    root.mkdir(parents=True)
    state={'implementation_hash':'same-code',
           'gates':[{'gate':i,'status':'passed'} for i in range(1,6)]}
    for runtime in (None,{'device':'mps'},{'device':'cuda','gpu':'A100'}):
        state['runtime']=runtime
        (root/'validation.json').write_text(json.dumps(state))
        with pytest.raises(RuntimeError,match='runtime changed or absent'):
            validation.require_gates()
    state['runtime']={'device':'cuda','gpu':'T4'}
    (root/'validation.json').write_text(json.dumps(state))
    assert validation.require_gates()==state

import json

import pytest
import torch

from research import memory_update_archive as updates
from tiered_kv import generate
from tiered_kv.adapter import DenseController
from tiered_kv.store import ColdStore, file_sha
from tiered_kv.types import digest
from test_attention import tiny


def test_physical_append_preserves_parent_and_matches_continuous_dense_forward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate, 'sync', lambda: None)
    monkeypatch.setattr(generate, 'empty_cache', lambda: None)
    monkeypatch.setattr(updates, 'sync', lambda: None)
    monkeypatch.setattr(updates, 'device_empty_cache', lambda: None)
    class Guard:
        def check(self): pass
    tokens = [i % 127 for i in range(278)]
    versions = [dict(id=f'v{i}', tokens=tokens[:n]+[5], prefix_tokens_count=n,
                     token_hash=digest(tokens[:n]+[5])) for i, n in enumerate((129, 148, 277))]
    engine = generate.RunEngine(tiny().half(), None, tmp_path/'results'/'append-test', Guard())
    try:
        checked = updates.validate_append(engine)
        assert checked['status'] == 'ok' and checked['next_token_logit_max_abs_error'] == 0
        assert len(checked['kv_checks']) == 2
        initial, _ = engine.prefix(versions[0])
        before = {p.name:file_sha(p) for p in initial.iterdir() if p.is_file()}
        hot, first = updates.append_version(engine, versions[0], versions[1], initial)
        aged, second = updates.append_version(engine, versions[1], versions[2], hot)
        assert before == {p.name:file_sha(p) for p in initial.iterdir() if p.is_file()}
        assert first['append_model_tokens'] == 19 and first['append_forward_calls'] == 1
        assert second['append_model_tokens'] == 129 and second['append_forward_calls'] == 2
        assert second['append_layer_kv_positions'] == 2*(129*148+129*130//2)
        assert first['cuda_peak_allocated_bytes'] is None  # CPU mechanics, not GPU measurement.
        saved = ColdStore.open(aged, readonly=True)
        try:
            assert saved.count == 277 and saved.prefix.count == 148 and saved.prefix.prefix.count == 129
            doc = json.loads((aged/'checkpoint.json').read_text())
            assert doc['sizes'] == [129*saved.record_bytes]*engine.layers
            # Init plus both sides of the audit read the entire earlier version,
            # including its grandparent. Checksum scans are a separate cost.
            assert second['logical_tensor_read_bytes_including_audit'] == 3*148*saved.record_bytes*engine.layers
            reference = DenseController(engine.layers)
            engine.adapter.set_controller(reference)
            # Identical causal chunk boundaries, including the partial first prefix.
            for start, end in ((0,128),(128,129),(129,148),(148,276),(276,277)):
                engine.adapter.forward_tokens(tokens[start:end], start)
            for layer in range(engine.layers):
                got = saved.read_range(layer, 0, 277)
                assert torch.equal(got.k, reference.keys[layer])
                assert torch.equal(got.v, reference.values[layer])
            expected = engine.adapter.step(7, 277)
            restored = DenseController(engine.layers)
            restored.initialize(saved, 'cpu', torch.float16)
            engine.adapter.set_controller(restored)
            actual = engine.adapter.step(7, 277)
            assert torch.equal(actual, expected)
        finally:
            saved.close()
        def no_forward(*args, **kwargs):
            raise AssertionError('completed append must not execute model tokens again')
        monkeypatch.setattr(engine.adapter, 'forward_tokens', no_forward)
        assert updates.append_version(engine, versions[1], versions[2], hot) == (aged, second)
        changed = {**versions[2], 'tokens':[-1]+versions[2]['tokens'][1:]}
        with pytest.raises(ValueError, match='historical token IDs'):
            updates.append_version(engine, versions[1], changed, hot)
    finally:
        engine.adapter.uninstall()

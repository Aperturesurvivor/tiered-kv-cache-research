import torch
import pytest

from research.reasoning_sessions import BatchedStockSession, CountedSession
from tiered_kv.adapter import ModelAdapter, DenseController
from tiered_kv.generate import decode_greedy
from test_attention import tiny


class EmptyArchive:
    count = 0
    layers = 0


def test_batched_stock_and_serial_hf_agree_with_correct_logical_work():
    model = tiny()
    batched = CountedSession(BatchedStockSession(model, EmptyArchive()), model)
    serial = CountedSession(BatchedStockSession(model, EmptyArchive()), model)
    tokens = [3, 7, 2, 11]
    left = batched.feed_prompt(tokens)
    for token in tokens:
        right = serial.feed_prompt([token])
    assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)
    assert batched.metrics()['model_token_evaluations'] == serial.metrics()['model_token_evaluations'] == 4
    assert batched.metrics()['transformer_forward_calls'] == 1
    assert serial.metrics()['transformer_forward_calls'] == 4
    assert batched.metrics()['layer_summed_valid_KV_positions'] == 2 * (1 + 2 + 3 + 4)
    assert batched.metrics()['logical_attention_head_pairs'] == serial.metrics()['logical_attention_head_pairs']


def test_counted_adapter_matches_stock_and_includes_pending_between_turns():
    model = tiny()
    stock = CountedSession(BatchedStockSession(model, EmptyArchive()), model)
    expected = stock.feed_prompt([3, 7, 2, 11])
    adapter = ModelAdapter.install(model, DenseController(model.config.num_hidden_layers))
    try:
        counted = CountedSession(adapter, model, active_limit=16)
        logits = counted.feed_prompt([3, 7, 2, 11])
        assert torch.allclose(logits, expected, atol=1e-5, rtol=1e-5)
        generated, _ = decode_greedy(counted, logits, 3, set(), synchronize=lambda: None)
        assert len(generated) == 3
        assert counted.position == 6 and counted.metrics()['model_token_evaluations'] == 6
        with pytest.raises(ValueError, match='preceding generated token'):
            counted.feed_prompt([5])
        counted.consume_pending()
        counted.feed_prompt([5, 6])
        assert counted.position == counted.metrics()['model_token_evaluations'] == 9
        assert counted.metrics()['prompt_token_evaluations'] == 6
        assert counted.metrics()['layer_summed_valid_KV_positions'] == 2 * sum(range(1, 10))
        assert counted.metrics()['max_active_KV_entries'] == 9
    finally:
        adapter.uninstall()


def test_active_budget_and_context_limit_fail_loudly():
    model = tiny()
    session = BatchedStockSession(model, EmptyArchive())
    with pytest.raises(ValueError, match='native context'):
        session.forward_tokens([1] * (model.config.max_position_embeddings + 1), 0)
    adapter = ModelAdapter.install(model, DenseController(model.config.num_hidden_layers))
    try:
        counted = CountedSession(adapter, model, active_limit=1)
        with pytest.raises(RuntimeError, match='working KV limit'):
            counted.feed_prompt([3, 4])
    finally:
        adapter.uninstall()


def test_real_tiered_cache_stays_bounded_across_reselection(tmp_path):
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore
    from tiered_kv.types import Rotary
    model = tiny()
    store = ColdStore(tmp_path/'kv', 2, 2, 16, 4, staged_tensor_writes=True)
    cfg = Config(W=3, C=0, B=4, R=4, k=1, sink_tokens=1,
                 retrieval='fixed', fixed_blocks=(0,), buffered_writes=True)
    cache = TieredCache(cfg, store, Rotary(model.model.rotary_emb.inv_freq.numpy()),
                        'cpu', dtype=torch.float32)
    adapter = ModelAdapter.install(model, cache)
    try:
        counted = CountedSession(adapter, model, active_limit=8)
        counted.feed_prompt([3, 4, 5, 6] * 4)
        cache.select_blocks((1,))
        counted.feed_prompt([7, 8, 9, 10] * 4)
        stats = counted.metrics()
        assert cache.position == store.count == stats['model_token_evaluations'] == 32
        assert stats['max_active_KV_entries'] <= 8
        assert stats['layer_summed_valid_KV_positions'] <= 2 * 8 * 32
        assert counted.trace_calls == 2 * 32
        assert cache.config.fixed_blocks == (1,)
    finally:
        store.close()
        adapter.uninstall()


@pytest.mark.parametrize('fail', [False, True])
def test_stock_session_releases_real_cache_without_cyclic_gc(fail):
    import gc
    import weakref

    model = tiny()
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        def run():
            session = BatchedStockSession(model, EmptyArchive())
            counted = CountedSession(session, model)
            counted.feed_prompt([3, 7, 2, 11])
            cache_ref = weakref.ref(session.past)
            tensor_ref = weakref.ref(session.past.key_cache[0])
            if fail:
                try:
                    counted.feed_prompt([])
                except ValueError:
                    pass
                else:
                    raise AssertionError('empty prompt must fail')
            return cache_ref, tensor_ref
        cache_ref, tensor_ref = run()
        assert cache_ref() is None
        assert tensor_ref() is None
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def test_new_prefix_charges_cleanup_before_first_forward(tmp_path, monkeypatch):
    import tiered_kv.generate as generate

    events = []
    engine = generate.RunEngine.__new__(generate.RunEngine)
    engine.scratch = tmp_path
    engine.layers, engine.h, engine.d, engine.model_revision = 2, 2, 16, 'revision'
    class Adapter:
        def set_controller(self, controller): pass
        def forward_tokens(self, tokens, offset): events.append('forward')
    class Guard:
        def check(self): pass
    class Store:
        def __init__(self, *args): pass
        def checkpoint(self, metadata): events.append(('checkpoint', metadata['prefill_s']))
        def close(self): pass
    engine.adapter, engine.guard = Adapter(), Guard()
    monkeypatch.setattr(generate, 'disk_preflight', lambda: None)
    monkeypatch.setattr(generate, 'sync', lambda: None)
    monkeypatch.setattr(generate, 'ColdStore', Store)
    monkeypatch.setattr(generate.gc, 'collect', lambda: events.append('collect'))
    monkeypatch.setattr(generate, 'empty_cache', lambda: events.append('empty_cache'))
    times = iter((100.0, 104.0))
    monkeypatch.setattr(generate.time, 'perf_counter', lambda: events.append('clock') or next(times))
    _, seconds = engine.prefix({'id': 'new', 'tokens': [3, 4], 'prefix_tokens_count': 1, 'token_hash': 'hash'})
    assert seconds == 4.0
    assert events[:5] == ['clock', 'collect', 'empty_cache', 'forward', 'clock']
    assert ('checkpoint', 4.0) in events


def test_cached_prefix_does_not_run_new_prefill_cleanup(tmp_path, monkeypatch):
    import tiered_kv.generate as generate

    engine = generate.RunEngine.__new__(generate.RunEngine)
    engine.scratch, engine.model_revision = tmp_path, 'revision'
    prefix = tmp_path/'cached-prefix'
    prefix.mkdir()
    (prefix/'checkpoint.json').write_text('{}')
    class Store:
        metadata = {'token_hash': 'hash', 'model_revision': 'revision', 'prefill_s': 3.0}
        count = 1
        def close(self): pass
        @classmethod
        def open(cls, path, readonly): return cls()
    monkeypatch.setattr(generate, 'disk_preflight', lambda: None)
    monkeypatch.setattr(generate, 'ColdStore', Store)
    monkeypatch.setattr(generate.gc, 'collect', lambda: pytest.fail('unexpected collection'))
    monkeypatch.setattr(generate, 'empty_cache', lambda: pytest.fail('unexpected cache cleanup'))
    assert engine.prefix({'id':'cached', 'tokens':[3,4], 'prefix_tokens_count':1, 'token_hash':'hash'}) == (prefix, 3.0)

"""Measured dense append into immutable-parent KV overlays for update probes.

This isolates archive updates from bounded serving. Dense ingestion is explicit;
it is not evidence that online ingestion has bounded GPU memory.
"""
import gc
import json
import math
from pathlib import Path
import time
import uuid

import torch

from research.resources import logical_store_reads
from tiered_kv.adapter import DenseController
from tiered_kv.generate import sync
from tiered_kv.store import ColdStore, atomic_json, file_sha
from tiered_kv.telemetry import device_empty_cache
from tiered_kv.types import digest


def audit_parent(parent, child, block=4096):
    checks = []
    for layer in range(parent.layers):
        for start in range(0, parent.count, block):
            end = min(parent.count, start + block)
            a, b = parent.read_range(layer, start, end), child.read_range(layer, start, end)
            if not torch.equal(a.k, b.k) or not torch.equal(a.v, b.v) or not (a.ids == b.ids).all():
                raise ValueError('historical KV changed during append')
        checks.append(dict(layer=layer, prefix_positions=parent.count, kv_bit_equal=True))
    return checks


def append_version(engine, earlier, later, parent_path):
    old_n, new_n = earlier['prefix_tokens_count'], later['prefix_tokens_count']
    if (not 0 < old_n < new_n < len(later['tokens'])
            or earlier['tokens'][:old_n] != later['tokens'][:old_n]):
        raise ValueError('append must preserve all historical token IDs')
    root = engine.scratch/(later['id']+'-append')
    identity = dict(parent_checkpoint_sha256=file_sha(Path(parent_path)/'checkpoint.json'),
                    parent_tokens_sha256=digest(earlier['tokens'][:old_n]),
                    tokens_sha256=digest(later['tokens'][:new_n]),
                    token_hash=later['token_hash'], model_revision=engine.model_revision,
                    old_positions=old_n, new_positions=new_n,
                    append_tokens=new_n-old_n, chunk=128)
    if (root/'checkpoint.json').exists():
        child = ColdStore.open(root, readonly=True)
        try:
            if child.metadata != identity or child.count != new_n:
                raise ValueError('existing archive version identity differs')
            receipt = json.loads((root/'append-receipt.json').read_text())
            if receipt['identity'] != identity or receipt['prefix_audit'] != audit_parent(child.prefix, child):
                raise ValueError('existing append receipt differs')
            return root, receipt
        finally:
            child.close()
    if root.exists():
        root = root.with_name(root.name+'-'+uuid.uuid4().hex[:8])
    start = time.perf_counter()
    parent = ColdStore.open(parent_path, readonly=True)
    mount_s = time.perf_counter()-start
    child, controller = None, None
    model_device = next(engine.model.parameters()).device
    is_cuda = model_device.type == 'cuda'
    try:
        if (parent.count != old_n or parent.metadata.get('model_revision') != engine.model_revision
                or parent.metadata.get('token_hash') != earlier['token_hash']):
            raise ValueError('parent KV does not match the declared earlier version')
        child = ColdStore(root, engine.layers, engine.h, engine.d, parent.R, prefix=parent,
                          staged_tensor_writes=True)
        sync()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        controller = DenseController(engine.layers, child)
        controller.initialize(parent, model_device, next(engine.model.parameters()).dtype)
        engine.adapter.set_controller(controller)
        sync()
        init_s = time.perf_counter()-start
        start = time.perf_counter()
        calls = 0
        for pos in range(old_n, new_n, 128):
            engine.guard.check()
            logits = engine.adapter.forward_tokens(later['tokens'][pos:min(pos+128,new_n)], pos)
            calls += 1
        sync()
        append_s = time.perf_counter()-start
        if controller.position != new_n or child.count != new_n or not torch.isfinite(logits).all():
            raise ValueError('append position or logits are invalid')
        peak = torch.cuda.max_memory_allocated() if is_cuda else None
        start = time.perf_counter()
        child.checkpoint(identity)
        checkpoint_s = time.perf_counter()-start
        start = time.perf_counter()
        checks = audit_parent(parent, child)
        audit_s = time.perf_counter()-start
        n = new_n-old_n
        receipt = dict(identity=identity, prefix_audit=checks,
                       checksum_mount_s=mount_s, dense_init_s=init_s,
                       append_s=append_s, checkpoint_s=checkpoint_s, prefix_audit_s=audit_s,
                       append_model_tokens=n, append_forward_calls=calls,
                       append_layer_kv_positions=engine.layers*(n*old_n+n*(n+1)//2),
                       cuda_peak_allocated_bytes=peak,
                       logical_append_write_bytes=child.write_bytes,
                       logical_tensor_read_bytes_including_audit=logical_store_reads(child),
                       scope='dense ingestion into immutable-parent overlay; not bounded online ingestion')
        if calls != math.ceil(n/128):
            raise ValueError('append forward accounting differs')
        atomic_json(root/'append-receipt.json', receipt)
        return root, receipt
    finally:
        engine.adapter.set_controller(DenseController(engine.layers))
        if controller:
            controller.keys.clear(); controller.values.clear()
        if child:
            child.close()
        else:
            parent.close()
        gc.collect(); device_empty_cache()


def validate_append(engine):
    """Actual loaded-model control across partial blocks and a two-parent chain."""
    tokens = [i % min(127,engine.model.config.vocab_size) for i in range(278)]
    versions = [dict(id=f'append-control-v{i}',tokens=tokens[:n]+[5],prefix_tokens_count=n,
                     token_hash=digest(tokens[:n]+[5])) for i,n in enumerate((129,148,277))]
    initial, _ = engine.prefix(versions[0])
    hot, first = append_version(engine,versions[0],versions[1],initial)
    aged, second = append_version(engine,versions[1],versions[2],hot)
    store = ColdStore.open(aged,readonly=True)
    model_device = next(engine.model.parameters()).device
    dtype = next(engine.model.parameters()).dtype
    reference = DenseController(engine.layers)
    restored = None
    try:
        engine.adapter.set_controller(reference)
        for start,end in ((0,128),(128,129),(129,148),(148,276),(276,277)):
            engine.guard.check()
            engine.adapter.forward_tokens(tokens[start:end],start)
        checks = []
        for layer in range(engine.layers):
            got = store.read_range(layer,0,277)
            same = (torch.equal(got.k,reference.keys[layer].cpu())
                    and torch.equal(got.v,reference.values[layer].cpu()))
            if not same:
                raise ValueError('actual-model appended KV differs from continuous dense schedule')
            checks.append(dict(layer=layer,positions=277,kv_bit_equal=True))
        expected = engine.adapter.step(7,277)
        restored = DenseController(engine.layers)
        restored.initialize(store,model_device,dtype)
        engine.adapter.set_controller(restored)
        actual = engine.adapter.step(7,277)
        if not torch.equal(actual,expected):
            raise ValueError('actual-model append reload logits differ')
        return dict(status='ok',scope='actual loaded model; matched causal chunk schedule',
                    prefixes=[129,148,277],appends=[first,second],kv_checks=checks,
                    next_token_logit_max_abs_error=float((actual-expected).abs().max()),
                    next_token_argmax_equal=int(actual.argmax())==int(expected.argmax()))
    finally:
        engine.adapter.set_controller(DenseController(engine.layers))
        reference.keys.clear(); reference.values.clear()
        if restored:
            restored.keys.clear(); restored.values.clear()
        store.close()
        gc.collect(); device_empty_cache()

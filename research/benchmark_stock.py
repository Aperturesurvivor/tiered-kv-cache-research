"""Practical stock HF baseline; same frozen latency fixtures, no KV disk writes.

Run from repo root after validation: python -m research.benchmark_stock
This control stays outside the 15-policy ablation ranking.
"""
import gc
import json
from pathlib import Path
import time
import torch
from tiered_kv.generate import load_model, weights_digest, append_jsonl
from tiered_kv.telemetry import DEVICE, ResourceGuard, device_synchronize, device_empty_cache, runtime_identity
from tiered_kv.validation import require_gates, implementation_hash
from tiered_kv.types import digest
from tiered_kv.store import atomic_json


@torch.inference_mode()
def measure(model,example,guard):
    device=next(model.parameters()).device
    def call(tokens,past):
        return model(torch.tensor([tokens],device=device),past_key_values=past,
                     use_cache=True,logits_to_keep=1)
    device_synchronize()
    start=time.perf_counter()
    past=None
    # Match exact_shared's prefix schedule, including its separate final token.
    query_start=example.get('prefix_tokens_count',len(example['tokens'])-1)
    prefix=example['tokens'][:query_start]
    for offset in range(0,len(prefix),128):
        guard.check()
        out=call(prefix[offset:offset+128],past)
        past=out.past_key_values
    device_synchronize()
    prefill_s=time.perf_counter()-start
    if DEVICE=='cuda':
        torch.cuda.reset_peak_memory_stats()
    query_timer_start=time.perf_counter()
    for token in example['tokens'][query_start:]:
        out=call([token],past)
        past=out.past_key_values
    first=int(out.logits[0,-1].argmax())
    device_synchronize()
    query_s=time.perf_counter()-query_timer_start
    selected=[]
    latency=[]
    all_latency=[]
    for i,token in enumerate(example['continuation']):
        guard.check()
        device_synchronize()
        start=time.perf_counter()
        out=call([token],past)
        past=out.past_key_values
        selected.append(int(out.logits[0,-1].argmax()))
        device_synchronize()
        elapsed=(time.perf_counter()-start)*1000
        all_latency.append(elapsed)
        if i>=16:
            latency.append(elapsed)
    row={'status':'ok','example_id':example['id'],'budget':example['budget'],
         'trial':example['trial'],'prompt_length':len(example['tokens']),
         'latency_ms':latency,'all_latency_ms':all_latency,'selected_hash':digest(selected),'first_token':first,
         'prefill_s':prefill_s,'query_s':query_s,'ttft_s':prefill_s+query_s,
         'warmup_calls':16,'measured_calls':len(latency),'kv_disk_writes':0}
    if DEVICE=='cuda':
        row.update(cuda_decode_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   cuda_decode_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return row


def main():
    validation=require_gates()
    fixtures=json.loads(Path('artifacts/data/pilot.json').read_text())
    identity={'implementation_hash':implementation_hash(),'runtime':runtime_identity(),
              'fixture_hash':fixtures['hash'],'protocol':'stock_hf_no_disk'}
    root=Path('results')/('stock-control-'+digest(identity)[:12])
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'manifest.json',{**identity,'validation':validation})
    atomic_json(root/'fixtures.json',fixtures)
    output=root/'samples.jsonl'
    done={json.loads(line)['example_id'] for line in output.read_text().splitlines()} if output.exists() else set()
    with ResourceGuard() as guard:
        model,_=load_model()
        before=weights_digest(model)
        for ex in fixtures['examples']:
            if ex['task']!='latency' or ex['id'] in done:
                continue
            row=measure(model,ex,guard)
            append_jsonl(output,row)
            print(json.dumps({'example_id':ex['id'],'median_ms':float(torch.tensor(row['latency_ms']).median()),
                              'cuda_peak':row.get('cuda_decode_peak_allocated_bytes')}),flush=True)
            gc.collect()
            device_empty_cache()
        if weights_digest(model)!=before:
            raise RuntimeError('stock model weights changed')
        atomic_json(root/'complete.json',{'weights_unchanged':True,'weights_sha256':before})
    print(root,flush=True)


if __name__=='__main__':
    main()

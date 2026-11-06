"""Capture six predeclared Q probes from a strictly replayed 32K oracle run."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from research.attention_trajectory_diagnostic import ROOT, WEIGHTS, file_sha, load_plan, validate_source
from research.attention_oracle_32k import CASE, CHECKPOINT_SHA, DIAGNOSTIC_SHA, PAGE, SINK, WINDOW, K

BASE=ROOT/'results/evidence/attention-fullkey-oracle-32k-1'
REPORT_SHA='a81d044950b70423b7216826a0228fe4def7c808966d620219a321a93cd2e2ed'
MARKER_SHA='546c9758f4f88cbd9119d89e38238b3dd3dffb7468293d066a6d81059ce64502'
SAMPLES_SHA='53bd6914d0555cde5bf053159561be0d64bb9ac392a5f52c3790b5440100cf03'
MAX_ARTIFACT=32*2**20


def probe_plan(rounds):
    """Final prompt token, then evaluated generated tokens 1 and 32 per round."""
    positions={}
    for turn,row in enumerate(rounds):
        events=row['selected_events'];assert events and len(events)%28==0
        first=events[0]['position']; prompt_start=first+(1 if turn else 0)
        final=prompt_start+len(row['prompt_token_ids'])-1
        positions[final]=f'round{turn}-final-prompt'
        for step in (1,32):
            if len(row['generated_token_ids'])>step:
                positions[final+step]=f'round{turn}-decode-{step}'
    assert len(positions)<=6
    return positions


def baseline():
    assert file_sha(BASE/'bridge.json')==REPORT_SHA and file_sha(BASE/'complete.json')==MARKER_SHA
    assert file_sha(BASE/'samples.jsonl')==SAMPLES_SHA
    marker=json.loads((BASE/'complete.json').read_text());assert marker['status']=='complete' and marker['weights_unchanged']
    assert marker['report_sha256']==REPORT_SHA and marker['samples_sha256']==SAMPLES_SHA
    doc=json.loads((BASE/'bridge.json').read_text());assert doc['case']==CASE and doc['weights_sha256']==WEIGHTS
    assert doc['checkpoint_sha256']==CHECKPOINT_SHA and len(doc['rows'])==1 and doc['rows'][0]['arm']=='oracle'
    rounds=doc['rows'][0]['rounds'];assert len(rounds)==2
    return rounds


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=Path,required=True);ap.add_argument('--diagnostic',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();plan=load_plan(CASE);source=validate_source();rounds=baseline()
    assert file_sha(args.prefix/'checkpoint.json')==CHECKPOINT_SHA and file_sha(args.diagnostic)==DIAGNOSTIC_SHA
    probes=probe_plan(rounds)
    if not args.execute:
        print(json.dumps({'case':CASE,'positions':probes,'baseline_sha256':REPORT_SHA,'prefix_checkpoint_sha256':CHECKPOINT_SHA}));return
    assert torch.cuda.is_available() and os.environ.get('HF_HUB_OFFLINE')=='1'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert shutil.disk_usage(args.output.parent).free>=12*2**30 and torch.cuda.mem_get_info()[0]>=8*2**30
    args.output.mkdir(parents=True,exist_ok=False)
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    from tiered_kv.adapter import ModelAdapter
    from tiered_kv.config import Config
    from tiered_kv.store import ColdStore,atomic_json
    from tiered_kv.telemetry import ResourceGuard,device_empty_cache,runtime_identity
    from tiered_kv.types import Rotary
    from tiered_kv.generate import weights_digest
    from tiered_kv.validation import parity
    from research.attention_page_cache import CompletePageCacheStore
    from research.attention_fullkey_oracle import FullKeyOracleCache,FullKeyOracleIndex
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    snapshot=snapshot_download(plan['model']['model_id'],revision=plan['model']['revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':'cuda'},attn_implementation='sdpa').eval().requires_grad_(False)
    model.config._commit_hash=plan['model']['revision'];assert weights_digest(model)==WEIGHTS
    checks=[parity(model,n) for n in (64,512)]
    assert all(all(c['greedy_agreement']) and max(c['max_errors'])<=1e-4 for c in checks)
    parent=ColdStore.open(args.prefix,readonly=True)
    assert parent.count==plan['fixture']['prefix_tokens_count'] and parent.metadata['token_hash']==plan['fixture']['token_hash']
    assert parent.metadata['model_revision']==plan['model']['revision']
    assert parent.layers==28 and parent.h==2 and parent.d==128 and model.config.num_attention_heads==12
    rotary=Rotary(model.model.rotary_emb.inv_freq.detach().float().cpu().numpy())
    child=CompletePageCacheStore(args.output/'overlay',parent.layers,parent.h,parent.d,PAGE,prefix=parent,
        staged_tensor_writes=True,cache_device='cuda',cache_pages=K,sink_tokens=SINK)
    controller=None;adapter=None;captured={}; cursor=0;expected=[]; checks_count=0
    try:
        cfg=Config(id='oracle-trace',W=WINDOW,C=0,R=PAGE,k=K,sink_tokens=SINK,retrieval='centroid',
                   model_revision=plan['model']['revision'],buffered_writes=True)
        controller=FullKeyOracleCache(cfg,child,rotary,'cuda',initialize=False)
        capacity=min(parent.count+768,model.config.max_position_embeddings)
        controller.index=FullKeyOracleIndex(parent,'cuda',capacity,sink=SINK,window=WINDOW,page=PAGE)
        controller._initialize();adapter=ModelAdapter.install(model,controller)
        def trace(layer,pos,q,view):
            nonlocal cursor
            assert cursor<len(expected)
            event=expected[cursor];cursor+=1
            assert event==dict(layer=layer,position=pos,pages=list(view.selected_blocks)), 'routing divergence'
            if pos in probes:
                name=f'p{pos}-l{layer}'
                assert name+'-q' not in captured and name+'-current' not in captured
                pending_pos,current=controller.index.pending[layer];assert pending_pos==pos
                captured[name+'-q']=q[0,:,0,:].detach().cpu().numpy().copy()
                captured[name+'-current']=current.detach().cpu().numpy().copy()
        controller.trace=trace
        prior_last=None
        with ResourceGuard() as guard:
            for turn,row in enumerate(rounds):
                expected=row['selected_events'];cursor=0
                if turn:
                    guard.check();adapter.step(prior_last,adapter.position)
                logits=None
                for token in row['prompt_token_ids']:
                    guard.check();logits=adapter.step(token,adapter.position)
                generated=row['generated_token_ids'];assert generated
                assert int(logits.argmax())==generated[0];checks_count+=1
                for i,token in enumerate(generated[:-1]):
                    guard.check();logits=adapter.step(token,adapter.position)
                    assert int(logits.argmax())==generated[i+1], 'generated token divergence'
                    checks_count+=1
                assert cursor==len(expected), 'missing routing events'
                prior_last=generated[-1]
                print(json.dumps({'phase':'replayed_round','turn':turn,'routing_events':cursor}),flush=True)
            assert weights_digest(model)==WEIGHTS
        assert len(captured)==2*28*len(probes)
        count=controller.position-parent.count;assert 0<count<=768
        for layer in range(parent.layers):
            assert controller.index.history_counts[layer]==controller.position
            captured[f'l{layer}-delta']=controller.index.keys[layer][:,parent.count:controller.position,:].cpu().numpy().copy()
        assert sum(value.nbytes for value in captured.values())<=MAX_ARTIFACT-2**20
        outfile=args.output/'trace.npz'
        with outfile.open('xb') as out:
            np.savez(out,**captured);out.flush();os.fsync(out.fileno())
        assert outfile.stat().st_size<=MAX_ARTIFACT
        metadata={'status':'complete','case':CASE,'positions':probes,'layers':parent.layers,'prefix_count':parent.count,
            'continuation_count':count,'prefix_path':str(args.prefix),'prefix_checkpoint_sha256':CHECKPOINT_SHA,
            'diagnostic_sha256':DIAGNOSTIC_SHA,'baseline_report_sha256':REPORT_SHA,'weights_sha256':WEIGHTS,
            'source_hashes':source,'trace_source_sha256':file_sha(Path(__file__)),
            'oracle_source_sha256':file_sha(ROOT/'research/attention_fullkey_oracle.py'),
            'trace_sha256':file_sha(outfile),'trace_bytes':outfile.stat().st_size,'parity':checks,
            'generated_argmax_checks':checks_count,'routing_events_checked':sum(len(r['selected_events']) for r in rounds),
            'runtime':runtime_identity(),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['original hidden states/Q and intermediate prompt argmaxes were not recorded; token and routing reproduction does not establish their bitwise equality',
                'sparse-oracle replay and full-key GPU history are diagnostic; not bounded retrieval or latency evidence']}
        atomic_json(args.output/'trace.json',metadata)
        print(json.dumps({'status':'complete','trace':str(outfile),'bytes':outfile.stat().st_size}),flush=True)
    finally:
        if controller:controller.trace=None
        if adapter:adapter.uninstall()
        child.prefix=None;child.close();parent.close();gc.collect();device_empty_cache()


if __name__=='__main__':main()

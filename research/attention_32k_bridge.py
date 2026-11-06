"""Cached-centroid matched-prefix answer-recovery diagnostic (32K paraphrase only).

This is a bounded research reference, not a serving implementation. No gold
is used for retrieval. Dense-trajectory tracing and free generation are separate.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from tiered_kv.retrieval import BlockIndex
from research.attention_trajectory_diagnostic import load_plan, validate_source, WEIGHTS, file_sha, EVIDENCE

CASE = 'reasoning-paraphrase-32768-163'
DIAGNOSTIC_SHA = '3059887c1ad8c0fb1e55563599620b3e8555393234294e0d85eb70431f529be9'
CHECKPOINT_SHA = '8e5885af3dedbcfe85f9975d39bdee8372e42f772a313a7df28ffb0b6b24ed5a'
K = 15
PAGE = 128
WINDOW = 2000
SINK = 128


class CompleteColdCentroid(BlockIndex):
    """Original float32 scores, rank all first, then filter complete cold pages."""
    def search(self, layer, q, k):
        ranked, scores = super().search(layer, q, max(1, len(self.counts[layer])))
        keep = np.array([b * PAGE >= SINK and self.counts[layer][b] == PAGE
                         for b in ranked], dtype=bool)
        return ranked[keep][:k], scores[keep][:k]


def filtered_text_blocks(blocks, position):
    """Keep only complete pages outside sink and the current recent window."""
    cutoff = max(0, position - WINDOW + 1)
    return [int(b) for b in blocks if b * PAGE >= SINK and (b + 1) * PAGE <= cutoff][:K]


def timed(fn):
    from tiered_kv.generate import sync
    sync(); start=time.perf_counter(); result=fn(); sync()
    return result, time.perf_counter()-start


def detach_session(session, child, adapter):
    """Break trace cycles and retain ownership of the shared parent archive."""
    if adapter:
        session.controller.trace=None
        adapter.uninstall()
    if child:
        child.prefix=None
        child.close()


def persist_stage(path, value):
    from tiered_kv.store import atomic_json
    atomic_json(path, value)


def persist_partial_row(path, row):
    encoded=json.dumps({'status':'partial','case':CASE,'row':row},sort_keys=True,allow_nan=False)+'\n'
    with path.open('a') as out:
        out.write(encoded)
        out.flush()
        os.fsync(out.fileno())


def progress(phase, **details):
    print(json.dumps({'phase':phase, **details},sort_keys=True),flush=True)


def compare_historical_round(observed, prior, fixed_text=False):
    matched=observed['prompt_token_ids']==prior['prompt_token_ids']
    comparison=dict(prompt_tokens_match=matched,
        generated_token_ids_match=(observed['generated_token_ids']==prior['generated_token_ids']) if matched else None,
        answer_match=(observed['answer']==prior['answer']) if matched else None)
    if fixed_text:
        old_pages=prior['selected_blocks'];new_pages=observed['selected_text_pages']
        comparison.update(historical_selected_pages=old_pages,current_selected_pages=new_pages,
            historical_page_zero_already_fixed_sink=(0 in old_pages),
            note='Page zero is already held as sink; page IDs alone do not prove exact same KV coverage.')
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',required=True,type=Path)
    parser.add_argument('--diagnostic',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();plan=load_plan(CASE); source=validate_source()
    assert file_sha(args.diagnostic)==DIAGNOSTIC_SHA
    prior=json.loads(args.diagnostic.read_text())
    assert prior['case']==CASE and prior['gate']=='all K/V bytes and all prompt logits matched'
    assert prior['weights_sha256']==WEIGHTS and prior['prefix_checkpoint_sha256']==CHECKPOINT_SHA
    assert file_sha(args.prefix/'checkpoint.json')==CHECKPOINT_SHA
    historical=[json.loads(line) for line in (EVIDENCE/'samples.jsonl').read_text().splitlines()]
    previous={r['policy']['attention']:r for r in historical if r['example_id']==CASE and r['purpose']=='quality'
              and r['policy']['id'] in ('stock-json-r2-cap128','sparse-json-r2-cap128')}
    assert set(previous)=={'stock','sparse'}
    assert len(plan['text_cold_pages'])==K
    if not args.execute:
        print(json.dumps(dict(case=CASE, prefix=str(args.prefix), checkpoint_sha256=CHECKPOINT_SHA,
                              diagnostic_sha256=DIAGNOSTIC_SHA, max_cold_pages=K, source_hashes=source)))
        return
    assert torch.cuda.is_available() and os.environ.get('HF_HUB_OFFLINE')=='1'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert shutil.disk_usage(args.output.parent).free>=12*2**30
    assert torch.cuda.mem_get_info()[0]>=8*2**30
    args.output.mkdir(parents=True,exist_ok=False)
    progress('model_load_and_short_parity')
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from research.budget_probe import expand_blocks
    from research.persistent_kv import next_question_tokens
    from research.reasoning_protocol import next_search, parse_response, round_request
    from research.reasoning_sessions import BatchedStockSession, CountedSession
    from research.resources import HostObservation, logical_store_reads
    from research.semantic_index import metadata_index
    from tiered_kv.adapter import DenseController, ModelAdapter
    from tiered_kv.cache import TieredCache
    from tiered_kv.config import Config
    from tiered_kv.generate import decode_greedy, sync, weights_digest
    from tiered_kv.store import ColdStore
    from research.attention_page_cache import CompletePageCacheStore
    from tiered_kv.telemetry import ResourceGuard, device_empty_cache, runtime_identity
    from tiered_kv.types import Rotary, digest
    from tiered_kv.validation import parity
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    snapshot=snapshot_download(plan['model']['model_id'],revision=plan['model']['revision'],local_files_only=True)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':'cuda'},attn_implementation='sdpa').eval().requires_grad_(False)
    model.config._commit_hash=plan['model']['revision']; assert weights_digest(model)==WEIGHTS
    checks=[parity(model,n) for n in (64,512)]
    assert all(all(c['greedy_agreement']) and max(c['max_errors'])<=1e-4 for c in checks)
    rotary=Rotary(model.model.rotary_emb.inv_freq.detach().float().cpu().numpy())
    ex=plan['fixture']; prompt=plan['prompt']
    (parent, mount_s)=timed(lambda:ColdStore.open(args.prefix,readonly=True))
    assert parent.count==ex['prefix_tokens_count'] and parent.metadata['token_hash']==ex['token_hash']
    assert parent.metadata['model_revision']==plan['model']['revision']
    (index,index_s)=timed(lambda:metadata_index(tok,ex,args.output,32))
    def lookup(query,position):
        seeds=index.search(query,None,4,'lexical_rare')
        expanded=expand_blocks(tuple(seeds),(ex['prefix_tokens_count']+127)//128,16) if seeds else []
        return filtered_text_blocks(expanded,position),list(seeds),list(expanded)
    first,first_seeds,first_raw=lookup(ex['query'],parent.count)
    assert first==plan['text_cold_pages'] and first_raw==plan['original_text_pages']
    assert tok.encode(round_request(ex['query'],0,2)+ex['first_suffix'],add_special_tokens=False)==prompt
    assert len(prompt)*model.config.vocab_size*4 <= 128*2**20
    eos=model.generation_config.eos_token_id; eos=set(eos if isinstance(eos,list) else [eos])
    counter=0
    def make_session(arm,guard):
        nonlocal counter
        counter+=1
        if arm=='stock':
            return CountedSession(BatchedStockSession(model,parent),model),None,None
        cfg=Config(id=arm,W=WINDOW,C=0,R=PAGE,k=K,sink_tokens=SINK,
                   retrieval='fixed' if arm=='fixedtext' else 'centroid',
                   model_revision=plan['model']['revision'],buffered_writes=True)
        store_type=CompletePageCacheStore if arm=='centroid' else ColdStore
        cache_args=dict(cache_device='cuda',cache_pages=K,sink_tokens=SINK) if arm=='centroid' else {}
        child=store_type(args.output/f'overlay-{counter}',parent.layers,parent.h,parent.d,PAGE,
                         prefix=parent,staged_tensor_writes=True,**cache_args)
        original_mask=model.model._update_causal_mask
        original_forwards=[layer.self_attn.forward for layer in model.model.layers]
        adapter=None
        controller=None
        try:
            controller=TieredCache(cfg,child,rotary,'cuda',initialize=False)
            if arm in ('centroid','centroid_uncached'): controller.index=CompleteColdCentroid(parent.layers,parent.h,parent.d,PAGE)
            controller._initialize()
            adapter=ModelAdapter.install(model,controller)
            session=CountedSession(adapter,model,active_limit=WINDOW+K*PAGE+SINK+1)
            counted_trace=controller.trace
            session.routing_stats={}
            def trace(layer,pos,q,view):
                counted_trace(layer,pos,q,view)
                state=session.routing_stats
                count=len(view.selected_blocks); retrieved=view.tiers.count('retrieved')
                assert 0<=count<=K and retrieved==count*PAGE
                assert all(b>=1 and (b+1)*PAGE<=max(0,pos-WINDOW+1) for b in view.selected_blocks)
                for key,value in [('page_count_histogram',count),('active_token_histogram',len(view.ids))]:
                    histogram=state.setdefault(key,{})
                    histogram[str(value)]=histogram.get(str(value),0)+1
                layers=state.setdefault('layers',{})
                entry=layers.setdefault(str(layer),{'first':list(view.selected_blocks),'last':None,'calls':0})
                entry['last']=list(view.selected_blocks);entry['calls']+=1
            controller.trace=trace
            return session,child,adapter
        except BaseException:
            if controller: controller.trace=None
            if adapter:
                adapter.uninstall()
            else:
                model.model._update_causal_mask=original_mask
                for layer,forward in zip(model.model.layers,original_forwards):
                    layer.self_attn.forward=forward
                if hasattr(model,'_tiered_adapter'):
                    del model._tiered_adapter
            child.prefix=None
            child.close()
            raise
    with ResourceGuard() as guard:
        # Native stock reference: full per-position logits stay only in bounded RAM.
        stock,_,_=make_session('stock',guard); native=[]
        for token in prompt:
            guard.check();native.append(stock.step(token,stock.position).detach().float().cpu())
        del stock;gc.collect();device_empty_cache()
        # Independently gate all archived K/V and every matched-prompt logit.
        dense=DenseController(parent.layers);dense.initialize(parent,'cuda',torch.float16)
        adapter=ModelAdapter.install(model,dense); max_error=0.0
        try:
            for layer in range(parent.layers):
                for start in range(0,parent.count,PAGE):
                    end=min(start+PAGE,parent.count);b=parent.read_range(layer,start,end)
                    assert torch.equal(dense.keys[layer][:,:,start:end].cpu(),b.k)
                    assert torch.equal(dense.values[layer][:,:,start:end].cpu(),b.v)
            for i,token in enumerate(prompt):
                actual=adapter.step(token,adapter.position).detach().float().cpu()
                max_error=max(max_error,float((actual-native[i]).abs().max()))
                assert torch.allclose(actual,native[i],atol=1e-4,rtol=1e-4)
        finally:
            adapter.uninstall();del adapter,dense;gc.collect();device_empty_cache()
        persist_stage(args.output/'validation.json',dict(status='partial',case=CASE,
            checkpoint_sha256=CHECKPOINT_SHA,diagnostic_sha256=DIAGNOSTIC_SHA,
            weights_sha256_before=WEIGHTS,source_hashes=source,bridge_sha256=file_sha(Path(__file__)),
            runtime=runtime_identity(),parity_checks=checks,all_kv_prompt_max_abs=max_error,
            prefix_prefill_s=parent.metadata['prefill_s'],archive_checksum_mount_s=mount_s,
            text_index_build_or_open_s=index_s,first_text_seeds=first_seeds))
        progress('validation_passed')
        teacher={}
        sparse_reference=[]
        for arm in ('centroid_uncached','centroid'):
            progress('teacher_forced_start',arm=arm)
            session,child,adapter=make_session(arm,guard)
            try:
                if arm=='fixedtext':session.controller.select_blocks(first)
                first_difference=None; max_abs=0.0
                for i,token in enumerate(prompt):
                    guard.check();actual=session.step(token,session.position).detach().float().cpu()
                    if arm=='centroid_uncached': sparse_reference.append(actual)
                    else: assert torch.equal(actual,sparse_reference[i]), 'cached sparse teacher logits differ'
                    err=float((actual-native[i]).abs().max());max_abs=max(max_abs,err)
                    if first_difference is None and int(actual.argmax())!=int(native[i].argmax()):
                        first_difference=dict(prompt_index=i,position=parent.count+i,
                                              stock_top=int(native[i].argmax()),sparse_top=int(actual.argmax()),max_abs=err)
                teacher[arm]=dict(first_argmax_difference=first_difference,max_abs_logit_error=max_abs,
                                  scope='teacher-forced first prompt; separate from free generation')
            finally:
                detach_session(session,child,adapter)
                session=child=adapter=None
                gc.collect();device_empty_cache()
            persist_stage(args.output/'teacher.json',dict(status='partial',case=CASE,arms=teacher))
            progress('teacher_forced_complete',arm=arm)
        del native,sparse_reference;gc.collect();device_empty_cache()
        rows=[]
        for arm in ('stock','fixedtext','centroid'):
            progress('arm_start',arm=arm)
            session=child=adapter=None
            with HostObservation(scope='initialization, both complete rounds and checkpoint') as host:
                try:
                    before_init_reads=logical_store_reads(parent)
                    (triple,init_s)=timed(lambda:make_session(arm,guard));session,child,adapter=triple
                    read_start=logical_store_reads(child if child else parent)
                    init_logical_reads=read_start-before_init_reads
                    rounds=[];prev='';last=None
                    sync();request_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                    for turn in range(2):
                        progress('round_start',arm=arm,turn=turn)
                        if child: session.routing_stats={}
                        if session.controller.pending_token is not None:session.consume_pending()
                        query,query_detail=next_search(ex['query'],prev,tok)
                        selected=[];seeds=[];raw=[];lookup_s=0.0
                        if arm=='fixedtext':
                            ((selected,seeds,raw),lookup_s)=timed(lambda:lookup(query,session.position))
                            session.controller.select_blocks(selected)
                            if turn==0:assert selected==first
                        text=round_request(ex['query'],turn,2)
                        tokens=tok.encode(text+ex['first_suffix'],add_special_tokens=False) if turn==0 else next_question_tokens(tok,text,last,eos)
                        assert session.position+len(tokens)+128<=min(ex['budget'],model.config.max_position_embeddings)
                        (logits,prompt_s)=timed(lambda:session.feed_prompt(tokens,guard))
                        ((generated,decode_ms),generation_s)=timed(lambda:decode_greedy(session,logits,128,eos,guard=guard))
                        prev=tok.decode(generated,skip_special_tokens=True);last=generated[-1];parsed=parse_response(prev)
                        rounds.append(dict(turn=turn,query=query,query_details=query_detail,prompt_tokens=len(tokens),prompt_token_ids=tokens,
                            selected_text_pages=selected,text_seed_pages=seeds,text_expanded_pages=raw,
                            generated_token_ids=generated,response=prev,parsed_response=parsed,
                            answer=parsed['answer'] if parsed else None,lookup_s=lookup_s,prompt_s=prompt_s,
                            generation_s=generation_s,decode_ms=decode_ms,work=session.metrics(),
                            routing_stats=dict(session.routing_stats) if child else None))
                    sync();request_s=time.perf_counter()-request_start
                    peak=torch.cuda.max_memory_allocated()
                    logical_reads=logical_store_reads(child if child else parent)-read_start
                    _,checkpoint_s=timed(lambda:session.controller.checkpoint(pending_token=session.controller.pending_token,token_hash=digest(ex)) if child else None)
                    row=dict(arm=arm,rounds=rounds,init_s=init_s,request_s=request_s,checkpoint_s=checkpoint_s,
                             logical_initialization_read_bytes=init_logical_reads,logical_request_read_bytes=logical_reads,logical_overlay_write_bytes=child.write_bytes if child else 0,
                             cuda_peak_allocated_bytes=peak,work=session.metrics(),
                             index_resident_bytes=child and session.controller.index.nbytes or 0,
                             index_searches_by_layer=list(session.controller.index.searches) if child else None,
                             page_cache_stats=child.stats if isinstance(child,CompletePageCacheStore) else None,
                             cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                             persistent_cache_per_layer_bound_bytes=K*PAGE*2*parent.h*parent.d*2 if isinstance(child,CompletePageCacheStore) else 0,
                             persistent_cache_bound_bytes=parent.layers*K*PAGE*2*parent.h*parent.d*2 if isinstance(child,CompletePageCacheStore) else 0,
                             peak_persistent_cache_bytes=child.peak_resident_pages_total*PAGE*2*parent.h*parent.d*2 if isinstance(child,CompletePageCacheStore) else 0)
                finally:
                    if session:detach_session(session,child,adapter)
                    session=child=adapter=triple=None
                    gc.collect();device_empty_cache()
            row['host']=host.metrics
            row['total_prompt_s']=sum(item['prompt_s'] for item in row['rounds'])
            row['total_generation_s']=sum(item['generation_s'] for item in row['rounds'])
            row['accounted_shared_prefix_plus_request_s']=(parent.metadata['prefill_s']+mount_s+init_s+request_s+checkpoint_s+(index_s if arm=='fixedtext' else 0))
            row['accounted_scope']='recorded prior prefix prefill + this run checksum mount + initialization + two-round request + checkpoint + text index build/open for fixed text; not a fresh end-to-end wall timer'
            for turn in row['rounds']:turn['exact_correct']=turn['answer']==ex['answer']
            row['exact_correct']=row['rounds'][-1]['exact_correct']
            if arm in ('stock','fixedtext'):
                old=previous['stock' if arm=='stock' else 'sparse']
                row['historical_round_comparison']=[]
                for observed,prior_round in zip(row['rounds'],old['rounds']):
                    row['historical_round_comparison'].append(compare_historical_round(observed,prior_round,fixed_text=arm=='fixedtext'))
            rows.append(row)
            persist_partial_row(args.output/'samples.jsonl',row)
            progress('arm_complete',arm=arm,completed_arms=len(rows))
        assert weights_digest(model)==WEIGHTS
    index.close();parent.close()
    report=dict(case=CASE,checkpoint_sha256=CHECKPOINT_SHA,diagnostic_sha256=DIAGNOSTIC_SHA,
        prefix_prefill_s=parent.metadata['prefill_s'],archive_checksum_mount_s=mount_s,text_index_build_or_open_s=index_s,text_index_database_bytes=(args.output/(ex['id']+'-metadata-32.sqlite3')).stat().st_size,
        parity_checks=checks,all_kv_prompt_max_abs=max_error,first_text_seeds=first_seeds,
        source_hashes=source,bridge_sha256=file_sha(Path(__file__)),weights_sha256=WEIGHTS,
        runtime=runtime_identity(),teacher_forced=teacher,teacher_sparse_exact_match=True,historical_samples_sha256=plan['source_files']['samples.jsonl'],cache_source_sha256=file_sha(Path(__file__).with_name('attention_page_cache.py')),rows=rows,
        limitations=['one correlated case; no aggregate quality conclusion',
            'cached centroid retains extra GPU K/V pages and temporary concatenation in addition to the same active attention positions; comparisons to prior process have uncontrolled cache and timing conditions',
            'centroid uses a linear full-metadata scan per layer/token; filtering changes original availability',
            'cache/page-cache conditions uncontrolled; single observations are not benchmark distributions',
            'logical read bytes exclude checksum scans and do not measure physical I/O',
            'dense prefill and text index preparation reported separately; no amortization assumed'])
    encoded=(json.dumps(report,indent=2)+'\n').encode();assert len(encoded)<=8*2**20
    persist_stage(args.output/'bridge.json',report)
    assert file_sha(args.output/'bridge.json')
    persist_stage(args.output/'complete.json',dict(status='complete',case=CASE,rows=len(rows),weights_unchanged=True,report_sha256=file_sha(args.output/'bridge.json'),samples_sha256=file_sha(args.output/'samples.jsonl')))
    print(json.dumps(dict(status='complete',case=CASE,output=str(args.output/'bridge.json'),bytes=len(encoded))))


if __name__=='__main__':main()

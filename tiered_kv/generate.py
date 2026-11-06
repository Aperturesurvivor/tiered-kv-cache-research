import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
import torch
from .adapter import ModelAdapter, DenseController
from .cache import TieredCache
from .config import MODEL_PATH, MODEL_REVISION
from .types import Rotary, digest
from .store import ColdStore, atomic_json, file_sha
from .telemetry import ResourceGuard, ResourceAbort, disk_preflight, DEVICE, device_synchronize, device_empty_cache
from .evals import HitTracker, score_needle, score_rouge


def sync():
    device_synchronize()


def empty_cache():
    device_empty_cache()


def append_jsonl(path,row):
    encoded=json.dumps(row,sort_keys=True,allow_nan=False)+'\n'
    with Path(path).open('a') as out:
        out.write(encoded)
        out.flush()
        os.fsync(out.fileno())


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if DEVICE=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
        raise RuntimeError('MPS CPU fallback must be disabled')
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,trust_remote_code=False,
           torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':DEVICE},attn_implementation='sdpa').eval().requires_grad_(False)
    if any(p.device.type!=DEVICE or p.dtype!=torch.float16 for p in model.parameters()):
        raise RuntimeError('retained CPU / non-fp16 model weights')
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,trust_remote_code=False)
    return model,tok


def weights_digest(model):
    hasher=hashlib.sha256()
    for name,p in model.named_parameters():
        hasher.update(name.encode())
        flat=p.detach().flatten()
        for start in range(0,flat.numel(),1024*1024):
            hasher.update(flat[start:start+1024*1024].cpu().numpy().tobytes())
    return hasher.hexdigest()


def decode_greedy(adapter,logits,cap,eos,synchronize=sync,guard=None,tracker=None,tokenizer=None):
    tokens=[]
    times=[]
    token=int(logits.argmax())
    for step in range(cap):
        tokens.append(token)
        adapter.controller.pending_token=token
        if token in eos or step==cap-1:
            break
        if guard:
            guard.check()
        synchronize()
        start=time.perf_counter()
        logits=adapter.consume_pending()
        token=int(logits.argmax())
        synchronize()
        times.append((time.perf_counter()-start)*1000)
    if tracker and tokenizer:
        tracker.finish_output(tokens,tokenizer)
    return tokens,times


class AttentionDiagnostic:
    """CPU recomputation outside headline timing; only aggregate rows retained."""
    def __init__(self,example,rotary):
        self.pos=len(example['tokens'])-1
        self.gold=set(range(*example.get('answer_span',[0,0])))
        self.rotary=rotary
        self.rows=[]

    @staticmethod
    def distribution(q,k,v):
        groups=q.shape[0]//k.shape[1]
        keys=np.repeat(k,groups,axis=1)
        values=np.repeat(v,groups,axis=1)
        scores=np.einsum('hd,shd->hs',q,keys)/math.sqrt(q.shape[-1])
        scores-=scores.max(-1,keepdims=True)
        probabilities=np.exp(scores)
        probabilities/=probabilities.sum(-1,keepdims=True)
        output=np.einsum('hs,shd->hd',probabilities,values)
        return probabilities,output

    def __call__(self,layer,pos,q,view):
        if pos!=self.pos:
            return
        queries=q[0,:,0].detach().float().cpu().numpy()
        keys=view.k[0].detach().float().cpu().numpy().transpose(1,0,2)
        values=view.v[0].detach().float().cpu().numpy().transpose(1,0,2)
        probs,_=self.distribution(queries,keys,values)
        labels=np.asarray(view.tiers)
        masses={tier:float(probs[:,labels==tier].sum(-1).mean()) for tier in ('compressed','retrieved','recent','sink')}
        gold=np.array([int(i) in self.gold and t!='compressed' for i,t in zip(view.ids,view.tiers)])
        norms={tier:np.percentile(np.linalg.norm(keys[labels==tier],axis=-1),[0,25,50,75,100]).tolist() if (labels==tier).any() else None for tier in masses}
        exact=np.flatnonzero(labels!='compressed')
        exact=exact[np.argsort(view.ids[exact])]
        ids=view.ids[exact]
        k,v=keys[exact],values[exact]
        original,original_out=self.distribution(queries,k,v)
        unrotated=self.rotary.apply(k,ids,inverse=True)
        compact_k=self.rotary.apply(unrotated,np.arange(len(ids)))
        compact_q=self.rotary.apply(self.rotary.apply(queries[None],np.array([pos]),inverse=True),np.array([len(ids)-1]))[0]
        compact,compact_out=self.distribution(compact_q,compact_k,v)
        gold_exact=np.array([int(i) in self.gold for i in ids])
        self.rows.append({'layer':layer,'pos':pos,'tier_mass':masses,'gold_mass':float(probs[:,gold].sum(-1).mean()),
             'entropy':float(-(probs*np.log(np.maximum(probs,1e-30))).sum(-1).mean()),'key_norm_percentiles':norms,
             'gap_relative_error':float(np.linalg.norm(compact_out-original_out)/max(np.linalg.norm(original_out),1e-8)),
             'gap_gold_mass_change':float(compact[:,gold_exact].sum(-1).mean()-original[:,gold_exact].sum(-1).mean()),
             'gap_size':int(pos-len(ids)+1),'exact_entries':len(ids)})


class RunEngine:
    def __init__(self,model,tokenizer,run_root,guard):
        self.model,self.tokenizer,self.root,self.guard=model,tokenizer,Path(run_root),guard
        self.scratch=Path('artifacts/scratch')/self.root.name
        self.scratch.mkdir(parents=True,exist_ok=True)
        inv=model.model.rotary_emb.inv_freq.detach().float().cpu().numpy()
        self.rotary=Rotary(inv)
        self.layers=model.config.num_hidden_layers
        self.h=model.config.num_key_value_heads
        self.d=getattr(model.config,'head_dim',model.config.hidden_size//model.config.num_attention_heads)
        self.model_revision=getattr(model.config,'_commit_hash',None) or MODEL_REVISION
        self.adapter=ModelAdapter.install(model,DenseController(self.layers))

    def prefix(self,example):
        disk_preflight()
        root=self.scratch/(example['id']+'-prefix')
        prefix_count=example.get('prefix_tokens_count',len(example['tokens'])-1)
        if not 0<prefix_count<len(example['tokens']):
            raise ValueError('invalid archive/query boundary')
        if (root/'checkpoint.json').exists():
            store=ColdStore.open(root,readonly=True)
            meta=store.metadata
            if meta.get('token_hash')!=example['token_hash'] or meta.get('model_revision')!=self.model_revision or store.count!=prefix_count:
                store.close()
                raise ValueError('shared prefix identity mismatch')
            store.close()
            return root,meta['prefill_s']
        # Never overwrite an interrupted uncommitted prefix. Keep it for audit.
        if root.exists():
            root=root.with_name(root.name+'-'+uuid.uuid4().hex[:8])
        store=ColdStore(root,self.layers,self.h,self.d)
        controller=DenseController(self.layers,store)
        self.adapter.set_controller(controller)
        sync()
        start=time.perf_counter()
        prefix_tokens=example['tokens'][:prefix_count]
        try:
            gc.collect()
            empty_cache()
            for offset in range(0,len(prefix_tokens),128):
                self.guard.check()
                self.adapter.forward_tokens(prefix_tokens[offset:offset+128],offset)
            sync()
            seconds=time.perf_counter()-start
            store.checkpoint({'token_hash':example['token_hash'],'model_revision':self.model_revision,'prefill_s':seconds,'chunk':128})
        finally:
            self.adapter.set_controller(DenseController(self.layers))
            controller.keys.clear()
            controller.values.clear()
            store.close()
            gc.collect()
            empty_cache()
        return root,seconds

    def run(self,config,example,protocol='exact_shared',prefix_info=None,diagnostic=False,oracle=False):
        record={'config_id':config.id,'config_hash':config.digest,'example_id':example['id'],'task':example['task'],
             'protocol':protocol,'prompt_length':len(example['tokens']),'budget':example['budget'],
             **{k:example[k] for k in ('depth','band','trial') if k in example}}
        store=None
        try:
            if config.model_revision!=self.model_revision:
                raise ValueError('cache policy revision differs from loaded model')
            disk_preflight()
            self.guard.check()
            sync()
            begin=time.perf_counter()
            if protocol not in ('exact_shared','online_tiered','query_replay'):
                raise ValueError('unknown execution protocol')
            if protocol=='online_tiered' and prefix_info is not None:
                raise ValueError('online execution cannot reuse a dense prefix')
            if protocol!='online_tiered':
                prefix_root,prefill_s=prefix_info or self.prefix(example)
            init_start=time.perf_counter()
            parent=ColdStore.open(prefix_root,readonly=True) if protocol!='online_tiered' else None
            mount_s=time.perf_counter()-init_start
            query_start=(example['prefix_tokens_count'] if protocol=='query_replay' or (protocol=='online_tiered' and 'prefix_tokens_count' in example) else len(example['tokens'])-1)
            if parent and parent.count!=query_start:
                raise ValueError('prefix does not end at the declared query boundary')
            prefix_read_start=parent.read_bytes if parent else 0
            # Fresh overlays on retries; preserve prior failed tails.
            overlay=self.scratch/f"{example['id']}-{config.id}-{uuid.uuid4().hex[:8]}"
            store=ColdStore(overlay,self.layers,self.h,self.d,config.R,prefix=parent,staged_tensor_writes=config.buffered_writes)
            if config.full:
                controller=DenseController(self.layers,store)
                if parent:
                    controller.initialize(parent,DEVICE,torch.float16)
            else:
                controller=TieredCache(config,store,self.rotary,DEVICE)
            self.adapter.set_controller(controller)
            sync()
            init_s=time.perf_counter()-init_start
            if protocol=='online_tiered':
                # A future question must not influence archive ingestion. Fixed
                # address retrieval activates only when that question arrives.
                deferred_blocks=None
                if isinstance(controller,TieredCache) and config.retrieval=='fixed':
                    deferred_blocks=controller.index.blocks
                    controller.index.blocks=np.array([],dtype=np.int64)
                if DEVICE=='cuda':
                    torch.cuda.reset_peak_memory_stats()
                ingest_start=time.perf_counter()
                for pos,token in enumerate(example['tokens'][:query_start]):
                    self.guard.check()
                    self.adapter.step(token,pos)
                sync()
                prefill_s=time.perf_counter()-ingest_start
                if deferred_blocks is not None:
                    controller.index.blocks=deferred_blocks
                    controller.retrieved_cache=[None]*self.layers
                    record['retrieval_activation_position']=query_start
                if DEVICE=='cuda':
                    record['cuda_online_ingestion_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                record['ingestion_schedule']='one token at a time from empty state; no dense prefix'
            tracker=HitTracker(example,config.W,config.k,config.full)
            observer=AttentionDiagnostic(example,self.rotary) if diagnostic else None
            def observe(layer,pos,q,view):
                tracker(layer,pos,q,view)
                if observer:
                    observer(layer,pos,q,view)
            if example['task']=='niah':
                controller.trace=observe
            if oracle and not config.full:
                controller.oracle_blocks=example['gold_blocks']
            sync()
            # Exclude dense shared prefill and previous configurations from this
            # policy's CUDA allocation peak. The one-second guard remains global.
            if DEVICE=='cuda':
                torch.cuda.reset_peak_memory_stats()
                record['cuda_decode_start_allocated_bytes']=torch.cuda.memory_allocated()
            query_timer_start=time.perf_counter()
            for pos in range(query_start,len(example['tokens'])):
                self.guard.check()
                logits=self.adapter.step(example['tokens'][pos],pos)
            first=int(logits.argmax())
            sync()
            query_s=time.perf_counter()-query_timer_start
            record.update(prefill_s=prefill_s,prefill_tokens_s=query_start/prefill_s,init_s=init_s,
                          archive_verify_open_s=mount_s,controller_init_s=init_s-mount_s,
                          archive_tokens=query_start,query_tokens=len(example['tokens'])-query_start,
                          ttft_s=prefill_s+init_s+query_s,query_s=query_s)
            if example['task']=='latency':
                latency=[]
                all_latency=[]
                selected=[]
                for i,token in enumerate(example['continuation']):
                    self.guard.check()
                    sync()
                    start=time.perf_counter()
                    logits=self.adapter.step(token,self.adapter.position)
                    selected.append(int(logits.argmax()))
                    sync()
                    elapsed=(time.perf_counter()-start)*1000
                    all_latency.append(elapsed)
                    if i>=16:
                        latency.append(elapsed)
                tokens=example['continuation']
                record.update(latency_ms=latency,all_latency_ms=all_latency,warmup_calls=16,measured_calls=len(latency),selected_hash=digest(selected))
                pending=None
            else:
                eos=set(self.model.generation_config.eos_token_id if isinstance(self.model.generation_config.eos_token_id,list) else [self.model.generation_config.eos_token_id])
                tokens,decode_times=decode_greedy(self.adapter,logits,example['max_new_tokens'],eos,guard=self.guard,tracker=tracker,tokenizer=self.tokenizer)
                prediction=self.tokenizer.decode(tokens,skip_special_tokens=True)
                pending=controller.pending_token
                record.update(prediction=prediction,references=example['references'],output_tokens=len(tokens),decode_ms=decode_times)
                if example['task']=='niah':
                    record.update(score_needle(prediction,example['answer']),retrieval=tracker.result())
                else:
                    rouge,error=score_rouge(prediction,example['references'])
                    record.update(rougeL=rouge,scorer_error=error)
            sync()
            if DEVICE=='cuda':
                record.update(cuda_decode_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              cuda_decode_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              cuda_memory_scope='final prompt query and decode; includes weights, active KV and workspace; excludes prefill and checkpoint')
            checkpoint_start=time.perf_counter()
            if isinstance(controller,TieredCache):
                controller.checkpoint(pending_token=pending,token_hash=example['token_hash'])
                if any(n!=controller.present_calls[0] for n in controller.present_calls):
                    raise ValueError('not every layer presented')
                if config.k and controller.index.searches!=controller.present_calls:
                    raise ValueError('retrieval did not fire every layer/call')
            else:
                store.checkpoint({'config_hash':config.digest,'model_revision':self.model_revision,'token_hash':example['token_hash'],'pending_token':pending,'next_position':controller.position})
            sync()
            record['checkpoint_s']=time.perf_counter()-checkpoint_start
            self.guard.check()
            active_counts=controller.max_entries_by_layer if isinstance(controller,TieredCache) else [controller.max_entries]*self.layers
            record.update(status='ok',token_ids=tokens,elapsed_s=time.perf_counter()-begin,
                 active_KV_entries=controller.max_entries,active_KV_entries_by_layer=active_counts,
                 active_KV_bytes=sum(active_counts)*2*self.h*self.d*2,
                 active_KV_scope='logical sum of per-layer view peaks; physical allocation reported separately',
                 index_bytes=controller.index.nbytes if isinstance(controller,TieredCache) and config.k else 0,
                 logical_read_bytes=store.read_bytes+(parent.read_bytes-prefix_read_start if parent else 0),logical_write_bytes=store.write_bytes,
                 cold_allocated_bytes=store.allocated_bytes+(parent.allocated_bytes if parent else 0),overlay=str(overlay),
                 shared_prefix_logical_bytes=parent.count*self.layers*parent.record_bytes if parent else 0,
                 present_calls=controller.present_calls if isinstance(controller,TieredCache) else None,
                 search_calls=controller.index.searches if isinstance(controller,TieredCache) else None,
                 **self.guard.metrics())
            if observer:
                record['attention_diagnostics']=observer.rows
                record['oracle']=oracle
                record['partial_oracle']=len(example['gold_blocks'])>config.k
        except Exception as exc:
            record.update(status='resource_failure' if isinstance(exc,ResourceAbort) else 'failure',failure_reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        finally:
            self.adapter.set_controller(DenseController(self.layers))
            if store:
                store.close()
            gc.collect()
            empty_cache()
        return record


def online_smoke(model,tokenizer,root,configs,guard,tokens_count=2048,checkpoint_pos=1024):
    """Validate every supplied policy; resume consumes the pending token once."""
    from dataclasses import replace
    from .evals import exact_prompt
    if not 0<checkpoint_pos<tokens_count:
        raise ValueError('checkpoint must be inside online sequence')
    _,encoded=exact_prompt(tokenizer,tokens_count,0,.5,41,question='Remember the archive and await the next conversation update.')
    tokens=encoded['input_ids']
    atomic_json(Path(root)/'online-tokens.json',{'tokens':tokens,'hash':digest(tokens),'checkpoint':checkpoint_pos})
    engine=RunEngine(model,tokenizer,root,guard)
    rows=[]
    for original in configs:
        config=replace(original,protocol='online_tiered')
        disk_preflight()
        row={'config_id':config.id,'config_hash':config.digest,'protocol':'online_tiered','processed_tokens':0}
        store=None
        try:
            run_dir=engine.scratch/f'online-{config.id}-{uuid.uuid4().hex[:8]}'
            store=ColdStore(run_dir,engine.layers,engine.h,engine.d,config.R,staged_tensor_writes=config.buffered_writes)
            controller=DenseController(engine.layers,store) if config.full else TieredCache(config,store,engine.rotary,DEVICE)
            engine.adapter.set_controller(controller)
            execution_tokens=tokens.copy()
            sync()
            start=time.perf_counter()
            for pos in range(len(execution_tokens)):
                guard.check()
                logits=engine.adapter.step(execution_tokens[pos],pos)
                if pos==checkpoint_pos-1:
                    # The checkpoint carries an actually emitted model token.
                    # Resume consumes it before the next conversation input.
                    execution_tokens[checkpoint_pos]=int(logits.argmax())
                    if config.full:
                        store.checkpoint({'config_hash':config.digest,'token_hash':digest(execution_tokens),'pending_token':execution_tokens[checkpoint_pos],'next_position':checkpoint_pos})
                    else:
                        controller.checkpoint(pending_token=execution_tokens[checkpoint_pos],token_hash=digest(execution_tokens))
            sync()
            uninterrupted_s=time.perf_counter()-start
            expected=int(logits.argmax())
            expected_logits=logits.detach().cpu()
            # Hash all 2K exact KV before reopening the older committed extent.
            exact_hashes=[]
            expected_kv=[]
            for layer in range(engine.layers):
                batch=store.read_range(layer,0,tokens_count)
                expected_kv.append(batch)
                exact_hashes.append(hashlib.sha256(batch.k.numpy().tobytes()+batch.v.numpy().tobytes()).hexdigest())
            store.close()
            if config.full:
                store=ColdStore.open(run_dir)
                restored=DenseController(engine.layers,store)
                restored.initialize(store,DEVICE,torch.float16)
                restored.pending_token=store.metadata['pending_token']
            else:
                restored=TieredCache.reopen(run_dir,config,engine.rotary,DEVICE)
                store=restored.store
                restored.verify_rebuild()
            engine.adapter.set_controller(restored)
            if restored.position!=checkpoint_pos:
                raise ValueError('online checkpoint wrong next position')
            sync()
            resume_start=time.perf_counter()
            logits=engine.adapter.consume_pending()
            for pos in range(checkpoint_pos+1,tokens_count):
                guard.check()
                logits=engine.adapter.step(execution_tokens[pos],pos)
            sync()
            resume_s=time.perf_counter()-resume_start
            max_error=float((logits.detach().cpu()-expected_logits).abs().max())
            if int(logits.argmax())!=expected or not torch.allclose(logits.detach().cpu(),expected_logits,atol=1e-2,rtol=1e-2):
                raise ValueError(f'online resumed logits mismatch {max_error}')
            resumed_hashes=[]
            kv_error=0.
            for layer in range(engine.layers):
                batch=store.read_range(layer,0,tokens_count)
                resumed_hashes.append(hashlib.sha256(batch.k.numpy().tobytes()+batch.v.numpy().tobytes()).hexdigest())
                for actual,want in ((batch.k,expected_kv[layer].k),(batch.v,expected_kv[layer].v)):
                    kv_error=max(kv_error,float((actual-want).abs().max()))
                    if not torch.allclose(actual,want,atol=1e-2,rtol=1e-2):
                        raise ValueError(f'online resumed KV differs beyond tolerance: {kv_error}')
            if config.full:
                store.checkpoint({'config_hash':config.digest,'model_revision':engine.model_revision,'token_hash':digest(execution_tokens),'next_position':tokens_count,'pending_token':expected})
            else:
                restored.checkpoint(pending_token=expected,token_hash=digest(execution_tokens))
            atomic_json(Path(root)/f'online-tokens-{config.id}.json',{'tokens':execution_tokens,'hash':digest(execution_tokens)})
            row.update(status='ok',processed_tokens=tokens_count,resume_forward_calls=tokens_count-checkpoint_pos,checkpoint_position=checkpoint_pos,
                 emitted_pending_consumed_once=True,uninterrupted_s=uninterrupted_s,resume_s=resume_s,
                 ingestion_tokens_s=tokens_count/uninterrupted_s,max_logit_error=max_error,kv_max_error=kv_error,kv_bit_identical=exact_hashes==resumed_hashes,
                 active_KV_entries=restored.max_entries,index_bytes=restored.index.nbytes if not config.full else 0,
                 next_token=expected,**guard.metrics())
        except Exception as exc:
            row.update(status='failure',failure_reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        finally:
            engine.adapter.set_controller(DenseController(engine.layers))
            if store:
                store.close()
        append_jsonl(Path(root)/'online.jsonl',row)
        rows.append(row)
    engine.adapter.uninstall()
    return rows

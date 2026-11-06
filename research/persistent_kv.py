"""Serve changing questions from one open archive; include actual disk stalls.

This is a three-turn, exact-identifier probe on Qwen2.5-1.5B. It isolates steady
service from one-time archive verification/loading. It is not a generic agent
or an evaluation beyond the model's native context limit.
"""
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import re
import time
import uuid
import numpy as np
import torch
from transformers import DynamicCache
from tiered_kv.adapter import ModelAdapter,DenseController
from tiered_kv.cache import TieredCache
from tiered_kv.config import Config,MODEL_REVISION
from tiered_kv.evals import make_needle,char_span,score_needle
from tiered_kv.generate import RunEngine,load_model,weights_digest,decode_greedy,append_jsonl,sync,online_smoke
from tiered_kv.store import ColdStore,atomic_json,file_sha
from tiered_kv.telemetry import ResourceGuard,runtime_identity,DEVICE,device_empty_cache
from tiered_kv.types import digest
from tiered_kv.validation import parity,implementation_hash
from research.demand_kv import query_boundary
from research.semantic_index import metadata_index,with_neighbors
from research.resources import HostObservation


def next_question_tokens(tok,question,previous_last,eos):
    marker='PREVIOUS_ASSISTANT_CONTENT_9c7d'
    rendered=tok.apply_chat_template([{'role':'user','content':'previous question'},
        {'role':'assistant','content':marker},{'role':'user','content':question}],
        tokenize=False,add_generation_prompt=True)
    suffix=rendered[rendered.index(marker)+len(marker):]
    if previous_last in eos:
        ending=tok.decode([previous_last],skip_special_tokens=False)
        if not suffix.startswith(ending):
            raise ValueError('generated EOS differs from serialized assistant ending')
        suffix=suffix[len(ending):]
    return tok.encode(suffix,add_special_tokens=False)


def make_session(tok,budget):
    # Reserve ample space for all three questions and generated answers.
    ex=make_needle(tok,budget-896,.1,83,distractors=True)
    original_query=query_boundary(tok,ex)
    text=tok.decode(ex['tokens'],skip_special_tokens=False)
    offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    candidates=[]
    cutoff=ex['prefix_tokens_count']-2000
    for match in re.finditer(r'The access code for (ITEM_[A-Z]+) is ([0-9]{8})\.',text[:text.rindex(original_query)]):
        span=char_span(offsets,match.start(),match.end())
        if span[0]>=128 and span[1]<cutoff-128:
            candidates.append({'item':match.group(1),'answer':match.group(2),'record_span':span})
    if len(candidates)<3:
        raise ValueError('not enough distinct cold facts for session')
    targets=[candidates[int(f*(len(candidates)-1))] for f in (.1,.5,.9)]
    if len({t['item'] for t in targets})!=3:
        raise ValueError('session targets must be distinct')
    suffix=text[text.rindex(original_query)+len(original_query):]
    if ex['tokens'][:ex['prefix_tokens_count']]+tok.encode(original_query+suffix,add_special_tokens=False)!=ex['tokens']:
        raise ValueError('first question boundary is not token-concatenable')
    for target in targets:
        target['question']=f"What is the access code for {target['item']}? Return only the code."
    ex.update(id='persistent-'+str(budget),budget=budget,queries=targets,first_suffix=suffix)
    return ex


class StockSession:
    def __init__(self,model,parent):
        self.model=model
        self.controller=self
        self.pending_token=None
        self.past=DynamicCache()
        self.position=parent.count
        for layer in range(parent.layers):
            batch=parent.read_range(layer,0,parent.count)
            self.past.update(batch.k.to(DEVICE),batch.v.to(DEVICE),layer)

    @torch.inference_mode()
    def step(self,token,pos):
        if pos!=self.position:
            raise ValueError('stock session absolute position mismatch')
        out=self.model(torch.tensor([[token]],device=DEVICE),past_key_values=self.past,use_cache=True,logits_to_keep=1)
        self.past=out.past_key_values
        self.position+=1
        return out.logits[0,-1]

    def consume_pending(self):
        token=self.pending_token
        if token is None:
            raise ValueError('no pending stock token')
        self.pending_token=None
        return self.step(token,self.position)


def advise_cold(parent):
    if not hasattr(os,'posix_fadvise'):
        raise RuntimeError('cold KV probe requires POSIX file-cache advice')
    for f in parent.files:
        os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)



def main():
    if DEVICE!='cuda':
        raise RuntimeError('persistent GPU/cache experiment requires CUDA; CPU fallback is not permitted')
    with ResourceGuard() as guard:
        model,tok=load_model()
        before=weights_digest(model)
        fixture={'examples':[make_session(tok,n) for n in (4096,8192,16384,32768)]}
        identity={'source':implementation_hash(),'script_sha256':file_sha(__file__),
                  'index_sha256':file_sha('research/semantic_index.py'),'fixture_sha256':digest(fixture),
                  'model_revision':MODEL_REVISION,'runtime':runtime_identity(),
                  'scope':'three distinct cold facts in consecutive questions; archive mounted once per session',
                  'cache_conditions':['warm','advised_cold_KV_only'],'index_cache':'warm metadata; no encoder needed for exact-identifier lookup'}
        root=Path('results')/('persistent-'+digest(identity)[:12])
        root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/'manifest.json',identity)
        atomic_json(root/'fixtures.json',fixture)
        checks=[parity(model,n) for n in (64,512,1025)]
        mechanics=[Config(id='full_buffered',full=True,W=128,C=0,R=32,B=32,k=2,buffered_writes=True),
                   Config(W=128,C=0,R=32,B=32,k=2,retrieval='fixed',fixed_blocks=(0,2),sink_tokens=4,buffered_writes=True)]
        online=online_smoke(model,tok,root,mechanics,guard,tokens_count=512,checkpoint_pos=256)
        if len(online)!=len(mechanics) or any(r['status']!='ok' for r in online):
            raise RuntimeError('persistent actual-model mechanics validation failed')
        atomic_json(root/'validation.json',{'status':'passed','parity':checks,'online':online,'weights_sha256':before})
        engine=RunEngine(model,tok,root,guard)
        output=root/'samples.jsonl'
        finished={r['session_id'] for line in output.read_text().splitlines()
                  if (r:=json.loads(line)).get('session_complete')} if output.exists() else set()
        eos=set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id,list) else [model.generation_config.eos_token_id])
        try:
            for ex in fixture['examples']:
                prefix,prefill_s=engine.prefix(ex)
                indices={padding:metadata_index(tok,ex,engine.scratch,padding) for padding in (0,32)}
                try:
                    for condition in ('warm','advised_cold_KV_only'):
                        for policy in ('stock_hf','sink_only','rare_fast','rare_context'):
                            session_id=f"{ex['id']}-{condition}-{policy}"
                            if session_id in finished:
                                continue
                            attempt=uuid.uuid4().hex[:8]
                            mount_start=time.perf_counter()
                            parent=ColdStore.open(prefix,readonly=True)
                            mount_s=time.perf_counter()-mount_start
                            store=None
                            cfg=None
                            init_start=time.perf_counter()
                            if policy=='stock_hf':
                                engine.adapter.uninstall()
                                adapter=StockSession(model,parent)
                            else:
                                cfg=Config(id=policy,C=0,k=4 if policy.startswith('rare_') else 0,retrieval='fixed',sink_tokens=4,buffered_writes=True)
                                store=ColdStore(engine.scratch/(session_id+'-'+attempt),engine.layers,engine.h,engine.d,128,prefix=parent,staged_tensor_writes=True)
                                controller=TieredCache(cfg,store,engine.rotary,DEVICE)
                                engine.adapter.set_controller(controller)
                                adapter=engine.adapter
                            sync()
                            init_s=time.perf_counter()-init_start
                            previous_last=None
                            try:
                                for turn,target in enumerate(ex['queries']):
                                    if condition=='advised_cold_KV_only':
                                        advise_cold(parent)
                                    tokens=(tok.encode(target['question']+ex['first_suffix'],add_special_tokens=False) if turn==0 else
                                            next_question_tokens(tok,target['question'],previous_last,eos))
                                    if adapter.position+len(tokens)+64>model.config.max_position_embeddings:
                                        raise ValueError('session would exceed native context')
                                    reads_start=parent.read_bytes+(store.read_bytes if store else 0)
                                    with HostObservation(scope='request only: pending token, lookup, question and generation; excludes mount, initialization and checkpoint') as host:
                                        sync()
                                        torch.cuda.reset_peak_memory_stats()
                                        start=time.perf_counter()
                                        if adapter.controller.pending_token is not None:
                                            adapter.consume_pending()
                                        lookup_s=0.
                                        selected=[]
                                        lookup_details={}
                                        if policy.startswith('rare_'):
                                            lookup_start=time.perf_counter()
                                            index=indices[32 if policy=='rare_context' else 0]
                                            selected=index.search(target['question'],None,4,'lexical_rare')
                                            lookup_details=dict(index.last_search)
                                            if policy=='rare_context':
                                                selected=with_neighbors(selected,(ex['prefix_tokens_count']+127)//128)
                                            controller.select_blocks(selected)
                                            lookup_s=time.perf_counter()-lookup_start
                                        for token in tokens:
                                            guard.check()
                                            logits=adapter.step(token,adapter.position)
                                        sync()
                                        first_token_s=time.perf_counter()-start
                                        generated,times=decode_greedy(adapter,logits,64,eos,guard=guard)
                                        sync()
                                        request_s=time.perf_counter()-start
                                        peak=torch.cuda.max_memory_allocated()
                                    previous_last=generated[-1]
                                    prediction=tok.decode(generated,skip_special_tokens=True)
                                    row={'status':'ok','session_id':session_id,'attempt':attempt,'turn':turn,'policy':policy,
                                         'budget':ex['budget'],'condition':condition,'archive_tokens':ex['prefix_tokens_count'],
                                         'query_tokens':len(tokens),'metadata_padding':32 if policy=='rare_context' else 0,
                                         'archive_mount_s':mount_s,'controller_init_s':init_s,'prefill_s':prefill_s,
                                         'index_search_s':lookup_s,'first_token_s':first_token_s,'request_s':request_s,
                                         'decode_ms':times,'output_tokens':len(generated),'prediction':prediction,'answer':target['answer'],
                                         'selected_blocks':selected,'lookup_details':lookup_details,
                                         'logical_read_bytes':parent.read_bytes+(store.read_bytes if store else 0)-reads_start,
                                         'cuda_peak_allocated_bytes':peak,'position_after':adapter.position,
                                         'active_KV_entries':controller.max_entries if store else adapter.position,
                                         'pending_finalize_included':turn>0,'policy_config':asdict(controller.config) if store else None,
                                         **score_needle(prediction,target['answer']),**host.metrics}
                                    append_jsonl(output,row)
                                    print(session_id,turn,row['em'],round(request_s,3),flush=True)
                                start=time.perf_counter()
                                resumed_error=None
                                if store:
                                    controller.checkpoint(pending_token=controller.pending_token,token_hash=digest(ex))
                                checkpoint_s=time.perf_counter()-start
                                if store:
                                    expected_position=controller.position+1
                                    expected=adapter.consume_pending().detach().float().cpu()
                                    current_config=controller.config
                                    store_root=store.root
                                    store.close()
                                    controller=TieredCache.reopen(store_root,current_config,engine.rotary,DEVICE)
                                    store=controller.store
                                    engine.adapter.set_controller(controller)
                                    actual=engine.adapter.consume_pending().detach().float().cpu()
                                    resumed_error=float((actual-expected).abs().max())
                                    if controller.position!=expected_position or not torch.allclose(actual,expected,atol=1e-4,rtol=1e-4):
                                        raise RuntimeError('persistent selection checkpoint continuation failed')
                                append_jsonl(output,{'session_id':session_id,'attempt':attempt,'session_complete':True,
                                    'checkpoint_s':checkpoint_s,'resumed_logit_max_abs':resumed_error,'status':'ok'})
                            finally:
                                if policy=='stock_hf':
                                    del adapter
                                    parent.close()
                                    engine.adapter=ModelAdapter.install(model,DenseController(engine.layers))
                                else:
                                    engine.adapter.set_controller(DenseController(engine.layers))
                                    store.close()
                                    del controller,adapter
                                gc.collect()
                                device_empty_cache()
                finally:
                    for index in indices.values():
                        index.close()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model)!=before:
            raise RuntimeError('frozen model weights changed')
        atomic_json(root/'complete.json',{'status':'complete','weights_unchanged':True})
    print('PERSISTENT_DONE',root,flush=True)


if __name__=='__main__':
    main()

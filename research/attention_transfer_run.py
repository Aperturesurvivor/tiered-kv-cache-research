"""New-seed transfer screen; predeclare plans before any CUDA generation."""
import argparse,gc,hashlib,json,os,random,shutil,subprocess,time,sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
import numpy as np
import torch
from research.attention_transfer_protocol import BUDGETS,REVISIONS,SEEDS,TASKS,POLICY,make_fixture
from research.attention_trajectory_diagnostic import validate_source,file_sha
from tiered_kv.types import digest,Rotary
from tiered_kv.store import atomic_json,ColdStore

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS={'Qwen/Qwen2.5-1.5B-Instruct':'fdf003e3fecca614b5f4cd70e02a7a0f7e28f9422bcd4fb509dcb025a8a4470a',
         'HuggingFaceTB/SmolLM2-1.7B-Instruct':'7c6fdf96c2489792315a773d9f8c5fc279ddf4fd5b4088734c4abf9237e53183'}
OWN_FILES=('research/attention_transfer_protocol.py','research/attention_transfer_run.py',
           'research/attention_rerank_live.py','research/attention_quest_reference.py','research/attention_page_cache.py',
           'research/attention_representations.py','research/attention_rerank_32k.py')


def source_identity():
    frozen=validate_source()
    return {'frozen_helpers':frozen,'transfer_helpers':{p:file_sha(ROOT/p) for p in OWN_FILES}}


def tokenizer_and_snapshot(model):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snap=Path(snapshot_download(model,revision=REVISIONS[model],allow_patterns=['*.json','*.model','merges.txt','vocab.txt'],local_files_only=True))
    tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
    tok.apply_chat_template=partial(tok.apply_chat_template,enable_thinking=False)
    return tok,snap


def make_plan(model):
    tok,snap=tokenizer_and_snapshot(model)
    specs=json.loads((ROOT/'research/model-manifest.json').read_text());spec=next(x for x in specs if x['model_id']==model)
    assert spec['revision']==REVISIONS[model] and max(BUDGETS[model])<=spec['config']['max_position_embeddings']
    examples=[make_fixture(tok,n,s,t) for n in BUDGETS[model] for s in SEEDS for t in TASKS]
    order=[];rng=random.Random(293)
    for ex in examples:
        arms=['stock','reranker'];rng.shuffle(arms)
        order.extend({'fixture_id':ex['id'],'arm':arm} for arm in arms)
    tokenizer_names=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json')
    tokenizer_files={name:file_sha(snap/name) for name in tokenizer_names if (snap/name).is_file()}
    assert all(name in tokenizer_files for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json'))
    return dict(status='frozen_plan',protocol='new-seed-transfer-1',model=spec,policy=POLICY,
        source_identity=source_identity(),weights_sha256=WEIGHTS[model],tokenizer_files=tokenizer_files,
        chat_template_sha256=digest(tok.chat_template),fixture_sha256=digest(examples),fixtures=examples,schedule=order,
        limits='one visible paired quality screen; fixed policy; no gold in routing; no latency or near-equivalence claim')


def validate_plan(doc,model):
    assert doc['status']=='frozen_plan' and doc['model']['model_id']==model
    assert doc['model']['revision']==REVISIONS[model] and doc['policy']==POLICY and doc['weights_sha256']==WEIGHTS[model]
    assert doc['source_identity']==source_identity() and digest(doc['fixtures'])==doc['fixture_sha256']
    assert len(doc['fixtures'])==len(BUDGETS[model])*len(SEEDS)*len(TASKS)
    assert len(doc['schedule'])==2*len(doc['fixtures'])
    assert {(x['fixture_id'],x['arm']) for x in doc['schedule']}=={(e['id'],a) for e in doc['fixtures'] for a in ('stock','reranker')}
    assert all(e['seed'] in SEEDS and e['task'] in TASKS and e['budget'] in BUDGETS[model] for e in doc['fixtures'])


def append_row(path,row):
    with path.open('a') as out:
        out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())


def persist_round(path,value,limit=8*2**20):
    """Bound and durably preserve a single round and its routing once."""
    data=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(data)>limit:raise ValueError('round telemetry exceeds declared byte limit')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as out:
        out.write(data);out.flush();os.fsync(out.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    assert path.stat().st_size==len(data)
    return len(data)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,choices=tuple(BUDGETS));ap.add_argument('--plan',required=True,type=Path)
    ap.add_argument('--prepare-plan',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--plan-sha256');ap.add_argument('--output',type=Path)
    args=ap.parse_args();assert args.prepare_plan!=args.execute
    os.chdir(ROOT)
    assert os.environ.get('HF_HUB_OFFLINE')=='1'
    if args.prepare_plan:
        assert not args.plan.exists();doc=make_plan(args.model);args.plan.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.plan,doc)
        print(json.dumps({'model':args.model,'fixtures':len(doc['fixtures']),'rows':len(doc['schedule']),'plan_sha256':file_sha(args.plan),'fixture_sha256':doc['fixture_sha256']}));return
    assert args.output is not None and args.plan_sha256 and file_sha(args.plan)==args.plan_sha256
    doc=json.loads(args.plan.read_text());validate_plan(doc,args.model)
    # Rebuild exact tokenizer/fixtures before loading weights or allocating GPU.
    assert make_plan(args.model)==doc
    assert torch.cuda.is_available() and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    config=doc['model']['config'];layers=config['num_hidden_layers'];h=config['num_key_value_heads'];d=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
    kv_bytes=layers*h*d*4;maxprefix=max(e['prefix_tokens_count'] for e in doc['fixtures'])
    disk=sum(e['prefix_tokens_count']*kv_bytes for e in doc['fixtures'])+len(doc['fixtures'])*2048*kv_bytes+20*2**30
    assert shutil.disk_usage(args.output.parent).free>=disk
    cache_bound=layers*32*128*2*h*d*2
    assert torch.cuda.mem_get_info()[0]>=doc['model']['parameters']*2+maxprefix*kv_bytes+cache_bound+2*2**30
    assert not (ROOT/'artifacts/scratch'/args.output.name).exists(), 'scratch namespace already exists'
    args.output.mkdir(parents=True,exist_ok=False)
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    from research.attention_rerank_live import CandidateRerankIndex,CandidatePageStore
    from research.attention_quest_reference import QuestBoundCache
    from research.reasoning_sessions import BatchedStockSession,CountedSession
    from research.reasoning_protocol import round_request,parse_response,next_search
    from research.persistent_kv import next_question_tokens
    from research.resources import HostObservation,logical_store_reads
    from tiered_kv.generate import RunEngine,weights_digest,decode_greedy,sync
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.config import Config
    from tiered_kv.telemetry import ResourceGuard,device_empty_cache,runtime_identity
    from tiered_kv.validation import parity
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tok,_=tokenizer_and_snapshot(args.model)
    snapshot=snapshot_download(args.model,revision=REVISIONS[args.model],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':'cuda'},attn_implementation='sdpa').eval().requires_grad_(False)
    model.config._commit_hash=REVISIONS[args.model];before=weights_digest(model);assert before==WEIGHTS[args.model]
    checks=[parity(model,n) for n in (64,512,1025)];assert all(all(x['greedy_agreement']) for x in checks)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    fixtures={e['id']:e for e in doc['fixtures']};rotary=Rotary(model.model.rotary_emb.inv_freq.detach().float().cpu().numpy())
    atomic_json(args.output/'validation.json',dict(status='partial',weights_sha256=before,parity=checks,plan_sha256=args.plan_sha256,runtime=runtime_identity(),disk_guard_bytes=disk))
    with ResourceGuard() as guard:
        engine=RunEngine(model,tok,args.output,guard)
        engine.adapter.uninstall()
        try:
            prefixes={}
            for row in doc['schedule']:
                ex=fixtures[row['fixture_id']];arm=row['arm'];print(json.dumps({'phase':'start',**row}),flush=True)
                if ex['id'] not in prefixes:
                    engine.adapter=ModelAdapter.install(model,DenseController(layers))
                    try:prefixes[ex['id']]=engine.prefix(ex)
                    finally:engine.adapter.uninstall()
                    prefix,_=prefixes[ex['id']];parent=ColdStore.open(prefix,readonly=True)
                    prompt=tok.encode(round_request(ex['query'],0,2)+ex['first_suffix'],add_special_tokens=False)
                    assert len(prompt)*model.config.vocab_size*4<=128*2**20
                    stock=BatchedStockSession(model,parent);native=[]
                    for token in prompt:guard.check();native.append(stock.step(token,stock.position).detach().float().cpu())
                    del stock;gc.collect();device_empty_cache()
                    dense=DenseController(layers);dense.initialize(parent,'cuda',torch.float16)
                    dense_adapter=ModelAdapter.install(model,dense);error=0.0
                    try:
                        for layer in range(layers):
                            for start_pos in range(0,parent.count,128):
                                end=min(start_pos+128,parent.count);batch=parent.read_range(layer,start_pos,end)
                                assert torch.equal(dense.keys[layer][:,:,start_pos:end].cpu(),batch.k)
                                assert torch.equal(dense.values[layer][:,:,start_pos:end].cpu(),batch.v)
                        for idx,token in enumerate(prompt):
                            actual=dense_adapter.step(token,dense_adapter.position).detach().float().cpu()
                            error=max(error,float((actual-native[idx]).abs().max()))
                            assert torch.allclose(actual,native[idx],atol=1e-4,rtol=1e-4)
                    finally:
                        dense_adapter.uninstall();del dense_adapter,dense,native;parent.close();gc.collect();device_empty_cache()
                    atomic_json(args.output/('parity-'+ex['id']+'.json'),dict(status='passed',fixture_id=ex['id'],all_kv_exact=True,prompt_max_abs=error,weights_sha256=before,plan_sha256=args.plan_sha256))
                prefix,prefill_s=prefixes[ex['id']];mount_start=time.perf_counter();parent=ColdStore.open(prefix,readonly=True);mount_s=time.perf_counter()-mount_start
                start_read=logical_store_reads(parent);init_start=time.perf_counter()
                session=child=adapter=controller=trace=observe=None
                original_mask=model.model._update_causal_mask
                original_forwards=[layer.self_attn.forward for layer in model.model.layers]
                host=HostObservation(scope='trajectory initialization, two rounds and checkpoint; sampled process RSS and process I/O')
                host.__enter__();host_active=True
                try:
                    if arm=='stock':session=CountedSession(BatchedStockSession(model,parent),model)
                    else:
                        cfg=Config(id='reranker',W=2000,C=0,R=128,k=15,sink_tokens=128,retrieval='centroid',model_revision=REVISIONS[args.model],buffered_writes=True)
                        child=CandidatePageStore(args.output/('overlay-'+ex['id']),layers,h,d,128,prefix=parent,staged_tensor_writes=True,cache_device='cuda',cache_pages=32,sink_tokens=128)
                        controller=QuestBoundCache(cfg,child,rotary,'cuda',initialize=False);controller.index=CandidateRerankIndex(layers,h,d,128,128,child,32);controller._initialize()
                        adapter=ModelAdapter.install(model,controller);session=CountedSession(adapter,model,active_limit=4048)
                        trace=session.controller.trace;session.events=[]
                        def observe(layer,pos,q,view):
                            trace(layer,pos,q,view)
                            if len(session.events)>=layers*768:raise ValueError('event cap exceeded')
                            session.events.append({'layer':layer,'position':pos,'pages':list(view.selected_blocks),'candidates':session.controller.index.last_candidates[layer], 'active':len(view.ids)})
                        session.controller.trace=observe
                    sync();init_s=time.perf_counter()-init_start
                    rounds=[];previous='';last=None;sync();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                    for turn in range(2):
                        event_start=len(session.events) if child else 0
                        if session.controller.pending_token is not None:session.consume_pending()
                        query,detail=next_search(ex['query'],previous,tok)
                        request=round_request(ex['query'],turn,2)
                        tokens=tok.encode(request+ex['first_suffix'],add_special_tokens=False) if turn==0 else next_question_tokens(tok,request,last,eos)
                        if session.position+len(tokens)+128>min(ex['budget'],model.config.max_position_embeddings):raise ValueError('native context/reserve exceeded')
                        t=time.perf_counter();logits=session.feed_prompt(tokens,guard);sync();prompt_s=time.perf_counter()-t
                        t=time.perf_counter();generated,decode_ms=decode_greedy(session,logits,128,eos,guard=guard);sync();gen_s=time.perf_counter()-t
                        previous=tok.decode(generated,skip_special_tokens=True);last=generated[-1];parsed=parse_response(previous)
                        rounds.append(dict(turn=turn,query=query,query_details=detail,prompt_token_ids=tokens,generated_token_ids=generated,response=previous,
                            parsed_response=parsed,answer=parsed['answer'] if parsed else None,exact_correct=bool(parsed and parsed['answer']==ex['answer']),prompt_s=prompt_s,generation_s=gen_s,decode_ms=decode_ms))
                        events=session.events[event_start:] if child else []
                        round_file=args.output/(ex['id']+'-'+arm+f'-round-{turn}.json')
                        persist_round(round_file,dict(status='partial',fixture_id=ex['id'],arm=arm,round=rounds[-1],selected_events=events))
                    sync();request_s=time.perf_counter()-start
                    checkpoint_start=time.perf_counter()
                    if child:child.checkpoint({'token_hash':ex['token_hash'],'scope':'append-only sparse trajectory; candidate index is not resumable'})
                    checkpoint_s=time.perf_counter()-checkpoint_start
                    host.__exit__(None,None,None);host_active=False
                    result=dict(status='partial',fixture_id=ex['id'],arm=arm,rounds=rounds,prefill_s=prefill_s,mount_s=mount_s,init_s=init_s,request_s=request_s,checkpoint_s=checkpoint_s,metrics=session.metrics(),
                        logical_read_bytes=logical_store_reads(child if child else parent)-start_read,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                        selected_events_scope='stored once in hash-bound per-round files',candidate_stats=controller.index.rerank_stats if child else None,
                        host=host.metrics,page_cache_stats=child.stats if child else None,bound_metadata_bytes=controller.index.nbytes if child else 0,
                        sink_cpu_bytes=sum(x.numel()*x.element_size() for x in controller.normalizer_sinks if x is not None) if child else 0,
                        normalizer_stats=controller.index.stats if child else None,
                        round_file_sha256={str(i):file_sha(args.output/(ex['id']+'-'+arm+f'-round-{i}.json')) for i in range(2)},
                        persistent_cache_bound_bytes=layers*32*128*2*h*d*2 if child else 0,host_scope='one task trajectory; request includes durable per-round serialization; not paired latency benchmark')
                    append_row(args.output/'samples.jsonl',result)
                finally:
                    if adapter:
                        if session:session.controller.trace=None
                        adapter.uninstall()
                    elif child:
                        model.model._update_causal_mask=original_mask
                        for layer,forward in zip(model.model.layers,original_forwards):layer.self_attn.forward=forward
                        if hasattr(model,'_tiered_adapter'):del model._tiered_adapter
                    if child:child.prefix=None;child.close()
                    session=child=adapter=controller=trace=observe=None;parent.close();gc.collect();device_empty_cache()
                    if host_active:host.__exit__(*sys.exc_info())
                print(json.dumps({'phase':'row_saved',**row}),flush=True)
        finally:
            if getattr(model,"_tiered_adapter",None):engine.adapter.uninstall()
        after=weights_digest(model);assert after==before
    assert source_identity()==doc['source_identity'] and file_sha(args.plan)==args.plan_sha256
    atomic_json(args.output/'integrity.json',dict(status='partial',weights_sha256_before=before,weights_sha256_after=after,plan_sha256=args.plan_sha256,
        samples_sha256=file_sha(args.output/'samples.jsonl'),source_identity=doc['source_identity'],
        fixture_parity_sha256={e['id']:file_sha(args.output/('parity-'+e['id']+'.json')) for e in doc['fixtures']}))
    rows=[json.loads(x) for x in (args.output/'samples.jsonl').read_text().splitlines()]
    assert len(rows)==len(doc['schedule']) and {(x['fixture_id'],x['arm']) for x in rows}=={(x['fixture_id'],x['arm']) for x in doc['schedule']}
    atomic_json(args.output/'complete.json',dict(status='complete',rows=len(rows),weights_unchanged=True,samples_sha256=file_sha(args.output/'samples.jsonl'),integrity_sha256=file_sha(args.output/'integrity.json'),plan_sha256=args.plan_sha256))

if __name__=='__main__':main()

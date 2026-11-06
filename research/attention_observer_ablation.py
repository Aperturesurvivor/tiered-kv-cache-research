"""Predeclared full-observer on/off device reranker cost ablation.

Separate from the frozen transfer study; only explicit --execute loads a model.
"""
import argparse,gc,hashlib,json,os,shutil,struct,subprocess,sys,time,traceback
from pathlib import Path
import torch
from research.attention_transfer_run import make_plan,source_identity,tokenizer_and_snapshot,WEIGHTS,append_row
from research.attention_transfer_protocol import REVISIONS
from research.attention_trajectory_diagnostic import file_sha
from tiered_kv.types import Rotary
from tiered_kv.store import ColdStore,atomic_json

ROOT=Path(__file__).resolve().parents[1]
MODEL='Qwen/Qwen2.5-1.5B-Instruct'
PLAN='dd31e825291215a5e35db43d11e79ae6de5b7a23e725f45e132066f96a943a29'
ARMS=('on','off')
EXTRA=('research/attention_device_cost.py','research/attention_device_rerank.py','research/attention_device_summaries.py')
EXPECTED={'research/attention_device_cost.py':'2e745374ecba6d6776de2215dc42354d3101de038a9e61731979f3677f645953','research/attention_device_rerank.py':'13be7a244ebf66731d4208a2ecf2b6b55abc75eef2ee0e73dd9cfb52929ba409','research/attention_device_summaries.py':'69a98c16452cf62e1bd555ebed758ea402ec9ead52bbbef0526f5f2b76c6e757'}

def cases(plan):
    out=[]
    for budget in (8192,32768):
        found=[e for e in plan['fixtures'] if (e['seed'],e['task'],e['budget'])==(283,'linked_fact',budget)]
        if len(found)!=1:raise ValueError('fixture identity mismatch')
        ex=found[0];n=ex['prefix_tokens_count'];tokens=ex['tokens']
        if n<64:raise ValueError('prefix too short')
        # Fixed archived tokens, independent of model output and answer.
        out.append((ex,tokens[n-64:n]))
    return out

def schedule():
    return [(budget,rep,arm) for budget in (8192,32768) for rep in range(3)
            for arm in (('on','off') if (rep+(budget==32768))%2==0 else ('off','on'))]


def check_equivalence(a,b):
    if a['events']!=b['events'] or a['kv_sha256']!=b['kv_sha256'] or a['metrics']!=b['metrics'] or len(a['logits'])!=len(b['logits']):
        raise ValueError('route or attention KV mismatch')
    errors=[float((x-y).abs().max()) for x,y in zip(a['logits'],b['logits'])]
    if not errors or not all(torch.isfinite(torch.tensor(e)).item() and e<=1e-4 for e in errors):
        raise ValueError(f'logit mismatch: max absolute differences {errors}')
    return max(errors)

def make_observer(controller,count_trace,events,hashes,validate):
    def observe(layer,pos,q,view):
        count_trace(layer,pos,q,view)
        ids=[int(x) for x in view.ids]
        if len(ids)!=len(set(ids)) or any(x<0 or x>pos for x in ids) or max(ids,default=-1)!=pos:
            raise ValueError('noncausal or duplicate attention positions')
        ids_sha=hashlib.sha256(b''.join(struct.pack('<q',x) for x in ids)).hexdigest()
        events.append([int(layer),int(pos),[int(x) for x in controller.index.last_candidates[layer]],
                       [int(x) for x in view.selected_blocks],len(ids),ids_sha])
        if validate:
            digest=hashlib.sha256()
            for tensor in (view.k,view.v):
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            hashes.append(digest.hexdigest())
    return observe

def select_observer(arm,controller,count_trace,events,hashes,validate):
    if arm=='off':
        if validate:raise ValueError('full validation requires observer on')
        return count_trace
    if arm=='on':return make_observer(controller,count_trace,events,hashes,validate)
    raise ValueError('unknown observer arm')


def source_hashes():return {x:file_sha(ROOT/x) for x in EXTRA}

def compare_final(a,b):
    if a.shape!=b.shape:raise ValueError('final logit geometry mismatch')
    err=float((a-b).abs().max())
    if not torch.isfinite(torch.tensor(err)).item() or err>1e-4:raise ValueError('final logits mismatch')
    return err


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    os.chdir(ROOT)
    if file_sha(a.plan)!=PLAN:raise ValueError('plan hash mismatch')
    doc=json.loads(a.plan.read_text());chosen=cases(doc)
    if source_hashes()!=EXPECTED:raise ValueError('frozen helper mismatch')
    declaration={'plan_sha256':PLAN,'fixtures':[e['id'] for e,_ in chosen],'schedule':schedule(),'forced_token_source':'last 64 archive-prefix tokens','sources':source_hashes(),'ablation_source_sha256':file_sha(Path(__file__)),'timing_scope':'device on/off; CountedSession always active; full route observer only on; separate untimed before/after full validations; prior stock timings are historical, not concurrent controls; warm caches; no timed-off per-event route proof'}
    if not a.execute:print(json.dumps(declaration));return
    if a.output is None or a.output.exists() or (ROOT/'artifacts/scratch'/a.output.name).exists():raise ValueError('fresh output and scratch required')
    if os.environ.get('HF_HUB_OFFLINE')!='1':raise ValueError('offline required')
    if make_plan(MODEL)!=doc:raise ValueError('frozen plan/source mismatch')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU busy')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4':raise ValueError('expected T4')
    config=doc['model']['config'];layers=config['num_hidden_layers'];h=config['num_key_value_heads'];d=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
    maxprefix=max(e['prefix_tokens_count'] for e,_ in chosen);kv=layers*h*d*4
    disk_guard=sum(e['prefix_tokens_count']*kv for e,_ in chosen)+22*768*kv+5*10**9+3*2**30
    if shutil.disk_usage(a.output.parent).free<disk_guard:raise ValueError('disk guard')
    declaration['disk_guard_bytes']=disk_guard
    if torch.cuda.mem_get_info()[0]<doc['model']['parameters']*2+maxprefix*kv+layers*32*128*2*h*d*2+2*2**30:raise ValueError('GPU lower-bound guard')
    a.output.mkdir(parents=True,exist_ok=False);atomic_json(a.output/'declaration.json',declaration)
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    from research.attention_rerank_live import CandidatePageStore
    from research.attention_device_rerank import DeviceQuestCache,DeviceRerankIndex
    from research.reasoning_sessions import CountedSession
    from research.reasoning_protocol import round_request
    from research.resources import HostObservation,logical_store_reads
    from tiered_kv.generate import RunEngine,weights_digest,sync
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.config import Config
    from tiered_kv.telemetry import ResourceGuard,device_empty_cache,runtime_identity
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tok,_=tokenizer_and_snapshot(MODEL);snap=snapshot_download(MODEL,revision=REVISIONS[MODEL],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(snap,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':'cuda'},attn_implementation='sdpa').eval().requires_grad_(False)
    model.config._commit_hash=REVISIONS[MODEL];before=weights_digest(model)
    if before!=WEIGHTS[MODEL]:raise ValueError('weights mismatch')
    frozen_source=source_identity()
    if frozen_source!=doc['source_identity']:raise ValueError('source identity mismatch')
    atomic_json(a.output/'initial-integrity.json',{'status':'partial','weights_sha256':before,'runtime':runtime_identity(),'source_identity':frozen_source,'extra_sources':source_hashes(),'plan_sha256':PLAN})
    rotary=Rotary(model.model.rotary_emb.inv_freq.detach().float().cpu().numpy());prefixes={};recorded={}
    with ResourceGuard() as guard,torch.inference_mode():
        engine=RunEngine(model,tok,a.output,guard);engine.adapter.uninstall()
        for ex,forced in chosen:
            engine.adapter=ModelAdapter.install(model,DenseController(layers))
            start=time.perf_counter()
            try:prefix,prefill=engine.prefix(ex)
            finally:engine.adapter.uninstall()
            construction=time.perf_counter()-start
            prompt=tok.encode(round_request(ex['query'],0,2)+ex['first_suffix'],add_special_tokens=False)
            if len(prompt)+64>768:raise ValueError('overlay estimate token bound exceeded')
            if len(prompt)+64+ex['prefix_tokens_count']>min(ex['budget'],model.config.max_position_embeddings):raise ValueError('native limit')
            prefixes[ex['budget']]=(ex,forced,prefix,prompt,construction,prefill)
            atomic_json(a.output/f'workload-{ex["budget"]}.json',{'fixture_id':ex['id'],'prompt_token_ids':prompt,'forced_token_ids':forced,'prefix_count':ex['prefix_tokens_count'],'token_hash':ex['token_hash']})
        def run(budget,arm,label,validate):
            ex,forced,prefix,prompt,construction,prefill=prefixes[budget];start=time.perf_counter();parent=ColdStore.open(prefix,readonly=True);mount=time.perf_counter()-start
            initial_reads=logical_store_reads(parent);child=controller=adapter=session=trace=None;events=[];hashes=[];logits=[]
            original_mask=model.model._update_causal_mask;original_forwards=[x.self_attn.forward for x in model.model.layers]
            host=HostObservation(scope='mount excluded; init through cleanup sampled');host.__enter__();host_active=True
            try:
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                cfg=Config(id='reranker',W=2000,C=0,R=128,k=15,sink_tokens=128,retrieval='centroid',model_revision=REVISIONS[MODEL],buffered_writes=True)
                child=CandidatePageStore(a.output/('overlay-'+label),layers,h,d,128,prefix=parent,staged_tensor_writes=True,cache_device='cuda',cache_pages=32,sink_tokens=128)
                controller=DeviceQuestCache(cfg,child,rotary,'cuda',initialize=False);controller.index=DeviceRerankIndex(layers,h,d,128,128,child,32);controller._initialize()
                adapter=ModelAdapter.install(model,controller);session=CountedSession(adapter,model,active_limit=4048);trace=controller.trace
                observe=select_observer(arm,controller,trace,events,hashes,validate)
                controller.trace=observe
                sync();init=time.perf_counter()-start;init_peak_allocated=torch.cuda.max_memory_allocated();init_peak_reserved=torch.cuda.max_memory_reserved();index_build_s=sum(x['absorb_s'] for x in controller.index.stats) if child else 0.;torch.cuda.reset_peak_memory_stats();sync();request_start=time.perf_counter();start=request_start
                value=session.feed_prompt(prompt,guard);sync();prompt_s=time.perf_counter()-start
                if validate:logits.append(value.detach().float().cpu())
                start=time.perf_counter();step_ms=[]
                for token in forced:
                    guard.check();sync();step_start=time.perf_counter();value=session.step(token,session.position);sync()
                    step_ms.append(1000*(time.perf_counter()-step_start))
                    if validate:logits.append(value.detach().float().cpu())
                sync();forced_s=time.perf_counter()-start
                serving=time.perf_counter()-request_start
                final_logits=value.detach().float().cpu() # outside serving timer
                start=time.perf_counter()
                if child:child.checkpoint({'token_hash':ex['token_hash'],'scope':'benchmark overlay; index not resumable'})
                checkpoint=time.perf_counter()-start
                metrics=session.metrics()
                if metrics['model_token_evaluations']!=len(prompt)+64 or metrics['prompt_token_evaluations']!=len(prompt):raise ValueError('workload count mismatch')
                reads=logical_store_reads(child if child else parent)-initial_reads
                stats={'metrics':metrics,'logical_read_bytes':reads,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'init_peak_allocated':init_peak_allocated,'init_peak_reserved':init_peak_reserved,'bound_metadata_bytes':controller.index.nbytes if child else 0,'index_build_s':index_build_s,'cache_bound_bytes':layers*32*128*2*h*d*2 if child else 0,'page_cache_stats':child.stats if child else None,'normalizer_stats':controller.index.stats if child else None,'selector_stats':controller.device_accounting()}
            finally:
                start=time.perf_counter()
                if adapter:
                    if session:session.controller.trace=None
                    adapter.uninstall()
                elif child:
                    model.model._update_causal_mask=original_mask
                    for layer,forward in zip(model.model.layers,original_forwards):layer.self_attn.forward=forward
                    if hasattr(model,'_tiered_adapter'):del model._tiered_adapter
                if child:child.prefix=None;child.close()
                session=child=controller=adapter=trace=observe=None;parent.close();gc.collect();device_empty_cache();sync();cleanup=time.perf_counter()-start
                if host_active:host.__exit__(*sys.exc_info())
            return {'events':events,'kv_sha256':hashes,'logits':logits,'final_logits':final_logits,'mount_s':mount,'init_s':init,'prompt_s':prompt_s,'forced_s':forced_s,'forced_step_ms':step_ms,'serving_s':serving,'checkpoint_s':checkpoint,'cleanup_s':cleanup,'host':host.metrics,**stats}
        for budget in (8192,32768):
            reference=run(budget,'on',f'validation-{budget}-before',True)
            before_evidence={'events':reference['events'],'kv_sha256':reference['kv_sha256'],'metrics':reference['metrics']}
            before_name=f'validation-{budget}-before.json';atomic_json(a.output/before_name,before_evidence)
            # The before-block full validation warms the device path.
            # Previous stock timings are historical and not a concurrent control.
            paired={}
            for _,rep,arm in (x for x in schedule() if x[0]==budget):
                result=run(budget,arm,f'{budget}-{rep}-{arm}',False)
                events=result.pop('events');actual_final=result.pop('final_logits');result.pop('logits');result.pop('kv_sha256')
                if arm=='on':
                    name=f'events-{budget}-{rep}-on.json';atomic_json(a.output/name,{'events':events});result['events_file']=name;result['events_sha256']=file_sha(a.output/name)
                if result['metrics']!=reference['metrics'] or (arm=='on' and events!=reference['events']):
                    atomic_json(a.output/f'timed-{budget}-{rep}-{arm}-failure.json',{'status':'failed','actual_metrics':result['metrics'],'expected_metrics':reference['metrics'],'actual_events_file':result.get('events_file'),'expected_events_file':before_name,'actual_events_sha256':result.get('events_sha256'),'expected_events_sha256':file_sha(a.output/before_name)})
                    raise ValueError('timed work or observed route mismatch')
                err=float((actual_final-reference['final_logits']).abs().max())
                if not torch.isfinite(torch.tensor(err)).item() or err>1e-4:
                    atomic_json(a.output/f'timed-{budget}-{rep}-{arm}-failure.json',{'status':'failed','kind':'final_logits_vs_reference','finite':bool(torch.isfinite(torch.tensor(err)).item()),'max_abs':err if torch.isfinite(torch.tensor(err)).item() else None})
                    raise ValueError('timed final logits mismatch')
                append_row(a.output/'samples.jsonl',{'budget':budget,'rep':rep,'arm':arm,'final_logit_max_abs':err,**result})
                paired.setdefault(rep,{})[arm]=actual_final
                if len(paired[rep])==2:
                    pair_error=float((paired[rep]['on']-paired[rep]['off']).abs().max());finite=bool(torch.isfinite(torch.tensor(pair_error)).item())
                    atomic_json(a.output/f'pair-{budget}-{rep}.json',{'status':'passed' if finite and pair_error<=1e-4 else 'failed','max_abs_logit':pair_error if finite else None,'scope':'final logits only; no timed-off route observation'})
                    if not finite or pair_error>1e-4:raise ValueError('paired final logits mismatch')
                    del paired[rep]
            after_check=run(budget,'on',f'validation-{budget}-after',True)
            after_name=f'validation-{budget}-after.json';atomic_json(a.output/after_name,{'events':after_check['events'],'kv_sha256':after_check['kv_sha256'],'metrics':after_check['metrics']})
            try:maxerr=check_equivalence(reference,after_check)
            except Exception:
                logit_errors=[float((x-y).abs().max()) for x,y in zip(reference['logits'],after_check['logits'])]
                finite=all(torch.isfinite(torch.tensor(e)).item() for e in logit_errors)
                atomic_json(a.output/f'validation-{budget}-failure.json',{'status':'failed','before_sha256':file_sha(a.output/before_name),'after_sha256':file_sha(a.output/after_name),'max_abs_logit':max(logit_errors) if logit_errors and finite else None,'compared_logits':len(logit_errors),'expected_logits':len(reference['logits']),'actual_logits':len(after_check['logits']),'error':traceback.format_exc()})
                raise
            atomic_json(a.output/f'validation-{budget}.json',{'status':'passed','max_abs_logit':maxerr,'events':len(reference['events']),'before_sha256':file_sha(a.output/before_name),'after_sha256':file_sha(a.output/after_name),'scope':'untimed full observer checks; timed-off routes not observed'})
            del reference,after_check;gc.collect();device_empty_cache()
        after=weights_digest(model)
        if after!=before or file_sha(Path(__file__))!=declaration['ablation_source_sha256'] or source_hashes()!=declaration['sources'] or source_identity()!=frozen_source or file_sha(a.plan)!=PLAN:raise ValueError('post-run identity changed')
        atomic_json(a.output/'complete.json',{'status':'complete','rows':12,'samples_sha256':file_sha(a.output/'samples.jsonl'),'weights_before':before,'weights_after':after,'runtime':runtime_identity(),'declaration':declaration,'prefix_construction':{str(k):{'wall_s':v[4],'prefill_s':v[5]} for k,v in prefixes.items()}})
if __name__=='__main__':
    try:main()
    except Exception:
        if '--output' in sys.argv:
            i=sys.argv.index('--output')
            if i+1<len(sys.argv):
                out=Path(sys.argv[i+1])
                if out.is_dir():atomic_json(out/'failure.json',{'status':'failed','error':traceback.format_exc()})
        raise

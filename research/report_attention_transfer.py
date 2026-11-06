"""Independent offline audit of the frozen new-seed transfer trajectories.

Partial results are labelled and never promoted to a completed cohort. Token
replay requires only the pinned local tokenizer; model inference is not used.
"""
import argparse,hashlib,json,math,os
from collections import Counter
from pathlib import Path
from research.attention_transfer_run import make_plan,tokenizer_and_snapshot,source_identity,WEIGHTS
from research.reasoning_protocol import round_request,parse_response,next_search
from research.persistent_kv import next_question_tokens

PLAN_HASHES={'qwen15':'dd31e825291215a5e35db43d11e79ae6de5b7a23e725f45e132066f96a943a29',
             'smol17':'acc9b8bf76c59b1f1c82746f46e223f8a669bce8f8709a0e25ca25eda8a4b66a'}
MODELS={'qwen15':'Qwen/Qwen2.5-1.5B-Instruct','smol17':'HuggingFaceTB/SmolLM2-1.7B-Instruct'}
CONTROLLER_SHA='93fc36fdb6762a72e4762ec087bff3219510e06c4de4c2d7948ce19357450b58'
BUNDLE_SHA='c57e86f82bae0997fda298c33793a08f035316488c214733887b53a94a2f8446'
REPAIRED_CONTROLLER_SHA='c3fa317882413b33615dc7d6a147c32f2b3235ec1670c53a2e99e2abbb0deac9'
REPAIRED_BUNDLE_SHA='0ae02a0be3ed5fbb413763dbb7d181c8648642120e15b36fa54b02d15496399c'
REPAIRED_HELPER_SHA='bc754b9ee17cad8fd799d8a97e0cf1bafcb6c5d5e20cf96bc67b7c07179a83ff'
PRIOR_FAILURE_STATE_SHA='4c25c717dc6774a43878a4d8dd7e746f071f15fb4f4aa348f96e1fc2f2067ad4'
FRESH_SMOL_CONTROLLER_SHA='160627c2358aaeefb01ba7659de44ee9a317cf5540fff62f5d4fad1074301bb9'
FRESH_BOOT='f2a4a03a-a73e-4bca-9557-f756da7da01c'
FRESH_PREDECESSORS={
 '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-1/state.json':PRIOR_FAILURE_STATE_SHA,
 '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-1/worker.log':'f51cbd87782b12b726dbe17df8389c78ea8e4b6ce49d19c42922a6915a51e423',
 '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-2/state.json':'f9c823264937806940b050e5897cd83859dfcccd9104c09e98690b0167c615a2',
 '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-2/output/complete.json':'cd41966e3193ac1bc02b65d93357f622512a9016f233a34d416c45827efc607c',
 '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-2/output/integrity.json':'ce26f7e197baf77c9bc742e193c44480a6baf11012c804fca7c53fab27b48403'}
FRESH_COST_TERMINAL={'receipt_sha256':'581220a508db47cdbd80bbb54e942e5ebb9500ea5bebf5fc4d1eeaf926fe4b03',
 'state_sha256':'0db0e03f77d93bd02e2f73a9a21c9f1dd7ed0cda66f1e4012265fb2e433134c2',
 'controller':[6116,1793782300.87],'worker':[6416,1793782357.41],
 'controller_shell_wait_returncode':0,'worker_returncode':0}
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)


def controller_provenance(controller,required=False,label=None):
    if controller is None:return None
    pair=(controller.get('bundle_sha256'),controller.get('controller_sha256'))
    if pair==(None,None) and not required:return None
    if pair==(BUNDLE_SHA,CONTROLLER_SHA):
        return {'package':'original','bundle_sha256':BUNDLE_SHA,'controller_sha256':CONTROLLER_SHA}
    fresh=pair==(REPAIRED_BUNDLE_SHA,FRESH_SMOL_CONTROLLER_SHA)
    need(pair==(REPAIRED_BUNDLE_SHA,REPAIRED_CONTROLLER_SHA) or fresh,'controller package identity differs')
    if fresh:
        need(label=='smol17' and controller.get('model')==MODELS['smol17'],'fresh controller is Smol-only')
        need(controller.get('backend_boot_id')==FRESH_BOOT,'fresh backend identity differs')
        need(controller.get('prior_receipts')==FRESH_PREDECESSORS,'fresh predecessor receipts differ')
        need(controller.get('cost_terminal')==FRESH_COST_TERMINAL,'fresh cost terminal binding differs')
    need(controller.get('extra_helper_sha256')==REPAIRED_HELPER_SHA,'repaired helper identity differs')
    smoke=controller.get('import_smoke',{})
    need(smoke.get('status')=='passed' and 'research.attention_fullkey_oracle' in smoke.get('modules',[]),'repaired import smoke missing')
    result={'package':'fresh_smol_70_member_export' if fresh else 'repaired_70_member_export','bundle_sha256':REPAIRED_BUNDLE_SHA,
            'controller_sha256':FRESH_SMOL_CONTROLLER_SHA if fresh else REPAIRED_CONTROLLER_SHA,'extra_helper_sha256':REPAIRED_HELPER_SHA,
            'prior_failed_attempt':'results/evidence/attention-transfer-qwen15-a712857-1',
            'prior_failed_state_sha256':PRIOR_FAILURE_STATE_SHA,
            'scope':'packaging-only repair; original frozen plan, runner and policy unchanged'}
    if fresh:
        result.update(backend_boot_id=FRESH_BOOT,prior_receipts=FRESH_PREDECESSORS,cost_terminal=FRESH_COST_TERMINAL)
    return result


def audit_events(events,first,steps,layers):
    need(len(events)==steps*layers,'missing or duplicated routing events')
    counts=Counter();active=Counter()
    for i,event in enumerate(events):
        pos=first+i//layers;layer=i%layers
        need(event['position']==pos and event['layer']==layer,'noncausal or reordered event')
        candidates=event['candidates'];pages=event['pages'];cutoff=pos-2000+1
        need(len(candidates)<=32 and len(candidates)==len(set(candidates)),'candidate budget/duplicates')
        need(len(pages)<=15 and len(pages)==len(set(pages)) and set(pages)<=set(candidates),'final page budget/subset')
        need(all(type(b) is int and b>=1 and (b+1)*128<=cutoff for b in candidates),'ineligible or future page')
        need(len(candidates)==min(32,max(0,cutoff//128-1)) and len(pages)==min(15,len(candidates)),'candidate or final underfill differs from eligibility')
        need(event['active']==2128+len(pages)*128 and event['active']<=4048,'active attention budget')
        counts[(len(candidates),len(pages))]+=1;active[event['active']]+=1
    return {'candidate_final_counts':{f'{a}/{b}':n for (a,b),n in sorted(counts.items())},'active_counts':dict(sorted(active.items()))}


def audit_model(label,plan_path,output,tok,eos,partial=False,controller=None):
    need(sha(plan_path)==PLAN_HASHES[label],'plan hash differs');plan=json.loads(plan_path.read_text());model=MODELS[label]
    need(plan['model']['model_id']==model and plan['weights_sha256']==WEIGHTS[model],'model or weights differ')
    need(plan['source_identity']==source_identity(),'source identity differs')
    need(make_plan(model)==plan,'tokenizer, fixture or schedule mismatch')
    rows_path=output/'samples.jsonl';rows=[json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    schedule=plan['schedule'];need(len(rows)<=len(schedule),'too many rows')
    need([(r['fixture_id'],r['arm']) for r in rows]==[(r['fixture_id'],r['arm']) for r in schedule[:len(rows)]],'partial rows differ from schedule')
    validation=output/'validation.json'
    if rows:
        need(validation.is_file(),'validation missing');v=json.loads(validation.read_text())
        need(v['weights_sha256']==plan['weights_sha256'] and v['plan_sha256']==PLAN_HASHES[label],'validation identity differs')
        need([x['length'] for x in v['parity']]==[64,512,1025] and all(len(x['greedy_agreement'])==17 and all(x['greedy_agreement']) and len(x['max_errors'])==17 and all(math.isfinite(e) and e>=0 for e in x['max_errors']) for x in v['parity']),'actual-model parity failed')
    fixtures={e['id']:e for e in plan['fixtures']};layers=plan['model']['config']['num_hidden_layers'];records=[]
    for row in rows:
        ex=fixtures[row['fixture_id']];arm=row['arm'];need(row['status']=='partial' and len(row['rounds'])==2,'invalid row')
        pp=output/('parity-'+ex['id']+'.json');pv=json.loads(pp.read_text())
        need(pv['fixture_id']==ex['id'] and pv['status']=='passed' and pv['all_kv_exact'] and pv['weights_sha256']==plan['weights_sha256'] and pv['plan_sha256']==PLAN_HASHES[label] and math.isfinite(pv['prompt_max_abs']) and pv['prompt_max_abs']>=0,'fixture parity missing/invalid')
        previous='';last=None;position=ex['prefix_tokens_count'];route=[];active_sum=0;active_max=0;generated_counts=[];layer_events=[[] for _ in range(layers)]
        for turn,roundrow in enumerate(row['rounds']):
            need(roundrow['turn']==turn,'wrong round index');request=round_request(ex['query'],turn,2)
            expected=tok.encode(request+ex['first_suffix'],add_special_tokens=False) if turn==0 else next_question_tokens(tok,request,last,eos)
            need(roundrow['prompt_token_ids']==expected,'prompt replay mismatch')
            tokens=roundrow['generated_token_ids'];need(0<len(tokens)<=128 and all(type(t) is int and t>=0 for t in tokens),'generated token cap/type')
            need(all(type(t) is int and 0<=t<plan['model']['config']['vocab_size'] for t in expected+tokens),'token ID outside vocabulary')
            need(position+int(turn>0)+len(expected)+128<=min(ex['budget'],plan['model']['config']['max_position_embeddings']),'native context reserve exceeded')
            need(not any(t in eos for t in tokens[:-1]),'tokens follow EOS')
            need(tokens[-1] in eos or len(tokens)==128,'generation stopped before EOS or cap')
            need(all(math.isfinite(roundrow[k]) and roundrow[k]>=0 for k in ('prompt_s','generation_s')),'invalid round timing')
            need(len(roundrow['decode_ms'])==len(tokens)-1 and all(math.isfinite(x) and x>=0 for x in roundrow['decode_ms']),'invalid decode timings')
            response=tok.decode(tokens,skip_special_tokens=True);parsed=parse_response(response);query,details=next_search(ex['query'],previous,tok)
            need(response==roundrow['response'] and parsed==roundrow['parsed_response'] and query==roundrow['query'] and details==roundrow['query_details'],'output or query replay mismatch')
            answer=parsed['answer'] if parsed else None;correct=bool(parsed and answer==ex['answer'])
            need(roundrow['answer']==answer and roundrow['exact_correct']==correct,'answer scoring mismatch')
            path=output/(ex['id']+'-'+arm+f'-round-{turn}.json');need(sha(path)==row['round_file_sha256'][str(turn)],'round hash mismatch');saved=json.loads(path.read_text())
            need(saved['status']=='partial' and saved['fixture_id']==ex['id'] and saved['arm']==arm and saved['round']==roundrow,'round artifact mismatch')
            steps=len(expected)+len(tokens)-1+(turn>0)
            if arm=='reranker':
                route.append(audit_events(saved['selected_events'],position,steps,layers))
                active_sum+=sum(e['active'] for e in saved['selected_events']);active_max=max(active_max,max((e['active'] for e in saved['selected_events']),default=0))
                for event in saved['selected_events']:layer_events[event['layer']].append(event)
            else:need(saved['selected_events']==[],'stock unexpectedly has routing events')
            position+=steps;previous=response;last=tokens[-1];generated_counts.append(len(tokens))
            records.append({'model':label,'fixture_id':ex['id'],'budget':ex['budget'],'task':ex['task'],'seed':ex['seed'],'arm':arm,'turn':turn,'correct':correct,'answer':answer,'response':response,'output_tokens':len(tokens)})
        for key in ('prefill_s','mount_s','init_s','request_s','checkpoint_s'):
            need(math.isfinite(row[key]) and row[key]>=0,'invalid timing')
        need(row['logical_read_bytes']>=0 and row['peak_allocated']>=0 and row['peak_reserved']>=row['peak_allocated'],'invalid resource counters')
        need('sampled' in row['host']['host_observation_scope'] and row['host']['host_rss_samples']>0,'host measurement scope missing')
        need(row['metrics']['model_token_evaluations']==position-ex['prefix_tokens_count'],'forward token accounting mismatch')
        n=position-ex['prefix_tokens_count'];metrics=row['metrics'];heads=plan['model']['config']['num_attention_heads']
        need(metrics['prompt_token_evaluations']==sum(len(r['prompt_token_ids']) for r in row['rounds']),'prompt accounting mismatch')
        expected_calls=n if arm=='reranker' else sum(generated_counts)+1
        need(metrics['transformer_forward_calls']==expected_calls,'forward call accounting mismatch')
        expected_positions=active_sum if arm=='reranker' else layers*(n*ex['prefix_tokens_count']+n*(n+1)//2)
        need(metrics['layer_summed_valid_KV_positions']==expected_positions and metrics['logical_attention_head_pairs']==expected_positions*heads,'logical attention count mismatch')
        need(metrics['max_active_KV_entries']==(active_max if arm=='reranker' else position),'active maximum mismatch')
        if route:
            cfg=plan['model']['config'];dim=cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
            need(row['persistent_cache_bound_bytes']==layers*32*128*2*cfg['num_key_value_heads']*dim*2,'cache bound mismatch')
            need(row['metrics']['max_active_KV_entries']<=4048,'active metric exceeds limit')
            need(len(row['page_cache_stats'])==layers and len(row['candidate_stats'])==layers and len(row['normalizer_stats'])==layers,'per-layer counters missing')
            for cache,stat,events in zip(row['page_cache_stats'],row['candidate_stats'],layer_events):
                need(cache['peak_resident_pages']<=32 and cache['calls']==stat['calls'],'cache bound or call count differs')
                need(sum(stat['candidate_counts'].values())==stat['calls'] and sum(stat['final_counts'].values())==stat['calls'],'routing histogram incomplete')
                need(stat['calls']==len(events) and stat['candidate_counts']==dict(Counter(str(len(e['candidates'])) for e in events)) and stat['final_counts']==dict(Counter(str(len(e['pages'])) for e in events)),'routing histograms differ from events')
                need(cache['cache_hits']+cache['cache_misses']==sum(len(e['candidates']) for e in events),'candidate cache hit/miss accounting differs')
            row['_audited_route']=route
    marker=output/'complete.json';integrity=output/'integrity.json';has_marker=marker.exists()
    complete=has_marker and controller is not None and controller.get('status')=='complete' and controller.get('worker_returncode')==0
    if has_marker:
        need(integrity.exists() and len(rows)==len(schedule),'completion without full schedule');m=json.loads(marker.read_text());v=json.loads(integrity.read_text())
        need(m['status']=='complete' and m['rows']==len(rows) and m['weights_unchanged'] is True and m['plan_sha256']==PLAN_HASHES[label] and m['integrity_sha256']==sha(integrity) and m['samples_sha256']==sha(rows_path),'completion marker mismatch')
        need(v['plan_sha256']==PLAN_HASHES[label] and v['samples_sha256']==sha(rows_path) and v['weights_sha256_before']==v['weights_sha256_after']==plan['weights_sha256'] and v['source_identity']==plan['source_identity'],'integrity mismatch')
        for ex in plan['fixtures']:need(v['fixture_parity_sha256'][ex['id']]==sha(output/('parity-'+ex['id']+'.json')),'parity hash mismatch')
    provenance=controller_provenance(controller,required=complete,label=label)
    if complete:
        need(controller['plan_sha256']==PLAN_HASHES[label] and controller['result']['rows']==len(rows) and controller['model']==model,'controller completion differs')
        actual={p.name for p in output.iterdir() if p.is_file() and not p.name.startswith('overlay-')}
        need(actual==set(controller['result']['files_sha256']),'controller file inventory mismatch')
        for name,h in controller['result']['files_sha256'].items():need(sha(output/name)==h,'controller backup hash mismatch')
    elif not partial:raise ValueError('no verified controller completion; use --partial for partial audit')
    return {'model':label,'status':'complete' if complete else 'partial','rows':len(rows),'expected_rows':len(schedule),'records':records,'controller_provenance':provenance,
            'route_summaries':[{'fixture_id':r['fixture_id'],'rounds':r['_audited_route']} for r in rows if '_audited_route' in r],
            'resources':[{'fixture_id':r['fixture_id'],'arm':r['arm'],'prefill_s':r['prefill_s'],'mount_s':r['mount_s'],'init_s':r['init_s'],'request_s':r['request_s'],'checkpoint_s':r['checkpoint_s'],'logical_read_bytes':r['logical_read_bytes'],'peak_allocated':r['peak_allocated'],'peak_reserved':r['peak_reserved'],'host':r['host'],'candidate_stats':r['candidate_stats'],'page_cache_stats':r['page_cache_stats'],'normalizer_stats':r['normalizer_stats']} for r in rows],
            'scope':'one trial per arm; host RSS sampled, process I/O not physical-device traffic; request includes round artifact serialization; prefill is shared and separately recorded'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--qwen-output',type=Path,required=True);ap.add_argument('--smol-output',type=Path,required=True);ap.add_argument('--qwen-state',type=Path,required=True);ap.add_argument('--smol-state',type=Path,required=True);ap.add_argument('--partial',action='store_true');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.chdir(ROOT);reports=[]
    args.output.parent.mkdir(parents=True,exist_ok=True);need(not args.output.exists(),'report already exists')
    try:
        for label,output,state_path in [('qwen15',args.qwen_output,args.qwen_state),('smol17',args.smol_output,args.smol_state)]:
            tok,snap=tokenizer_and_snapshot(MODELS[label]);cfg=json.loads((snap/'generation_config.json').read_text());eos=cfg['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
            need(state_path.exists() or args.partial,'controller state missing')
            controller=json.loads(state_path.read_text()) if state_path.exists() else None
            need(controller is not None or not output.exists(),'output exists without controller state')
            report=audit_model(label,ROOT/'results/evidence/attention-transfer-preparation'/(label+'-plan.json'),output,tok,eos,args.partial,controller)
            report['controller_status']=controller.get('status') if controller else 'not_started';report['controller_state_sha256']=sha(state_path) if controller else None;reports.append(report)
        paired={}
        for report in reports:
            for r in report['records']:
                key=(r['model'],r['fixture_id'],r['budget'],r['task'],r['seed'],r['turn'])
                paired.setdefault(key,{})[r['arm']]=r['correct']
        outcomes=[dict(model=k[0],fixture_id=k[1],budget=k[2],task=k[3],seed=k[4],turn=k[5],**v) for k,v in paired.items() if set(v)=={'stock','reranker'}]
        grouped={}
        for item in outcomes:
            key=(item['model'],item['budget'],item['task'],item['turn'])
            d=grouped.setdefault(key,{'pairs':0,'stock_correct':0,'reranker_correct':0});d['pairs']+=1;d['stock_correct']+=item['stock'];d['reranker_correct']+=item['reranker']
        groups=[dict(model=k[0],budget=k[1],task=k[2],turn=k[3],**v) for k,v in sorted(grouped.items())]
        complete=all(r['status']=='complete' for r in reports)
        result={'status':'complete' if complete else 'partial','models':reports,'paired_outcomes':outcomes,'auditor_sha256':sha(Path(__file__)),
                'paired_groups':groups if complete else [],
                'scope':'fixed new-seed screen; route checks establish budget and causality, not selector-score replay; no statistical equivalence or single-trial speed claim'}
        if result['status']=='complete':need(len(outcomes)==36,'paired outcome matrix incomplete')
    except Exception as exc:
        args.output.write_text(json.dumps({'status':'failed_audit','error':repr(exc),'completed_model_audits':reports},indent=2,sort_keys=True,allow_nan=False)+'\n')
        raise
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'rows':sum(r['rows'] for r in reports)}))

if __name__=='__main__':main()

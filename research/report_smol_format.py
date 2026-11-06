"""Independent offline audit of the separately frozen Smol short-answer control.

Partial results are labelled and never promoted to a completed cohort. Token
replay requires only the pinned local tokenizer; model inference is not used.
"""
import argparse,hashlib,json,math,os
from collections import Counter
from pathlib import Path
from research.attention_smol_format_run import make_plan,tokenizer_and_snapshot,source_identity,WEIGHTS
from research.attention_smol_format_protocol import round_request,score_response
from research.persistent_kv import next_question_tokens

PLAN_HASHES={'smol17':'98b4f3083ded2b3180192d38671944a2b057ac74be120cc4e9cb4332c02ebfb4'}
MODELS={'smol17':'HuggingFaceTB/SmolLM2-1.7B-Instruct'}
# Must be bound to a separately reviewed deployment before completion claims.
FORMAT_CONTROLLER_SHA='5f8d8b678edda71f06791142f9e8f253854cec089beebd60e90ac6f2afe0fd30'
FORMAT_BUNDLE_SHA='aadadd93d0ed7aa1109d69cd8c63f38c1d7464688034d64a22e23705c268a223'
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)


def controller_provenance(controller,required=False,label=None):
    if controller is None:return None
    if not required and controller.get('controller_sha256') is None:return None
    need(FORMAT_CONTROLLER_SHA and FORMAT_BUNDLE_SHA,'format deployment not bound')
    need(controller.get('controller_sha256')==FORMAT_CONTROLLER_SHA and controller.get('bundle_sha256')==FORMAT_BUNDLE_SHA and label=='smol17','format deployment identity differs')
    predecessor={'receipt_sha256':'7e2e20f8f4a1fe0606d87296270a9773f522e6eb4f7b9686844f389f4e484d5c','state_sha256':'64056293bd8e1589b665c9481c0e35c9bcccdbbd7fa2853a134c0e7956f40fdf','controller':[30436,1793787725.62],'worker':[30741,1793787783.43]}
    prior={
      '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-1/state.json':'4c25c717dc6774a43878a4d8dd7e746f071f15fb4f4aa348f96e1fc2f2067ad4',
      '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-1/worker.log':'f51cbd87782b12b726dbe17df8389c78ea8e4b6ce49d19c42922a6915a51e423',
      '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-2/state.json':'f9c823264937806940b050e5897cd83859dfcccd9104c09e98690b0167c615a2',
      '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-2/output/complete.json':'cd41966e3193ac1bc02b65d93357f622512a9016f233a34d416c45827efc607c',
      '/redacted/runtime/tiered-kv-results/transfer-qwen15-a712857-2/output/integrity.json':'ce26f7e197baf77c9bc742e193c44480a6baf11012c804fca7c53fab27b48403',
      '/redacted/runtime/tiered-kv-results/transfer-smol17-a712857-fresh-1/state.json':'b888bf57128045e51022cd44243f57835e021a2b4dcb961684f6ee4ae41690d6'}
    need(controller.get('backend_boot_id')=='f2a4a03a-a73e-4bca-9557-f756da7da01c' and controller.get('predecessor')==predecessor and controller.get('prior_receipts')==prior,'observer or prior binding differs')
    need(controller.get('extra_helper_sha256')=='bc754b9ee17cad8fd799d8a97e0cf1bafcb6c5d5e20cf96bc67b7c07179a83ff','frozen helper differs')
    smoke=controller.get('import_smoke',{})
    need(smoke.get('status')=='passed' and 'research.attention_smol_format_run' in smoke.get('modules',[]) and 'research.attention_fullkey_oracle' in smoke.get('modules',[]),'format import smoke missing')
    return {'controller_sha256':FORMAT_CONTROLLER_SHA,'bundle_sha256':FORMAT_BUNDLE_SHA,'predecessor':predecessor,'scope':'separate format control; recorded terminal provenance, not independent predecessor audit'}


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
            need(roundrow['turn']==turn,'wrong round index');request=round_request(ex['query'],turn)
            expected=tok.encode(request+ex['first_suffix'],add_special_tokens=False) if turn==0 else next_question_tokens(tok,request,last,eos)
            need(roundrow['prompt_token_ids']==expected,'prompt replay mismatch')
            tokens=roundrow['generated_token_ids'];need(0<len(tokens)<=32 and all(type(t) is int and t>=0 for t in tokens),'generated token cap/type')
            need(all(type(t) is int and 0<=t<plan['model']['config']['vocab_size'] for t in expected+tokens),'token ID outside vocabulary')
            need(position+int(turn>0)+len(expected)+32<=min(ex['budget'],plan['model']['config']['max_position_embeddings']),'native context reserve exceeded')
            need(not any(t in eos for t in tokens[:-1]),'tokens follow EOS')
            need(tokens[-1] in eos or len(tokens)==32,'generation stopped before EOS or cap')
            need(all(math.isfinite(roundrow[k]) and roundrow[k]>=0 for k in ('prompt_s','generation_s')),'invalid round timing')
            need(len(roundrow['decode_ms'])==len(tokens)-1 and all(math.isfinite(x) and x>=0 for x in roundrow['decode_ms']),'invalid decode timings')
            response=tok.decode(tokens,skip_special_tokens=True);parsed=score_response(response,ex['answer']);query=ex['query'];details={'source':'original_question','uses_gold':False}
            need(response==roundrow['response'] and parsed==roundrow['parsed_response'] and query==roundrow['query'] and details==roundrow['query_details'],'output or query replay mismatch')
            answer=parsed['answer'];correct=parsed['exact_correct']
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


def paired_final(report):
    pairs={}
    for item in report['records']:
        if item['turn']!=1:continue
        key=(item['fixture_id'],item['task'],item['seed'])
        arms=pairs.setdefault(key,{})
        need(item['arm'] not in arms,'duplicate final-round arm')
        arms[item['arm']]=item['correct']
    need(len(pairs)==6 and all(set(v)=={'stock','reranker'} for v in pairs.values()),'final-round pair matrix incomplete')
    return [dict(fixture_id=k[0],task=k[1],seed=k[2],**v) for k,v in sorted(pairs.items())]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--partial',action='store_true');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    tok,snap=tokenizer_and_snapshot(MODELS['smol17']);cfg=json.loads((snap/'generation_config.json').read_text());eos=cfg['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
    controller=json.loads(args.state.read_text()) if args.state.exists() else None
    if not args.partial:need(controller is not None,'controller missing')
    report=audit_model('smol17',args.plan,args.run,tok,eos,args.partial,controller)
    outcomes=paired_final(report) if report['status']=='complete' else []
    result={'status':report['status'],'model':report,'paired_final_round':outcomes,'auditor_sha256':sha(Path(__file__)),'scope':'separate short-answer protocol; final round primary, first round descriptive; no pooling with original JSON scores'}
    if args.output.exists():raise ValueError('report already exists')
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'rows':report['rows']}))
if __name__=='__main__':main()

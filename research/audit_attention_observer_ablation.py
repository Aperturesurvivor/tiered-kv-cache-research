"""Independent saved-evidence audit for the declared observer ablation.

No inference or model load. Timed-off per-event routes cannot be reconstructed.
"""
import argparse,hashlib,json,math,os,sys
from pathlib import Path
from research.report_attention_observer_ablation import PLAN,RUNNER,WEIGHTS,schedule

ROOT=Path(__file__).resolve().parents[1]
CONTROLLER='3fcc70b48951ad626500e8fcb23cab6aa83b331691697b968e53b09def95e121'
BUNDLE='259fba2a50b0f60a15b7d1492f85fb4738f4a0dff96bcd1e65708f0f91ef1ad5'
BOOT='f2a4a03a-a73e-4bca-9557-f756da7da01c'
MANIFEST_SHA='01c794e56a75426eede82eccd2a0a2709ba908431816358c3d97524dab600a08'
MANIFEST_COUNT=74
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def need(ok,message):
 if not ok:raise ValueError(message)
def checked_manifest():
 manifest_file=ROOT/'results/evidence/attention-observer-ablation-preparation/source-manifest.json'
 need(sha(manifest_file)==MANIFEST_SHA,'manifest hash')
 manifest=read(manifest_file);need(len(manifest['members_sha256'])==MANIFEST_COUNT,'manifest member count')
 need(manifest['bundle_sha256']==BUNDLE,'manifest bundle identity')
 for name in ('research/reasoning_protocol.py','research/attention_transfer_run.py'):
  need(sha(ROOT/name)==manifest['members_sha256'][name],'local replay helper differs from frozen source')
 return manifest
def check_events(events,prefix,n,layers):
 need(len(events)==layers*n,'event count')
 for step in range(n):
  for layer in range(layers):
   l,pos,candidates,chosen,active,ids=events[step*layers+layer];cutoff=pos-1999
   need(l==layer and pos==prefix+step,'event ordering')
   need(len(candidates)==min(32,max(0,cutoff//128-1)) and len(candidates)==len(set(candidates)),'candidate count')
   need(len(chosen)==min(15,len(candidates)) and len(chosen)==len(set(chosen)) and set(chosen)<=set(candidates),'final subset')
   need(all(type(x) is int and x>=1 and (x+1)*128<=cutoff for x in candidates) and active==2128+128*len(chosen),'causality or active budget')
   need(isinstance(ids,str) and len(ids)==64 and all(c in '0123456789abcdef' for c in ids),'position hash')
 return True

def audit(output,state_path,terminal_path,plan_path,tokenizer):
 output=Path(output);state_path=Path(state_path);terminal_path=Path(terminal_path);plan_path=Path(plan_path)
 state=read(state_path);terminal=read(terminal_path);marker=read(output/'complete.json');initial=read(output/'initial-integrity.json')
 need(state.get('status')=='complete' and state.get('worker_returncode')==0,'controller not complete')
 need(state.get('controller_sha256')==CONTROLLER and state.get('bundle_sha256')==BUNDLE and state.get('plan_sha256')==PLAN,'controller source identity')
 need(state.get('backend_boot_id')==BOOT and (state.get('worker_pid'),state.get('worker_create_time'))==(30741,1793787783.43),'worker identity')
 need((terminal.get('controller_pid'),terminal.get('controller_create_time'))==(30436,1793787725.62),'controller identity')
 need((terminal.get('worker_pid'),terminal.get('worker_create_time'))==(30741,1793787783.43) and terminal.get('backend_boot_id')==BOOT,'terminal worker identity')
 need(terminal.get('controller_shell_wait_returncode')==0 and terminal.get('controller_pid_exists') is False and terminal.get('worker_pid_exists') is False,'direct terminal missing')
 files={p.name:sha(p) for p in output.iterdir() if p.is_file() and not p.is_symlink() and not p.name.endswith('.tmp')}
 need(files==state['result']['files_sha256'] and state['result']['rows']==12,'backup file inventory')
 need(marker.get('status')=='complete' and marker.get('rows')==12 and marker.get('samples_sha256')==files['samples.jsonl'],'marker identity')
 need(marker.get('weights_before')==marker.get('weights_after')==initial.get('weights_sha256')==WEIGHTS,'weights differ')
 need(sha(plan_path)==PLAN and initial.get('plan_sha256')==PLAN and marker['declaration']['plan_sha256']==PLAN,'plan hash')
 need(marker['declaration']['ablation_source_sha256']==RUNNER,'runner identity')
 manifest=checked_manifest()
 need(state['manifest']==manifest['members_sha256'],'source manifest')
 need(state['manifest']['research/attention_observer_ablation.py']==RUNNER and all(state['manifest'][k]==v for k,v in initial['extra_sources'].items()),'source helpers')
 plan=read(plan_path);need(initial['source_identity']==plan['source_identity'],'frozen source identity')
 rows=[json.loads(x) for x in (output/'samples.jsonl').read_text().splitlines()]
 need([(r['budget'],r['rep'],r['arm']) for r in rows]==schedule(),'schedule mismatch')
 from research.reasoning_protocol import round_request
 fixtures={e['budget']:e for e in plan['fixtures'] if e['seed']==283 and e['task']=='linked_fact'};need(set(fixtures)=={8192,32768},'fixture set')
 refs={};validations={}
 for budget in (8192,32768):
  e=fixtures[budget];w=read(output/f'workload-{budget}.json');v=read(output/f'validation-{budget}.json');n=len(w['prompt_token_ids'])+64
  need(w['fixture_id']==e['id'] and w['token_hash']==e['token_hash'] and w['prefix_count']==e['prefix_tokens_count'],'fixture identity')
  need(w['prompt_token_ids']==tokenizer.encode(round_request(e['query'],0,2)+e['first_suffix'],add_special_tokens=False) and w['forced_token_ids']==e['tokens'][e['prefix_tokens_count']-64:e['prefix_tokens_count']],'workload tokens')
  need(v['status']=='passed' and math.isfinite(v['max_abs_logit']) and 0<=v['max_abs_logit']<=1e-4,'validation logit')
  before=output/f'validation-{budget}-before.json';after=output/f'validation-{budget}-after.json'
  need(sha(before)==v['before_sha256'] and sha(after)==v['after_sha256'],'validation ledger hashes')
  a=read(before);b=read(after);need(a==b and v['events']==len(a['events'])==len(a['kv_sha256']),'validation equality')
  need(check_events(a['events'],w['prefix_count'],n,plan['model']['config']['num_hidden_layers']),'validation routes')
  need(all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in a['kv_sha256']),'KV digest')
  for rep in range(3):
   pair=read(output/f'pair-{budget}-{rep}.json');need(pair['status']=='passed' and math.isfinite(pair['max_abs_logit']) and 0<=pair['max_abs_logit']<=1e-4,'pair logits')
  refs[budget]=(a,w,n);validations[str(budget)]={'events':v['events'],'max_abs_logit':v['max_abs_logit']}
 for row in rows:
  ref,w,n=refs[row['budget']];need(row['metrics']==ref['metrics'] and row['metrics']['model_token_evaluations']==n and row['metrics']['prompt_token_evaluations']==n-64,'work counts')
  need(math.isfinite(row['final_logit_max_abs']) and 0<=row['final_logit_max_abs']<=1e-4,'final logit')
  need(len(row['forced_step_ms'])==64 and all(math.isfinite(x) and x>=0 for x in row['forced_step_ms']),'forced step times')
  need(all(math.isfinite(row[k]) and row[k]>=0 for k in ('mount_s','init_s','prompt_s','forced_s','serving_s','checkpoint_s','cleanup_s')),'timing values')
  if row['arm']=='on':
   name=f"events-{row['budget']}-{row['rep']}-on.json";need(row['events_file']==name and row['events_sha256']==files[name],'timed-on ledger')
   events=read(output/name)['events'];need(events==ref['events'] and check_events(events,w['prefix_count'],n,plan['model']['config']['num_hidden_layers']),'timed-on routes')
  else:need('events_file' not in row and 'events_sha256' not in row,'off route claim')
 return {'status':'verified_complete','rows':12,'samples_sha256':files['samples.jsonl'],'complete_sha256':files['complete.json'],'state_sha256':sha(state_path),'terminal_sha256':sha(terminal_path),'validation_passed':True,'timed_on_routes_verified':True,'timed_off_routes_observed':False,'source_files':len(state['manifest']),'output_files':len(files),'validations':validations,'scope':'Saved scalar logit checks and KV/position hashes are reconciled, but raw logits, tensors and IDs are not reconstructed. Off-arm per-event routing is unobserved. No inference or model load.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 need(not a.report.exists(),'report already exists');os.environ['HF_HUB_OFFLINE']='1';checked_manifest()
 from research.attention_transfer_run import tokenizer_and_snapshot
 tok,_=tokenizer_and_snapshot('Qwen/Qwen2.5-1.5B-Instruct');result=audit(a.output,a.state,a.terminal,a.plan,tok)
 a.report.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':result['status'],'rows':result['rows']}))
if __name__=='__main__':main()

"""Describe a separately audited complete matched-workload cost attempt.

This is a derivation, not an integrity auditor. It requires the independently
produced verified-complete receipt and the exact files to which it is bound.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path

ARMS=('stock','cpu','device')
WEIGHTS='fdf003e3fecca614b5f4cd70e02a7a0f7e28f9422bcd4fb509dcb025a8a4470a'
PLAN='dd31e825291215a5e35db43d11e79ae6de5b7a23e725f45e132066f96a943a29'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def describe(values):
 if not values or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid observed values')
 return {'values':values,'mean':statistics.mean(values),'median':statistics.median(values),'min':min(values),'max':max(values)}
def schedule():return [(b,r,ARMS[(i+r)%3]) for b in (8192,32768) for r in range(3) for i in range(3)]

def derive(output,state_path,terminal_path,audit_path):
 output=Path(output);state_path=Path(state_path);terminal_path=Path(terminal_path);audit_path=Path(audit_path)
 audit=read(audit_path);marker_path=output/'complete.json';samples_path=output/'samples.jsonl'
 required={'samples_sha256':sha(samples_path),'complete_sha256':sha(marker_path),'state_sha256':sha(state_path),'terminal_sha256':sha(terminal_path)}
 if audit.get('status')!='verified_complete' or audit.get('rows')!=18 or any(audit.get(k)!=v for k,v in required.items()):raise ValueError('independent complete audit or exact input hashes missing')
 state=read(state_path);terminal=read(terminal_path);marker=read(marker_path)
 if state.get('status')!='complete' or state.get('worker_returncode')!=0 or terminal.get('controller_shell_wait_returncode')!=0 or terminal.get('controller_pid_exists') is not False or terminal.get('worker_pid_exists') is not False:raise ValueError('terminal success not established')
 if marker.get('status')!='complete' or marker.get('rows')!=18 or marker.get('samples_sha256')!=required['samples_sha256'] or marker.get('weights_before')!=WEIGHTS or marker.get('weights_after')!=WEIGHTS or marker['declaration']['plan_sha256']!=PLAN:raise ValueError('completion identity mismatch')
 rows=[json.loads(line) for line in samples_path.read_text().splitlines()]
 if [(r['budget'],r['rep'],r['arm']) for r in rows]!=schedule():raise ValueError('incomplete or changed schedule')
 table={}; groups=[]; pairs=[]
 fields=('prompt_s','forced_s','serving_s','mount_s','init_s','checkpoint_s','cleanup_s','index_build_s','logical_read_bytes','peak_allocated','peak_reserved','init_peak_allocated','init_peak_reserved','bound_metadata_bytes','cache_bound_bytes')
 for r in rows:
  steps=r['forced_step_ms'];describe(steps)
  if len(steps)!=64:raise ValueError('wrong fixed continuation length')
  item={key:r[key] for key in fields}
  item['step_mean_ms']=statistics.mean(steps);item['step_median_ms']=statistics.median(steps);item['step_min_ms']=min(steps);item['step_max_ms']=max(steps)
  item['accounted_service_s']=sum(r[k] for k in ('mount_s','init_s','serving_s','checkpoint_s','cleanup_s'))
  item['host_sampled_peak_rss_bytes']=r['host']['host_rss_sampled_peak_bytes']
  item['process_read_bytes']=r['host']['process_io_delta']['read_bytes'];item['process_read_chars']=r['host']['process_io_delta']['read_chars']
  item['forced_step_ms']=steps;item['model_token_evaluations']=r['metrics']['model_token_evaluations'];item['prompt_token_evaluations']=r['metrics']['prompt_token_evaluations'];item['max_active_KV_entries']=r['metrics']['max_active_KV_entries']
  for k,v in item.items():
   if k!='forced_step_ms':describe([v])
  item['telemetry']={'page_cache_stats':r['page_cache_stats'],'normalizer_stats':r['normalizer_stats'],'selector_stats':r['selector_stats']}
  table[(r['budget'],r['rep'],r['arm'])]=item
 for budget in (8192,32768):
  per_arm={}
  for arm in ARMS:
   trials=[table[(budget,rep,arm)] for rep in range(3)]
   summaries={k:describe([x[k] for x in trials]) for k in trials[0] if k not in ('forced_step_ms','telemetry')}
   per_arm[arm]={'replicates':trials,'summary':summaries}
  groups.append({'budget':budget,'arms':per_arm,'prefix_construction':marker['prefix_construction'][str(budget)]})
  for rep in range(3):
   trials={a:table[(budget,rep,a)] for a in ARMS}
   ratios={}
   for label,numerator,denominator in (('cpu_over_stock','cpu','stock'),('device_over_stock','device','stock'),('device_over_cpu','device','cpu')):
    ratios[label]={}
    for key in ('prompt_s','forced_s','serving_s','accounted_service_s','step_mean_ms','step_median_ms'):
     if trials[denominator][key]<=0:raise ValueError('zero ratio denominator')
     ratios[label][key]=trials[numerator][key]/trials[denominator][key]
   pairs.append({'budget':budget,'rep':rep,'ratios':ratios})
 ratio_summaries=[]
 for budget in (8192,32768):
  matched=[p for p in pairs if p['budget']==budget]
  ratio_summaries.append({'budget':budget,'ratios':{label:{key:describe([p['ratios'][label][key] for p in matched]) for key in matched[0]['ratios'][label]} for label in matched[0]['ratios']}})
 return {'status':'derived_from_independently_audited_complete_run','input_hashes':{**required,'audit_sha256':sha(audit_path)},'groups':groups,'matched_rep_ratios':pairs,'ratio_summaries':ratio_summaries,
 'definitions':{'serving_s':'synchronized entire prompt plus fixed 64-token teacher-forced serving; includes sparse route/position-hash instrumentation','forced_s':'64 synchronized forced steps plus loop/guard overhead; forced_step_ms times each synchronized model step','accounted_service_s':'sum of separately timed mount, init, serving, checkpoint and cleanup; not a continuous wall measurement and excludes shared prefix construction, inter-component bookkeeping and post-timer evidence serialization','index_build_s':'reported absorb timer inside init_s; not added again','prefix_construction':'one shared per-length wall and prefill time; not added to per-request total','cuda':'init allocated/reserved peaks are separate; post-reset peaks include serving and checkpoint, with model and allocator included; not whole-system memory','host':'sampled process RSS from init through cleanup; excludes mount and shared construction','logical_read_bytes':'store logical read accounting, not physical SSD traffic','process_read_bytes':'OS process I/O accounting, not guaranteed physical SSD traffic','cache_bound_bytes':'configured upper bound for extra resident candidate KV, not measured whole-device memory'},
 'scope':'Three warm balanced-order matched teacher-forced requests per arm and length, each with 64 forced tokens. Tokens are identical across arms within a length; the two length fixtures can differ in prefix, prompt and forced IDs, so cross-length comparisons do not isolate context length or establish asymptotic behavior. OS/model caches are not purged, serialization excluded and sparse-only observation adds overhead. No statistical significance, answer-quality, 1M-context or physical-SSD claim.'}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);ap.add_argument('--terminal',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 result=derive(a.output,a.state,a.terminal,a.audit)
 if a.report.exists():raise ValueError('refusing to overwrite a report')
 a.report.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()

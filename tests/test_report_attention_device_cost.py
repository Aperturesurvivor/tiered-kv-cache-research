import hashlib,json
import pytest
from research.report_attention_device_cost import derive,schedule,WEIGHTS,PLAN

def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,sort_keys=True))

def prepare(tmp):
 out=tmp/'output';out.mkdir();rows=[]
 for b,rep,arm in schedule():
  factor={'stock':1,'cpu':4,'device':2}[arm];duration=float(factor+rep/10)
  rows.append(dict(budget=b,rep=rep,arm=arm,prompt_s=duration,forced_s=duration,serving_s=duration*2,mount_s=1,init_s=2,checkpoint_s=3,cleanup_s=4,index_build_s=0,logical_read_bytes=0,peak_allocated=1,peak_reserved=2,init_peak_allocated=1,init_peak_reserved=2,bound_metadata_bytes=0,cache_bound_bytes=0,forced_step_ms=[duration]*64,host={'host_rss_sampled_peak_bytes':1,'process_io_delta':{'read_bytes':0,'read_chars':1}},metrics={'model_token_evaluations':160,'prompt_token_evaluations':96,'max_active_KV_entries':4048},page_cache_stats=None,normalizer_stats=None,selector_stats=None))
 samples=out/'samples.jsonl';samples.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 marker=out/'complete.json';write(marker,{'status':'complete','rows':18,'samples_sha256':h(samples),'weights_before':WEIGHTS,'weights_after':WEIGHTS,'declaration':{'plan_sha256':PLAN},'prefix_construction':{'8192':{'wall_s':10,'prefill_s':8},'32768':{'wall_s':20,'prefill_s':18}}})
 state=tmp/'state.json';write(state,{'status':'complete','worker_returncode':0});terminal=tmp/'terminal.json';write(terminal,{'controller_shell_wait_returncode':0,'controller_pid_exists':False,'worker_pid_exists':False});audit=tmp/'audit.json'
 def reset():write(audit,{'status':'verified_complete','rows':18,'samples_sha256':h(samples),'complete_sha256':h(marker),'state_sha256':h(state),'terminal_sha256':h(terminal)})
 reset();return out,state,terminal,audit,rows,reset

def test_summary_and_no_double_count(tmp_path):
 out,state,term,audit,rows,_=prepare(tmp_path);x=derive(out,state,term,audit)
 assert len(x['groups'])==2 and len(x['matched_rep_ratios'])==6
 assert x['groups'][0]['arms']['stock']['replicates'][0]['accounted_service_s']==12
 assert x['matched_rep_ratios'][0]['ratios']['cpu_over_stock']['serving_s']==4
 assert x['groups'][0]['prefix_construction']['wall_s']==10

def test_reject_missing_or_mismatched_independent_audit(tmp_path):
 out,state,term,audit,rows,reset=prepare(tmp_path);x=json.loads(audit.read_text());x['status']='partial';write(audit,x)
 with pytest.raises(ValueError):derive(out,state,term,audit)
 reset();(out/'samples.jsonl').write_text((out/'samples.jsonl').read_text()+' ')
 with pytest.raises(ValueError):derive(out,state,term,audit)

def test_reject_partial_and_changed_schedule_even_with_rebound_hashes(tmp_path):
 out,state,term,audit,rows,reset=prepare(tmp_path);rows.pop();samples=out/'samples.jsonl';samples.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 marker=out/'complete.json';x=json.loads(marker.read_text());x['samples_sha256']=h(samples);write(marker,x);reset()
 with pytest.raises(ValueError):derive(out,state,term,audit)

def test_reject_terminal_without_direct_shell_wait(tmp_path):
 out,state,term,audit,rows,reset=prepare(tmp_path)
 write(term,{'controller_returncode':0,'controller_pid_exists':False,'worker_pid_exists':False});reset()
 with pytest.raises(ValueError):derive(out,state,term,audit)

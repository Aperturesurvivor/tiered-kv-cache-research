import hashlib,json
import pytest
import research.audit_attention_observer_ablation as audit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,sort_keys=True))
class Tok:
 def encode(self,*a,**kw):return [7]

def setup_run(tmp,monkeypatch):
 root=tmp/'root';root.mkdir();monkeypatch.setattr(audit,'ROOT',root)
 manifestfile=root/'results/evidence/attention-observer-ablation-preparation/source-manifest.json';manifestfile.parent.mkdir(parents=True)
 helperdir=root/'research';helperdir.mkdir();(helperdir/'reasoning_protocol.py').write_text('prompt');(helperdir/'attention_transfer_run.py').write_text('tokenizer')
 manifest={'research/attention_observer_ablation.py':audit.RUNNER,'research/x.py':'a'*64,'research/reasoning_protocol.py':sha(helperdir/'reasoning_protocol.py'),'research/attention_transfer_run.py':sha(helperdir/'attention_transfer_run.py')};write(manifestfile,{'bundle_sha256':audit.BUNDLE,'members_sha256':manifest});monkeypatch.setattr(audit,'MANIFEST_SHA',sha(manifestfile));monkeypatch.setattr(audit,'MANIFEST_COUNT',4)
 fixtures=[{'budget':b,'seed':283,'task':'linked_fact','id':str(b),'token_hash':'x','prefix_tokens_count':2304,'query':'q','first_suffix':'','tokens':list(range(2305))} for b in (8192,32768)]
 plan={'fixtures':fixtures,'source_identity':{'x':'y'},'model':{'config':{'num_hidden_layers':2}}};planfile=tmp/'plan.json';write(planfile,plan);monkeypatch.setattr(audit,'PLAN',sha(planfile))
 out=tmp/'output';out.mkdir();rows=[]
 for b in (8192,32768):
  w={'fixture_id':str(b),'token_hash':'x','prefix_count':2304,'prompt_token_ids':[7],'forced_token_ids':fixtures[(0 if b==8192 else 1)]['tokens'][2240:2304]};write(out/f'workload-{b}.json',w)
  events=[]
  for t in range(65):
   pos=2304+t;eligible=max(0,(pos-1999)//128-1);c=list(range(1,min(32,eligible)+1));chosen=c[:15]
   for layer in range(2):events.append([layer,pos,c,chosen,2128+128*len(chosen),'a'*64])
  data={'events':events,'kv_sha256':['b'*64]*len(events),'metrics':{'model_token_evaluations':65,'prompt_token_evaluations':1}}
  for stage in ('before','after'):write(out/f'validation-{b}-{stage}.json',data)
  write(out/f'validation-{b}.json',{'status':'passed','max_abs_logit':0,'events':len(events),'before_sha256':sha(out/f'validation-{b}-before.json'),'after_sha256':sha(out/f'validation-{b}-after.json')})
  for rep in range(3):write(out/f'pair-{b}-{rep}.json',{'status':'passed','max_abs_logit':0})
  for budget,rep,arm in audit.schedule():
   if budget!=b:continue
   row={'budget':b,'rep':rep,'arm':arm,'metrics':data['metrics'],'final_logit_max_abs':0,'forced_step_ms':[1.]*64,**{k:1. for k in ('mount_s','init_s','prompt_s','forced_s','serving_s','checkpoint_s','cleanup_s')}}
   if arm=='on':
    name=f'events-{b}-{rep}-on.json';write(out/name,{'events':events});row.update(events_file=name,events_sha256=sha(out/name))
   rows.append(row)
 samples=out/'samples.jsonl';samples.write_text(''.join(json.dumps(x)+'\n' for x in rows));write(out/'initial-integrity.json',{'weights_sha256':audit.WEIGHTS,'plan_sha256':audit.PLAN,'source_identity':plan['source_identity'],'extra_sources':{'research/x.py':'a'*64}})
 write(out/'complete.json',{'status':'complete','rows':12,'samples_sha256':sha(samples),'weights_before':audit.WEIGHTS,'weights_after':audit.WEIGHTS,'declaration':{'plan_sha256':audit.PLAN,'ablation_source_sha256':audit.RUNNER}})
 state=tmp/'state.json';term=tmp/'terminal.json'
 def save_state():
  files={p.name:sha(p) for p in out.iterdir()};write(state,{'status':'complete','worker_returncode':0,'controller_sha256':audit.CONTROLLER,'bundle_sha256':audit.BUNDLE,'plan_sha256':audit.PLAN,'backend_boot_id':audit.BOOT,'worker_pid':30741,'worker_create_time':1793787783.43,'manifest':manifest,'result':{'rows':12,'files_sha256':files}})
 save_state();write(term,{'controller_pid':30436,'controller_create_time':1793787725.62,'worker_pid':30741,'worker_create_time':1793787783.43,'backend_boot_id':audit.BOOT,'controller_shell_wait_returncode':0,'controller_pid_exists':False,'worker_pid_exists':False})
 return out,state,term,planfile,save_state

def test_complete_and_evidence_tampering(tmp_path,monkeypatch):
 out,state,term,plan,save=setup_run(tmp_path,monkeypatch);x=audit.audit(out,state,term,plan,Tok());assert x['status']=='verified_complete' and x['timed_off_routes_observed'] is False
 f=out/'pair-8192-0.json';write(f,{'status':'failed','max_abs_logit':0});save()
 with pytest.raises(ValueError,match='pair logits'):audit.audit(out,state,term,plan,Tok())

def test_routes_reject_future_and_wrong_count():
 assert audit.check_events([[0,2304,[1],[1],2256,'a'*64]],2304,1,1)
 with pytest.raises(ValueError):audit.check_events([[0,2304,[2],[2],2256,'a'*64]],2304,1,1)
 with pytest.raises(ValueError):audit.check_events([],2304,1,1)


def test_reject_drifted_local_prompt_helper(tmp_path,monkeypatch):
 out,state,term,plan,save=setup_run(tmp_path,monkeypatch)
 (audit.ROOT/'research/reasoning_protocol.py').write_text('changed')
 with pytest.raises(ValueError,match='local replay helper'):audit.audit(out,state,term,plan,Tok())

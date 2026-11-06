import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from .config import load_matrix, MODEL_PATH, MODEL_REVISION, load_eval
from .types import digest
from .store import atomic_json, file_sha
from .telemetry import doctor, disk_preflight, ResourceGuard, ResourceAbort, runtime_identity
from .evals import prepare
from .generate import RunEngine, append_jsonl, load_model, online_smoke, weights_digest
from .validation import validate, require_gates, implementation_hash
from .report import build_report, read_jsonl


def fixtures(profile):
    path=Path('artifacts/data')/f'{profile}.json'
    if not path.exists():
        return prepare(profile)
    doc=json.loads(path.read_text())
    identity={k:v for k,v in doc.items() if k!='hash'}
    if digest(identity)!=doc['hash'] or doc['spec']!=load_eval()[profile]:
        raise ValueError('fixture hash/profile mismatch; preserve and investigate')
    return doc


def make_run(args,configs,fixture):
    identity={'profile':args.profile,'configs':[asdict(c) for c in configs],
              'fixture_hash':fixture['hash'],'implementation_hash':implementation_hash(),
              'lock_sha256':file_sha('requirements.lock'),'model_revision':MODEL_REVISION,
              'runtime':runtime_identity()}
    identity_hash=digest(identity)
    if args.resume:
        candidates=[]
        for path in Path('results').glob('*/manifest.json'):
            doc=json.loads(path.read_text())
            if doc.get('identity_hash')==identity_hash:
                candidates.append((path,doc))
        if candidates:
            path,doc=sorted(candidates,key=lambda p:p[0].parent.name)[-1]
            return path.parent,doc
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+identity_hash[:8]
    root=Path('results')/run_id
    root.mkdir(parents=True)
    def git_read(arguments):
        p=subprocess.run(['git',*arguments],capture_output=True,text=True)
        return p.stdout.strip() if p.returncode==0 else 'unavailable'
    manifest={**identity,'identity_hash':identity_hash,'run_id':run_id,'created_utc':datetime.now(timezone.utc).isoformat(),
        'git_revision':git_read(['rev-parse','HEAD']),'git_status':git_read(['status','--short']),
        'dirty_diff_hash':digest(git_read(['diff','--no-ext-diff'])),'seeds':[11,23,37,41],
        'system':{'OS':platform.platform(),'machine':platform.machine()},'precision':'float16','kernel':'explicit-mask torch SDPA; manually expanded GQA',
        'protocol':'online_tiered' if args.profile=='online-smoke' else 'exact_shared','prefill_chunk':128,
        'data':{k:v for k,v in fixture.items() if k!='examples'},
        'examples':[{k:v for k,v in e.items() if k not in ('tokens','continuation','references','answer','item')} for e in fixture['examples']],
        'scratch_policy':'preserve everything; no automatic deletion','filesystem_cache':'OS-cache-uncontrolled',
        'model_config_sha256':file_sha(MODEL_PATH/'config.json'),'start_disk':disk_preflight(),
        'validation':None}
    atomic_json(root/'manifest.json',manifest)
    atomic_json(root/'fixtures.json',fixture)
    for name in ('samples.jsonl','tokens.jsonl','retrieval.jsonl','latency.jsonl','resources.jsonl','diagnostics.jsonl'):
        (root/name).touch(exist_ok=False)
    return root,manifest


def run(args):
    disk_preflight()
    configs=load_matrix(args.matrix)
    if args.profile=='online-smoke':
        configs=[replace(c,protocol='online_tiered') for c in configs if c.id in ('full','default')]
    fixture=fixtures('pilot' if args.profile=='online-smoke' else args.profile)
    root,manifest=make_run(args,configs,fixture)
    print(f'Run {root}',flush=True)
    env=doctor(root/'doctor.json')
    try:
        if env['status']!='passed':
            raise RuntimeError('gate 1 failed: '+'; '.join(env['errors']))
        state=require_gates()
        manifest['validation']=state
    except Exception as exc:
        manifest['blocked_reason']=str(exc)
        atomic_json(root/'manifest.json',manifest)
        build_report(root)
        print(manifest['blocked_reason'],file=sys.stderr)
        return 1
    # Clear only a prior blocked status in the current manifest; retain doctor logs.
    manifest.pop('blocked_reason',None)
    atomic_json(root/'manifest.json',manifest)
    try:
        return execute_run(args,configs,fixture,root,manifest)
    except (Exception,KeyboardInterrupt) as exc:
        manifest['blocked_reason']=f'{type(exc).__name__}: {exc}'
        atomic_json(root/'manifest.json',manifest)
        build_report(root)
        print(manifest['blocked_reason'],file=sys.stderr)
        return 1


def execute_run(args,configs,fixture,root,manifest):
    with ResourceGuard(sink=lambda s:append_jsonl(root/'resources.jsonl',s)) as guard:
        model,tok=load_model()
        frozen=weights_digest(model)
        if args.profile=='online-smoke':
            rows=online_smoke(model,tok,root,[c for c in configs if c.id in ('full','default')],guard)
            unchanged=weights_digest(model)==frozen
            atomic_json(root/'online-result.json',{'rows':rows,'weights_unchanged':unchanged})
            (root/'report.md').write_text('Online 2K ingestion/resume, separate from shared-prefill ranking.\n'+json.dumps(rows,indent=2)+'\n')
            return 0 if unchanged and len(rows)==2 and all(r['status']=='ok' for r in rows) else 1
        engine=RunEngine(model,tok,root,guard)
        done={(s['config_hash'],s['example_id']) for s in read_jsonl(root/'samples.jsonl') if s['status']=='ok'}
        remaining=sum((c.digest,e['id']) not in done for e in fixture['examples'] for c in configs)
        print(f'{remaining} resumable configuration/example rows remain.',flush=True)
        rng=np.random.default_rng(41)
        calibrated=False
        interrupted=False
        try:
            for example in fixture['examples']:
                order=rng.permutation(len(configs))
                pending=[configs[int(i)] for i in order if (configs[int(i)].digest,example['id']) not in done]
                if not pending:
                    continue
                disk_preflight()
                guard.check()
                # Preserve-all disk forecast includes the shared prefix plus all
                # overlays/state. No cleanup is performed to make a run fit.
                import shutil
                estimate=(len(example['tokens'])-1)*28672+len(pending)*(example['max_new_tokens']*28672+16*1024**2)
                if shutil.disk_usage('.').free-estimate<5*10**9:
                    raise ResourceAbort(f'next example needs approximately {estimate/1024**3:.2f} GiB plus 5 GB reserve; all existing files retained')
                prefix=engine.prefix(example)
                for config in pending:
                    guard.check()
                    row=engine.run(config,example,prefix_info=prefix)
                    row.update(manifest_hash=manifest['identity_hash'],resource_scope='run cumulative peak')
                    tokens=row.pop('token_ids',None)
                    if tokens is not None:
                        append_jsonl(root/'tokens.jsonl',{'config_id':config.id,'example_id':example['id'],'tokens':tokens,'prompt_token_hash':example['token_hash']})
                    if row.get('retrieval'):
                        append_jsonl(root/'retrieval.jsonl',{'config_id':config.id,'example_id':example['id'],**row['retrieval']})
                    if row.get('latency_ms'):
                        append_jsonl(root/'latency.jsonl',{'config_id':config.id,'example_id':example['id'],'budget':example['budget'],'trial':example['trial'],'ms':row['latency_ms']})
                    append_jsonl(root/'samples.jsonl',row)
                    print(f"{example['id']} / {config.id}: {row['status']}",flush=True)
                    if row['status']=='ok':
                        done.add((config.digest,example['id']))
                    else:
                        # Correctness failures stop the matrix; avoid treating
                        # an implementation failure as a negative quality score.
                        raise RuntimeError(row['failure_reason'])
                    if not calibrated and example['task']=='niah' and example['budget']==8192:
                        remain=sum((c.digest,e['id']) not in done for e in fixture['examples'] for c in configs)
                        forecast=remain*(row['elapsed_s']+row['prefill_s']/len(configs))
                        calibration={'calibration_example':example['id'],'config':config.id,'measured_row_s':row['elapsed_s'],
                             'shared_prefill_s':row['prefill_s'],'remaining_rows':remain,'estimated_remaining_hours':forecast/3600,
                             'limitations':'single 8K extrapolation; 32K prefill, 512-token summaries and k32 can cost more'}
                        atomic_json(root/'runtime-estimate.json',calibration)
                        print(json.dumps(calibration),flush=True)
                        calibrated=True
                build_report(root)
            # Outside headline measurements: one fixed 50%-depth needle / length.
            default=next(c for c in configs if c.id=='default')
            for ex in fixture['examples']:
                if ex['task']=='niah' and ex['depth']==.5 and ex['seed']==11:
                    prefix=engine.prefix(ex)
                    for oracle in (False,True):
                        row=engine.run(default,ex,prefix_info=prefix,diagnostic=True,oracle=oracle)
                        append_jsonl(root/'diagnostics.jsonl',row)
                        if row['status']!='ok':
                            raise RuntimeError('diagnostic failed: '+row['failure_reason'])
            if weights_digest(model)!=frozen:
                raise ValueError('frozen weights changed')
            manifest['weights_sha256']=frozen
            manifest['weights_unchanged']=True
            atomic_json(root/'manifest.json',manifest)
            atomic_json(root/'validation.json',{'gates':manifest['validation']['gates']+[
                {'gate':6,'status':'passed','samples_expected':len(fixture['examples'])*len(configs),
                 'weights_unchanged':True,'diagnostics':'diagnostics.jsonl',
                 'online_evidence':'../validation/online.jsonl'}]})
        except (Exception,KeyboardInterrupt) as exc:
            interrupted=True
            manifest['blocked_reason']=f'{type(exc).__name__}: {exc}'
            atomic_json(root/'manifest.json',manifest)
        finally:
            engine.adapter.uninstall()
            ranked=build_report(root)
    return 0 if not interrupted and all(r['status']=='complete' for r in ranked) else 1


def main():
    parser=argparse.ArgumentParser(description='Frozen Qwen2 tiered cache experiment; timed accelerator runs only (CUDA/MPS, CPU prohibited)')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    prep=sub.add_parser('prepare')
    prep.add_argument('--profile',choices=['pilot','confirm'],default='pilot')
    val=sub.add_parser('validate')
    val.add_argument('--through',type=int,choices=range(1,6),default=5)
    exe=sub.add_parser('run')
    exe.add_argument('--matrix',default='configs/matrix.yaml')
    exe.add_argument('--profile',choices=['pilot','confirm','online-smoke'],default='pilot')
    exe.add_argument('--resume',action='store_true')
    rep=sub.add_parser('report')
    rep.add_argument('--run',required=True)
    args=parser.parse_args()
    if args.command=='doctor':
        result=doctor()
        print(json.dumps(result,indent=2))
        return 0 if result['status']=='passed' else 1
    if args.command=='prepare':
        doc=prepare(args.profile)
        print(json.dumps({'profile':args.profile,'fixture_hash':doc['hash'],'examples':len(doc['examples']),'band_available':doc['band_available']},indent=2))
        return 0
    if args.command=='validate':
        state=validate(args.through)
        print(json.dumps(state,indent=2))
        return 0 if len(state['gates'])==args.through and all(g['status']=='passed' for g in state['gates']) else 1
    if args.command=='report':
        rows=build_report(args.run)
        print(f'{len(rows)} configuration rows; {sum(r["status"]=="complete" for r in rows)} complete')
        return 0
    return run(args)


if __name__=='__main__':
    sys.exit(main())

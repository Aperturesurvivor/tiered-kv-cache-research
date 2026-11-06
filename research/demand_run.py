"""Sequential demand-KV runs with durable backups and explicit failure state."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from tiered_kv.store import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--semantic-root',type=Path)
    p.add_argument('--stage',choices=['ablate','screen','confirm','online','persistent','summaries','eviction'],default='ablate')
    args=p.parse_args()
    root=Path('results')/('demand-dispatch-'+args.stage)
    root.mkdir(parents=True,exist_ok=True)
    seen={}
    state={'stage':args.stage,'jobs':[],'started':time.time(),'status':'running'}
    def backup():
        for source in Path('results').rglob('*'):
            if source.is_file():
                stat=source.stat()
                signature=(stat.st_size,stat.st_mtime_ns)
                if seen.get(str(source))!=signature:
                    target=args.backup/source
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
                    seen[str(source)]=signature
    def run(label,command):
        row={'label':label,'command':command,'started':time.time(),'status':'running'}
        state['jobs'].append(row)
        atomic_json(root/'state.json',state)
        print('JOB_START',label,flush=True)
        with (root/(label+'.log')).open('a') as log:
            job=subprocess.Popen([sys.executable,'-u',*command],stdout=log,stderr=subprocess.STDOUT)
            try:
                while job.poll() is None:
                    backup()
                    try:
                        job.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                job.terminate()
                job.wait(timeout=30)
                raise
        row.update(returncode=job.returncode,status='complete' if job.returncode==0 else 'failed',finished=time.time())
        atomic_json(root/'state.json',state)
        backup()
        print('JOB_END',label,job.returncode,flush=True)
        return job.returncode
    if run('cuda-tests',['-m','pytest','-q']):
        raise RuntimeError('CUDA mechanics tests failed')
    if args.stage=='ablate':
        if not args.semantic_root:
            p.error('ablate requires --semantic-root')
        indices=list((args.semantic_root/'results/semantic-index-screen').glob('*/complete.json'))
        finished=list((args.semantic_root/'results').glob('semantic-kv-*/complete.json'))
        if len(indices)!=1 or len(finished)!=1:
            raise RuntimeError('expected exactly one completed semantic study before ablation')
        run('qwen-ablation',['-m','research.demand_kv','--stage','ablate','--reuse-index',str(indices[0].parent),
            '--reuse-prefixes',str(args.semantic_root/'artifacts/scratch'/finished[0].parent.name)])
    elif args.stage=='persistent':
        run('persistent-queries',['-m','research.persistent_kv'])
    elif args.stage=='summaries':
        run('complete-report-summary',['-m','research.demand_kv','--stage','summaries'])
    elif args.stage=='eviction':
        if not args.semantic_root:
            p.error('eviction probe requires --semantic-root')
        run('eviction-pairs',['-m','research.eviction_probe','--semantic-root',str(args.semantic_root)])
    else:
        specs=json.loads(Path('research/model-manifest.json').read_text())
        order=[0,3,5,1,4,6,2] if args.stage=='screen' else [1,4,5]
        for i in order:
            model=specs[i]['model_id']
            run(model.replace('/','--'),['-m','research.demand_kv','--model',model,'--stage',args.stage])
    state.update(finished=time.time(),status='complete' if all(r['returncode']==0 for r in state['jobs']) else 'partial_failure')
    atomic_json(root/'state.json',state)
    backup()
    if state['status']!='complete':
        raise SystemExit(1)
    print('DEMAND_STAGE_DONE',args.stage,flush=True)


if __name__=='__main__':
    main()

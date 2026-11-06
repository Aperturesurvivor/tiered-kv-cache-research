"""Sequential GPU study with durable small-file backups during real computation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from tiered_kv.store import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backup',type=Path,required=True)
    parser.add_argument('--stage',choices=['screen','online','confirm'],default='screen')
    args=parser.parse_args()
    root=Path('results')/('batch-'+args.stage)
    root.mkdir(parents=True,exist_ok=True)
    specs=json.loads(Path('research/model-manifest.json').read_text())
    # Cover the three families early, then increase size.
    order=[0,3,5,1,4,6,2] if args.stage=='screen' else [1,4,5]
    seen={}
    state={'stage':args.stage,'jobs':[],'started':time.time()}
    def backup():
        for source in Path('results').rglob('*'):
            if not source.is_file():
                continue
            stat=source.stat()
            signature=(stat.st_size,stat.st_mtime_ns)
            if seen.get(str(source))==signature:
                continue
            dest=args.backup/source
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
            seen[str(source)]=signature
    for i in order:
        spec=specs[i]
        logfile=root/(spec['model_id'].replace('/','--')+'.log')
        row={'model_id':spec['model_id'],'started':time.time(),'status':'running','log':str(logfile)}
        state['jobs'].append(row)
        atomic_json(root/'state.json',state)
        print('JOB_START',args.stage,spec['model_id'],flush=True)
        with logfile.open('a') as log:
            job=subprocess.Popen([sys.executable,'-u','-m','research.generalize','--model',spec['model_id'],'--stage',args.stage],stdout=log,stderr=subprocess.STDOUT)
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
        row.update(status='complete' if job.returncode==0 else 'failed',returncode=job.returncode,finished=time.time())
        atomic_json(root/'state.json',state)
        backup()
        print('JOB_END',spec['model_id'],job.returncode,flush=True)
    state.update(finished=time.time(),status='complete' if all(j['returncode']==0 for j in state['jobs']) else 'partial_failure')
    atomic_json(root/'state.json',state)
    backup()
    if state['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()

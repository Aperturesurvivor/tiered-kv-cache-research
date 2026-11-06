"""Sequential semantic-KV experiment dispatcher with incremental evidence copy."""
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
    parser.add_argument('--pilot-prefix',type=Path)
    args=parser.parse_args()
    root=Path('results/semantic-dispatch')
    root.mkdir(parents=True,exist_ok=True)
    state={'jobs':[],'started':time.time()}
    seen={}
    def backup():
        for source in Path('results').rglob('*'):
            if not source.is_file():
                continue
            stat=source.stat()
            signature=(stat.st_size,stat.st_mtime_ns)
            if seen.get(str(source))==signature:
                continue
            target=args.backup/source
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            seen[str(source)]=signature
    def stage(label,command,required=True):
        row={'label':label,'command':command,'started':time.time(),'status':'running'}
        state['jobs'].append(row)
        atomic_json(root/'state.json',state)
        print('STAGE_START',label,flush=True)
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
        row.update(status='complete' if job.returncode==0 else 'failed',returncode=job.returncode,finished=time.time())
        atomic_json(root/'state.json',state)
        backup()
        print('STAGE_END',label,job.returncode,flush=True)
        if required and job.returncode:
            raise RuntimeError(label+' failed; see durable log')
    stage('cuda-tests',['-m','pytest','-q'])
    if args.pilot_prefix:
        stage('storage-probe',['-m','research.storage_probe','--prefix',str(args.pilot_prefix),
              '--output',str(root/'storage-probe.json')],required=False)
    stage('semantic-index',['-m','research.semantic_probe','--expanded'])
    completed=list(Path('results/semantic-index-screen').glob('*/complete.json'))
    if len(completed)!=1:
        raise RuntimeError('expected exactly one completed index identity in this isolated root')
    stage('semantic-kv',['-m','research.semantic_kv','--index-results',str(completed[0].parent)])
    state.update(finished=time.time(),status='complete')
    atomic_json(root/'state.json',state)
    backup()
    print('SEMANTIC_STUDY_DONE',flush=True)


if __name__=='__main__':
    main()

"""Warm/cold OS-cache sensitivity of exact disk KV reads; outside GPU timing.

Run only while the inference benchmark is idle. Advisory cache eviction applies
only to the specified read-only experiment prefix, never to global OS caches.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import psutil
from tiered_kv.store import ColdStore,atomic_json,file_sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if platform.system()!='Linux' or not hasattr(os,'posix_fadvise'):
        raise RuntimeError('cold-cache probe requires Linux posix_fadvise and process I/O counters')
    store=ColdStore.open(args.prefix,readonly=True)
    if store.prefix is not None:
        raise ValueError('probe needs an owned immutable base prefix, not an overlay')
    process=psutil.Process()
    rng=np.random.default_rng(41)
    rows=[]
    try:
        blocks=store.count//store.R
        for k in (4,32):
            if k>blocks:
                continue
            for trial in range(5):
                selected=[rng.choice(blocks,k,replace=False).tolist() for _ in range(store.layers)]
                for mode in rng.permutation(['warm','cold_advised']):
                    if mode=='warm':
                        for layer,chosen in enumerate(selected):
                            store.read(layer,chosen,store.count)
                    else:
                        for f in store.files:
                            # Read-only committed files are already durable; fsync
                            # also ensures the advisory request sees clean pages.
                            os.fsync(f.fileno())
                            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                    io_start=process.io_counters()._asdict()
                    logical_start=store.read_bytes
                    start=time.perf_counter()
                    for layer,chosen in enumerate(selected):
                        batch=store.read(layer,chosen,store.count)
                        if len(batch.ids)!=k*store.R:
                            raise ValueError('incomplete block read')
                    elapsed=(time.perf_counter()-start)*1000
                    io_end=process.io_counters()._asdict()
                    row={'k':k,'trial':trial,'mode':str(mode),'all_layers_read_ms':elapsed,
                         'logical_read_bytes':store.read_bytes-logical_start,
                         'process_io_delta':{key:io_end[key]-value for key,value in io_start.items()}}
                    rows.append(row)
                    print(json.dumps(row),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output,{'scope':'exact KV read/conversion microbenchmark only; no inference, query search or GPU transfers',
            'prefix':str(args.prefix),'prefix_sha256':file_sha(args.prefix/'checkpoint.json'),
            'platform':platform.platform(),'archive_tokens':store.count,'layers':store.layers,
            'kv_heads':store.h,'head_dim':store.d,'block_size':store.R,
            'cache_advice':'POSIX_FADV_DONTNEED is advisory, not guaranteed; inspect actual process read_bytes',
            'storage_hardware':'not established by this virtual-machine measurement','rows':rows})
    finally:
        store.close()


if __name__=='__main__':
    main()

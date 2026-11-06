"""CPU-only synthetic index cost; never presented as language-model quality."""
import argparse
import gc
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from tiered_kv.retrieval import BlockIndex
from tiered_kv.store import atomic_json,file_sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/evidence/index-scaling.json'))
    args=parser.parse_args()
    rng=np.random.default_rng(41)
    layers,h,d,R,W,heads=28,2,128,128,2000,12
    rows=[]
    for n in (8192,32768,131072,524288,1048576):
        evicted=n-W
        blocks=math.ceil(evicted/R)
        capacity=max(16,2**math.ceil(math.log2(blocks)))
        index=BlockIndex(layers,h,d,R,capacity=capacity)
        for layer in range(layers):
            # Synthetic block sums with roughly independent Gaussian keys.
            index.sums[layer][:blocks]=rng.standard_normal((blocks,h,d),dtype=np.float32)*np.sqrt(R)
            index.counts[layer][:blocks]=R
            index.counts[layer][blocks-1]=evicted-(blocks-1)*R
            index.next_pos[layer]=evicted
        q=torch.from_numpy(rng.standard_normal((1,heads,1,d),dtype=np.float32))
        for layer in range(layers):
            index.search(layer,q,4)
        trials=[]
        for _ in range(7):
            start=time.perf_counter()
            for layer in range(layers):
                index.search(layer,q,4)
            trials.append((time.perf_counter()-start)*1000)
        rows.append({'archive_tokens':n,'evicted_tokens':evicted,'blocks':blocks,
            'capacity':capacity,'index_bytes':index.nbytes,
            'all_layers_search_ms':trials,'median_ms':float(np.median(trials)),
            'exact_fp16_disk_bytes':n*layers*h*d*4})
        print(json.dumps(rows[-1]),flush=True)
        del index
        gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,{'scope':'synthetic CPU index search only; no model, GPU transfer, disk retrieval or quality measurements',
        'platform':platform.platform(),'machine':platform.machine(),'numpy':np.__version__,
        'torch':str(torch.__version__),'seed':41,'layers':layers,'kv_heads':h,
        'query_heads':heads,'head_dim':d,'R':R,'W':W,'k':4,
        'source_sha256':file_sha('tiered_kv/retrieval.py'),'rows':rows})


if __name__=='__main__':
    main()

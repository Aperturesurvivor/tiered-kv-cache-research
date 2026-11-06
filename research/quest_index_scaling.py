"""Synthetic CPU QuestBoundIndex scaling; no model, disk KV, or quality claim."""
import argparse,gc,json,math,os,platform,signal,time
from pathlib import Path
import numpy as np
import psutil
import torch
from research.attention_quest_reference import QuestBoundIndex
from research.resources import HostObservation
from tiered_kv.store import atomic_json,file_sha

LENGTHS=(4096,8192,32768,131072,524288,1048576)
LAYERS=28;KV_HEADS=2;HEADS=12;DIM=128;PAGE=128;WINDOW=2000;SINK=128;POOL=32;TRIALS=7
SEED=317;RUNTIME_FLOOR=2*2**30;ALLOCATION_ALLOWANCE=1*2**30;INITIAL_FLOOR=RUNTIME_FLOOR+ALLOCATION_ALLOWANCE
DISK_RESERVE=1*2**30;RSS_LIMIT=2*2**30;DEADLINE=600
THREAD_ENV=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
QUEST_SHA='a97ecdfcc3fb5652b02649ac00b10899e7a2f851a111e4c93342ff66f49c6ce1'
REPRESENTATIONS_SHA='2a9765aa4ea39a2ee46ccb99974027c444172b0453a57d7bf66f5856ca7eab85'
RETRIEVAL_SHA='0303b4d4049359fdbd481e9810e20b901ffa014145e95e2bc529139729bf08a0'


def build_index(length,seed=SEED,layers=LAYERS,h=KV_HEADS,d=DIM,page=PAGE,window=WINDOW,sink=SINK):
    """Synthetic finite coordinate extrema and exact complete/partial counts."""
    cutoff=length-window+1
    if cutoff<sink:raise ValueError('sink/recent overlap')
    full,partial=divmod(cutoff,page);blocks=full+bool(partial)
    capacity=1<<(blocks-1).bit_length()
    rng=np.random.default_rng(seed+length)
    index=QuestBoundIndex(layers,h,d,page,sink)
    for layer in range(layers):
        lo=-np.abs(rng.standard_normal((capacity,h,d),dtype=np.float32))
        hi=np.abs(rng.standard_normal((capacity,h,d),dtype=np.float32))
        if not (np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo<=hi)):
            raise ValueError('invalid synthetic page bounds')
        counts=np.zeros(capacity,dtype=np.int64);counts[:full]=page
        if partial:counts[full]=partial
        index.sums[layer]=lo;index.maxima[layer]=hi;index.counts[layer]=counts;index.next_pos[layer]=cutoff
        if psutil.Process().memory_info().rss>RSS_LIMIT:raise RuntimeError('process RSS exceeded 2 GiB observed guard during metadata build')
        if psutil.virtual_memory().available<RUNTIME_FLOOR:raise RuntimeError('available memory fell below runtime 2 GiB floor during metadata build')
    return index,{'cutoff':cutoff,'complete_pages':full,'partial_tokens':partial,'capacity_pages_per_layer':capacity,
                  'eligible_complete_pages':max(0,full-(sink+page-1)//page)}


def check_resources(output,*,initial=False):
    m=psutil.virtual_memory();free=__import__('shutil').disk_usage(output.parent).free
    threshold=INITIAL_FLOOR if initial else RUNTIME_FLOOR
    if m.available<threshold:raise RuntimeError('available system memory below declared guard')
    if free<DISK_RESERVE:raise RuntimeError('disk free below 1 GiB guard')
    if psutil.Process().memory_info().rss>RSS_LIMIT:raise RuntimeError('process RSS exceeded 2 GiB observed guard')
    return {'available_bytes':m.available,'disk_free_bytes':free,'process_rss_bytes':psutil.Process().memory_info().rss}


def all_layers(index,queries,fixed):
    result=[]
    for layer in range(index.layers if hasattr(index,'layers') else len(index.sums)):
        index.stage(layer,fixed[layer]);ids,_=index.search(layer,queries[layer],POOL)
        result.append(ids.tolist())
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    if (file_sha('research/attention_quest_reference.py')!=QUEST_SHA or
            file_sha('research/attention_representations.py')!=REPRESENTATIONS_SHA or
            file_sha('tiered_kv/retrieval.py')!=RETRIEVAL_SHA):raise ValueError('frozen Quest scorer or dependency differs')
    if any(os.environ.get(k)!='1' for k in THREAD_ENV):raise ValueError('set declared thread environment to 1 before importing numeric libraries')
    if args.output.exists():raise ValueError('output already exists')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    guard=check_resources(args.output,initial=True)
    plan={'lengths':LENGTHS,'layers':LAYERS,'kv_heads':KV_HEADS,'query_heads':HEADS,'head_dim':DIM,'page':PAGE,'window':WINDOW,'sink':SINK,
          'candidate_ceiling':POOL,'warmup_all_layer_passes':1,'trials':TRIALS,'seed':SEED,'rss_limit_bytes':RSS_LIMIT,'initial_minimum_available_bytes':INITIAL_FLOOR,'runtime_minimum_available_bytes':RUNTIME_FLOOR,'additional_allocation_allowance_bytes':ALLOCATION_ALLOWANCE,
          'disk_reserve_bytes':DISK_RESERVE,'deadline_s':DEADLINE,'thread_environment':{k:os.environ[k] for k in THREAD_ENV},'preflight':guard}
    if not args.execute:print(json.dumps({'status':'plan',**plan}));return
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rng=np.random.default_rng(SEED);queries=[torch.from_numpy(rng.standard_normal((1,HEADS,1,DIM),dtype=np.float32)) for _ in range(LAYERS)]
    # Cutoff modulo page is identical for the six power-of-two lengths.
    boundary=(LENGTHS[0]-WINDOW+1)%PAGE;fixed_count=SINK+boundary+WINDOW
    fixed=[torch.from_numpy(rng.standard_normal((KV_HEADS,fixed_count,DIM),dtype=np.float32).astype(np.float16)) for _ in range(LAYERS)]
    partial=args.output.with_suffix('.partial.jsonl');assert not partial.exists()
    handler=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('synthetic CPU lookup exceeded 600 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(DEADLINE);rows=[]
    try:
        for length in LENGTHS:
            check_resources(args.output)
            with HostObservation(scope='synthetic metadata build, warmup and lookup trials; timed trials exclude metadata build') as host:
                index,geometry=build_index(length)
                assert geometry['partial_tokens']==boundary
                warm=all_layers(index,queries,fixed)
                assert all(len(x)==min(POOL,geometry['eligible_complete_pages']) for x in warm)
                trials=[]
                for _ in range(TRIALS):
                    check_resources(args.output);start=time.perf_counter();selected=all_layers(index,queries,fixed);trials.append((time.perf_counter()-start)*1000)
                    if selected!=warm:raise ValueError('fixed query produced unstable rankings')
            row={'archive_tokens':length,**geometry,'metadata_bytes':index.nbytes,'fixed_key_count_per_layer':fixed_count,
                 'fixed_key_tensor_bytes':sum(x.numel()*x.element_size() for x in fixed),
                 'actual_candidate_counts':sorted({len(x) for x in warm}),'all_layer_search_ms':trials,'median_ms':float(np.median(trials)),
                 'host':host.metrics,'post_trial_resource_guard':check_resources(args.output)}
            with partial.open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row);print(json.dumps({'length':length,'median_ms':row['median_ms']}),flush=True)
            del index;gc.collect()
        atomic_json(args.output,{'status':'complete','plan':plan,'rows':rows,'platform':platform.platform(),'machine':platform.machine(),
            'numpy_version':np.__version__,'torch_version':str(torch.__version__),'torch_threads':torch.get_num_threads(),
            'source_sha256':file_sha(__file__),'quest_source_sha256':QUEST_SHA,'representations_source_sha256':REPRESENTATIONS_SHA,'retrieval_source_sha256':RETRIEVAL_SHA,
            'partial_sha256':file_sha(partial),'scope':'synthetic CPU scan and fixed-key normalization only; no GPU, KV disk reads, model inference, generation or quality; above-native lengths represent index only'})
    except BaseException as exc:
        atomic_json(args.output.with_suffix('.failure.json'),{'status':'failed','completed_lengths':[x['archive_tokens'] for x in rows],'error':repr(exc)})
        raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,handler)

if __name__=='__main__':main()

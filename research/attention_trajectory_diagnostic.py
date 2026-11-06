"""Post-hoc matched dense-trajectory routing diagnostic; no performance claim.

Default mode only validates and prints a plan. --execute requires an idle CUDA
GPU, locally cached model weights and a new output directory; it never changes
the frozen experiment or appends to its evidence. One case is executed per run.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from research.attention_counterfactual import (PAGE, eligible_pages, oracle_pages,
                                                original_centroid_ranking, retained_common_cold_fraction, streamed_metrics)

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'results/evidence/reasoning-Qwen--Qwen2.5-1.5B-Instruct-8862fb5f28f2'
SOURCE='8b37d6b4b3dac49d207e93c003531e3d15e93c55'
WEIGHTS='fdf003e3fecca614b5f4cd70e02a7a0f7e28f9422bcd4fb509dcb025a8a4470a'
INPUTS={'samples.jsonl':'7407fb4d2dd3b6389e5042bad7cb544ce26d8e085383e23c65e859b8ced7afc8',
        'fixtures.json':'fc70d08e250550a6718fa2990cf19843f0da3e6f972ec78a2556d571a92f978f',
        'manifest.json':'0404071a61e7f2a321c52271dbfc0472f582be57dec8f6d9276213dfd88ef3bd',
        'validation-6a7e2608.json':'bbda4388087840d6d5a80f92cc9dd9972ddcbfa5990b270851346c4f33be3221',
        'complete.json':'349a0c7e478bbd0684322e917aa64a7b9d12c6c84702e253186c54f4a67d5f0a'}
CASES=tuple(f'reasoning-{task}-{budget}-163' for budget in (8192,32768) for task in ('literal','paraphrase'))
MAX_OUTPUT=8*1024*1024
MAX_PROMPT=160


def file_sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(case):
    assert case in CASES
    # The completion marker has a full 256-bit digest, as do the other inputs.
    expected=INPUTS
    assert all(len(h)==64 and file_sha(EVIDENCE/name)==h for name,h in expected.items())
    manifest=json.loads((EVIDENCE/'manifest.json').read_text())
    fixtures=json.loads((EVIDENCE/'fixtures.json').read_text())
    rows=[json.loads(line) for line in (EVIDENCE/'samples.jsonl').read_text().splitlines()]
    assert len(rows)==220 and json.loads((EVIDENCE/'complete.json').read_text())['weights_unchanged']
    assert json.loads((EVIDENCE/'validation-6a7e2608.json').read_text())['weights_sha256']==WEIGHTS
    fixture=next(e for e in fixtures if e['id']==case)
    matched={r['policy']['attention']:r for r in rows if r['example_id']==case and r['purpose']=='quality'
             and r['policy']['id'] in ('stock-json-r2-cap128','sparse-json-r2-cap128')}
    assert set(matched)=={'stock','sparse'}
    stock,sparse=matched['stock'],matched['sparse']
    prompt=stock['rounds'][0]['prompt_token_ids']
    assert prompt==sparse['rounds'][0]['prompt_token_ids'] and 0<len(prompt)<=MAX_PROMPT
    position=fixture['prefix_tokens_count']+len(prompt)-1
    universe=eligible_pages(position+1,position)
    recorded=sparse['rounds'][0]['selected_blocks']
    text=[b for b in recorded if b in universe]
    assert len(set(text))==len(text) and 0<len(text)<=16
    return dict(case=case,model=manifest['model'],fixture=fixture,prompt=prompt,
                original_text_pages=recorded,text_cold_pages=text,
                source_outcomes=dict(stock=stock['exact_correct'],sparse=sparse['exact_correct']),
                selected_query=sparse['rounds'][0]['query'],source_files=expected)


PINNED_HELPERS = {'research/archive_payloads.py': '252dfd8f1e08d6b698dfe8b00c2eb039bafd6728c0d4bf8802ff754f95164803',
 'research/batch.py': '1dd62fee6d9dd2271ab79d8f5a10c63e478e8e37fcdffc9354a7a6b4397f098a',
 'research/benchmark_stock.py': '71356ff9eea980740fb4653467b46070e486e13fc17255d2ed1b504f41be4cae',
 'research/boundary_probe.py': 'b88b274f06407fb36864b3062a50b3451ed4c6994f25cbf12630edbead5d4467',
 'research/budget_probe.py': '1e78e6adcfcaef5730cc5aef3242c683838959fc9b99b7144bd3c98453a9188b',
 'research/demand_kv.py': '87fa68bda06d16d5866aad0798c3311c70d013aa9df7d61c9c113f8b1f68044f',
 'research/demand_run.py': '6e6c638f92b9ae7f675dfcee2a94e6033ad54f39a9351b3629b0284ea5796bfb',
 'research/eviction_probe.py': 'c371060698d1bcb7d2b8a6d953569ba65ea03b4fccde7757e7fd75e63c2e1e44',
 'research/generalize.py': '26119816a9c8994683c414423fccd4db2ee8af5c96308a7884a22091dda76f23',
 'research/index_scaling.py': '52f29d553dd06f4eefd7d3816160d29002a02a2018dc79e233dc3f2357265012',
 'research/lsh_index.py': '25b057f6178a0df7d4167e3e7a5ed1d741c17d6d2aa57cbe1c2a810eb8e649eb',
 'research/lsh_scaling.py': 'c841dcaedc3eae811616966c37ed37b604d0221aded70f5c4e7178dd3e9b6ed2',
 'research/lsh_text_probe.py': '8af9dbd9cb42c4c44ef9f47f9e7ae1bd19abdc0af911e47a9f957e287a5f7369',
 'research/memory_harness.py': '38c0eab82cbb522d9e3ba2334bc419a36381dee0fd5e7e2ac9faef95bba43762',
 'research/memory_protocol.py': '4c1d63f0ab36f504cfc78bc75479214ba841f0234d373fa2ee683a8f4ac6b2ab',
 'research/memory_run.py': 'ccfc073fd9b213aa149dce04354b362c1041dae2aa5e08d0b1072e45dc1282ed',
 'research/memory_update_archive.py': 'fcebbb9a71ef7e7153d00266b6376afda841d147e1a6814a5c623e77a1fb09f7',
 'research/memory_update_protocol.py': 'bdec5fa9886820cb7d8addba6db76cd460b51834e0e343257ee4f42ed8db7298',
 'research/memory_update_run.py': '757b2f0fd2febf9d191581d3bf06fbb7be8a9bedf0c9f545d87589c405534156',
 'research/persistent_kv.py': '94fe254c0e0b89308842ad7d3956ad6cffc05d0356d2c1b5425662fce0ea0089',
 'research/persistent_probe.py': '51ddd4b381c6e636463b6014d38692d42e885b7a25181c9037ee83b7b101746a',
 'research/prefix_probe.py': '18b5632a1d68957a530609ddc52ba04994ad4b4d312b05b2b959e7bd7c837a52',
 'research/prefix_sessions.py': '232b5d37f2ae112601a9b347a7016a3b4cedafc7c2dbd44dc51ed2f815f021f9',
 'research/reasoning_protocol.py': '8033fd1b18b0dc9e3b2c94c2b3fedd61718cc54c0bc277e75802751cddfac873',
 'research/reasoning_run.py': '09a5f102df898b5457aa81577c863895a8cd1f9822a624fd374dbdfe4938dcbd',
 'research/reasoning_sessions.py': 'b2b2959b95839606d51ddbc68bf3f5b1acf04e00f94d6d6d24702e7b2c43dfce',
 'research/report_budget.py': '6990f422955863d87a6ade9eeb0f3b51ea1bdc4d310fa484b47e6da2b0e57b99',
 'research/report_cross.py': '492c75088ca822494467f122d6c3dd266ae9c113491ef645ad974e23c651243c',
 'research/report_demand.py': '0f983c47b03c3a050680158b4932acd028201c51d8f1ea8a7d3d051dd5bae3de',
 'research/report_lsh_scaling.py': 'b221fd5bc2fda3c138db54ad8369b495940dddb6228c2a54d3d545e8e6853929',
 'research/report_lsh_text.py': '8320ff0bd76ad534eac919738a35d21220f0b8547a57b94dfa037b685b0701c3',
 'research/report_memory.py': '7e7fa212362d8e13b353a31a6b22a03e1c9eb335b9c40f688530c1b88661dd63',
 'research/report_memory_update.py': '8e3ff9aa19cebbf0b4c431f7f9c4d21d10116d12752509f6bc692c0d5d6ba05c',
 'research/report_reasoning.py': 'e12750eaa0cc677f091be87c2a74d58c1331339bb6973d8d2d3c3cf260e6467f',
 'research/report_semantic_reasoning.py': '74e4dea5b018f376036fcab0ad1f1142d469766a76757c0a28a8272e26a67595',
 'research/resources.py': '1425b9923353683dfbff094ec54ec1ab1893a25de7fed1f77602cca3dbdbe3e2',
 'research/semantic_index.py': '00bb7c5be5723d6c2779687b7a7bd47b26e2d56f49778a069dc996cf88b69c4c',
 'research/semantic_kv.py': '28adeeb129b01e6c1cd35570092d95da6fe20d0e839cf6d587ce9b5a59fe3f41',
 'research/semantic_probe.py': 'cfbb2f978d72b7da4e94f257bf3b5c1fb4c217883ddf22c5169b6ce90d2df1d7',
 'research/semantic_reasoning.py': '152feddfdb107cd4d4df2c7f1ab752e5312464fd2844bf80fec1a862bd9e94ee',
 'research/semantic_run.py': '6ed7dd3091add2a42d214c6127732fdd425479080f16439dd97f1252e7a1963c',
 'research/storage_probe.py': '9eb8a052fe821807082639970e79e171c9baf6600bb29bb6cf1e7c23faf10951',
 'research/text_index_scaling.py': 'aa97dbe0f0cbd77fd214b85d530dec13d0124d5a81f5d21abff08949d9e28e8b',
 'tiered_kv/__init__.py': '2c33c673ae490df7cd13a08c6749a3d65dab960627ebc189b174ed53c69f1d9f',
 'tiered_kv/__main__.py': 'c6c1e13d224fb72b43483174e95385966688ddaae482aee598b1bca50a65f85e',
 'tiered_kv/adapter.py': '97f4398c0b1a38b1c3074a1d58e8468e533436040395d4ec7bc4c452a69cf200',
 'tiered_kv/cache.py': 'c9ac733a62ed7aa85b1bb93b58055861a592d05a1fb216633966107c378f02d6',
 'tiered_kv/config.py': '4c2d38f79ff09e5fb749e2a124caadfb03fb452bdbd0839520e4e1508c1fd978',
 'tiered_kv/evals.py': 'f288108e35675b56bbc3539a7750ce67ffa0e101ef4d986acb55f0cb4e44f968',
 'tiered_kv/generate.py': 'b511ae47a47772cac874c0128b294860de9e8f9e5df7be3ec67f6de5aa760553',
 'tiered_kv/report.py': '0f89d7599e064551cb36e13517d305f85ac5d7a7b63e32bc2e3f0362fa16ed18',
 'tiered_kv/retrieval.py': '0303b4d4049359fdbd481e9810e20b901ffa014145e95e2bc529139729bf08a0',
 'tiered_kv/store.py': '91ab47937b16a705265cb7323a1d82780b07f578a7a192cbf3a58caa07ff016e',
 'tiered_kv/telemetry.py': '2cf046a3f994a5818c13c19f7c8458ff79b0a726236cc719d185391ed27bcc44',
 'tiered_kv/tiers.py': 'c57047a0ce2db37b9109aae80b5bbeaee290e0b8ee94aa04eb075f6108e03afc',
 'tiered_kv/types.py': 'e75bfbc5a3c09b8018a3d328370895f41ec4bb176a1151c10fcd6817e5b427b0',
 'tiered_kv/validation.py': 'e7ec7f0d203e123013e54377d7068dbe6dc04308b7b05b1cb2725fccdca1bddc'}


def validate_source():
    """Verify frozen source bytes without requiring a Git checkout."""
    result = {}
    for name, expected in PINNED_HELPERS.items():
        actual = file_sha(ROOT / name)
        assert actual == expected, "source differs: " + name
        result[name] = actual
    result["research/attention_counterfactual.py"] = file_sha(ROOT / "research/attention_counterfactual.py")
    result["research/attention_trajectory_diagnostic.py"] = file_sha(Path(__file__))
    return result


def score_layer(q,view,position,text,case,layer):
    """One dense trace layer, one shared cold-page budget for every head."""
    ids=np.asarray(view.ids);assert np.array_equal(ids,np.arange(position+1))
    q=q.detach().float().cpu().numpy()[0,:,0,:]
    assert q.shape[0]<=32 and q.shape[1]<=128, 'workspace bound exceeded'
    def reader(start,end):
        assert 0<=start<end<=len(ids) and end-start<=PAGE
        return (view.k[0,:,start:end,:].detach().float().cpu().numpy(),
                view.v[0,:,start:end,:].detach().float().cpu().numpy())
    candidates=eligible_pages(len(ids),position); assert all(b in candidates for b in text)
    k=len(text);assert 0<k<=16 and len(candidates)>=k
    cutoff=max(0,position-2000+1)
    rank,_=original_centroid_ranking(q,reader,cutoff)
    filtered=[b for b in rank if b in candidates][:k];assert len(filtered)==k
    seed=int.from_bytes(hashlib.sha256(f'{case}:{layer}'.encode()).digest()[:8],'big')
    random=np.random.default_rng(seed).choice(candidates,size=k,replace=False).tolist()
    oracle=oracle_pages(q,reader,len(ids),candidates,k)
    fixed=set(range(min(128,position+1)))|set(range(cutoff,position+1))
    selections=dict(text=text,centroid_filtered=filtered,random=random,oracle_shared=oracle)
    metrics=streamed_metrics(q,reader,len(ids),fixed,selections)
    cold_fractions=retained_common_cold_fraction(q,reader,candidates,selections)
    for name in metrics: metrics[name]['common_cold_retained_fraction']=cold_fractions[name]
    # A separate equal-budget sink tradeoff: common cold eligibility and the
    # same 16 pages in all four arms, changing only sink/recent allocation.
    sink_candidates=eligible_pages(len(ids),position,sink=128,window=2128)
    sink_pages=[b for b in rank if b in sink_candidates][:16]
    assert len(sink_pages)==16
    sink_metrics={}
    for sink in (0,4,16,128):
        window=2128-sink
        sink_fixed=set(range(sink)) | set(range(position-window+1,position+1))
        sink_metrics[str(sink)]=streamed_metrics(q,reader,len(ids),sink_fixed,{'shared_centroid':sink_pages})['shared_centroid']
    assert len({m['active_tokens'] for m in sink_metrics.values()})==1
    cold_fraction=retained_common_cold_fraction(q,reader,sink_candidates,{'shared_centroid':sink_pages})['shared_centroid']
    sink_mass={str(n):streamed_metrics(q,reader,len(ids),set(range(n)),{'prefix':[]})['prefix']['retained_mass'] for n in (1,4,16,128)}
    assert len({x['active_tokens'] for x in metrics.values()})==1
    return dict(layer=layer,position=position,query_heads=q.shape[0],page_budget=k,
        fixed_tokens=len(fixed),candidate_pages=len(candidates),original_centroid_top16=rank[:16],
        original_centroid_top16_eligible=[b for b in rank[:16] if b in candidates],
        filtered_centroid_note='rank all original eligible centroids, then restrict to complete non-sink cold pages and take matched budget',
        random_seed=seed,metrics=metrics, sink_ablation=dict(cold_pages=sink_pages, candidate_pages=len(sink_candidates), recent_windows={str(n):2128-n for n in (0,4,16,128)}, arms=sink_metrics, dense_prefix_mass=sink_mass, common_cold_retained_fraction=cold_fraction, note='Equal total active tokens; sink size trades off against recent window. Shared cold candidates exclude first 128 and last 2128 positions.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();plan=load_plan(args.case)
    visible={k:v for k,v in plan.items() if k not in ('fixture','prompt')}
    visible.update(source_revision=SOURCE,prompt_tokens=len(plan['prompt']),
                   probe='last token of matched first-round JSON prompt',
                   scope='post-hoc dense-trajectory counterfactual; no sparse-trajectory or latency claim')
    if not args.execute:
        print(json.dumps(visible,indent=2));return
    assert torch.cuda.is_available() and os.environ.get('HF_HUB_OFFLINE')=='1', 'CUDA and explicit offline mode required'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU must be externally idle'
    assert shutil.disk_usage(ROOT).free>=12*2**30, '12 GiB free disk required for retained diagnostic scratch'
    free,_=torch.cuda.mem_get_info();assert free>=8*2**30, '8 GiB free GPU required'
    source_hashes=validate_source()
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from research.reasoning_sessions import BatchedStockSession
    from tiered_kv.adapter import DenseController,ModelAdapter
    from tiered_kv.generate import RunEngine,weights_digest
    from tiered_kv.store import ColdStore,file_sha as stored_file_sha
    from tiered_kv.telemetry import ResourceGuard,device_empty_cache,runtime_identity
    from tiered_kv.validation import parity
    from tiered_kv.types import digest
    out=ROOT/'artifacts/attention-trajectory'/args.case
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    snapshot=snapshot_download(plan['model']['model_id'],revision=plan['model']['revision'],local_files_only=True)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':'cuda'},attn_implementation='sdpa').eval().requires_grad_(False)
    model.config._commit_hash=plan['model']['revision'];assert weights_digest(model)==WEIGHTS
    checks=[parity(model,n) for n in (64,512)]
    assert all(all(c['greedy_agreement']) and max(c['max_errors'])<=1e-4 for c in checks)
    with ResourceGuard() as guard:
        engine=RunEngine(model,tok,out,guard)
        prefix,prefill=engine.prefix(plan['fixture']);engine.adapter.uninstall()
        parent=ColdStore.open(prefix,readonly=True)
        assert parent.count==plan['fixture']['prefix_tokens_count']
        assert parent.metadata['token_hash']==plan['fixture']['token_hash'] and parent.metadata['model_revision']==plan['model']['revision']
        checkpoint_sha=stored_file_sha(Path(prefix)/'checkpoint.json')
        native=[]
        try:
            # Save bounded native logits only in RAM, for a strict all-token gate.
            vocab=model.config.vocab_size;assert len(plan['prompt'])*vocab*4<=128*2**20
            stock=BatchedStockSession(model,parent);native=[]
            for token in plan['prompt']:
                guard.check();native.append(stock.forward_tokens([token],stock.position).detach().float().cpu())
            del stock;gc.collect();device_empty_cache()
            def run_dense(trace=None):
                dense=DenseController(parent.layers);dense.initialize(parent,'cuda',torch.float16)
                adapter=ModelAdapter.install(model,dense);maximum=0.0; records=[]
                try:
                    # The all-KV gate checks every original K/V byte, not a 16-page helper.
                    for layer in range(parent.layers):
                        for start in range(0,parent.count,PAGE):
                            end=min(start+PAGE,parent.count);batch=parent.read_range(layer,start,end)
                            assert torch.equal(dense.keys[layer][:,:,start:end].cpu(),batch.k)
                            assert torch.equal(dense.values[layer][:,:,start:end].cpu(),batch.v)
                    if trace:
                        def observer(layer,pos,q,view):
                            if pos==parent.count+len(plan['prompt'])-1:
                                records.append(score_layer(q,view,pos,plan['text_cold_pages'],args.case,layer))
                        dense.trace=observer
                    for i,token in enumerate(plan['prompt']):
                        guard.check();actual=adapter.step(token,adapter.position).detach().float().cpu()
                        maximum=max(maximum,float((actual-native[i]).abs().max()))
                        assert torch.allclose(actual,native[i],atol=1e-4,rtol=1e-4),'all-KV stock parity failed'
                    if trace: assert len(records)==parent.layers
                    return maximum,records
                finally:
                    dense.trace=None;adapter.uninstall();del adapter,dense;gc.collect();device_empty_cache()
            maximum,_=run_dense();traced_max,layers=run_dense(trace=True)
            assert weights_digest(model)==WEIGHTS
            report=dict(**visible,source_hashes=source_hashes,weights_sha256=WEIGHTS,
                fixture_digest=digest(plan['fixture']),prefix_checkpoint_sha256=checkpoint_sha,
                prefix_path=str(prefix),prefill_s=prefill,all_kv_stock_max_abs=maximum,
                traced_stock_max_abs=traced_max,gate='all K/V bytes and all prompt logits matched',
                runtime=runtime_identity(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),layers=layers,
                limitations=['float64 diagnostic softmax/output uses original fp16 Q/K/V and is not actual fp16 SDPA output',
                    'centroid arm filters original ranking to complete non-sink cold pages; original top-16 also recorded',
                    'text selection is not padded; every arm shares its per-layer page count',
                    'same dense Q/K/V only; no sparse hidden-state trajectory or final-quality claim',
                    'prefill, dense scans, index rebuild and transfers are diagnostic work, not sparse serving latency'])
            encoded=(json.dumps(report,indent=2)+'\n').encode();assert len(encoded)<=MAX_OUTPUT
            (out/'diagnostic.json').write_bytes(encoded)
            print(json.dumps(dict(status='complete',case=args.case,layers=len(layers),output=str(out/'diagnostic.json'),bytes=len(encoded))))
        finally:
            parent.close();del native;gc.collect();device_empty_cache()


if __name__=='__main__': main()

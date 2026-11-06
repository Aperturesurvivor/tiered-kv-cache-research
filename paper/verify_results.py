#!/usr/bin/env python3
"""Recalculate the manuscript's four tables from saved metrics; no inference."""
import csv
import hashlib
import json
import statistics as st
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / 'results/evidence'
P = ROOT / 'paper'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text())
def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line]
def fmt(x, n=3):
    return f'{x:.{n}f}'
def table(label):
    s = (P / 'manuscript.tex').read_text()
    end = s.index('\\label{tab:' + label + '}')
    start = s.rfind('\\begin{tabular}', 0, end)
    body = s[start:end]
    out = {}
    for line in body.splitlines():
        if '&' not in line or '\\\\' not in line: continue
        fields = [v.strip() for v in line.split('\\\\')[0].split('&')]
        out[fields[0]] = fields[1:]
    return out
def check_table(label, expected):
    actual = table(label)
    for name, cells in expected.items():
        assert actual.get(name) == cells, (label, name, actual.get(name), cells)

def confirm(name, raw_relative=None):
    folder = E / name
    samples = folder / (raw_relative or 'samples.jsonl')
    r = rows(samples)
    assert len(r) == 264
    assert sum(x['task'] == 'niah' for x in r) == 180
    assert sum(x['task'] == 'latency' for x in r) == 84
    quality = {}
    for cfg in ('full_buffered', 'rare_context'):
        z = [x for x in r if x['task'] == 'niah' and x['config_id'] == cfg]
        quality[cfg] = [sum(x['em'] == 100 for x in z), len(z)]
        q32 = [x for x in z if x['budget'] == 32768]
        quality[cfg + '_32'] = [sum(x['em'] == 100 for x in q32), len(q32)]
    with (folder / 'latency.csv').open(newline='') as f:
        lat = list(csv.DictReader(f))
    metrics = {}
    for cfg in ('stock_hf', 'rare_context'):
        z = [x for x in lat if x['policy'] == cfg and int(x['budget']) == 32768]
        assert len(z) == 3
        metrics[cfg] = (st.mean(float(x['mean_ms']) for x in z), max(float(x['cuda_peak_GiB']) for x in z))
    return quality, metrics, [samples, folder / 'latency.csv']

q15 = 'demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235'
q06 = 'demand-Qwen--Qwen3-0.6B-confirm-479bee540057'
a, am, sources = confirm(q15)
b, bm, paths = confirm(q06, 'raw/' + q06 + '/samples.jsonl'); sources += paths
frac = lambda x: f'{x[0]}/{x[1]}'
check_table('confirm', {
    'All factual cases': [frac(a['full_buffered']),frac(a['rare_context']),frac(b['full_buffered']),frac(b['rare_context'])],
    '32K factual cases': [frac(a['full_buffered_32']),frac(a['rare_context_32']),frac(b['full_buffered_32']),frac(b['rare_context_32'])],
    '32K decode (ms/token)': [fmt(am['stock_hf'][0],2),fmt(am['rare_context'][0],2),fmt(bm['stock_hf'][0],2),fmt(bm['rare_context'][0],2)],
    '32K CUDA peak (GiB)': [fmt(am['stock_hf'][1]),fmt(am['rare_context'][1]),fmt(bm['stock_hf'][1],2),fmt(bm['rare_context'][1],2)],
})
verified = read(P/'verified-results.json')
assert verified['selected_results']['qwen15_confirmation']['quality_all']['full_buffered'] == a['full_buffered']
assert verified['selected_results']['qwen15_confirmation']['quality_all']['rare_context'] == a['rare_context']
assert verified['selected_results']['qwen3_06_confirmation']['quality_all']['full_buffered'] == b['full_buffered']
assert verified['selected_results']['qwen3_06_confirmation']['quality_all']['rare_context'] == b['rare_context']
for policy, label in [('stock_hf','native_stock'),('rare_context','rare_context')]:
    assert round(bm[policy][0],2)==verified['selected_results']['qwen3_06_confirmation']['steady_decode_32k_mean_ms'][label]
    assert round(bm[policy][1],2)==verified['selected_results']['qwen3_06_confirmation']['cuda_allocated_peak_gib'][label]
for key, quality in [('qwen15_confirmation',a),('qwen3_06_confirmation',b)]:
    for policy in ('full_buffered','rare_context'):
        assert verified['selected_results'][key]['quality_32k'][policy] == quality[policy+'_32']

p = E/'attention-device-cost-af741c7-attempt2/raw/samples.jsonl'; sources.append(p); cost = rows(p); assert len(cost)==18
expected = {}
for budget in (8192,32768):
    for arm, label in [('stock','Native stock'),('cpu','CPU reranker'),('device','Device reranker')]:
        z = [x for x in cost if x['budget']==budget and x['arm']==arm]; assert len(z)==3
        key = ('8K' if budget==8192 else '32K')
        expected.setdefault(key, None)
        # Each cost row has the budget as its first cell; duplicate keys need explicit checking.
        actual = [x.strip() for x in next(line for line in (P/'manuscript.tex').read_text().splitlines() if line.startswith(key+' & '+label+' &')).split('\\\\')[0].split('&')]
        values = [key,label]+[fmt(st.median(x[k] for x in z)) for k in ('prompt_s','forced_s','serving_s')]+[fmt(st.median(st.mean(x['forced_step_ms']) for x in z))]
        assert actual == values, (actual, values)
        if budget==32768:
            assert round(st.median(x['serving_s'] for x in z),3)==verified['selected_results']['attention_device_cost_32k']['serving_median_s'][{'stock':'native_stock','cpu':'cpu_reranker','device':'device_reranker'}[arm]]

ratios=[]
for rep in range(3):
    d=next(x for x in cost if x['budget']==32768 and x['arm']=='device' and x['rep']==rep)
    stock=next(x for x in cost if x['budget']==32768 and x['arm']=='stock' and x['rep']==rep)
    ratios.append(d['serving_s']/stock['serving_s'])
assert round(st.median(ratios),3)==verified['selected_results']['attention_device_cost_32k']['paired_device_over_stock_median']

reason_names = ['reasoning-Qwen--Qwen2.5-1.5B-Instruct-8862fb5f28f2','reasoning-Qwen--Qwen3-4B-Instruct-2507-67a2e095096b']
reason = []
for i,n in enumerate(reason_names):
    p = E/n/('samples.jsonl' if i==0 else 'run/samples.jsonl'); sources.append(p); z=rows(p); assert len(z)==220
    assert sum(x['purpose']=='quality' for x in z)==176 and sum(x['purpose']=='repeated_timing' for x in z)==44
    reason.append(z)
rtable={}
for label,policy in [('Stock direct','stock-direct-r1-cap128'),('Stock JSON, 1 round','stock-json-r1-cap128'),('Stock JSON, 2 rounds','stock-json-r2-cap128'),('Stock JSON, 4 rounds','stock-json-r4-cap128'),('Sparse JSON, 1 round','sparse-json-r1-cap128'),('Sparse JSON, 2 rounds','sparse-json-r2-cap128'),('Sparse JSON, 4 rounds','sparse-json-r4-cap128')]:
    vals=[]
    for z in reason:
        v=[x for x in z if x['purpose']=='quality' and x['archive_tokens']>20000 and x['policy']['id']==policy]; assert len(v)==8
        vals.extend([f"{sum(x['exact_correct'] for x in v)}/8",fmt(st.mean(x['request_s'] for x in v))])
    rtable[label]=vals
check_table('reason',rtable)
for label,key in [('Stock direct','stock_direct'),('Sparse JSON, 1 round','sparse_r1'),('Sparse JSON, 2 rounds','sparse_r2'),('Sparse JSON, 4 rounds','sparse_r4')]:
    for i,model in enumerate(('qwen15','qwen4')):
        assert int(rtable[label][i*2].split('/')[0])==verified['selected_results']['reasoning_32k_exact_per_8'][model][key]

mem_names=['memory-Qwen--Qwen2.5-1.5B-Instruct-566c642d5409','memory-Qwen--Qwen3-4B-Instruct-2507-991d4e87104b']; memories=[]
for i,n in enumerate(mem_names):
    p=E/n/('samples.jsonl' if i==0 else 'run/samples.jsonl'); sources.append(p); z=rows(p); assert len(z)==160
    assert sum(x['purpose']=='natural_model_memory' for x in z)==120 and sum(x['purpose']=='synthetic_stale_memory_stress' for x in z)==40
    assert len({x['episode_id'] for x in z})==4
    memories.append(z)
mtable={}
for label,policy in [('Stock, none','stock-memory-none-r2-cap128'),('Stock, notes','stock-memory-notes-r2-cap128'),('Sparse, none','sparse-memory-none-r2-cap128'),('Sparse, notes','sparse-memory-notes-r2-cap128'),('Sparse, addresses','sparse-memory-addresses-r2-cap128'),('Sparse, both','sparse-memory-both-r2-cap128')]:
    vals=[]
    for high in (False,True):
        z=[x for x in memories[1] if x['purpose']=='natural_model_memory' and (x['archive_tokens']>20000)==high and x['policy']['id']==policy]; assert len(z)==10 and len({x['episode_id'] for x in z})==2
        vals.extend([f"{sum(x['exact_correct'] for x in z)}/10",fmt(st.mean(x['request_s'] for x in z)),fmt(max(x['cuda_peak_allocated_bytes'] for x in z)/2**30)])
    mtable[label]=vals
check_table('memory',mtable)
for label,key in [('Stock, none','stock_none'),('Sparse, none','sparse_none'),('Sparse, notes','sparse_notes'),('Sparse, addresses','sparse_addresses'),('Sparse, both','sparse_both')]:
    assert int(mtable[label][3].split('/')[0])==verified['selected_results']['memory_32k_exact_per_10']['qwen4'][key]
for policy,key in [('stock-memory-none-r2-cap128','stock_none'),('sparse-memory-none-r2-cap128','sparse_none'),('sparse-memory-addresses-r2-cap128','sparse_addresses')]:
    z=[x for x in memories[0] if x['purpose']=='natural_model_memory' and x['archive_tokens']>20000 and x['policy']['id']==policy]
    assert len(z)==10 and sum(x['exact_correct'] for x in z)==verified['selected_results']['memory_32k_exact_per_10']['qwen15'][key]

obs=E/'attention-observer-ablation-fresh-1/observer-samples.jsonl'; sources.append(obs)
z=rows(obs)
assert len(z)==12
for arm in ('on','off'):
    v=[x for x in z if x['budget']==32768 and x['arm']==arm]; assert len(v)==3
    assert round(st.median(x['serving_s'] for x in v),3)==verified['selected_results']['observer_ablation_32k']['serving_median_s'][arm]
ratios=[]
for rep in range(3):
    off=next(x for x in z if x['budget']==32768 and x['arm']=='off' and x['rep']==rep)
    on=next(x for x in z if x['budget']==32768 and x['arm']=='on' and x['rep']==rep)
    ratios.append(off['serving_s']/on['serving_s'])
assert round(st.median(ratios),3)==verified['selected_results']['observer_ablation_32k']['median_paired_off_over_on']
for n in (reason_names[1],mem_names[1]):
    p=E/n/'run/manifest.json'; sources.append(p); model=read(p)['model']
    assert model['model_id']=='Qwen/Qwen3-4B-Instruct-2507' and model['revision']=='cdbee75f17c01a7cc42f958dc650907174af0554'
for item in verified['source_files']:
    p=ROOT/item['path']; assert p.is_file() and sha(p)==item['sha256'], item['path']
output={'status':'passed','manuscript_sha256':sha(P/'manuscript.tex'),'validator_sha256':sha(Path(__file__)),'verified_results_sha256':sha(P/'verified-results.json'),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'raw_counts':{'qwen15_confirmation':264,'qwen3_confirmation':264,'matched_reranker':18,'observer_ablation':12,'qwen15_reasoning':220,'qwen4_reasoning':220,'qwen15_memory':160,'qwen4_memory':160},'checked_tables':['confirm','cost','reason','memory'],'model_identity':'Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554','limits':['Saved scores and measurements are aggregated, not re-decoded or rescored.','No model inference, tensor reconstruction, or full replay was performed.','Hash matches and aggregates do not establish independent replication.','Timing and cohort independence limits remain in the manuscript and evidence audit.']}
(P/'validation-results.json').write_text(json.dumps(output,indent=2)+'\n')
print('PASS: four manuscript tables and selected saved-result values; report written')

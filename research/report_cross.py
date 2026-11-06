"""Compare completed model screens without pooling their quality cases."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from tiered_kv.store import file_sha, atomic_json
from tiered_kv.types import digest


def collect(runs):
    sources, table, curves = [], [], []
    identities = set()
    models = set()
    for run in runs:
        manifest = json.loads((run/'manifest.json').read_text())
        fixture = json.loads((run/'fixtures.json').read_text())
        complete = json.loads((run/'complete.json').read_text())
        rows = [json.loads(line) for line in (run/'samples.jsonl').read_text().splitlines()]
        if manifest['stage'] != 'screen' or not complete['weights_unchanged']:
            raise ValueError('only completed frozen-weight screening runs are comparable here')
        if digest(fixture) != manifest['fixture_hash']:
            raise ValueError('fixture differs from its frozen manifest')
        successful = [row for row in rows if row.get('status') == 'ok']
        if len(successful) != complete['rows'] or len(successful) != len(rows):
            raise ValueError('failed/incomplete rows require explicit separate analysis')
        keys = {(row['example_id'], row['config_id']) for row in rows}
        if len(keys) != len(rows):
            raise ValueError('duplicate records')
        identities.add(tuple(manifest[key] for key in
            ('implementation_hash', 'script_sha256', 'index_sha256', 'stock_sha256', 'host_sha256'))
            + (json.dumps(manifest['runtime'],sort_keys=True),))
        model = manifest['model']['model_id']
        if model in models:
            raise ValueError('multiple runs of one model cannot be silently pooled')
        models.add(model)
        sources.append({'run':str(run), 'model':model, 'rows':len(rows),
                        'samples_sha256':file_sha(run/'samples.jsonl'),
                        'manifest_sha256':file_sha(run/'manifest.json'),
                        'fixtures_sha256':file_sha(run/'fixtures.json')})
        facts = [row for row in rows if row.get('task') == 'niah']
        examples = {ex['id']:ex for ex in fixture['examples']}
        for row in facts:
            cfg=row['policy']
            ex=examples[row['example_id']]
            if digest(cfg)!=row['config_hash'] or cfg['C']!=0:
                raise ValueError('policy identity differs or this exact-position audit needs a separate compression scope')
            end=len(ex['tokens'])
            gold=set(range(*ex['answer_span']))
            if cfg['full']:
                available=set(range(end))
            else:
                if cfg['retrieval']!='fixed':
                    raise ValueError('this screen audit requires fixed-address selections')
                cutoff=max(0,end-cfg['W'])
                available=set(range(cutoff,end))|set(range(min(cfg['sink_tokens'],cutoff)))
                for block in cfg['fixed_blocks']:
                    available.update(range(block*cfg['R'],min((block+1)*cfg['R'],cutoff)))
            if not gold or abs(len(gold&available)/len(gold)-row['retrieval']['exact_availability'])>1e-12:
                raise ValueError('traced final-question answer availability differs from exact position arithmetic')
        full = {row['example_id']:row for row in facts if row['config_id'] == 'full_buffered'}
        expected_policies = {'full_buffered','sink_only','lexical_fast','rare_fast','semantic_fast'}
        if {row['config_id'] for row in facts} != expected_policies:
            raise ValueError('screen is missing a declared quality policy')
        if set(full) != {ex['id'] for ex in fixture['examples']}:
            raise ValueError('screen does not cover every declared quality fixture')
        for policy in sorted({row['config_id'] for row in facts}):
            chosen = [row for row in facts if row['config_id'] == policy]
            if {row['example_id'] for row in chosen} != set(full):
                raise ValueError('unpaired factual fixture set')
            retained = sum(row['code_found']==100 and full[row['example_id']]['code_found']==100 for row in chosen)
            table.append({'model':model, 'policy':policy, 'cases':len(chosen),
                'strict_correct':sum(row['em']==100 for row in chosen),
                'code_present':sum(row['code_found']==100 for row in chosen),
                'full_correct':sum(row['code_found']==100 for row in full.values()),
                'full_correct_retained':retained})
        forced = [row for row in rows if row.get('task') == 'latency']
        stock = {row['example_id']:row for row in forced if row['config_id'] == 'stock_hf'}
        expected_timing = {(policy,budget,0) for policy in expected_policies|{'stock_hf'} for budget in fixture['budgets']}
        if {(row['config_id'],row['budget'],row.get('trial',0)) for row in forced} != expected_timing:
            raise ValueError('screen does not cover the declared timing matrix')
        for row in forced:
            reference = stock[row['example_id']]
            if len(row['latency_ms']) != len(reference['latency_ms']):
                raise ValueError('unequal paired timing sample counts')
            mean = statistics.mean(row['latency_ms'])
            baseline = statistics.mean(reference['latency_ms'])
            curves.append({'model':model, 'policy':row['config_id'], 'budget':row['budget'],
                'trial':row.get('trial',0), 'measured_tokens':len(row['latency_ms']),
                'full_stock_forced_argmax_equal': (row['selected_hash']==reference['selected_hash']
                                                  if row['config_id']=='full_buffered' else None),
                'mean_ms':mean, 'stock_mean_ms':baseline, 'throughput_ratio':baseline/mean,
                'cuda_peak_GiB':row['cuda_decode_peak_allocated_bytes']/2**30,
                'stock_cuda_peak_GiB':reference['cuda_decode_peak_allocated_bytes']/2**30})
    if len(identities) != 1:
        raise ValueError('source revisions differ; keep experiments separate')
    return sources, table, curves


def plot(sources, latency, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'tiered-kv-cross-screen'
    models = [source['model'] for source in sources]
    families = list(dict.fromkeys(model.split('/')[-1].split('-')[0] for model in models))
    groups = [[model for model in models if model.split('/')[-1].split('-')[0] == family]
              for family in families]
    columns = max(map(len, groups))
    nrows = len(groups)
    figure, axes = plt.subplots(nrows, columns, figsize=(4.5*columns,3.0*nrows), squeeze=False)
    styles = [('lexical_fast','Lexical','#2563eb'),('rare_fast','Bounded rare-term','#d97706'),
              ('semantic_fast','Exact semantic scan','#7c3aed')]
    upper = max(1.3, max(row['throughput_ratio'] for row in latency)*1.12)
    for axis in axes.flat:
        axis.set_visible(False)
    panels = [(axes[row, column], model) for row, group in enumerate(groups)
              for column, model in enumerate(group)]
    for axis, model in panels:
        axis.set_visible(True)
        axis.axhline(1,color='#475569',linewidth=1,linestyle='--')
        for policy,label,color in styles:
            points = sorted((row for row in latency if row['model']==model and row['policy']==policy),key=lambda row:row['budget'])
            axis.plot([row['budget']/1024 for row in points],[row['throughput_ratio'] for row in points],
                      marker='o',label=label,color=color,linewidth=1.7,markersize=4)
        axis.set_title(model.split('/')[-1].replace('-Instruct',''),fontsize=11)
        axis.set_xscale('log',base=2)
        axis.set_xticks([4,8,16,32],['4','8','16','32'])
        axis.set_xlim(3.7,34.5)
        axis.set_ylim(0,upper)
        axis.set_xlabel('Context budget (Ki tokens)',fontsize=9)
        axis.set_ylabel('Throughput / stock HF',fontsize=9)
        axis.grid(axis='y',alpha=.2)
        axis.spines[['top','right']].set_visible(False)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.955))
    figure.suptitle('Decode crossover depends on model and context',fontsize=15,y=.995)
    figure.text(.5,.012,'Above 1: faster decoding. One trial per point; retrieval/setup/checkpoints excluded.\nFrozen fp16, batch one, T4. Quality is reported separately; no points beyond native limits.',ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.07,1,.90))
    figure.savefig(output/'crossover.png',dpi=170)
    figure.savefig(output/'crossover.svg',metadata={'Date':None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sources, quality, latency = collect(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output/'summary.json', {'sources':sources, 'quality':quality, 'latency':latency})
    for name, rows in [('quality',quality), ('latency',latency)]:
        with (args.output/(name+'.csv')).open('w',newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    plot(sources,latency,args.output)
    controls=[r for r in latency if r['policy']=='full_buffered']
    lines = ['# Cross-model exploratory screening', '',
             f'Completed models included: {len(sources)}. Each model is compared with its own full-attention control.',
             'Do not pool counts across models: tokenization, native limits and baseline task capability differ.', '',
             f"Full-buffered and stock forced-continuation argmax hashes match in {sum(r['full_stock_forced_argmax_equal'] for r in controls)}/{len(controls)} paired model/context controls.",
             'This supports the full-cache control through the tested native lengths; it is not full-logit equivalence or free-generation accuracy.','',
             f"All {sum(r['cases'] for r in quality)} factual records pass an independent position-set check of traced final-question answer-span availability, with frozen fixture and policy hashes verified.",
             'This checks position metadata, not the bitwise contents of every presented tensor.','',
             '| Model | Policy | Correct code / strict | Retained full-correct cases | Longest tested context | Mean ms: policy / stock | Throughput ratio | CUDA GiB: policy / stock |',
             '|---|---|---|---|---:|---|---:|---|']
    for row in quality:
        points = [x for x in latency if (x['model'],x['policy'])==(row['model'],row['policy'])]
        last = max(points, key=lambda x:x['budget'])
        lines.append(f"| {row['model']} | {row['policy']} | {row['code_present']}/{row['cases']} / {row['strict_correct']}/{row['cases']} | {row['full_correct_retained']}/{row['full_correct']} | {last['budget']} | {last['mean_ms']:.2f} / {last['stock_mean_ms']:.2f} | {last['throughput_ratio']:.2f} | {last['cuda_peak_GiB']:.3f} / {last['stock_cuda_peak_GiB']:.3f} |")
    lines += ['', 'One latency trial per context; correlated token measurements are not independent trials.',
              'Stock HF runs last within each fixture. Timings exclude index search, archive mounting, setup and checkpointing.',
              'All models are frozen fp16, batch one, on one T4 VM. CUDA allocations do not measure whole-device or whole-host memory.',
              'Sources and raw-file hashes are listed in summary.json. This comparison does not establish general architecture superiority.',
              '', '![Decoding crossover by model](crossover.png)']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'report.md')


if __name__ == '__main__':
    main()

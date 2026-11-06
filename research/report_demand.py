"""Summarize one immutable demand/semantic run without pooling experiments."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from tiered_kv.store import atomic_json,file_sha
from tiered_kv.types import digest


def validate_matrix(manifest,fixture,rows,complete):
    """Check the declared demand matrix, not just a self-reported row count."""
    stage=manifest.get('stage')
    policy_sets={
        'screen':{'full_buffered','sink_only','lexical_fast','rare_fast','semantic_fast'},
        'confirm':{'full_buffered','sink_only','lexical_fast','rare_fast','rare_context','semantic_fast'},
        'online':{'full_buffered','sink_only','rare_context'},
        'summaries':{'full_buffered','sink_only','lexical_fast','semantic_fast','semantic_k32','semantic_compressed'}}
    if stage not in policy_sets:
        return None
    if digest(fixture)!=manifest['fixture_hash']:
        raise ValueError('fixture differs from its frozen manifest')
    policies=policy_sets[stage]
    expected={(policy,ex['id']):ex['task'] for ex in fixture['examples'] for policy in policies}
    first_seed=min((ex['seed'] for ex in fixture['examples'] if 'seed' in ex),default=None)
    for ex in fixture['examples']:
        if ex.get('depth')==.5 and ex['id'].startswith('niah-') and ex['seed']==first_seed:
            for trial in range(3 if stage=='confirm' else 1):
                expected.update({(policy,ex['id']+f'-forced-{trial}'):'latency' for policy in policies|{'stock_hf'}})
    actual={(row['config_id'],row['example_id']) for row in rows}
    if actual-set(expected) or (complete and actual!=set(expected)):
        raise ValueError('successful records do not cover the declared policy/fixture/timing matrix')
    if any(row['task']!=expected[row['config_id'],row['example_id']] for row in rows):
        raise ValueError('record task differs from declared matrix')
    return {'expected_rows':len(expected),'observed_rows':len(actual),'missing_rows':len(set(expected)-actual)}


def trial_ranges(latency):
    """Summarize trial means, never treating individual tokens as replicates."""
    result=[]
    stock={(r['budget'],r['trial']):r['mean_ms'] for r in latency if r['policy']=='stock_hf'}
    for policy,budget in sorted({(r['policy'],r['budget']) for r in latency}):
        rows=sorted((r for r in latency if (r['policy'],r['budget'])==(policy,budget)),key=lambda r:r['trial'])
        if len({r['trial'] for r in rows})!=len(rows):
            raise ValueError('duplicate timing trials')
        means=[r['mean_ms'] for r in rows]
        ratios=[stock[(budget,r['trial'])]/r['mean_ms'] for r in rows if (budget,r['trial']) in stock]
        result.append({'policy':policy,'budget':budget,'trials':len(rows),
            'mean_of_trial_means_ms':statistics.mean(means),'min_trial_mean_ms':min(means),
            'max_trial_mean_ms':max(means),'paired_stock_trials':len(ratios),
            'mean_paired_throughput_ratio':statistics.mean(ratios) if ratios else None,
            'min_paired_throughput_ratio':min(ratios) if ratios else None,
            'max_paired_throughput_ratio':max(ratios) if ratios else None})
    return result


def context_scaling(latency):
    """Describe measured context growth within each trial, without extrapolation."""
    result=[]
    for policy,trial in sorted({(r['policy'],r['trial']) for r in latency}):
        points=sorted((r for r in latency if (r['policy'],r['trial'])==(policy,trial)),key=lambda r:r['budget'])
        by_budget={r['budget']:r for r in points}
        if len(by_budget)!=len(points):
            raise ValueError('duplicate context in one timing trial')
        if len(points)<2:
            continue
        x=np.log([r['prompt_tokens'] for r in points])
        if not np.isfinite(x).all() or len(set(x))!=len(points):
            raise ValueError('invalid or repeated actual context lengths')
        record={'policy':policy,'trial':trial,'budgets':[r['budget'] for r in points],
                'actual_prompt_tokens':[r['prompt_tokens'] for r in points]}
        for measure in ('mean','median'):
            values=[r[measure+'_ms'] for r in points]
            if any(v<=0 or not np.isfinite(v) for v in values):
                raise ValueError('invalid timing statistic')
            record[measure+'_log_log_slope']=float(np.polyfit(x,np.log(values),1)[0])
            record[measure+'_32K_over_8K']=(by_budget[32768][measure+'_ms']/by_budget[8192][measure+'_ms']
                                          if {8192,32768}<=set(by_budget) else None)
        ratio=record['median_32K_over_8K']
        record['median_ratio_at_or_below_1_25']=ratio<=1.25 if ratio is not None else None
        result.append(record)
    return result


def report_sessions(run,rows,complete):
    """Only complete three-turn attempts contribute to persistent-session tables."""
    markers=[r for r in rows if r.get('session_complete') and r.get('status')=='ok']
    sessions={r['session_id']:r for r in markers}
    if len(sessions)!=len(markers):
        raise ValueError('duplicate completed sessions; do not silently pool attempts')
    selected=[]
    for session,marker in sessions.items():
        queries=[r for r in rows if r.get('session_id')==session and r.get('attempt')==marker['attempt'] and 'turn' in r]
        if sorted(r['turn'] for r in queries)!=[0,1,2] or any(r.get('status')!='ok' for r in queries):
            raise ValueError('completed session lacks three successful distinct turns')
        selected.extend(queries)
    manifest=json.loads((run/'manifest.json').read_text())
    fixture=json.loads((run/'fixtures.json').read_text())
    policies=('stock_hf','sink_only','rare_fast','rare_context')
    expected={f"{ex['id']}-{condition}-{policy}" for ex in fixture['examples']
              for condition in manifest['cache_conditions'] for policy in policies}
    if complete and set(sessions)!=expected:
        raise ValueError('completion marker lacks declared policy/context/cache-condition sessions')
    # Cache conditions repeat the same questions; use only warm sessions for
    # paired factual quality and keep those repeats out of the denominator.
    warm={(r['budget'],r['turn'],r['policy']):r for r in selected if r['condition']=='warm'}
    quality=[]
    coverage=[]
    for policy in policies:
        group=[r for r in warm.values() if r['policy']==policy]
        paired=[r for r in group if warm.get((r['budget'],r['turn'],'stock_hf'),{}).get('code_found')==100]
        quality.append({'policy':policy,'unique_queries':len(group),
            'correct_code':sum(r['code_found']==100 for r in group),
            'strict_correct':sum(r['em']==100 for r in group),
            'stock_correct_cases':len(paired),'stock_correct_retained':sum(r['code_found']==100 for r in paired)})
        if policy.startswith('rare_'):
            for row in group:
                ex=next(e for e in fixture['examples'] if e['budget']==row['budget'])
                target=ex['queries'][row['turn']]
                start,end=target['record_span']
                block_size=row['policy_config']['R']
                needed=list(range(start//block_size,(end-1)//block_size+1))
                coverage.append({'budget':row['budget'],'turn':row['turn'],'policy':policy,
                    'record_span':[start,end],'record_blocks':needed,'selected_blocks':row['selected_blocks'],
                    'selected_record_complete':set(needed)<=set(row['selected_blocks']),
                    'answer':target['answer'],'prediction':row['prediction'],'code_found':row['code_found']})
    repeated=[(row,warm[(row['budget'],row['turn'],row['policy'])]) for row in selected
              if row['condition']=='advised_cold_KV_only' and (row['budget'],row['turn'],row['policy']) in warm]
    table=[]
    for key in sorted({(r['budget'],r['condition'],r['policy']) for r in selected}):
        budget,condition,policy=key
        group=[r for r in selected if (r['budget'],r['condition'],r['policy'])==key]
        marker=sessions[group[0]['session_id']]
        table.append({'budget':budget,'condition':condition,'policy':policy,'queries':len(group),
            'strict_correct':sum(r['em']==100 for r in group),'code_present':sum(r['code_found']==100 for r in group),
            'median_first_token_s':statistics.median(r['first_token_s'] for r in group),
            'median_request_s':statistics.median(r['request_s'] for r in group),
            'output_tokens':[r['output_tokens'] for r in sorted(group,key=lambda r:r['turn'])],
            'lookup_max_ms':max(r['index_search_s'] for r in group)*1000,
            'logical_read_MiB':sum(r['logical_read_bytes'] for r in group)/2**20,
            'physical_read_MiB':sum(r.get('process_io_delta',{}).get('read_bytes',0) for r in group)/2**20,
            'cuda_peak_GiB':max(r['cuda_peak_allocated_bytes'] for r in group)/2**30,
            'host_sampled_peak_GiB':max(r['host_rss_sampled_peak_bytes'] for r in group)/2**30,
            'active_KV_entries_max':max(r['active_KV_entries'] for r in group),
            'archive_mount_s':group[0]['archive_mount_s'],'controller_init_s':group[0]['controller_init_s'],
            'checkpoint_s':marker['checkpoint_s'],'resume_max_abs':marker.get('resumed_logit_max_abs')})
    summary={'status':'complete_run' if complete else 'partial_run','samples_sha256':file_sha(run/'samples.jsonl'),
        'manifest_sha256':file_sha(run/'manifest.json'),'source_rows':len(rows),'complete_sessions':len(sessions),
        'expected_sessions':len(expected),'complete_queries':len(selected),'table':table,
        'unique_warm_quality':quality,'selected_span_coverage':coverage,
        'warm_cold_pairs':len(repeated),
        'warm_cold_prediction_and_length_matches':sum(a['prediction']==b['prediction'] and a['output_tokens']==b['output_tokens'] for a,b in repeated),
        'fixtures_sha256':file_sha(run/'fixtures.json'),
        'uncompleted_query_rows':sum('turn' in r for r in rows)-len(selected),
        'limitations':['three different questions per context, not independent repeated latency trials',
            'warm and advised-cold repeat the same facts; do not pool as independent quality cases',
            'request time includes lookup, pending token, question and variable-length generated answer',
            'both policies feed question tokens serially; this does not benchmark a stock server with batched question prefill',
            'prefill, verified archive mount, controller initialization and final checkpoint are separate',
            'only KV files receive cold-cache advice; metadata remains warm; inspect physical read counters',
            'file-cache advice does not evict selected KV already resident on the GPU; inspect per-query reads for actual misses',
            'cache conditions and policies use a fixed order, not randomized repeated request trials',
            'older raw host_observation_scope text is generic; source code samples the request only',
            'Colab backing storage is not a qualified physical SSD; native-context and one-model limits apply']}
    atomic_json(run/'summary.json',summary)
    with (run/'sessions.csv').open('w',newline='') as out:
        if table:
            writer=csv.DictWriter(out,fieldnames=list(table[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(table)
    lines=['# '+('Completed' if complete else 'Partial')+' persistent-query experiment','',
        f"Complete sessions: {len(sessions)}/{len(expected)}; complete query rows: {len(selected)}.",
        'This is one Qwen2.5-1.5B protocol experiment, not completion of the architecture research.','',
        '| Policy | Correct code / unique questions | Strict output | Stock-correct answers retained |',
        '|---|---|---|---|',
        *[f"| {r['policy']} | {r['correct_code']}/{r['unique_queries']} | {r['strict_correct']}/{r['unique_queries']} | {r['stock_correct_retained']}/{r['stock_correct_cases']} |" for r in quality],
        '', 'Quality uses warm sessions once. Warm/cold repeats are not independent facts.',
        'The JSON also records complete-record coverage inferred from selected block IDs; this is not a direct per-layer payload audit.','',
        '| Budget | Cache condition | Policy | Correct code / strict | Median first token / request s | Output tokens by turn | Peak CUDA GiB | Total logical / physical reads MiB |',
        '|---:|---|---|---|---|---|---:|---|']
    for r in table:
        lines.append(f"| {r['budget']} | {r['condition']} | {r['policy']} | {r['code_present']}/{r['queries']} / {r['strict_correct']}/{r['queries']} | {r['median_first_token_s']:.3f} / {r['median_request_s']:.3f} | {r['output_tokens']} | {r['cuda_peak_GiB']:.3f} | {r['logical_read_MiB']:.2f} / {r['physical_read_MiB']:.2f} |")
    lines+=['','The CSV preserves one-time mount/setup, checkpoint, resume, index lookup and host-memory observations.',
        'Variable-length answers and early EOS make request times task-cost observations; use the forced-token studies for decoding throughput.',
        '', '## Limits','',*['- '+s+'.' for s in summary['limitations']]]
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print(run/'report.md')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    complete=(args.run/'complete.json').exists()
    if not complete and not args.allow_partial:
        p.error('run has no completion marker; use --allow-partial for an explicitly interim report')
    source=args.run/'samples.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    if any('session_id' in r for r in rows):
        report_sessions(args.run,rows,complete)
        return
    successful=[r for r in rows if r.get('status')=='ok' and 'config_id' in r]
    keys=[(r['config_id'],r['example_id']) for r in successful]
    if len(keys)!=len(set(keys)):
        raise ValueError('duplicate successful rows; do not silently pool retries')
    manifest=json.loads((args.run/'manifest.json').read_text()) if (args.run/'manifest.json').exists() else {}
    if complete:
        marker=json.loads((args.run/'complete.json').read_text())
        if marker.get('rows',len(successful))!=len(successful):
            raise ValueError('completion marker and downloaded samples disagree')
        if not (args.run/'manifest.json').exists():
            raise ValueError('completed report requires the run manifest')
    policies=sorted({r['config_id'] for r in successful})
    by_key={(r['config_id'],r['example_id']):r for r in successful}
    full=next((name for name in ('full_buffered','full') if name in policies),None)
    fixture_path=args.run/'fixtures.json'
    fixture=json.loads(fixture_path.read_text()) if fixture_path.exists() else {}
    fixtures={e['id']:e for e in fixture.get('examples',[])}
    matrix=validate_matrix(manifest,fixture,successful,complete)
    if complete and matrix and not marker.get('weights_unchanged'):
        raise ValueError('completed matrix requires verified frozen weights')
    def needs_cold(r):
        ex=fixtures.get(r['example_id'])
        if ex is None or 'record_span' not in ex:
            return None
        window=r.get('policy',{}).get('W',2000)
        return ex['record_span'][0]<max(0,ex['prefix_tokens_count']-window+1)
    quality=[]
    latency=[]
    quality_by_budget=[]
    summaries=[{'policy':r['config_id'],'example_id':r['example_id'],'prompt_tokens':r['prompt_length'],
                'rougeL':r['rougeL'],'scorer_error':r.get('scorer_error')} for r in successful if r.get('task')=='gov_report']
    for policy in policies:
        measured=[r for r in successful if r['config_id']==policy]
        factual=[r for r in measured if r.get('task')=='niah']
        if factual:
            paired=[r for r in factual if by_key.get((full,r['example_id']),{}).get('code_found')==100.]
            cold=[r for r in factual if needs_cold(r)]
            quality.append({'policy':policy,'cases':len(factual),'strict_correct':sum(r['em']==100 for r in factual),
                'code_present':sum(r['code_found']==100 for r in factual),
                'cold_cases':len(cold),'cold_code_present':sum(r['code_found']==100 for r in cold),
                'full_correct_cases':len(paired),'full_correct_retained':sum(r['code_found']==100 for r in paired),
                'oracle':any(r.get('oracle') for r in factual)})
            for budget in sorted({r['budget'] for r in factual}):
                group=[r for r in factual if r['budget']==budget]
                retained=[r for r in group if by_key.get((full,r['example_id']),{}).get('code_found')==100.]
                quality_by_budget.append({'policy':policy,'budget':budget,'cases':len(group),
                    'strict_correct':sum(r['em']==100 for r in group),
                    'code_present':sum(r['code_found']==100 for r in group),
                    'full_correct_cases':len(retained),
                    'full_correct_retained':sum(r['code_found']==100 for r in retained)})
        for r in measured:
            if not r.get('latency_ms'):
                continue
            times=r['latency_ms']
            all_times=r.get('all_latency_ms',times)
            latency.append({'policy':policy,'budget':r['budget'],'prompt_tokens':r['prompt_length'],
                'trial':r.get('trial',0),'n':len(times),'mean_ms':statistics.mean(times),
                'median_ms':statistics.median(times),'p95_ms':float(np.percentile(times,95)),
                'p99_ms':float(np.percentile(times,99)),'max_ms':max(times),
                'all_calls_mean_ms':statistics.mean(all_times),'all_calls_n':len(all_times),
                'all_calls_includes_warmup':'all_latency_ms' in r,
                'cuda_peak_GiB':r.get('cuda_decode_peak_allocated_bytes',0)/2**30,
                'query_s':r.get('query_s'),'init_s':r.get('init_s'),
                'archive_verify_open_s':r.get('archive_verify_open_s'),
                'controller_init_s':r.get('controller_init_s'),
                'checkpoint_s':r.get('checkpoint_s'),'retrieval_event_s':r.get('retrieval_event_s')})
    ranges=trial_ranges(latency)
    scaling=context_scaling(latency)
    ingestion=[{'policy':r['config_id'],'example_id':r['example_id'],'archive_tokens':r['archive_tokens'],
        'ingestion_s':r['prefill_s'],'ingestion_tokens_s':r['prefill_tokens_s'],
        'cuda_ingestion_peak_GiB':r['cuda_online_ingestion_peak_allocated_bytes']/2**30,
        'cuda_query_decode_peak_GiB':r['cuda_decode_peak_allocated_bytes']/2**30,
        'active_KV_entries':r['active_KV_entries'],'checkpoint_s':r['checkpoint_s'],
        'logical_write_MiB':r['logical_write_bytes']/2**20,
        'retrieval_activation_position':r.get('retrieval_activation_position')}
        for r in successful if r.get('protocol')=='online_tiered']
    for row in ingestion:
        if row['policy']!='full_buffered' and row['retrieval_activation_position']!=row['archive_tokens']:
            raise ValueError('online retrieval was not activated at the declared query boundary')
    study_label='held-out confirmation' if manifest.get('stage')=='confirm' else 'exploratory run'
    summary={'status':'complete_run' if complete else 'partial_run','samples_sha256':file_sha(source),
             'manifest_sha256':file_sha(args.run/'manifest.json') if (args.run/'manifest.json').exists() else None,
             'source_rows':len(rows),'successful_rows':len(successful),'failures':[r for r in rows if r.get('status')!='ok'],
             'matrix_coverage':matrix,'fixtures_sha256':file_sha(fixture_path) if fixture_path.exists() else None,
             'quality':quality,'quality_by_budget':quality_by_budget,'latency':latency,'latency_trial_ranges':ranges,'context_scaling':scaling,'summaries':summaries,'online_ingestion':ingestion,
             'limitations':['one run is not cross-model validation','token timings within a trial are correlated',
                            'quality cases reuse some target IDs; case counts are not independent factual coverage',
                            'decode timing excludes index lookup, archive mounting and prefill; separate costs remain visible',
                            'stock HF is the practical no-disk full-cache speed baseline']}
    if any(r.get('protocol') in ('query_replay','exact_shared') for r in successful):
        summary['limitations'].append('shared-archive ttft_s adds historical prefill to setup/query; it is not measured fresh end-to-end latency and excludes index build and prefix checkpoint')
    if ingestion:
        summary['limitations']+=['online ingestion starts empty and feeds every token serially; its full baseline also writes KV to disk',
            'online text indices and fixture tokens were prepared offline; incremental indexing and constant whole-process RAM are not demonstrated',
            'online cases are different fixtures, not repeated timing trials; stored KV is exact for each policy trajectory, not necessarily equal to dense-ingestion KV']
    atomic_json(args.run/'summary.json',summary)
    with (args.run/'latency.csv').open('w',newline='') as out:
        if latency:
            writer=csv.DictWriter(out,fieldnames=list(latency[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(latency)
    if ingestion:
        with (args.run/'online-ingestion.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(ingestion[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(ingestion)
    lines=['# '+('Completed ' if complete else 'Partial ')+study_label,'',
           'This report describes one experiment, not completion of the architecture research.',
           f'Raw rows: {len(rows)}; successful rows: {len(successful)}. Source: `{args.run.name}`.',
           *([f"Declared matrix: {matrix['observed_rows']}/{matrix['expected_rows']} successful records."] if matrix else []),
           '', '## Factual retrieval', '',
           '| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |',
           '|---|---:|---:|---:|---:|']
    for q in quality:
        label=q['policy']+(' (gold-assisted diagnostic)' if q['oracle'] else '')
        lines.append(f"| {label} | {q['strict_correct']}/{q['cases']} | {q['code_present']}/{q['cases']} | {q['cold_code_present']}/{q['cold_cases']} | {q['full_correct_retained']}/{q['full_correct_cases']} |")
    if 'oracle_k4' in policies:
        lines+=['','The `oracle_k4` answer-block-only control is underfilled and is not an accuracy upper bound.']
    lines+=['','Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.']
    if quality_by_budget:
        factual_fixtures=[e for e in fixture.get('examples',[]) if e.get('task')=='niah']
        associations={(e.get('item'),e.get('answer')) for e in factual_fixtures}
        if factual_fixtures and len(associations)==1 and all(e.get('item') and e.get('answer') for e in factual_fixtures):
            lines+=['',f"These {len(factual_fixtures)} fixtures are layouts of one target item/code association, not independent target facts. A multikey fixture adds distractor keys and requests one code."]
        lines+=['','## Factual quality by context budget','',
            'Case mixes may differ by budget; inspect individual fixtures before attributing differences to length.',
            '', '| Budget | Policy | Correct code | Strict output | Full-correct answers retained |',
            '|---:|---|---:|---:|---:|']
        for q in sorted(quality_by_budget,key=lambda r:(r['budget'],r['policy'])):
            lines.append(f"| {q['budget']} | {q['policy']} | {q['code_present']}/{q['cases']} | {q['strict_correct']}/{q['cases']} | {q['full_correct_retained']}/{q['full_correct_cases']} |")
    if summaries:
        lines+=['','## Complete-report summaries','','ROUGE-L measures reference overlap, not factual correctness. Inspect generated claims against the complete source report.',
                '', '| Report | Tokens | Policy | ROUGE-L | Scorer error |','|---|---:|---|---:|---|']
        for r in sorted(summaries,key=lambda r:(r['example_id'],r['policy'])):
            lines.append(f"| {r['example_id']} | {r['prompt_tokens']} | {r['policy']} | {r['rougeL']:.2f} | {r['scorer_error'] or ''} |")
    if ingestion:
        lines+=['','## Empty-start online ingestion','',
            'The full-buffered control also writes every KV record to disk. These are matched ingestion trajectories, not a stock no-disk serving benchmark.',
            'Index preparation remains offline. The CSV preserves retrieval activation, active KV, writes and checkpoint cost.',
            '', '| Fixture | Policy | Archive tokens | Ingestion s / tokens per s | Ingestion CUDA GiB | Query/decode CUDA GiB | Checkpoint s |',
            '|---|---|---:|---|---:|---:|---:|']
        for r in ingestion:
            lines.append(f"| {r['example_id']} | {r['policy']} | {r['archive_tokens']} | {r['ingestion_s']:.2f} / {r['ingestion_tokens_s']:.2f} | {r['cuda_ingestion_peak_GiB']:.3f} | {r['cuda_query_decode_peak_GiB']:.3f} | {r['checkpoint_s']:.3f} |")
        lines+=['','### Query and generation scope','','Query time is final prompt processing only; generated length and summed decode-step times vary with the answer. Neither is an entire request wall time, and ingestion, index preparation, setup and checkpoint are separate.', '', '| Fixture | Policy | Query s | Output tokens | Decode-step sum s |', '|---|---|---:|---:|---:|']
        for r in successful:
            if r.get('protocol')=='online_tiered':
                lines.append(f"| {r['example_id']} | {r['config_id']} | {r['query_s']:.3f} | {r['output_tokens']} | {sum(r['decode_ms'])/1000:.3f} |")
    if latency:
        lines+=['', '## Forced-token latency', '',
                '| Context budget | Policy | Trial | Mean / median ms | p95 / max ms | CUDA peak GiB |',
                '|---:|---|---:|---:|---:|---:|']
        for r in sorted(latency,key=lambda r:(r['budget'],r['policy'],r['trial'])):
            lines.append(f"| {r['budget']} | {r['policy']} | {r['trial']} | {r['mean_ms']:.2f} / {r['median_ms']:.2f} | {r['p95_ms']:.2f} / {r['max_ms']:.2f} | {r['cuda_peak_GiB']:.3f} |")
        lines+=['','These are synchronized forced forwards after 16 warmups. The CSV also preserves means over all calls when available, including buffered-write flushes.',
                'Physical CUDA allocations include weights, active KV and workspace. They do not measure whole-device memory or disk page-cache residency.',
                'Archive construction, verified mounting, controller setup, index lookup and checkpoints are not included in steady decode; their available timings remain in the CSV and raw rows.',
                'A short successful answer or early EOS is not used to establish speed. Compare separate trial means; tokens within a trial are not independent repetitions.']
    if scaling:
        lines+=['','## Measured context growth','',
            'Ratios compare 32K with 8K within each numbered trial. The original finite-range ratio threshold is 1.25; this describes decode timing only, not retrieval cost, accuracy, or asymptotic complexity.',
            'Slopes fit log median latency against log actual prompt length over the available native-context points. Trial ranges are not confidence intervals. Models tested only through 8K have no 32K ratio.',
            '', '| Policy | 32K/8K median ratios by trial | Median log-log slopes by trial |',
            '|---|---|---|']
        for policy in sorted({r['policy'] for r in scaling}):
            group=[r for r in scaling if r['policy']==policy]
            ratios=', '.join(f"{r['median_32K_over_8K']:.3f}" if r['median_32K_over_8K'] is not None else 'N/A' for r in group)
            slopes=', '.join(f"{r['median_log_log_slope']:.3f}" for r in group)
            lines.append(f'| {policy} | {ratios} | {slopes} |')
    lines+=['', '## Limitations', '',*['- '+item+'.' for item in summary['limitations']]]
    if any(r['trials']>1 for r in ranges):
        lines+=['','## Variation across timing trials','',
                'Ranges below are observed trial-to-trial variation, not confidence intervals. Stock runs last within each fixture.',
                '', '| Budget | Policy | Trials | Mean of trial means ms | Min–max trial mean ms | Mean paired throughput / stock |',
                '|---:|---|---:|---:|---|---:|']
        for r in sorted(ranges,key=lambda r:(r['budget'],r['policy'])):
            ratio=r['mean_paired_throughput_ratio']
            lines.append(f"| {r['budget']} | {r['policy']} | {r['trials']} | {r['mean_of_trial_means_ms']:.2f} | {r['min_trial_mean_ms']:.2f}–{r['max_trial_mean_ms']:.2f} | {ratio:.2f} |" if ratio is not None else
                         f"| {r['budget']} | {r['policy']} | {r['trials']} | {r['mean_of_trial_means_ms']:.2f} | {r['min_trial_mean_ms']:.2f}–{r['max_trial_mean_ms']:.2f} | unavailable |")
    if latency:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['svg.hashsalt']='tiered-kv-demand-report'
        chosen=[name for name in ('stock_hf','default','lexical_k4','semantic_k4','lexical_fast','rare_fast','rare_context','semantic_fast') if name in policies]
        fig,axes=plt.subplots(1,2,figsize=(11,4.4))
        for policy in chosen:
            subset=[r for r in latency if r['policy']==policy]
            budgets=sorted({r['budget'] for r in subset})
            means=[statistics.mean(r['mean_ms'] for r in subset if r['budget']==b) for b in budgets]
            lower=[m-min(r['mean_ms'] for r in subset if r['budget']==b) for b,m in zip(budgets,means)]
            upper=[max(r['mean_ms'] for r in subset if r['budget']==b)-m for b,m in zip(budgets,means)]
            memory=[statistics.mean(r['cuda_peak_GiB'] for r in subset if r['budget']==b) for b in budgets]
            axes[0].errorbar(np.array(budgets)/1024,means,yerr=[lower,upper],fmt='o-',capsize=3,label=policy)
            axes[1].plot(np.array(budgets)/1024,memory,'o-',label=policy)
        for ax in axes:
            ax.set_xlabel('Context budget (Ki tokens; prompt is shorter by output reserve)')
            ax.set_xticks(sorted({r['budget']/1024 for r in latency}))
            ax.grid(alpha=.25)
        axes[0].set_ylabel('Mean forced-token ms (bars: range of trial means)')
        axes[1].set_ylabel('Peak allocated CUDA memory (GiB)')
        axes[0].legend(fontsize=8)
        fig.suptitle(('Completed ' if complete else 'Partial ')+study_label+' — '+args.run.name,fontsize=10)
        fig.tight_layout()
        fig.savefig(args.run/'scaling.png',dpi=180)
        fig.savefig(args.run/'scaling.svg',metadata={'Date':None})
        svg=args.run/'scaling.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
        lines+=['','![Measured scaling](scaling.png)']
    (args.run/'report.md').write_text('\n'.join(lines)+'\n')
    print(args.run/'report.md')


if __name__=='__main__':
    main()

"""Audit frozen multi-round traces before making quality or speed claims.

Partial mode checks available observations but never emits a completed report.
Logical attention work is checked independently from the measured counters.
It remains an operation-count proxy, not measured hardware FLOPs.
"""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import tempfile

from research.reasoning_protocol import parse_response, next_search, round_request
from research.persistent_kv import next_question_tokens


SOURCE = '5d19c3ca3c48d8ed9b0c76773ef9b9996dec4243'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def declared_policies():
    policies=[dict(attention='stock',rounds=1,cap_per_round=128,format='direct',schedule_labels=['direct_vanilla'])]
    for attention in ('stock','sparse'):
        for rounds,cap,labels in [(1,128,['128_per_round']),(2,128,['128_per_round']),
                (4,128,['128_per_round','512_total']),(1,512,['512_total']),(2,256,['512_total'])]:
            policies.append(dict(attention=attention,rounds=rounds,cap_per_round=cap,format='json',schedule_labels=labels))
    return [{**p,'id':f"{p['attention']}-{p['format']}-r{p['rounds']}-cap{p['cap_per_round']}"} for p in policies]


def expected_schedule(fixtures):
    rng=random.Random(173)
    schedule=[]
    for ex in fixtures:
        for trial in range(3 if ex['task']=='linked_fact' and ex['seed']==157 else 1):
            policies=declared_policies();rng.shuffle(policies)
            schedule.extend(dict(example_id=ex['id'],trial=trial,policy=p,
                purpose='quality' if trial==0 else 'repeated_timing') for p in policies)
    return schedule


def verify_source(manifest, fixtures, repo, revision):
    need(bool(re.fullmatch('[0-9a-f]{40}',revision)),'full immutable source revision required')
    def source(name):
        return subprocess.check_output(['git','show',revision+':'+name],cwd=repo)
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',revision],cwd=repo,text=True).splitlines()
    core=sorted(f for f in files if len(Path(f).parts)==2 and Path(f).parts[0] in ('tiered_kv','tests') and f.endswith('.py'))
    core+=['configs/matrix.yaml','configs/eval.yaml','requirements.lock']
    need(digest({f:hashlib.sha256(source(f)).hexdigest() for f in core})==manifest['implementation_hash'],
         'frozen implementation/test/dependency identity differs')
    need(hashlib.sha256(source('research/reasoning_run.py')).hexdigest()==manifest['script_sha256'],
         'frozen generator source differs')
    expected_paths={'research/reasoning_protocol.py','research/reasoning_sessions.py','research/persistent_kv.py',
                    'research/budget_probe.py','research/semantic_index.py','research/resources.py'}
    need(set(manifest['dependencies_sha256'])==expected_paths,'incomplete research dependency identity')
    for name,sha in manifest['dependencies_sha256'].items():
        need(hashlib.sha256(source(name)).hexdigest()==sha,'frozen dependency differs: '+name)
    # These are shared formatting functions used by the auditor; require the
    # local copies to match the immutable generator source before using them.
    for name in ('research/reasoning_protocol.py','research/persistent_kv.py',
                 'research/semantic_index.py','research/budget_probe.py'):
        need((repo/name).read_bytes()==source(name),'audit formatting helper differs from frozen source')
    expected=json.loads(source('research/reasoning-fixture-expectations.json'))
    model=manifest['model']
    pinned=next(m for m in expected['models'] if m['model_id']==model['model_id'])
    specs=json.loads(source('research/model-manifest.json'))+json.loads(source('research/long-context-candidates.json'))['models']
    need(model==next(m for m in specs if m['model_id']==model['model_id']),'model specification differs')
    need(model['revision']==pinned['revision'],'model revision differs')
    need(digest(fixtures)==manifest['fixture_sha256']==pinned['fixture_sha256'],'fixture payload differs')
    need(manifest['schedule']==expected_schedule(fixtures),'randomized declared schedule differs')
    need(len(fixtures)==16 and len(manifest['schedule'])==220,'unexpected cohort extent')
    need(manifest['runtime']['device']=='cuda','not CUDA generator evidence')
    return pinned


def audit_validation(value, model, *, parameter_count_required=True):
    # The frozen memory runner asserts parameter count but did not save that
    # count in its validation JSON. Do not invent a recorded measurement.
    if parameter_count_required or 'loaded_parameters' in value:
        need(value['loaded_parameters']==model['parameters'],'loaded parameter count differs')
    need(bool(re.fullmatch('[0-9a-f]{64}',value['weights_sha256'])),'missing actual weight identity')
    need([v['length'] for v in value['parity']]==[64,512,1025],'missing actual-model parity lengths')
    for check in value['parity']:
        need(check['atol']==check['rtol']==1e-4,'changed CUDA parity tolerance')
        need(len(check['max_errors'])==len(check['greedy_agreement'])==17,'incomplete parity probes')
        need(all(math.isfinite(v) and 0<=v<=1e-4 for v in check['max_errors'])
             and all(check['greedy_agreement']),'matched-schedule parity failed')
        labels=[f'prefill-{min(n+128,check["length"])}' for n in range(0,check['length'],128)]
        labels += [f'decode-{n}' for n in range(1,17)]
        need([v['label'] for v in check['same_chunk_stock']]==labels,'incomplete same-chunk stock controls')
        need(all(0<=v['max_abs']<=1e-4 and v['greedy_agreement'] for v in check['same_chunk_stock']),
             'same-chunk stock control failed')
    online=value['online']
    need(len(online)==2 and {r['config_id'] for r in online}=={'full_buffered','default'},'resume controls missing')
    need(all(r['status']=='ok' and r['processed_tokens']==512 and r['checkpoint_position']==256
             and r['emitted_pending_consumed_once'] and r['kv_bit_identical']
             and 0<=r['max_logit_error']<=1e-4 and r['kv_max_error']==0 for r in online),
         'actual-model resume mechanics failed')


def check_work(work, stock, prefix, evaluated, prompts, calls, layers, heads):
    need(work['model_token_evaluations']==evaluated,'evaluated-token accounting differs')
    need(work['prompt_token_evaluations']==prompts,'prompt-token accounting differs')
    need(work['transformer_forward_calls']==calls,'forward-call accounting differs')
    positions=work['layer_summed_valid_KV_positions']
    need(type(positions) is int and positions>=0,'invalid attention position count')
    need(work['logical_attention_head_pairs']==positions*heads,'head-pair accounting differs')
    if stock:
        need(positions==layers*(evaluated*prefix+evaluated*(evaluated+1)//2),'dense causal attention count differs')
        need(work['max_active_KV_entries']==(prefix+evaluated if evaluated else 0),'dense active cache extent differs')
    else:
        need(evaluated*layers<=positions<=4177*evaluated*layers,'sparse attention count outside declared bound')
        need(0<=work['max_active_KV_entries']<=4177,'sparse active cache bound exceeded')


def audit_rows(manifest, fixtures, rows, marker=None, tokenizer=None, eos_ids=None, memory_prompts=None):
    schedule=manifest['schedule']
    key=lambda r:(r['example_id'],r['policy']['id'],r['trial'])
    need(len(rows)<=len(schedule) and [key(r) for r in rows]==[key(r) for r in schedule[:len(rows)]],
         'missing, duplicate, reordered or unexpected observations')
    if marker is not None:
        need(marker==dict(rows=220,quality_rows=176,repeated_timing_rows=44,weights_unchanged=True)
             and len(rows)==220,'completion marker or exact matrix differs')
    examples={e['id']:e for e in fixtures}
    model=manifest['model']; layers=model['config']['num_hidden_layers'];heads=model['config']['num_attention_heads']
    records=[]
    for planned,row in zip(schedule,rows):
        ex=examples[row['example_id']];p=row['policy']; stock=p['attention']=='stock'
        need(row['status']=='ok' and p==planned['policy'] and row['purpose']==planned['purpose']
             and row['task']==ex['task'],'failed or mislabeled observation')
        need(row['archive_tokens']==ex['prefix_tokens_count'] and row['gold_answer']==ex['answer'],'fixture identity differs')
        need(len(row['rounds'])==p['rounds'],'round count differs')
        if stock:
            need(row['policy_config'] is None,'stock observation carries sparse configuration')
        else:
            cfg=row['policy_config']
            need(all(cfg[k]==v for k,v in dict(W=2000,C=0,R=128,k=16,sink_tokens=128,retrieval='fixed',
                full=False,buffered_writes=True,fixed_blocks=[],model_revision=model['revision'],id=p['id']).items()),
                'sparse policy differs')
        generated=prompts=evaluated=calls=0; previous='';last_token=None;elapsed=0
        for turn,round in enumerate(row['rounds']):
            need(round['turn']==turn,'round index differs')
            if turn:
                evaluated+=1;calls+=1  # pending final generated token is consumed once
            check_work(round['work_before_prompt'],stock,row['archive_tokens'],evaluated,prompts,calls,layers,heads)
            token_ids=round['generated_token_ids'];prompt_ids=round['prompt_token_ids']
            need(prompt_ids and 1<=len(token_ids)<=p['cap_per_round'],'empty prompt or wrong generation cap')
            need(all(type(t) is int and 0<=t<model['config']['vocab_size'] for t in prompt_ids+token_ids),'invalid token ID')
            need(len(round['decode_ms'])==len(token_ids)-1,'generated-token/decode-step count differs')
            need(all(math.isfinite(t) and t>=0 for t in round['decode_ms']),'invalid decode timer')
            if eos_ids is not None:
                need(round['stopped_by_eos']==(token_ids[-1] in eos_ids)
                     and not any(t in eos_ids for t in token_ids[:-1]),'EOS stopping differs')
                need(round['stopped_by_eos'] or len(token_ids)==p['cap_per_round'],'unexplained early generation stop')
            if tokenizer is not None:
                need(tokenizer.decode(token_ids,skip_special_tokens=True)==round['response'],'response differs from saved generated tokens')
                query,detail=next_search(ex['query'],previous,tokenizer)
                need(round['query']==query and round['query_details']==detail,'query rewrite differs from model output')
                prompt=round_request(ex['query'],turn,p['rounds'],direct=p['format']=='direct')
                if turn==0:
                    extra=memory_prompts[key(row)] if memory_prompts is not None else ''
                    if extra:prompt+='\n'+extra
                    expected=tokenizer.encode(prompt+ex['first_suffix'],add_special_tokens=False)
                else:
                    expected=next_question_tokens(tokenizer,prompt,last_token,
                        {last_token} if row['rounds'][turn-1]['stopped_by_eos'] else set())
                need(prompt_ids==expected,'prompt replay differs from frozen control messages')
            parsed=parse_response(round['response']) if p['format']=='json' else None
            answer=round['response'].strip() if p['format']=='direct' else parsed['answer'] if parsed else None
            need(round['parsed_response']==parsed and round['answer']==answer,'parsed or selected answer differs')
            need(round['exact_correct']==(answer==ex['answer']),'round scoring differs')
            need(round['answer_contains_gold_diagnostic_only']==bool(answer and ex['answer'] in answer),'diagnostic scoring differs')
            selected=round['selected_blocks']
            need(len(selected)==len(set(selected)) and len(selected)<=16,'retrieval budget or distinctness differs')
            need(all(type(b) is int and 0<=b<math.ceil(ex['prefix_tokens_count']/128) for b in selected),'ineligible retrieval address')
            if stock:need(not selected,'stock row has selected blocks')
            prompts+=len(prompt_ids);generated+=len(token_ids)
            evaluated+=len(prompt_ids)+len(token_ids)-1
            calls+=(1 if stock else len(prompt_ids))+len(token_ids)-1
            need(ex['prefix_tokens_count']+evaluated<min(ex['budget'],model['config']['max_position_embeddings']),
                 'native or declared context budget exceeded')
            check_work(round['cumulative_work'],stock,row['archive_tokens'],evaluated,prompts,calls,layers,heads)
            need(round['cumulative_generated_tokens']==generated,'cumulative generated count differs')
            need(elapsed<=round['cumulative_first_token_s']<=round['cumulative_request_s']<=row['request_s'],
                 'nonmonotonic or inconsistent request timing')
            need(0<=round['lookup_and_selection_s']<=round['cumulative_request_s']-elapsed,'invalid lookup duration')
            elapsed=round['cumulative_request_s']; previous=round['response'];last_token=token_ids[-1]
            # Positional coverage is inferred from the selected address set at
            # round start; it is not a new per-layer tensor-payload audit.
            offset=round['work_before_prompt']['model_token_evaluations']+ex['prefix_tokens_count']
            def covered(span):
                return stock or all(i<128 or i>=offset-1999 or i//128 in selected for i in range(*span))
            coverage=(all(covered(ex['record_spans'][i]) for i in ex['required_record_indices'])
                      if 'required_record_indices' in ex else None)
            records.append(dict(example_id=ex['id'],budget=ex['budget'],task=ex['task'],policy=p['id'],trial=row['trial'],
                turn=turn,exact_correct=round['exact_correct'],whole_required_records=coverage,
                selected_blocks=selected,query=round['query'],prediction=answer))
        check_work(row,stock,row['archive_tokens'],evaluated,prompts,calls,layers,heads)
        need(row['generated_tokens']==generated and row['prediction']==row['rounds'][-1]['answer']
             and row['exact_correct']==row['rounds'][-1]['exact_correct'],'final draft/count differs')
        need(math.isclose(row['request_plus_checkpoint_s'],row['request_s']+row['checkpoint_s'],abs_tol=1e-8),
             'checkpoint-inclusive time differs')
        need(all(math.isfinite(row[k]) and row[k]>=0 for k in ('request_s','checkpoint_s','controller_init_s','archive_checksum_mount_s','prefill_s','index_build_or_open_s')),
             'invalid time component')
    return {'status':'complete' if marker is not None else 'incomplete', 'observed_rows':len(rows),
            'expected_rows':len(schedule),'tokenizer_verified':tokenizer is not None,
            'round_record_coverage':records}


def replay_retrieval(fixtures, rows, tokenizer):
    """Rebuild public metadata and replay every recorded query without gold."""
    from research.semantic_index import metadata_index
    from research.budget_probe import expand_blocks
    checked=0
    with tempfile.TemporaryDirectory(prefix='reasoning-index-audit-') as temp:
        for ex in fixtures:
            relevant=[r for r in rows if r['example_id']==ex['id'] and r['policy']['attention']=='sparse']
            if not relevant:continue
            index=metadata_index(tokenizer,ex,Path(temp),32)
            try:
                for row in relevant:
                    for round in row['rounds']:
                        seeds=index.search(round['query'],None,4,'lexical_rare')
                        selected=list(expand_blocks(tuple(seeds),math.ceil(ex['prefix_tokens_count']/128),16)) if seeds else []
                        need(selected==round['selected_blocks'],'query-index selection replay differs')
                        need(index.last_search==round['index_details'],'query-index candidate telemetry differs')
                        checked+=1
            finally:
                index.close()
    return checked


def accounted_service_s(row):
    """Sum recorded serving components; not a fresh deployment wall timer."""
    return sum(row[k] for k in ('archive_checksum_mount_s','controller_init_s',
                                'request_s','checkpoint_s'))


def summarize(manifest, fixtures, rows, audit):
    need(audit['status']=='complete' and audit['tokenizer_verified'],'a completed token-verified audit is required')
    examples={e['id']:e for e in fixtures}
    quality=[];timing=[];paired=[];curves=[]
    bykey={(r['example_id'],r['policy']['id'],r['trial']):r for r in rows}
    for budget in (8192,32768):
        for p in declared_policies():
            group=[r for r in rows if r['trial']==0 and r['policy']['id']==p['id'] and examples[r['example_id']]['budget']==budget]
            refs=[bykey[r['example_id'],'stock-direct-r1-cap128',0] for r in group]
            quality.append(dict(budget=budget,policy=p['id'],cases=len(group),exact_correct=sum(r['exact_correct'] for r in group),
                direct_stock_correct=sum(r['exact_correct'] for r in refs),
                lost_vs_direct_stock=sum(b['exact_correct'] and not r['exact_correct'] for r,b in zip(group,refs)),
                gained_vs_direct_stock=sum(not b['exact_correct'] and r['exact_correct'] for r,b in zip(group,refs)),
                mean_request_s=statistics.mean(r['request_s'] for r in group),
                mean_accounted_service_s=statistics.mean(accounted_service_s(r) for r in group),
                mean_mount_s=statistics.mean(r['archive_checksum_mount_s'] for r in group),
                mean_init_s=statistics.mean(r['controller_init_s'] for r in group),
                mean_checkpoint_s=statistics.mean(r['checkpoint_s'] for r in group),
                mean_shared_prefill_s=statistics.mean(r['prefill_s'] for r in group),
                mean_shared_index_build_or_open_s=statistics.mean(r['index_build_or_open_s'] for r in group),
                mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in group),
                mean_model_token_evaluations=statistics.mean(r['model_token_evaluations'] for r in group),
                max_active_KV_entries=max(r['max_active_KV_entries'] for r in group)))
            repeated=[r for r in rows if r['example_id']==f'reasoning-linked_fact-{budget}-157' and r['policy']['id']==p['id']]
            need(len(repeated)==3,'missing repeated timing trials')
            control=p['id'].replace('sparse-','stock-',1)
            ratios=[bykey[r['example_id'],control,r['trial']]['request_s']/r['request_s'] for r in repeated]
            timing.append(dict(budget=budget,policy=p['id'],trials=3,
                request_mean_s=statistics.mean(r['request_s'] for r in repeated),
                request_range_s=[min(r['request_s'] for r in repeated),max(r['request_s'] for r in repeated)],
                paired_matching_stock_speed_ratios=ratios,paired_ratio_mean=statistics.mean(ratios),
                accounted_service_mean_s=statistics.mean(accounted_service_s(r) for r in repeated),
                paired_accounted_service_ratios=[accounted_service_s(bykey[r['example_id'],control,r['trial']])/accounted_service_s(r) for r in repeated],
                cuda_peak_GiB=max(r['cuda_peak_allocated_bytes'] for r in repeated)/2**30,
                host_peak_GiB=max(r['host_rss_sampled_peak_bytes'] for r in repeated)/2**30))
    for row in (r for r in rows if r['trial']==0 and r['policy']['attention']=='sparse'):
        same=bykey[row['example_id'],row['policy']['id'].replace('sparse-','stock-',1),0]
        direct=bykey[row['example_id'],'stock-direct-r1-cap128',0]
        paired.append(dict(example_id=row['example_id'],policy=row['policy']['id'],budget=examples[row['example_id']]['budget'],
            sparse_correct=row['exact_correct'],matching_stock_correct=same['exact_correct'],direct_stock_correct=direct['exact_correct'],
            matching_stock_speed_ratio=same['request_s']/row['request_s'],direct_stock_speed_ratio=direct['request_s']/row['request_s'],
            matching_stock_accounted_service_ratio=accounted_service_s(same)/accounted_service_s(row),
            direct_stock_accounted_service_ratio=accounted_service_s(direct)/accounted_service_s(row),
            model_evaluation_ratio_vs_matching_stock=row['model_token_evaluations']/same['model_token_evaluations'],
            logical_attention_ratio_vs_matching_stock=row['logical_attention_head_pairs']/same['logical_attention_head_pairs']))
        for control,baseline in [('matching_stock',same),('direct_stock',direct)]:
            for metric,limit in [('elapsed_time',baseline['request_s']),('generated_tokens',baseline['generated_tokens'])]:
                field='cumulative_request_s' if metric=='elapsed_time' else 'cumulative_generated_tokens'
                eligible=[r for r in row['rounds'] if r[field]<=limit]
                last=eligible[-1] if eligible else None
                curves.append(dict(example_id=row['example_id'],policy=row['policy']['id'],control=control,
                    budget_metric=metric,limit=limit,completed_round=last['turn']+1 if last else None,
                    exact_correct=last['exact_correct'] if last else False,
                    baseline_correct=baseline['exact_correct'],scope='retrospective latest completed draft; not interrupted execution or best-answer selection'))
    return dict(quality=quality,repeated_timing=timing,paired_cases=paired,retrospective_budgets=curves)


def write_report(root, manifest, audit):
    need(audit['status']=='complete' and audit['tokenizer_verified'],'cannot publish an incomplete comparison')
    lines=['# Multi-round quality, request and serving-cost comparison','',
        f"Model: `{manifest['model']['model_id']}` at `{manifest['model']['revision']}`.",
        'All 220 declared rows are present: 176 quality observations and 44 separately labeled timing repetitions.',
        'The audit checks frozen source/fixtures, actual-model validation, decoded output tokens, prompt replay,',
        'pending-token accounting, causal attention counts, final-draft scoring and every query/index selection.',
        '', 'There are eight underlying seed/task questions repeated at two archive lengths. This is a screening',
        'cohort, not evidence of near-equivalent quality or a functionally unlimited context.', '',
        '## Quality and cost', '',
        'The 8K/32K labels are nominal budgets; each archive reserves 2048 tokens for reasoning.',
        'Request time starts from initialized KV and includes every prompt, lookup and generated draft.',
        'These task-mix means include failures; faster incorrect answers are not an improvement.', '',
        '| Budget | Policy | Exact | Lost/gained vs direct stock | Mean request s | Mean generated tokens | Mean evaluated tokens | Max active KV |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in audit['quality']:
        lines.append(f"| {r['budget']} | {r['policy']} | {r['exact_correct']}/{r['cases']} | {r['lost_vs_direct_stock']}/{r['gained_vs_direct_stock']} | {r['mean_request_s']:.3f} | {r['mean_generated_tokens']:.1f} | {r['mean_model_token_evaluations']:.1f} | {r['max_active_KV_entries']} |")
    lines+=['','## Archive setup and serving costs','',
        'Accounted service sums checksum mount, controller initialization, initialized-cache request and checkpoint.',
        'It excludes uninstrumented gaps and cleanup, so it is not a directly measured whole-call wall time.',
        'Shared prefill and index build/open are shown separately: the same preparation is reused across policies',
        'and trials, not independently repeated for each row. These means must not be summed as independent work.',
        'Stock also reloads from the shared disk archive here; this is not a resident vanilla deployment comparison.', '',
        '| Budget | Policy | Mount s | Init s | Checkpoint s | Accounted service s | Shared prefill s | Shared index build/open s |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in audit['quality']:
        lines.append(f"| {r['budget']} | {r['policy']} | {r['mean_mount_s']:.3f} | {r['mean_init_s']:.3f} | {r['mean_checkpoint_s']:.3f} | {r['mean_accounted_service_s']:.3f} | {r['mean_shared_prefill_s']:.3f} | {r['mean_shared_index_build_or_open_s']:.3f} |")
    lines+=['','## Repeated linked-fact timing','','Three trials per policy and length; ranges are not confidence intervals.',
        'Speed ratios divide matching stock request time by sparse request time. Ratios above one favor sparse.',
        'Stock uses the same round count and output allowance. Quality must be assessed alongside speed.', '',
        '| Budget | Policy | Mean request s (range) | Mean paired stock/request ratio | Peak CUDA GiB | Peak host RSS GiB |',
        '|---|---|---:|---:|---:|---:|']
    for r in audit['repeated_timing']:
        lo,hi=r['request_range_s']
        lines.append(f"| {r['budget']} | {r['policy']} | {r['request_mean_s']:.3f} ({lo:.3f}–{hi:.3f}) | {r['paired_ratio_mean']:.3f} | {r['cuda_peak_GiB']:.3f} | {r['host_peak_GiB']:.3f} |")
    lines+=['','## Equal-budget interpretation','',
        '`summary.json` retains per-case comparisons with direct stock and matching-round stock, plus',
        'retrospective equal-time and equal-generated-token curves. Each curve chooses the latest completed',
        'draft, never the best answer; a budget that cannot finish the first draft has no completed answer.',
        'These curves are not separately executed interruption experiments. Evaluated model tokens include',
        'prompt replay and consumed generated tokens, and differ from output-token counts.', '',
        'Per-round coverage records report whether every required record lies in the retained address set.',
        'Coverage is inferred from positions; the separate model parity/resume tests check mechanics.', '',
        '## Limits','',*['- '+limit for limit in audit['limits']], '',
        'Raw hashes, source revision, weight identity and the audit script hash are retained in `summary.json`.',
        'Paired accounted-service ratios are also retained there; they include the measured mount/init/checkpoint components.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def plot_comparison(root, audit):
    """Plot the complete paired screen without hiding wrong-answer costs."""
    need(audit['status']=='complete' and audit['tokenizer_verified'],'cannot plot incomplete comparison')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='tiered-kv-reasoning-screen'
    protocols=[(1,128),(2,128),(4,128),(1,512),(2,256)]
    labels=[f'{rounds} × {cap}' for rounds,cap in protocols]
    lookup={(r['budget'],r['policy']):r for r in audit['quality']}
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex='col')
    for col,budget in enumerate((8192,32768)):
        direct=lookup[budget,'stock-direct-r1-cap128']
        for attention,offset,color in [('stock',-.18,'#3366a5'),('sparse',.18,'#cc7722')]:
            rows=[lookup[budget,f'{attention}-json-r{rounds}-cap{cap}'] for rounds,cap in protocols]
            x=[i+offset for i in range(len(rows))]
            axes[0,col].bar(x,[r['exact_correct'] for r in rows],width=.35,color=color,label=attention)
            axes[1,col].bar(x,[r['mean_accounted_service_s'] for r in rows],width=.35,color=color,label=attention)
        axes[0,col].axhline(direct['exact_correct'],color='#333333',linestyle='--',label='direct stock')
        axes[1,col].axhline(direct['mean_accounted_service_s'],color='#333333',linestyle='--',label='direct stock')
        axes[0,col].set(title=f'{budget//1024}K nominal context',ylabel='Correct answers / 8',ylim=(0,8.6),yticks=range(9))
        axes[1,col].set(ylabel='Mean accounted service (s)',xticks=range(5),xticklabels=labels,
                        xlabel='Rounds × output-token cap per round')
        for ax in axes[:,col]:
            ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(loc='upper left',bbox_to_anchor=(0,1.26),ncol=3,frameon=False)
    fig.suptitle('Quality and serving cost of additional reasoning',y=.995)
    fig.text(.5,.02,'Eight correlated questions repeated at two lengths; not a quality-equivalence estimate.\n'
             'Serving cost sums mount + initialization + request + checkpoint; shared prefill/index and cleanup excluded.\n'
             'All answers, including failures, contribute to time. Stock also reloads the shared archive.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.96))
    fig.savefig(root/'quality-cost.png',dpi=160)
    fig.savefig(root/'quality-cost.svg',metadata={'Date':None})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--partial',action='store_true')
    parser.add_argument('--source-revision',default=SOURCE)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1];root=args.run
    manifest=json.loads((root/'manifest.json').read_text());fixtures=json.loads((root/'fixtures.json').read_text())
    rows=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    pinned=verify_source(manifest,fixtures,repo,args.source_revision)
    validation_names={r['validation_file'] for r in rows};weights=set()
    for name in validation_names:
        need(Path(name).name==name and name.startswith('validation-') and name.endswith('.json'),'unsafe validation filename')
        value=json.loads((root/name).read_text());audit_validation(value,manifest['model']);weights.add(value['weights_sha256'])
    need(len(weights)==1,'missing or inconsistent actual weight identity')
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot=snapshot_download(manifest['model']['model_id'],revision=pinned['revision'],local_files_only=True)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    tok.apply_chat_template=partial(tok.apply_chat_template,enable_thinking=False)
    need(digest(tok.chat_template)==pinned['chat_template_sha256'],'tokenizer chat template differs')
    generation_path=Path(snapshot)/'generation_config.json'
    generation=json.loads(generation_path.read_text())
    eos=generation['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
    marker=json.loads((root/'complete.json').read_text()) if (root/'complete.json').exists() else None
    need(args.partial or marker is not None,'matrix incomplete; use --partial for non-conclusive audit')
    audit=audit_rows(manifest,fixtures,rows,marker,tokenizer=tok,eos_ids=eos)
    audit['retrieval_rounds_replayed']=replay_retrieval(fixtures,rows,tok)
    audit.update(source_revision=args.source_revision,weights_sha256=next(iter(weights)),
                 files_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                               ['manifest.json','fixtures.json','samples.jsonl',*sorted(validation_names)]+(['complete.json'] if marker else [])},
                 audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    audit['generation_config_sha256']=hashlib.sha256(generation_path.read_bytes()).hexdigest()
    if marker is not None:
        audit.update(summarize(manifest,fixtures,rows,audit))
        audit['limits']=['Eight underlying seed/task questions repeated at two lengths; not sixteen independent quality examples.',
            'Screening only; no near-equivalence or 1M-context claim.',
            'Total initialized-cache request includes all prompt and generation forwards, query selection and parsing; mount/init/prefill/index/checkpoint are reported separately.',
            'Stock batches prompts; current sparse implementation evaluates prompt tokens serially.',
            'Three repeated trials cover one linked-fact fixture per length; ranges are not confidence intervals.',
            'Coverage is inferred from address selections, not a new per-layer payload audit.',
            'OS page-cache conditions uncontrolled; periodic result backup may introduce host-I/O noise.',
            'Logical attention pairs are not measured GPU FLOPs; generated-token savings are distinct from model-token evaluations.']
    name='summary.json' if marker else 'audit-partial.json'
    (root/name).write_text(json.dumps(audit,indent=2)+'\n')
    if marker is not None:
        write_report(root,manifest,audit)
        plot_comparison(root,audit)
        with (root/'report.md').open('a') as f:
            f.write('\n![Quality and accounted serving cost](quality-cost.png)\n')
    print(json.dumps({k:audit[k] for k in ('status','observed_rows','expected_rows','tokenizer_verified')}))


if __name__=='__main__':main()

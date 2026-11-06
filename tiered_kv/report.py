import csv
import json
from pathlib import Path
import numpy as np
from .config import Config
from .store import atomic_json

QUALITY=('niah_em_8k','niah_em_32k','rougeL_low','rougeL_high')


def rank_rows(rows):
    full=next((r for r in rows if r['config_id']=='full' and r['status']=='complete'),None)
    for row in rows:
        complete=row['status']=='complete' and all(row.get(k) is not None for k in QUALITY)
        row['quality_score']=float(np.mean([row[k] for k in QUALITY])) if complete else None
        row['rank']=''
        row['practical_candidate']=bool(complete and full and all(row[k]>=full[k]-(5 if k.startswith('niah') else 2) for k in QUALITY))
        row['pareto']=False
    ordered=sorted(rows,key=lambda r:(r['quality_score'] is None,-(r['quality_score'] or 0),r.get('median_ms_32k') or float('inf'),r.get('active_KV_bytes') or float('inf')))
    valid=[r for r in ordered if r['quality_score'] is not None]
    for i,row in enumerate(valid,1):
        row['rank']=i
        if row.get('median_ms_32k') is not None and row.get('active_KV_bytes') is not None:
            point=(row['quality_score'],-row['median_ms_32k'],-row['active_KV_bytes'])
            row['pareto']=not any(all(a>=b for a,b in zip((r['quality_score'],-r['median_ms_32k'],-r['active_KV_bytes']),point)) and any(a>b for a,b in zip((r['quality_score'],-r['median_ms_32k'],-r['active_KV_bytes']),point)) for r in valid if r is not row and r.get('median_ms_32k') is not None and r.get('active_KV_bytes') is not None)
    return ordered


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows=[]
    with Path(path).open() as src:
        for n,line in enumerate(src,1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{n}: incomplete record; preserve file and recover explicitly') from exc
    return rows


def csv_write(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def bootstrap_delta(samples,baseline,metric,stratum):
    pairs=[(s[metric]-baseline[s['example_id']][metric],s[stratum]) for s in samples if s['example_id'] in baseline]
    if not pairs:
        return None
    rng=np.random.default_rng(41)
    groups=[[v for v,s in pairs if s==key] for key in sorted(set(s for _,s in pairs))]
    values=[np.mean(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])) for _ in range(2000)]
    return np.percentile(values,[2.5,97.5]).tolist()


def build_report(run):
    root=Path(run)
    if not root.exists():
        root=Path('results')/run
    manifest=json.loads((root/'manifest.json').read_text())
    samples=read_jsonl(root/'samples.jsonl')
    # Only terminal records count; one key may have explicit superseding retries.
    latest={(s['config_hash'],s['example_id']):s for s in samples}
    samples=list(latest.values())
    expected=manifest.get('examples',[])
    full={s['example_id']:s for s in samples if s['config_id']=='full' and s['status']=='ok'}
    rows=[]
    avg=lambda vals:float(np.mean(vals)) if vals else None
    for expanded in manifest['configs']:
        cfg=Config(**expanded)
        mine=[s for s in samples if s['config_hash']==cfg.digest]
        good=[s for s in mine if s['status']=='ok']
        row={'run_id':manifest['run_id'],'config_id':cfg.id,'config_hash':cfg.digest,
             'protocol':cfg.protocol,'profile':manifest['profile'],'model_revision':cfg.model_revision,
             'rope_variant':cfg.rope_variant,**{k:getattr(cfg,k) for k in ('W','C','B','R','k','merge')},
             'status':'complete' if len(good)==len(expected) and expected else 'incomplete',
             'completed':len(good),'expected':len(expected),'failure_reason':'; '.join(sorted(set(s.get('failure_reason','') for s in mine if s['status']!='ok'))),
             'quality_ci':'N/A: pilot sample size' if manifest['profile']=='pilot' else 'paired stratified bootstrap 2000 seed 41'}
        if manifest.get('blocked_reason') and row['status']!='complete':
            row.update(status='blocked',failure_reason=manifest['blocked_reason'])
        lengths=[s['prompt_length'] for s in good]
        row.update(prompt_min=min(lengths) if lengths else None,prompt_median=avg(lengths) if not lengths else float(np.median(lengths)),prompt_max=max(lengths) if lengths else None)
        coverage={}
        for budget,label in [(8192,'8k'),(32768,'32k')]:
            selected=[s for s in good if s['task']=='niah' and s['budget']==budget]
            exp=[e for e in expected if e['task']=='niah' and e['budget']==budget]
            row[f'niah_em_{label}']=avg([s['em'] for s in selected])
            row[f'niah_correct_{label}']=sum(s['em']==100 for s in selected)
            row[f'niah_total_{label}']=len(selected)
            row[f'niah_expected_{label}']=len(exp)
            row[f'code_found_{label}']=avg([s['code_found'] for s in selected])
            pairs=[s['em']-full[s['example_id']]['em'] for s in selected if s['example_id'] in full]
            row[f'niah_delta_{label}']=avg(pairs)
            hits=[s['retrieval'] for s in selected if s.get('retrieval')]
            for name,num,den in [('query_hit','common_hits','common_opportunities'),('query_hit_all','query_hits','query_opportunities'),('step_hit','step_hits','step_opportunities')]:
                total=sum(h[den] for h in hits)
                row[f'{name}_{label}']=sum(h[num] for h in hits)/total if total else None
                row[f'{name}_opportunities_{label}']=total
            for name in ('full_span_coverage','exact_availability','recent_excluded','candidate_mean','chance_mean','gold_fraction'):
                row[f'{name}_{label}']=avg([h[name] for h in hits if h.get(name) is not None])
            coverage[label]={str(d):{'correct':sum(s['em']==100 for s in selected if s['depth']==d),'total':sum(s['depth']==d for s in selected)} for d in sorted(set(s['depth'] for s in selected))}
            if manifest['profile']=='confirm':
                row[f'niah_delta_ci_{label}']=bootstrap_delta(selected,full,'em','depth')
        for band in ('low','high'):
            selected=[s for s in good if s['task']=='gov_report' and s['band']==band]
            row[f'rougeL_{band}']=avg([s['rougeL'] for s in selected])
            row[f'rougeL_total_{band}']=len(selected)
            row[f'rougeL_expected_{band}']=sum(e['task']=='gov_report' and e['band']==band for e in expected)
            row[f'rougeL_delta_{band}']=avg([s['rougeL']-full[s['example_id']]['rougeL'] for s in selected if s['example_id'] in full])
            if manifest['profile']=='confirm':
                row[f'rougeL_delta_ci_{band}']=bootstrap_delta(selected,full,'rougeL','band')
        latency=[s for s in good if s['task']=='latency']
        for budget in sorted(set(e['budget'] for e in expected if e['task']=='latency')):
            group=[s for s in latency if s['budget']==budget]
            times=[x for s in group for x in s['latency_ms']]
            label=f'{budget//1024}k'
            row[f'median_ms_{label}']=float(np.median(times)) if times else None
            row[f'p95_ms_{label}']=float(np.percentile(times,95)) if times else None
            row[f'tokens_s_{label}']=1000/avg(times) if times else None
            row[f'trial_means_{label}']=[avg(s['latency_ms']) for s in group]
            row[f'latency_completed_{label}']=len(group)
            peaks=[s['cuda_decode_peak_allocated_bytes'] for s in group if 'cuda_decode_peak_allocated_bytes' in s]
            row[f'cuda_decode_peak_allocated_bytes_{label}']=max(peaks) if peaks else None
        ratios=[]
        for trial in range(3):
            low=next((s for s in latency if s['budget']==8192 and s['trial']==trial),None)
            high=next((s for s in latency if s['budget']==32768 and s['trial']==trial),None)
            if low and high:
                ratios.append(float(np.median(high['latency_ms'])/np.median(low['latency_ms'])))
        row['paired_trial_ratios']=ratios
        row['trial_ratio_range']=[min(ratios),max(ratios)] if ratios else None
        lo,hi=row.get('median_ms_8k'),row.get('median_ms_32k')
        row['ratio_32k_8k']=hi/lo if lo and hi else None
        xs=[s['prompt_length']+17+i for s in latency for i in range(len(s['latency_ms']))]
        ys=[ms for s in latency for ms in s['latency_ms']]
        row['log_slope']=float(np.polyfit(np.log(xs),np.log(ys),1)[0]) if len(set(xs))>=2 else None
        for field in ('prefill_s','init_s','ttft_s'):
            row[field]=avg([s[field] for s in good if field in s])
        for field in ('peak_rss','peak_mps_allocated','peak_mps_driver','swap_delta','pressure_max','active_KV_entries','active_KV_bytes','index_bytes','cold_allocated_bytes','cuda_decode_peak_allocated_bytes','cuda_decode_peak_reserved_bytes'):
            values=[s[field] for s in good if field in s]
            row[field]=max(values) if values else None
        row['active_KV_bytes']=row.get('active_KV_bytes')
        row['approximately_flat']=bool(row['ratio_32k_8k'] is not None and row['ratio_32k_8k']<=1.25 and len(ratios)==3 and row.get('swap_delta',0)<=512*1024**2 and row.get('pressure_max',1)<2)
        for field in ('logical_read_bytes','logical_write_bytes'):
            row[field]=sum(s.get(field,0) for s in good)
        row['depth_breakdown']='depth_breakdown.json'
        row['diagnostics']='diagnostics.jsonl' if (root/'diagnostics.jsonl').exists() and (root/'diagnostics.jsonl').stat().st_size else 'pending'
        rows.append(row)
        atomic_json(root/f'depth-{cfg.id}.json',coverage)
    ranked=rank_rows(rows)
    csv_write(root/'summary.csv',rows)
    csv_write(root/'ranking.csv',ranked)
    atomic_json(root/'depth_breakdown.json',{r['config_id']:json.loads((root/f"depth-{r['config_id']}.json").read_text()) for r in rows})
    complete=[r for r in ranked if r['rank']!='']
    interaction={}
    by_id={r['config_id']:r for r in rows}
    if all(k in by_id and by_id[k]['status']=='complete' for k in ('default','retrieval_only','compression_only','recent_only')):
        interaction={q:by_id['default'][q]-by_id['retrieval_only'][q]-by_id['compression_only'][q]+by_id['recent_only'][q] for q in QUALITY}
    count_task=lambda task:sum(s['status']=='ok' and s['task']==task for s in samples)
    text=[f"# Tiered KV {manifest['profile']} — {manifest['run_id']}",'',
          f"Measured runs: NIAH {count_task('niah')}; gov_report {count_task('gov_report')}; latency {count_task('latency')}. "
          f"Complete configurations: {len(complete)}/{len(rows)}.",'',
          ('Best measured config: '+complete[0]['config_id']+f" (four-metric mean {complete[0]['quality_score']:.3f}).") if complete else 'No measured winner. Incomplete and blocked configurations are unranked.',
          '',manifest.get('blocked_reason',''),'',
          'Pilot quality intervals are N/A; confirmation uses paired stratified bootstrap. Latency is teacher-forced with OS-cache-uncontrolled filesystem reads. Legacy peak_mps fields represent the selected accelerator and are cumulative run peaks, not per-policy memory. RSS and MPS memory overlap and are never added.',
          'CUDA decode allocation peaks are measured separately for each policy after initialization, include weights/active KV/workspace, and exclude shared prefill and checkpoint. Reserved bytes include allocator caching. Missing CUDA fields mean unmeasured; logical KV bytes alone do not establish a physical VRAM bound.',
          'Flatness is only the predeclared 32K/8K ratio ≤1.25 over the measured range, without a resource abort. It is not an unlimited-history claim.',
          '',f'Component interaction (EM / ROUGE-L points): {interaction or "unavailable"}.',
          '', 'Practical shortlist, sorted by 32K latency: '+(', '.join(r['config_id'] for r in sorted((r for r in complete if r['practical_candidate']),key=lambda r:r.get('median_ms_32k') or float('inf'))) or 'none'),
          '', 'Pareto frontier: '+(', '.join(r['config_id'] for r in complete if r['pareto']) or 'unavailable'),
          '', 'Online ingestion/resume, attention/oracle/gap diagnostics, and qualitative summary inspection require their own evidence files. See validation.json and diagnostics.jsonl; missing files mean pending.',
          'Confirmation and 128K stretch are pending unless separately executed. This is a LongBench v1 gov_report subset plus custom NIAH, not a full LongBench or RULER score.']
    (root/'report.md').write_text('\n'.join(text)+'\n')
    _plots(root,rows)
    return ranked


def _plots(root,rows):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(Path('artifacts/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for row in rows:
        pairs=[(n,row.get(f'median_ms_{n}k')) for n in (1,4,8,16,32) if row.get(f'median_ms_{n}k') is not None]
        if pairs:
            ax.plot(*zip(*pairs),marker='o',label=row['config_id'])
    if ax.lines:
        ax.legend(fontsize=6,ncol=3)
    else:
        ax.text(.5,.5,'No measured latency — validation blocked',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Context budget (Ki tokens)',ylabel='Median ms / forced token')
    fig.tight_layout()
    fig.savefig(root/'latency.png')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    measured=[r for r in rows if r.get('active_KV_bytes') is not None]
    if measured:
        ax.bar([r['config_id'] for r in measured],[r['active_KV_bytes']/1024**2 for r in measured])
        ax.tick_params(axis='x',rotation=60)
    else:
        ax.text(.5,.5,'No measured active KV',ha='center',transform=ax.transAxes)
    ax.set(ylabel='Logical active KV (MiB; sum of layer peaks)')
    fig.tight_layout()
    fig.savefig(root/'memory.png')
    plt.close(fig)

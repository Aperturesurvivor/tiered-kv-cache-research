"""Independent version, memory, retrieval and work audit of archive updates."""
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

from research.report_memory import MemoryReplay
from research.report_reasoning import audit_rows, audit_validation, digest, need
from research.budget_probe import expand_blocks
from research.semantic_index import metadata_index


def policies():
    return [dict(id=f'{a}-update-{m}-r2-cap128',attention=a,memory=m,rounds=2,
        cap_per_round=128,format='json',memory_seed='empty')
        for a,m in [('stock','none'),('stock','notes'),('sparse','none'),('sparse','both')]]


def verify_source(manifest, episodes, repo, revision):
    need(bool(re.fullmatch('[0-9a-f]{40}',revision)),'full immutable source revision required')
    source=lambda name:subprocess.check_output(['git','show',revision+':'+name],cwd=repo)
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',revision],cwd=repo,text=True).splitlines()
    core=sorted(f for f in files if len(Path(f).parts)==2 and Path(f).parts[0] in ('tiered_kv','tests') and f.endswith('.py'))
    core+=['configs/matrix.yaml','configs/eval.yaml','requirements.lock']
    need(digest({f:hashlib.sha256(source(f)).hexdigest() for f in core})==manifest['implementation_hash'],'implementation identity differs')
    names=('memory_update_run','memory_update_archive','memory_update_protocol','memory_run',
           'memory_harness','memory_protocol','reasoning_run','reasoning_sessions',
           'reasoning_protocol','persistent_kv','budget_probe','semantic_index','resources')
    need(set(manifest['sources_sha256'])=={f'research/{n}.py' for n in names},'incomplete source identity')
    for filename, sha in manifest['sources_sha256'].items():
        need(hashlib.sha256(source(filename)).hexdigest()==sha,'frozen source differs: '+filename)
    for name in ('reasoning_protocol','persistent_kv','budget_probe','semantic_index'):
        filename=f'research/{name}.py'
        need((repo/filename).read_bytes()==source(filename),'audit helper differs from frozen source')
    declared=json.loads(source('research/memory-update-fixture-expectations.json'))
    need(hashlib.sha256(source('research/memory_update_protocol.py')).hexdigest()==declared['protocol_sha256'],'fixture protocol differs')
    model=manifest['model']; pin=next(m for m in declared['models'] if m['model_id']==model['model_id'])
    specs=json.loads(source('research/model-manifest.json'))+json.loads(source('research/long-context-candidates.json'))['models']
    need(model==next(m for m in specs if m['model_id']==model['model_id']) and model['revision']==pin['revision'],'model differs')
    need(digest(episodes)==manifest['fixture_sha256']==pin['fixture_sha256'],'fixture payload differs')
    need(len(episodes)==4 and all(len(e['queries'])==5 for e in episodes),'episode extent differs')
    rng=random.Random(239);schedule=[]
    for ep in episodes:
        ordered=policies();rng.shuffle(ordered)
        schedule.extend(dict(episode_id=ep['id'],policy=p,queries=5) for p in ordered)
    need(manifest['schedule']==schedule and manifest['runtime']['device']=='cuda','schedule or runtime differs')
    return pin


def audit_append(receipt, model, old_n, new_n):
    c=model['config'];layers=c['num_hidden_layers'];n=new_n-old_n
    identity=receipt['identity']
    need(identity['old_positions']==old_n and identity['new_positions']==new_n
         and identity['append_tokens']==n and identity['chunk']==128
         and identity['model_revision']==model['revision'],'append identity differs')
    for key in ('parent_checkpoint_sha256','parent_tokens_sha256','tokens_sha256','token_hash'):
        need(bool(re.fullmatch('[0-9a-f]{64}',identity[key])),'invalid append identity digest')
    need(receipt['prefix_audit']==[dict(layer=i,prefix_positions=old_n,kv_bit_equal=True) for i in range(layers)],'historical KV audit incomplete')
    need(receipt['append_model_tokens']==n and receipt['append_forward_calls']==math.ceil(n/128)
         and receipt['append_layer_kv_positions']==layers*(n*old_n+n*(n+1)//2),'append work differs')
    d=c.get('head_dim',c['hidden_size']//c['num_attention_heads'])
    need(receipt['logical_append_write_bytes']==n*layers*4*c['num_key_value_heads']*d,'append write extent differs')
    # One dense initialization plus two sides of the historical-prefix audit.
    need(receipt['logical_tensor_read_bytes_including_audit']==3*old_n*layers*4*c['num_key_value_heads']*d,'ancestor tensor reads omitted')
    need(type(receipt['cuda_peak_allocated_bytes']) is int and receipt['cuda_peak_allocated_bytes']>0,'missing CUDA append allocation')
    for key in ('checksum_mount_s','dense_init_s','append_s','checkpoint_s','prefix_audit_s'):
        need(math.isfinite(receipt[key]) and receipt[key]>=0,'invalid append timer')


def audit_control(value, model):
    audit_validation(value,model)
    a=value['append_control'];layers=model['config']['num_hidden_layers']
    need(a['status']=='ok' and a['prefixes']==[129,148,277],'actual-model append control missing')
    need(a['kv_checks']==[dict(layer=i,positions=277,kv_bit_equal=True) for i in range(layers)],'actual-model appended KV mismatch')
    need(a['next_token_logit_max_abs_error']==0 and a['next_token_argmax_equal'],'actual-model append reload logits differ')
    need(len(a['appends'])==2,'append control chain incomplete')
    for receipt,lo,hi in zip(a['appends'],(129,148),(148,277)):
        audit_append(receipt,model,lo,hi)


def audit_archives(prepared, ep, model):
    need(len(prepared)==3 and [r['version'] for r in prepared]==[0,1,2],'archive versions missing')
    for vi, (record,version) in enumerate(zip(prepared,ep['versions'])):
        n=version['prefix_tokens_count']
        need(record['index_audit']==dict(blocks=(n+127)//128,visible_positions=n,exact_version_metadata=True),'version index audit differs')
        for key in ('archive_prepare_or_open_wall_s','index_build_or_open_s','version_setup_wall_s'):
            need(math.isfinite(record[key]) and record[key]>=0,'invalid version preparation timer')
        need(record['version_setup_wall_s']>=record['archive_prepare_or_open_wall_s']+record['index_build_or_open_s'],'version setup omits measured work')
        if vi:
            earlier=ep['versions'][vi-1];old_n=earlier['prefix_tokens_count']
            audit_append(record['receipt'],model,old_n,n)
            identity=record['receipt']['identity']
            need(identity['parent_tokens_sha256']==digest(earlier['tokens'][:old_n])
                 and identity['tokens_sha256']==digest(version['tokens'][:n])
                 and identity['token_hash']==version['token_hash'],'append uses wrong version tokens')
        else:
            need(math.isfinite(record['receipt']['prefill_s']) and record['receipt']['prefill_s']>=0,'invalid initial prefill time')


def replay_versions(manifest, episodes, rows, tok, eos_ids, index_factory=None):
    if index_factory is None:index_factory=lambda version,root:metadata_index(tok,version,root,32)
    byid={e['id']:e for e in episodes};fixtures=[];plan=[]
    for ep in episodes:
        for qi,q in enumerate(ep['queries']):
            v=ep['versions'][q['version']]
            fixtures.append({**v,'id':ep['id']+f'-q{qi}','task':q['kind'],
                'query':q['query'],'answer':q['answer'],'required_record_indices':[q['record']]})
    for item in manifest['schedule']:
        plan.extend(dict(example_id=item['episode_id']+f'-q{qi}',trial=0,policy=item['policy'],
                         purpose='natural_model_memory_archive_update') for qi in range(5))
    key=lambda r:(r['example_id'],r['policy']['id'],r['trial'])
    need(len(rows)<=len(plan) and [key(r) for r in rows]==[key(r) for r in plan[:len(rows)]],'update row order differs')
    memories={};prompts={};trace=[];retrieval_rounds=0
    with tempfile.TemporaryDirectory(prefix='update-index-audit-') as temp:
        indices={}
        try:
            for row in rows:
                ep=byid[row['episode_id']];p=row['policy'];qi=row['query_index']
                need(type(qi) is int and 0<=qi<5 and row['example_id']==ep['id']+f'-q{qi}','query identity differs')
                q=ep['queries'][qi];vi=q['version'];v=ep['versions'][vi]
                need(row['archive_version']==vi and row['archive_tokens']==v['prefix_tokens_count'],'wrong archive version served')
                mk=(ep['id'],p['id'])
                if mk not in memories:memories[mk]=(MemoryReplay(tok,p['memory']),0)
                memory,next_q=memories[mk]
                need(qi==next_q and row['memory_before']==memory.snapshot(),'broken versioned memory chain')
                prompt=memory.prompt();prompts[key(row)]=prompt
                if p['attention']=='sparse':
                    ik=(ep['id'],vi)
                    if ik not in indices:indices[ik]=index_factory(v,Path(temp))
                    index=indices[ik];eligible=math.ceil(v['prefix_tokens_count']/128)
                    for turn,rr in enumerate(row['rounds']):
                        seeds=index.search(rr['query'],None,4,'lexical_rare');details=dict(index.last_search)
                        if turn==0:
                            remembered=memory.addresses(q['query'],eligible)
                            details['remembered_seed_blocks']=remembered
                            seeds=list(dict.fromkeys([*seeds,*remembered]))[:16]
                        selected=list(expand_blocks(tuple(seeds),eligible,16)) if seeds else []
                        need(selected==rr['selected_blocks'] and details==rr['index_details'],'version-specific retrieval differs')
                        retrieval_rounds+=1
                rr=row['rounds'][-1];update=memory.update(q['query'],rr['response'],rr['selected_blocks'])
                need(row['memory_update']==update and row['memory_after']==memory.snapshot(),'incorrect update or repaired model memory')
                memories[mk]=(memory,qi+1)
                need(math.isfinite(row['policy_call_wall_s']) and row['policy_call_wall_s']>=row['request_plus_checkpoint_s'],'whole-call timer omits request work')
                trace.append(dict(episode_id=ep['id'],policy=p['id'],query_index=qi,version=vi,
                    prompt_tokens=len(tok.encode(prompt,add_special_tokens=False)),
                    stored_text_tokens=memory.size(),stored_addresses=memory.snapshot()['stored_addresses'],
                    saved=update['saved'],saved_incorrect=update['saved'] and not row['exact_correct']))
        finally:
            for index in indices.values():index.close()
    audit=audit_rows({**manifest,'schedule':plan},fixtures,rows,tokenizer=tok,eos_ids=eos_ids,memory_prompts=prompts)
    return {**audit,'memory_trace':trace,'retrieval_rounds_replayed':retrieval_rounds}


def summarize(episodes, rows, archive_files):
    need(len(rows)==80,'complete update matrix required')
    byid={e['id']:e for e in episodes};lookup={(r['episode_id'],r['policy']['id'],r['query_index']):r for r in rows}
    quality=[];totals=[]
    for p in policies():
        control='stock-update-'+('notes' if p['memory'] in ('notes','both') else 'none')+'-r2-cap128'
        for budget in (8192,32768):
            for qi in range(5):
                group=[r for r in rows if r['policy']['id']==p['id'] and r['query_index']==qi and byid[r['episode_id']]['budget']==budget]
                need(len(group)==2,'missing update pair')
                baseline=[lookup[r['episode_id'],control,qi] for r in group]
                quality.append(dict(budget=budget,policy=p['id'],kind=group[0]['task'],cases=2,
                    correct=sum(r['exact_correct'] for r in group),matching_stock=control,
                    stock_correct=sum(r['exact_correct'] for r in baseline),
                    mean_request_s=statistics.mean(r['request_s'] for r in group),
                    mean_whole_call_s=statistics.mean(r['policy_call_wall_s'] for r in group),
                    mean_stock_whole_call_s=statistics.mean(r['policy_call_wall_s'] for r in baseline),
                    mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in group),
                    mean_evaluated_tokens=statistics.mean(r['model_token_evaluations'] for r in group),
                    cuda_peak_bytes=max(r['cuda_peak_allocated_bytes'] for r in group)))
        for ep in episodes:
            group=[lookup[ep['id'],p['id'],qi] for qi in range(5)]
            names=sorted({r['archive_evidence_file'] for r in group})
            setup=sum(v['version_setup_wall_s'] for name in names for v in archive_files[name])
            calls=sum(r['policy_call_wall_s'] for r in group)
            totals.append(dict(episode_id=ep['id'],policy=p['id'],correct=sum(r['exact_correct'] for r in group),
                whole_serving_calls_s=calls,shared_preparation_observed_s=setup,
                charged_episode_s=calls+setup,archive_evidence_files=names,
                cost_note='Shared preparation charged once per referenced preparation attempt to each policy; not an independently timed cold deployment.'))
    return dict(quality=quality,episode_totals=totals)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--source-revision',required=True)
    parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    root=args.run;repo=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'manifest.json').read_text());episodes=json.loads((root/'fixtures.json').read_text())
    rows=[json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    pin=verify_source(manifest,episodes,repo,args.source_revision)
    weights=set();files=['manifest.json','fixtures.json','samples.jsonl'];archives={}
    def safe(name,prefix):
        need(Path(name).name==name and name.startswith(prefix) and name.endswith('.json'),'unsafe evidence filename')
        if name not in files:files.append(name)
        return json.loads((root/name).read_text())
    for name in {r['validation_file'] for r in rows}:
        value=safe(name,'validation-');audit_control(value,manifest['model']);weights.add(value['weights_sha256'])
    need(len(weights)==1,'missing or inconsistent actual weight identity')
    byid={e['id']:e for e in episodes}
    for row in rows:
        name=row['archive_evidence_file']
        if name not in archives:archives[name]=safe(name,'archives-')
        prepared=archives[name];ep=byid[row['episode_id']]
        audit_archives(prepared,ep,manifest['model'])
        need(row['prefill_s']==prepared[0]['receipt']['prefill_s'] and
             row['index_build_or_open_s']==prepared[row['archive_version']]['index_build_or_open_s'],'shared setup attribution differs')
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot=Path(snapshot_download(manifest['model']['model_id'],revision=pin['revision'],local_files_only=True))
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    tok.apply_chat_template=partial(tok.apply_chat_template,enable_thinking=False)
    need(digest(tok.chat_template)==pin['chat_template_sha256'],'tokenizer template differs')
    generation=json.loads((snapshot/'generation_config.json').read_text());eos=generation['eos_token_id']
    marker=safe('complete.json','complete') if (root/'complete.json').exists() else None
    need(args.partial or marker is not None,'incomplete update matrix; use --partial')
    if marker is not None:need(marker==dict(rows=80,weights_unchanged=True) and len(rows)==80,'completion marker differs')
    audit=replay_versions(manifest,episodes,rows,tok,set(eos if isinstance(eos,list) else [eos]))
    audit.update(status='complete' if marker else 'incomplete',source_revision=args.source_revision,
        weights_sha256=next(iter(weights)),files_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files},
        audit_sources_sha256={n:hashlib.sha256((repo/'research'/n).read_bytes()).hexdigest()
                             for n in ('report_memory_update.py','report_memory.py','report_reasoning.py')})
    audit['limits']=['Four correlated episodes, with two underlying seeds; no near-equivalence or 1M claim.',
        'Hot updates, aged updates, historical facts and new entities are reported separately.',
        'Shared dense ingestion and full version-index rebuilds; bounded serving does not imply bounded ingestion.',
        'Whole-call time includes mount/init/checkpoint; shared preparation is separately charged, not a cold-deployment benchmark.',
        'No repeated timing trials; OS page-cache condition uncontrolled and backups may add I/O noise.',
        'Logical tensor reads exclude checksum scans; physical storage traffic is not established.',
        'Append equality is checked against recorded per-layer controls from the pinned generator; this report does not rerun CUDA or reload full KV payloads.',
        'Record coverage is positional inference, not a new per-layer serving-payload audit.']
    if marker:audit.update(summarize(episodes,rows,archives))
    (root/('summary.json' if marker else 'audit-partial.json')).write_text(json.dumps(audit,indent=2)+'\n')
    if marker:
        lines=['# Archive-update memory comparison','',f"Model: {manifest['model']['model_id']}. All 80 rows audited.",
            '', '| Budget | Policy | Question | Exact | Matching stock | Whole call / stock s |',
            '|---|---|---|---:|---:|---:|']
        for r in audit['quality']:
            lines.append(f"| {r['budget']} | {r['policy']} | {r['kind']} | {r['correct']}/2 | {r['stock_correct']}/2 | {r['mean_whole_call_s']:.3f} / {r['mean_stock_whole_call_s']:.3f} |")
        lines+=['','Episode totals and separately charged preparation are in `summary.json`.','',*['- '+s for s in audit['limits']]]
        (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:audit[k] for k in ('status','observed_rows','expected_rows','tokenizer_verified','retrieval_rounds_replayed')}))


if __name__=='__main__':main()

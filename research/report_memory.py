"""Audit bounded cross-question memory against the frozen, fallible protocol.

State, prompt and saved-address replay below do not call BoundedMemory. Gold
answers enter only scoring after the decisions have been independently replayed.
Partial runs never produce a completed quality comparison.
"""
import argparse
from copy import deepcopy
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

from research.report_reasoning import audit_rows, audit_validation, digest, need
from research.reasoning_protocol import parse_response


SOURCE='e68b6debc341c95a370e8db65c4bbe4182f0bb34'
NOTICE=('Unverified memory from earlier requests follows. It may be wrong or '
        'outdated. Check it against the archive and the current question.\n')
FIELDS={'question':32,'answer':16,'search':32,'notes':48}
STOP={'a','an','the','what','which','is','was','are','for','of','in','at',
      'to','and','or','please','answer','code','access','current','now'}


def declared_policies():
    result=[]
    for attention,modes in [('stock',['none','notes']),('sparse',['none','notes','addresses','both'])]:
        for mode in modes:
            result.append(dict(id=f'{attention}-memory-{mode}-r2-cap128',attention=attention,
                memory=mode,rounds=2,cap_per_round=128,format='json',memory_seed='empty'))
    for attention,mode in [('stock','notes'),('sparse','both')]:
        result.append(dict(id=f'{attention}-memory-{mode}-stale-r2-cap128',attention=attention,
            memory=mode,rounds=2,cap_per_round=128,format='json',memory_seed='synthetic_stale'))
    return result


def expected_schedule(episodes):
    rng=random.Random(197);result=[]
    for ep in episodes:
        ordered=declared_policies();rng.shuffle(ordered)
        result.extend(dict(episode_id=ep['id'],policy=p,queries=5) for p in ordered)
    return result


def verify_source(manifest, episodes, repo, revision):
    need(bool(re.fullmatch('[0-9a-f]{40}',revision)),'full immutable source revision required')
    def source(name):return subprocess.check_output(['git','show',revision+':'+name],cwd=repo)
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',revision],cwd=repo,text=True).splitlines()
    core=sorted(f for f in files if len(Path(f).parts)==2 and Path(f).parts[0] in ('tiered_kv','tests') and f.endswith('.py'))
    core+=['configs/matrix.yaml','configs/eval.yaml','requirements.lock']
    need(digest({f:hashlib.sha256(source(f)).hexdigest() for f in core})==manifest['implementation_hash'],
         'frozen implementation/test/dependency identity differs')
    paths={'research/'+name+'.py' for name in ('memory_run','memory_protocol','memory_harness','reasoning_run',
        'reasoning_sessions','reasoning_protocol','budget_probe','persistent_kv','semantic_index','resources')}
    need(set(manifest['sources_sha256'])==paths,'incomplete source identity')
    for name,sha in manifest['sources_sha256'].items():
        need(hashlib.sha256(source(name)).hexdigest()==sha,'frozen source differs: '+name)
    for name in ('reasoning_protocol','persistent_kv','semantic_index','budget_probe'):
        filename='research/'+name+'.py'
        need((repo/filename).read_bytes()==source(filename),'audit helper differs from frozen source')
    expected=json.loads(source('research/memory-fixture-expectations.json'))
    model=manifest['model'];pinned=next(m for m in expected['models'] if m['model_id']==model['model_id'])
    need(hashlib.sha256(source('research/memory_protocol.py')).hexdigest()==expected['protocol_sha256'],
         'predeclared memory protocol differs')
    specs=json.loads(source('research/model-manifest.json'))+json.loads(source('research/long-context-candidates.json'))['models']
    need(model==next(m for m in specs if m['model_id']==model['model_id']),'model specification differs')
    need(model['revision']==pinned['revision'],'model revision differs')
    need(digest(episodes)==manifest['fixture_sha256']==pinned['fixture_sha256'],'fixture payload differs')
    need(len(episodes)==4 and all(len(e['queries'])==5 for e in episodes),'episode extent differs')
    need(manifest['schedule']==expected_schedule(episodes),'declared randomized schedule differs')
    need(manifest['memory_state_token_cap']==manifest['memory_prompt_token_cap']==256,'declared memory cap differs')
    need(manifest['runtime']['device']=='cuda','not CUDA generator evidence')
    # The chat template declaration is shared with the original reasoning plan.
    templates=json.loads(source('research/reasoning-fixture-expectations.json'))
    return next(m for m in templates['models'] if m['model_id']==model['model_id'])


class MemoryReplay:
    """An independent interpreter of the fixed memory contract, not a model."""
    def __init__(self, tokenizer, mode):
        need(mode in ('none','notes','addresses','both'),'unexpected memory mode')
        self.tok=tokenizer;self.mode=mode;self.entries=[]

    def size(self):
        return sum(len(self.tok.encode(e[k],add_special_tokens=False)) for e in self.entries for k in FIELDS)

    def clip(self, text, cap):
        tokens=self.tok.encode(text,add_special_tokens=False)[:cap]
        while True:
            text=self.tok.decode(tokens,skip_special_tokens=True)
            if len(self.tok.encode(text,add_special_tokens=False))<=cap:return text
            tokens=tokens[:-1]

    def snapshot(self):
        return dict(mode=self.mode,entries=deepcopy(self.entries),stored_text_tokens=self.size(),
                    stored_addresses=sum(len(e['blocks']) for e in self.entries))

    def update(self, question, response, blocks):
        parsed=parse_response(response)
        if self.mode=='none' or parsed is None:
            return dict(saved=False,valid_response=parsed is not None,evicted=0)
        values=dict(question=question,**parsed)
        entry={k:self.clip(values[k],cap) for k,cap in FIELDS.items()}
        need(all(type(b) is int and b>=0 for b in blocks),'invalid selected address')
        entry['blocks']=list(dict.fromkeys(blocks))[:4]
        self.entries.append(entry);evicted=0
        while len(self.entries)>4 or self.size()>256:
            self.entries=self.entries[1:];evicted+=1
        return dict(saved=True,valid_response=True,evicted=evicted)

    def prompt(self):
        if self.mode not in ('notes','both') or not self.entries:return ''
        text=NOTICE+'\n'.join(f"Question: {e['question']}\nModel answer: {e['answer']}\nModel notes: {e['notes']}"
                              for e in self.entries[::-1])
        return self.clip(text,256)

    def addresses(self, question, eligible):
        if self.mode not in ('addresses','both'):return []
        terms=lambda s:set(re.findall('[a-z0-9_]+',s.lower()))-STOP
        scored=[(len(terms(question)&terms(e['question'])),i) for i,e in enumerate(self.entries)]
        if not scored:return []
        overlap,i=max(scored)
        return [b for b in self.entries[i]['blocks'] if b<eligible] if overlap else []


def question_coverage(episodes, tok):
    """Post-hoc scoring metadata only; never passed into memory/index decisions."""
    result={}
    for ep in episodes:
        prefix=ep['tokens'][:ep['prefix_tokens_count']]
        text=tok.decode(prefix,skip_special_tokens=False)
        encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        need(encoded['input_ids']==prefix,'coverage text does not round-trip to the actual prefix')
        def span(record):
            need(text.count(record)==1,'coverage record is missing or ambiguous')
            start=text.index(record);end=start+len(record)
            hits=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<end and b>start]
            need(bool(hits),'coverage record has no token positions')
            return [min(hits),max(hits)+1]
        original=[span(record) for record in ep['records']]
        need(original==ep['record_spans'],'original archive span metadata differs')
        relation=next(r for r in ep['records'] if r.startswith('The designated station'))
        for qi,q in enumerate(ep['queries']):
            if q['kind'] in ('bridge_learning','bridge_reuse'):needed=[relation]
            elif q['kind'] in ('linked_reuse','paraphrased_reuse'):needed=ep['records']
            elif q['kind']=='new_entity':
                entity=re.search(r'CABINET_[0-9]+',q['query'])
                need(entity is not None,'missing cabinet identity')
                matches=list(re.finditer(r'The access code for '+re.escape(entity[0])+r' is ([0-9]{8})\.',text))
                need(len(matches)==1 and matches[0][1]==q['answer'],'new-entity source answer differs')
                needed=[matches[0][0]]
            else:raise ValueError('unknown memory question kind')
            result[ep['id']+f'-memory-q{qi}']=[span(record) for record in needed]
    return result


def flat_plan(manifest, episodes, coverage=None):
    questions=[];plan=[]
    byid={ep['id']:ep for ep in episodes}
    for ep in episodes:
        for i,q in enumerate(ep['queries']):
            ex={**ep,'id':ep['id']+f'-memory-q{i}','task':q['kind'],'query':q['query'],'answer':q['answer']}
            # The parent fixture's required spans describe its original linked
            # query. They are not gold coverage for all five new questions.
            ex.pop('required_record_indices',None)
            if coverage is not None:
                ex['record_spans']=coverage[ex['id']]
                ex['required_record_indices']=list(range(len(ex['record_spans'])))
            questions.append(ex)
    for item in manifest['schedule']:
        ep=byid[item['episode_id']];p=item['policy']
        for i in range(item['queries']):
            plan.append(dict(example_id=ep['id']+f'-memory-q{i}',trial=0,policy=p,
                purpose='synthetic_stale_memory_stress' if p['memory_seed']=='synthetic_stale' else 'natural_model_memory'))
    return questions,plan


def replay_memory(manifest, episodes, rows, tok, *, eos_ids=None, index_factory=None, coverage=None):
    from research.semantic_index import metadata_index
    from research.budget_probe import expand_blocks
    if index_factory is None:index_factory=lambda ep,root:metadata_index(tok,ep,root,32)
    questions,plan=flat_plan(manifest,episodes,coverage)
    key=lambda r:(r['example_id'],r['policy']['id'],r['trial'])
    need(len(rows)<=160 and [key(r) for r in rows]==[key(r) for r in plan[:len(rows)]],
         'missing, duplicate, reordered or unexpected memory observations')
    byid={ep['id']:ep for ep in episodes};memories={};prompts={};records=[];retrieval_rounds=0
    with tempfile.TemporaryDirectory(prefix='memory-index-audit-') as temp:
        indices={}
        try:
            for row in rows:
                ep=byid[row['episode_id']];p=row['policy'];qi=row['query_index']
                need(type(qi) is int and 0<=qi<5 and row['example_id']==ep['id']+f'-memory-q{qi}',
                     'episode or query identity differs')
                mk=(ep['id'],p['id'])
                if mk not in memories:
                    need(qi==0,'episode starts after its first question')
                    memory=MemoryReplay(tok,p['memory']);memories[mk]=(memory,0)
                    if p['memory_seed']=='synthetic_stale':
                        memory.update(ep['queries'][0]['query'],json.dumps(dict(answer='STATION_OLD_MEMORY',search='',
                            notes='An older note says the assigned station was STATION_OLD_MEMORY.')),[])
                    else:need(p['memory_seed']=='empty','unknown memory initialization')
                memory,expected_qi=memories[mk]
                need(qi==expected_qi,'memory query chain skips or repeats a question')
                need(row['memory_before']==memory.snapshot(),'memory before-state differs from prior model output')
                prompt=memory.prompt();prompts[key(row)]=prompt
                prompt_tokens=len(tok.encode(prompt,add_special_tokens=False))
                need(prompt_tokens<=256 and memory.size()<=256,'memory text budget exceeded')
                need(row['comparison']==('hybrid KV-plus-model-text-memory' if p['memory'] in ('notes','both')
                                        else 'no text-memory prompt'),'memory comparison label differs')
                if p['attention']=='sparse':
                    if ep['id'] not in indices:indices[ep['id']]=index_factory(ep,Path(temp))
                    index=indices[ep['id']];eligible=math.ceil(ep['prefix_tokens_count']/128)
                    for turn,round in enumerate(row['rounds']):
                        seeds=index.search(round['query'],None,4,'lexical_rare');detail=dict(index.last_search)
                        if turn==0:
                            remembered=memory.addresses(ep['queries'][qi]['query'],eligible)
                            detail['remembered_seed_blocks']=remembered
                            seeds=list(dict.fromkeys([*seeds,*remembered]))[:16]
                        selected=list(expand_blocks(tuple(seeds),eligible,16)) if seeds else []
                        need(selected==round['selected_blocks'] and detail==round['index_details'],
                             'retrieval or remembered-address replay differs')
                        retrieval_rounds+=1
                final=row['rounds'][-1]
                update=memory.update(ep['queries'][qi]['query'],final['response'],final['selected_blocks'])
                need(row['memory_update']==update,'memory update decision or FIFO eviction differs')
                need(row['memory_after']==memory.snapshot(),'memory after-state differs from latest model output')
                memories[mk]=(memory,qi+1)
                records.append(dict(episode_id=ep['id'],policy=p['id'],query_index=qi,
                    prompt_text_tokens=prompt_tokens,stored_text_tokens=memory.size(),
                    stored_addresses=memory.snapshot()['stored_addresses'],entries=len(memory.entries),
                    saved=update['saved'],valid_response=update['valid_response'],evicted=update['evicted'],
                    saved_incorrect_final_answer=update['saved'] and not row['exact_correct']))
        finally:
            for index in indices.values():index.close()
    common=audit_rows({**manifest,'schedule':plan},questions,rows,tokenizer=tok,
                      eos_ids=eos_ids,memory_prompts=prompts)
    return common,records,retrieval_rounds


def summarize(episodes, rows, records):
    need(len(rows)==160,'complete memory matrix required')
    byid={ep['id']:ep for ep in episodes}
    bykey={(r['episode_id'],r['policy']['id'],r['query_index']):r for r in rows}
    quality=[];paired=[];episode_totals=[]
    for p in declared_policies():
        stale=p['memory_seed']=='synthetic_stale'
        matching_stock=('stock-memory-notes-stale-r2-cap128' if stale else
                        'stock-memory-'+('notes' if p['memory'] in ('notes','both') else 'none')+'-r2-cap128')
        bare=p['attention']+'-memory-none-r2-cap128'
        for budget in (8192,32768):
            group=[r for r in rows if r['policy']['id']==p['id'] and byid[r['episode_id']]['budget']==budget]
            need(len(group)==10,'missing paired memory questions')
            controls=[bykey[r['episode_id'],matching_stock,r['query_index']] for r in group]
            bare_rows=[bykey[r['episode_id'],bare,r['query_index']] for r in group]
            quality.append(dict(budget=budget,policy=p['id'],memory_seed=p['memory_seed'],questions=10,episodes=2,
                exact_correct=sum(r['exact_correct'] for r in group),
                lost_vs_matching_stock=sum(b['exact_correct'] and not r['exact_correct'] for r,b in zip(group,controls)),
                gained_vs_matching_stock=sum(not b['exact_correct'] and r['exact_correct'] for r,b in zip(group,controls)),
                matching_stock=matching_stock,bare_control=bare,
                lost_vs_bare=sum(b['exact_correct'] and not r['exact_correct'] for r,b in zip(group,bare_rows)),
                gained_vs_bare=sum(not b['exact_correct'] and r['exact_correct'] for r,b in zip(group,bare_rows)),
                mean_request_s=statistics.mean(r['request_s'] for r in group),
                mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in group),
                mean_evaluated_tokens=statistics.mean(r['model_token_evaluations'] for r in group),
                cuda_peak_GiB=max(r['cuda_peak_allocated_bytes'] for r in group)/2**30,
                host_peak_GiB=max(r['host_rss_sampled_peak_bytes'] for r in group)/2**30,
                max_active_KV_entries=max(r['max_active_KV_entries'] for r in group)))
        for ep in episodes:
            group=[bykey[ep['id'],p['id'],qi] for qi in range(5)]
            controls=[bykey[ep['id'],matching_stock,qi] for qi in range(5)]
            episode_totals.append(dict(episode_id=ep['id'],budget=ep['budget'],policy=p['id'],
                request_s=sum(r['request_s'] for r in group),matching_stock=matching_stock,
                matching_stock_request_s=sum(r['request_s'] for r in controls),
                paired_episode_speed_ratio=sum(r['request_s'] for r in controls)/sum(r['request_s'] for r in group)))
            for row,control in zip(group,controls):
                baseline=bykey[ep['id'],bare,row['query_index']]
                paired.append(dict(episode_id=ep['id'],budget=ep['budget'],policy=p['id'],query_index=row['query_index'],
                    kind=row['task'],correct=row['exact_correct'],matching_stock_correct=control['exact_correct'],
                    bare_correct=baseline['exact_correct'],request_s=row['request_s'],
                    matching_stock_request_s=control['request_s'],bare_request_s=baseline['request_s'],
                    prediction=row['prediction']))
    return dict(quality=quality,paired_questions=paired,episode_totals=episode_totals,
                memory_trace=records,max_stored_text_tokens=max(r['stored_text_tokens'] for r in records),
                max_memory_prompt_tokens=max(r['prompt_text_tokens'] for r in records),
                max_stored_addresses=max(r['stored_addresses'] for r in records))


def write_report(root, manifest, audit):
    need(audit['status']=='complete','incomplete memory comparison')
    lines=['# Bounded memory-harness comparison','',f"Model: `{manifest['model']['model_id']}` at `{manifest['model']['revision']}`.",
        'All 160 scheduled rows are audited: 120 natural-memory rows and 40 synthetic-stale rows.',
        'Four reused archive episodes each contain five sequential questions. The questions within an episode',
        'and the two archive lengths are correlated; these are not 160 independent quality examples.', '',
        'The audit replays every saved memory transition, inserted note prompt, address reuse and query selection.',
        'Generated token decoding, prompt replay, pending-token work and final-draft scoring are checked separately.',
        'All policies use two 128-token rounds. Only bounded declared memory persists between fresh per-question KV sessions.', '']
    for seed,title in [('empty','Natural model memory'),('synthetic_stale','Synthetic stale-note stress')]:
        lines+=['## '+title,'',
            '| Budget | Policy | Exact | Lost/gained vs same-attention no-memory | Mean request s | Mean output/evaluated tokens | CUDA peak GiB |',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in audit['quality']:
            if r['memory_seed']==seed:
                lines.append(f"| {r['budget']} | {r['policy']} | {r['exact_correct']}/10 | {r['lost_vs_bare']}/{r['gained_vs_bare']} | {r['mean_request_s']:.3f} | {r['mean_generated_tokens']:.1f}/{r['mean_evaluated_tokens']:.1f} | {r['cuda_peak_GiB']:.3f} |")
        lines+=['']
    lines+=['## Interpretation and limits','',
        'A correct answer and lower total request cost must occur together to support an improvement.',
        'No-memory baselines also receive two reasoning rounds; they emit the same JSON fields but discard notes.',
        'The stale rows begin with an explicitly injected obsolete note, not a claimed earlier model answer.',
        '`summary.json` contains each paired question and five-question episode total, including matching-memory stock controls.',
        'There are no repeated timing trials in this mechanism screen, and task-mix averages include failures.', '',
        f"Observed maxima: {audit['max_stored_text_tokens']} stored text tokens, {audit['max_memory_prompt_tokens']} memory-prompt-body tokens before serialization, {audit['max_stored_addresses']} stored address integers.",
        'The exact serialized model prompts, including separators, are replayed and counted in evaluated tokens.',
        'These fixed logical state bounds do not establish constant host-process RSS or index size.', '',
        'Per-round record coverage is recomputed for each question type from token-aligned archive text.',
        'It describes retained positions at round start, not a fresh per-layer payload check. A correct answer',
        'can also come from model notes, so coverage alone does not identify the cause of a recovery.', '',
        *['- '+x for x in audit['limits']]]
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--partial',action='store_true')
    parser.add_argument('--source-revision',default=SOURCE)
    args=parser.parse_args();root=args.run;repo=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'manifest.json').read_text());episodes=json.loads((root/'fixtures.json').read_text())
    rows=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    pinned=verify_source(manifest,episodes,repo,args.source_revision)
    validation_names={r['validation_file'] for r in rows};weights=set();parameter_counts_recorded=[]
    for name in validation_names:
        need(Path(name).name==name and name.startswith('validation-') and name.endswith('.json'),'unsafe validation filename')
        value=json.loads((root/name).read_text());audit_validation(value,manifest['model'],parameter_count_required=False)
        weights.add(value['weights_sha256']);parameter_counts_recorded.append('loaded_parameters' in value)
    need(len(weights)==1,'missing or inconsistent weight identity')
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot=Path(snapshot_download(manifest['model']['model_id'],revision=pinned['revision'],local_files_only=True))
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    tok.apply_chat_template=partial(tok.apply_chat_template,enable_thinking=False)
    need(digest(tok.chat_template)==pinned['chat_template_sha256'],'pinned tokenizer template differs')
    generation=json.loads((snapshot/'generation_config.json').read_text());eos=generation['eos_token_id']
    eos=set(eos if isinstance(eos,list) else [eos])
    marker=json.loads((root/'complete.json').read_text()) if (root/'complete.json').exists() else None
    need(args.partial or marker is not None,'matrix incomplete; use --partial for non-conclusive audit')
    if marker is not None:
        need(marker==dict(rows=160,weights_unchanged=True,natural_memory_rows=120,synthetic_stale_rows=40)
             and len(rows)==160,'completion marker or exact matrix differs')
    coverage=question_coverage(episodes,tok)
    audit,records,replayed=replay_memory(manifest,episodes,rows,tok,eos_ids=eos,coverage=coverage)
    audit.update(status='complete' if marker is not None else 'incomplete',retrieval_rounds_replayed=replayed,
        source_revision=args.source_revision,weights_sha256=next(iter(weights)),question_record_spans=coverage,
        parameter_count_independently_recorded=all(parameter_counts_recorded),
        files_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ['manifest.json','fixtures.json','samples.jsonl',*sorted(validation_names)]+(['complete.json'] if marker else [])},
        audit_sources_sha256={n:hashlib.sha256((repo/'research'/n).read_bytes()).hexdigest()
                              for n in ('report_memory.py','report_reasoning.py')},
        generation_config_sha256=hashlib.sha256((snapshot/'generation_config.json').read_bytes()).hexdigest())
    audit['limits']=['Four correlated episodes; no near-equivalence or 1M-context claim.',
        'Model-generated note prompts make this a hybrid KV-plus-text-memory system; no retrieved source text is inserted.',
        'Fresh immutable archive KV per question; this screen does not test newly appended facts or continuing KV conversations.',
        'Initialized-cache request time includes memory prompts and updates; mount/init/prefill/index/checkpoint remain separate.',
        'Stock batches prompts; sparse prompts are currently evaluated serially.',
        'Memory-run source asserts loaded parameter count but its frozen validation JSON does not separately record that count.',
        'Question-specific record coverage is inferred from retained addresses, not a new per-layer KV-payload audit.',
        'No repeated timing trials; uncontrolled OS page cache and periodic result backups limit latency interpretation.',
        'Logical attention pairs are operation-count proxies, not measured GPU FLOPs.']
    if marker is not None:audit.update(summarize(episodes,rows,records))
    else:audit['memory_trace']=records
    (root/('summary.json' if marker is not None else 'audit-partial.json')).write_text(json.dumps(audit,indent=2)+'\n')
    if marker is not None:write_report(root,manifest,audit)
    print(json.dumps({k:audit[k] for k in ('status','observed_rows','expected_rows','tokenizer_verified','retrieval_rounds_replayed')}))


if __name__=='__main__':main()

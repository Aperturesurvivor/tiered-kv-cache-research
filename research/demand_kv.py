"""Question-triggered exact-KV retrieval, with cross-model and held-out stages.

Archive construction is separate from each question. No retrieved source text
enters the model. All question tokens use the evaluated policy. Selection is
fixed during the answer: this tests one retrieval event per question, not a
learned token-level trigger or arbitrary multi-hop retrieval controller.
"""
import argparse
from dataclasses import asdict
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
from tiered_kv.adapter import ModelAdapter,DenseController
from tiered_kv.config import Config
from tiered_kv.config import DATA_PATH,DATA_REVISION
from tiered_kv.evals import make_needle,char_span,exact_prompt,render,GOV_PROMPT
from tiered_kv.generate import RunEngine,online_smoke,weights_digest,append_jsonl
from tiered_kv.store import atomic_json,file_sha
from tiered_kv.telemetry import ResourceGuard,runtime_identity,DEVICE,device_empty_cache,disk_preflight
from tiered_kv.types import digest
from tiered_kv.validation import parity,implementation_hash
from research.semantic_index import Encoder,DiskTextIndex,ENCODER_ID,ENCODER_REVISION,metadata_index,with_neighbors
from research.benchmark_stock import measure as stock_measure
from research.resources import HostObservation


def query_boundary(tok,ex):
    query=ex.get('query') or f"What is the access code for {ex['item']}? Return only the code."
    text=tok.decode(ex['tokens'],skip_special_tokens=False)
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=ex['tokens']:
        raise ValueError('query boundary failed to round-trip')
    start=text.rindex(query)
    ex['prefix_tokens_count']=char_span(encoded['offset_mapping'],start,start+len(query))[0]
    ex['token_hash']=digest(ex['tokens'])
    return query


def fact_probe(tok,budget,seed,paraphrase):
    answer=str(int(np.random.default_rng(seed).integers(10000000,99999999)))
    record=f'The emergency radio channel at the seaside watchpost is {answer}.'
    query=('Which eight-digit frequency should rescuers use to contact the coastal observation station? Reply only with the number.' if paraphrase else
           'What is the emergency radio channel at the seaside watchpost? Return only the number.')
    source=''.join(f'The record for cabinet {i} is the report with the blue folders. The report for the archive is recorded with {i+17} pages.\n' for i in range(budget//16+100))
    text,encoded=exact_prompt(tok,budget,128,.5,seed,record,query,source_text=source)
    if text.count(answer)!=1:
        raise ValueError('fact probe answer collision')
    a=text.index(answer)
    answer_span=char_span(encoded['offset_mapping'],a,a+len(answer))
    start=text.index(record)
    record_span=char_span(encoded['offset_mapping'],start,start+len(record))
    return {'id':f'fact-{"paraphrase" if paraphrase else "literal"}-{budget}-{seed}',
            'task':'niah','budget':budget,'depth':.5,'seed':seed,'query':query,
            'tokens':encoded['input_ids'],'answer':answer,'references':[answer],
            'record_span':record_span,'answer_span':answer_span,
            'gold_blocks':list(range(answer_span[0]//128,(answer_span[1]-1)//128+1)),
            'actual_depth':record_span[0]/len(encoded['input_ids']),'max_new_tokens':128,
            'probe':'natural wording; invented numeric fact; literal/paraphrase pair'}


def fixtures(tok,spec,stage):
    limit=spec['config']['max_position_embeddings']
    budgets=[n for n in (4096,8192,16384,32768) if n<=limit]
    if stage=='summaries':
        import zipfile
        with zipfile.ZipFile(DATA_PATH) as archive:
            docs=[json.loads(line) for line in archive.read('data/gov_report.jsonl').decode().splitlines()]
        bands={'short':[],'long':[]}
        for doc in docs:
            text=render(tok,GOV_PROMPT.format(context=doc['context']))
            tokens=tok.encode(text,add_special_tokens=False)
            band='short' if 4096<=len(tokens)<=7680 else ('long' if 16384<=len(tokens)<=min(limit-512,32256) else None)
            if band is None:
                continue
            source_id=str(doc.get('_id',doc.get('id',digest(doc))))
            ex={'id':'gov-'+source_id,'source_id':source_id,'task':'gov_report','tokens':tokens,
                'references':doc['answers'],'max_new_tokens':512,'band':band,'depth':None,
                'budget':next(n for n in budgets if n>=len(tokens)+512),
                'query':'Now, write a one-page summary of the report.'}
            query_boundary(tok,ex)
            bands[band].append(ex)
        if not all(bands.values()):
            raise ValueError('summary probe requires complete reports in both declared length bands')
        examples=[min(candidates,key=lambda e:hashlib.sha256((e['source_id']+'demand-summary-97').encode()).hexdigest()) for candidates in bands.values()]
        return {'examples':examples,'budgets':budgets,'data_revision':DATA_REVISION,
                'data_archive_sha256':file_sha(DATA_PATH),'eligible_reports':{k:len(v) for k,v in bands.items()},
                'selection':'one complete report per declared band, minimum source-ID hash with salt demand-summary-97',
                'chat_template_hash':digest(tok.chat_template),'thinking':False}
    seeds=[41] if stage=='screen' else [53,71]
    depths=(.1,.5,.9) if stage=='confirm' else (.1,.5)
    if stage=='online':
        budgets=[4096,min(8192,limit)]
        seeds=[83]
        depths=(.1,)
    examples=[make_needle(tok,n,d,s) for n in budgets for d in depths for s in seeds]
    examples.extend(make_needle(tok,max(budgets),.5,s,distractors=True) for s in seeds)
    if stage!='online':
        examples.extend(fact_probe(tok,max(budgets),s,paraphrase) for s in seeds for paraphrase in (False,True))
    for ex in examples:
        query_boundary(tok,ex)
    return {'examples':examples,'budgets':budgets,'seeds':seeds,'depths':depths,
            'chat_template_hash':digest(tok.chat_template),'thinking':False}


def prepare_index(tok,fixture,root,boundary_control=False):
    encoder=Encoder()
    rows=[]
    for ex in fixture['examples']:
        query=query_boundary(tok,ex)
        cutoff=max(0,ex['prefix_tokens_count']-2000+1)
        path=root/(ex['id']+'.sqlite3')
        marker=path.with_suffix('.complete.json')
        if path.exists() and not marker.exists():
            raise RuntimeError('incomplete index retained: '+str(path))
        start=time.perf_counter()
        index=DiskTextIndex(path,create=not path.exists())
        try:
            if not marker.exists():
                for offset in range(0,cutoff,16*128):
                    chunks=[(s//128,s,min(s+128,cutoff),tok.decode(ex['tokens'][s:min(s+128,cutoff)],skip_special_tokens=False))
                            for s in range(offset,min(offset+16*128,cutoff),128)]
                    index.add(chunks,encoder.encode([c[3] for c in chunks]))
                atomic_json(marker,{'token_hash':ex['token_hash'],'database_sha256':file_sha(path)})
            else:
                meta=json.loads(marker.read_text())
                if meta['token_hash']!=ex['token_hash'] or meta['database_sha256']!=file_sha(path):
                    raise ValueError('index identity/checksum mismatch')
            build_s=time.perf_counter()-start
            start=time.perf_counter()
            vector=encoder.encode([query])[0]
            encode_s=time.perf_counter()-start
            for method in ('lexical','lexical_rare','semantic','hybrid_bounded'):
                for k in ((4,16,32) if ex['task']=='gov_report' else (4,16)):
                    start=time.perf_counter()
                    selected=index.search(query,vector,k,method)
                    search_s=time.perf_counter()-start
                    exact={i for b in selected for i in range(b*128,min((b+1)*128,cutoff))}
                    available=exact|set(range(cutoff,len(ex['tokens'])))
                    rows.append({'example_id':ex['id'],'method':method,'k':k,'selected_blocks':selected,
                        'query_encode_s':0. if method.startswith('lexical') else encode_s,'index_search_s':search_s,
                        'index_build_or_open_s':build_s,'index_disk_bytes':path.stat().st_size,
                        'gold_fully_available':set(range(*ex['answer_span']))<=available if 'answer_span' in ex else None,
                        'lookup_details':dict(index.last_search)})
        finally:
            index.close()
        if boundary_control:
            start=time.perf_counter()
            contextual=metadata_index(tok,{**ex,'prefix_tokens_count':cutoff},root,32)
            build_s=time.perf_counter()-start
            try:
                start=time.perf_counter()
                found=contextual.search(query,None,4,'lexical_rare')
                selected=with_neighbors(found,(cutoff+127)//128)
                search_s=time.perf_counter()-start
                exact={i for b in selected for i in range(b*128,min((b+1)*128,cutoff))}
                available=exact|set(range(cutoff,len(ex['tokens'])))
                rows.append({'example_id':ex['id'],'method':'lexical_rare_context','k':4,
                    'selected_blocks':selected,'query_encode_s':0.,'index_search_s':search_s,
                    'index_build_or_open_s':build_s,
                    'index_disk_bytes':(root/(ex['id']+'-metadata-32.sqlite3')).stat().st_size,
                    'gold_fully_available':set(range(*ex['answer_span']))<=available,
                    'lookup_details':dict(contextual.last_search)|{'metadata_padding':32,'neighbors':True,'before_neighbors':found}})
            finally:
                contextual.close()
    del encoder
    gc.collect()
    return rows


def policies(ex,lookup,revision,stage):
    common={'protocol':'online_tiered' if stage=='online' else 'query_replay',
            'model_revision':revision,'C':0,'sink_tokens':4,'buffered_writes':True}
    def selected(method,k=4):
        return tuple(lookup[ex['id'],method,k]['selected_blocks'])
    rows=[Config(id='full_buffered',full=True,**common),
          Config(id='sink_only',k=0,retrieval='fixed',**common),
          Config(id='lexical_fast',retrieval='fixed',fixed_blocks=selected('lexical'),**common),
          Config(id='semantic_fast',retrieval='fixed',fixed_blocks=selected('semantic'),**common)]
    if stage=='ablate':
        for name,extra in [('lexical_unbuffered',{'buffered_writes':False}),
                           ('lexical_no_sinks',{'sink_tokens':0}),
                           ('lexical_compressed',{'C':32})]:
            rows.append(Config(id=name,retrieval='fixed',fixed_blocks=selected('lexical'),**(common|extra)))
        # Label as a gold-assisted control, never as a deployable policy or an
        # accuracy upper bound. Fill its budget with the same semantic candidates.
        cutoff=max(0,ex['prefix_tokens_count']-2000+1)
        gold=[b for b in ex['gold_blocks'] if b*128<cutoff]
        combined=list(dict.fromkeys(gold+list(selected('semantic'))))[:4]
        rows.append(Config(id='gold_assisted',retrieval='fixed',fixed_blocks=tuple(combined),**common))
    elif stage=='summaries':
        rows.extend([Config(id='semantic_k32',k=32,retrieval='fixed',fixed_blocks=selected('semantic',32),**common),
                     Config(id='semantic_compressed',retrieval='fixed',fixed_blocks=selected('semantic'),**(common|{'C':32}))])
    else:
        rows.append(Config(id='rare_fast',retrieval='fixed',fixed_blocks=selected('lexical_rare'),**common))
    if stage in ('confirm','online'):
        rows.append(Config(id='rare_context',retrieval='fixed',fixed_blocks=selected('lexical_rare_context'),**common))
    if stage=='online':
        rows=[r for r in rows if r.id in ('full_buffered','sink_only','rare_context')]
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--stage',choices=['ablate','screen','confirm','online','summaries'],default='ablate')
    p.add_argument('--reuse-index',type=Path)
    p.add_argument('--reuse-prefixes',type=Path)
    args=p.parse_args()
    if args.stage=='ablate' and not args.reuse_index:
        p.error('ablation requires --reuse-index to keep its exploratory fixtures fixed')
    torch.set_num_threads(1)
    spec=next(s for s in json.loads(Path('research/model-manifest.json').read_text()) if s['model_id']==args.model)
    disk_preflight()
    snapshot=snapshot_download(spec['model_id'],revision=spec['revision'],
        allow_patterns=['*.safetensors','*.json','*.model','merges.txt','vocab.txt'])
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    tok.apply_chat_template=partial(tok.apply_chat_template,enable_thinking=False)
    if args.reuse_index:
        if args.model!='Qwen/Qwen2.5-1.5B-Instruct' or args.stage!='ablate':
            raise ValueError('existing index is restricted to its original model and exploratory ablation')
        source_fixture=json.loads((args.reuse_index/'fixtures.json').read_text())
        fixture={'examples':[e for e in source_fixture['examples'] if e['depth']==.5 or (e['budget']==32768 and e['depth']==.9)]}
        coverage=json.loads((args.reuse_index/'coverage.json').read_text())
    else:
        fixture=fixtures(tok,spec,args.stage)
        coverage=None
    identity={'model':spec,'stage':args.stage,'runtime':runtime_identity(),
        'implementation_hash':implementation_hash(),'script_sha256':file_sha(__file__),
        'stock_sha256':file_sha('research/benchmark_stock.py'),'host_sha256':file_sha('research/resources.py'),
        'index_sha256':file_sha('research/semantic_index.py'),'encoder':[ENCODER_ID,ENCODER_REVISION],
        'fixture_hash':digest(fixture),'reuse_index_sha256':file_sha(args.reuse_index/'coverage.json') if args.reuse_index else None,
        'scope':'one lookup per question; fixed original-position exact KV retained during answer',
        'limitation':'shared dense archive except online stage; archive checksum/open costs included in init, not steady decode'}
    root=Path('results')/('demand-'+args.model.replace('/','--')+'-'+args.stage+'-'+digest(identity)[:12])
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'manifest.json',identity)
    atomic_json(root/'fixtures.json',fixture)
    if coverage is None:
        index_root=Path('artifacts/demand-index')/root.name
        index_root.mkdir(parents=True,exist_ok=True)
        with HostObservation(scope='index preparation and query lookup; generator not loaded') as preparation:
            coverage=prepare_index(tok,fixture,index_root,boundary_control=args.stage in ('confirm','online'))
        atomic_json(root/'index-resources.json',preparation.metrics|{'scope':'index preparation and query lookup; generator not loaded'})
    atomic_json(root/'coverage.json',coverage)
    lookup={(r['example_id'],r['method'],r['k']):r for r in coverage}
    output=root/'samples.jsonl'
    done={(r['config_id'],r['example_id']) for line in output.read_text().splitlines()
          if (r:=json.loads(line))['status']=='ok'} if output.exists() else set()
    print('DEMAND_START',args.model,args.stage,root,flush=True)
    with ResourceGuard() as guard:
        model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':DEVICE},attn_implementation='sdpa').eval().requires_grad_(False)
        if getattr(model.config,'_commit_hash',None) not in (None,spec['revision']):
            raise ValueError('loaded model revision differs from pinned snapshot')
        model.config._commit_hash=spec['revision']
        before=weights_digest(model)
        checks=[parity(model,n) for n in (64,512,1025)]
        natural=tok.encode('Read the archive and identify the relevant evidence. Cities, rivers and public reports are described in these records. '*200)
        checks.append(parity(model,1025,natural[:1025],probe='natural_text'))
        mechanics=[Config(id='full_buffered',full=True,W=128,C=0,B=32,R=32,k=2,buffered_writes=True,model_revision=spec['revision']),
                   Config(W=128,C=0,B=32,R=32,k=2,retrieval='fixed',fixed_blocks=(0,2),sink_tokens=4,buffered_writes=True,model_revision=spec['revision'])]
        online=online_smoke(model,tok,root,mechanics,guard,tokens_count=512,checkpoint_pos=256)
        if len(online)!=len(mechanics) or any(r['status']!='ok' for r in online):
            raise RuntimeError('buffered actual-model resume failed')
        atomic_json(root/'validation.json',{'status':'passed','parity':checks,'online':online,'weights_sha256':before,
            'loaded_parameters':sum(p.numel() for p in model.parameters())})
        engine=RunEngine(model,tok,root,guard)
        rng=np.random.default_rng(113)
        try:
            for original in fixture['examples']:
                configs=policies(original,lookup,spec['revision'],args.stage)
                examples=[original]
                if original['depth']==.5 and original['id'].startswith('niah-') and original['seed']==min(e['seed'] for e in fixture['examples']):
                    for trial in range(3 if args.stage=='confirm' else 1):
                        examples.append({**original,'id':original['id']+f'-forced-{trial}','task':'latency','trial':trial,
                            'continuation':tok.encode(' The archive contains numbered cabinets and blue folders.'*40,add_special_tokens=False)[:min(144,model.config.max_position_embeddings-len(original['tokens']))]})
                prefix=None
                for ex in examples:
                    measured_configs=configs
                    if ex['task']=='latency' and args.stage=='ablate':
                        measured_configs=[c for c in configs if c.id in ('full_buffered','lexical_fast','lexical_unbuffered','lexical_compressed')]
                    pending=[measured_configs[int(i)] for i in rng.permutation(len(measured_configs)) if (measured_configs[int(i)].id,ex['id']) not in done]
                    if pending and prefix is None and args.stage!='online':
                        existing=args.reuse_prefixes/(original['id']+'-prefix') if args.reuse_prefixes else None
                        if existing and (existing/'checkpoint.json').exists():
                            doc=json.loads((existing/'checkpoint.json').read_text())
                            if doc['metadata']['token_hash']!=original['token_hash'] or doc['metadata']['model_revision']!=spec['revision']:
                                raise ValueError('reused prefix identity mismatch')
                            prefix=(existing,doc['metadata']['prefill_s'])
                        else:
                            prefix=engine.prefix(original)
                    for cfg in pending:
                        with HostObservation() as host:
                            row=engine.run(cfg,ex,prefix_info=prefix,protocol=cfg.protocol)
                        row.update(host.metrics,model_id=args.model,model_revision=spec['revision'],stage=args.stage,policy=asdict(cfg),external_index_memory_included=False)
                        method=('lexical' if cfg.id.startswith('lexical') else 'semantic' if cfg.id.startswith('semantic_') else
                                'lexical_rare' if cfg.id=='rare_fast' else 'lexical_rare_context' if cfg.id=='rare_context' else None)
                        if method:
                            evidence=lookup[original['id'],method,cfg.k]
                            row['index_query_evidence']=evidence
                            row['retrieval_event_s']=evidence['index_search_s']+(evidence['query_encode_s'] if not method.startswith('lexical') else 0.)
                        row['oracle']=cfg.id=='gold_assisted'
                        append_jsonl(output,row)
                        print(json.dumps({k:row.get(k) for k in ('example_id','config_id','status','em','code_found','failure_reason')}),flush=True)
                        if row['status']!='ok':
                            raise RuntimeError(row['failure_reason'])
                        done.add((cfg.id,ex['id']))
                    if ex['task']=='latency' and ('stock_hf',ex['id']) not in done:
                        engine.adapter.uninstall()
                        try:
                            with HostObservation() as host:
                                row=stock_measure(model,ex,guard)
                            row.update(host.metrics,config_id='stock_hf',task='latency',model_id=args.model,stage=args.stage)
                            append_jsonl(output,row)
                            done.add(('stock_hf',ex['id']))
                        finally:
                            engine.adapter=ModelAdapter.install(model,DenseController(engine.layers))
                    device_empty_cache()
        finally:
            engine.adapter.uninstall()
        if weights_digest(model)!=before:
            raise RuntimeError('frozen weights changed')
        atomic_json(root/'complete.json',{'rows':len(done),'weights_unchanged':True})
    print('DEMAND_DONE',root,flush=True)


if __name__=='__main__':
    main()

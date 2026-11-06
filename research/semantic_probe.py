"""Index-only screening: retrieve KV addresses from source-text metadata.

This measures block coverage, not generated answer accuracy. The query contains
only information present in the user's question; gold spans are scoring-only.
"""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from tiered_kv.config import MODEL_PATH,MODEL_REVISION
from tiered_kv.store import atomic_json,file_sha
from tiered_kv.types import digest
from tiered_kv.evals import make_needle,char_span
from research.semantic_index import Encoder,DiskTextIndex,ENCODER_ID,ENCODER_REVISION


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures',type=Path,default=Path('artifacts/data/pilot.json'))
    parser.add_argument('--output',type=Path,default=Path('results/semantic-index-screen'))
    parser.add_argument('--expanded',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    fixture=json.loads(args.fixtures.read_text())
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    if args.expanded:
        fixture={'examples':[make_needle(tok,b,d,23) for b in (4096,8192,16384,32768) for d in (.1,.5,.9)]
                 +[make_needle(tok,32768,.5,37,distractors=True)]}
    for ex in fixture['examples']:
        if ex['task']!='niah':
            continue
        query=ex.get('query') or f"What is the access code for {ex['item']}? Return only the code."
        text=tok.decode(ex['tokens'],skip_special_tokens=False)
        encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        if encoded['input_ids']!=ex['tokens']:
            raise ValueError('query boundary tokenization failed to round-trip')
        start=text.rindex(query)
        ex['prefix_tokens_count']=char_span(encoded['offset_mapping'],start,start+len(query))[0]
        ex['token_hash']=digest(ex['tokens'])
    encoder=Encoder()
    identity={'fixture_hash':digest(fixture),'generator_revision':MODEL_REVISION,
        'encoder_id':ENCODER_ID,'encoder_revision':ENCODER_REVISION,'encoder_device':'cpu',
        'encoder_dtype':'float32','block_size':128,'recent_window':2000,
        'query_policy':'original question only; fixed selection for the answer',
        'semantic_search':'streamed exact cosine scan; not ANN','fusion':'RRF constant 60',
        'source_sha256':file_sha('research/semantic_index.py'),
        'probe_sha256':file_sha(__file__),'platform':platform.platform()}
    output=args.output/digest(identity)[:12]
    output.mkdir(parents=True,exist_ok=True)
    index_root=Path('artifacts/semantic-index')/digest(identity)[:12]
    index_root.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'manifest.json',identity)
    atomic_json(output/'fixtures.json',fixture)
    rows=[]
    for ex in fixture['examples']:
        if ex['task']!='niah':
            continue
        # Same cold/recent boundary used by the last prompt-token attention call.
        cutoff=max(0,ex['prefix_tokens_count']-2000+1)
        query=ex.get('query') or f"What is the access code for {ex['item']}? Return only the code."
        path=index_root/(ex['id']+'.sqlite3')
        complete=path.with_suffix('.complete.json')
        start=time.perf_counter()
        if path.exists() and not complete.exists():
            raise RuntimeError('incomplete index retained: '+str(path))
        index=DiskTextIndex(path,create=not path.exists())
        try:
            if not complete.exists():
                for offset in range(0,cutoff,16*128):
                    blocks=[(s//128,s,min(s+128,cutoff),tok.decode(ex['tokens'][s:min(s+128,cutoff)],skip_special_tokens=False))
                            for s in range(offset,min(offset+16*128,cutoff),128)]
                    index.add(blocks,encoder.encode([b[3] for b in blocks]))
                atomic_json(complete,{'token_hash':ex['token_hash'],'cutoff':cutoff,'database_sha256':file_sha(path)})
            else:
                meta=json.loads(complete.read_text())
                if meta['token_hash']!=ex['token_hash'] or meta['database_sha256']!=file_sha(path):
                    raise ValueError('index identity/checksum mismatch')
            build_or_open_s=time.perf_counter()-start
            start=time.perf_counter()
            q=encoder.encode([query])[0]
            query_encode_s=time.perf_counter()-start
            for method in ('lexical','lexical_rare','semantic','hybrid','hybrid_bounded'):
                for k in (4,16,32):
                    start=time.perf_counter()
                    selected=index.search(query,q,k,method)
                    search_s=time.perf_counter()-start
                    exact={i for b in selected for i in range(b*128,min((b+1)*128,cutoff))}
                    recent=set(range(cutoff,len(ex['tokens'])))
                    gold=set(range(*ex['answer_span']))
                    row={'example_id':ex['id'],'token_hash':ex['token_hash'],'budget':ex['budget'],
                         'depth':ex['depth'],'seed':ex['seed'],'method':method,'k':k,
                         'query':query,'selected_blocks':selected,'eligible_blocks':(cutoff+127)//128,
                         'gold_blocks':ex['gold_blocks'],'gold_fraction_retrieved':len(gold&exact)/len(gold),
                         'gold_fully_available':gold<=(exact|recent),'gold_in_recent':gold<=recent,
                         'index_build_or_open_s':build_or_open_s,'query_encode_s':0. if method.startswith('lexical') else query_encode_s,
                         'index_search_s':search_s,'index_disk_bytes':path.stat().st_size,
                         'index_path':str(path),'lookup_details':dict(index.last_search)}
                    rows.append(row)
                    print(json.dumps({key:row[key] for key in ('example_id','method','k','gold_fully_available','index_search_s')}),flush=True)
            atomic_json(output/'coverage.json',rows)
        finally:
            index.close()
    atomic_json(output/'complete.json',{'rows':len(rows),'scope':'index coverage only; exact KV generation is a separate test'})
    print('INDEX_RESULTS',output,flush=True)


if __name__=='__main__':
    main()

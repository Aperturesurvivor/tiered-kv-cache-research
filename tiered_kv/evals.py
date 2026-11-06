"""Immutable token fixtures and official LongBench v1 English ROUGE-L."""
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from rouge import Rouge
from .config import DATA_PATH, DATA_REVISION, MODEL_PATH, MODEL_REVISION, load_eval
from .types import digest
from .store import atomic_json, file_sha
from .telemetry import disk_preflight

UPSTREAM_REVISION='c5ea10bcd06285223c58dfed76bbc92d22273709'
GOV_PROMPT='You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:'


def score_needle(prediction,answer):
    return {'em':100.*(prediction.strip()==answer),'code_found':100.*bool(re.search(r'(?<!\d)'+re.escape(answer)+r'(?!\d)',prediction))}


def score_rouge(prediction,references):
    if not prediction.strip():
        return 0.,None
    try:
        return 100*max(Rouge().get_scores([prediction],[r],avg=True)['rouge-l']['f'] for r in references),None
    except Exception as exc:
        return 0.,f'{type(exc).__name__}: {exc}'


def render(tokenizer,content):
    return tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)


def filler(seed,budget):
    rng=np.random.default_rng(seed)
    return ''.join(f'Archive fact {i}: cabinet {int(rng.integers(10,9999))} held {int(rng.integers(10,9999))} blue folders.\n' for i in range(budget//12+100))


def exact_prompt(tok,budget,reserve,depth,seed,record='',question='Continue the archive with one brief sentence.',source_text=None):
    target=budget-reserve
    source=tok.encode(filler(seed,budget) if source_text is None else source_text,add_special_tokens=False)
    front_count=max(0,int(target*depth)-40)
    front=tok.decode(source[:front_count])
    back=tok.decode(source[front_count:front_count+max(0,target-front_count-100)])
    def assemble():
        return render(tok,'Read the following archive.\n\n'+front+'\n'+record+'\n'+back+'\n\n'+question)
    for _ in range(30):
        text=assemble()
        encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        delta=target-len(encoded['input_ids'])
        if delta==0:
            return text,encoded
        if delta>0:
            back+=' x'*delta
        else:
            ids=tok.encode(back,add_special_tokens=False)
            if len(ids)+delta<=0:
                raise ValueError('budget too short for fixture')
            back=tok.decode(ids[:delta])
    raise ValueError('exact prompt assembly did not converge')


def char_span(offsets,start,end):
    indices=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    if not indices:
        raise ValueError('span has no tokens')
    return [min(indices),max(indices)+1]


def make_needle(tok,budget,depth,seed,distractors=False):
    rng=np.random.default_rng(seed)
    item='ITEM_'+''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'),12))
    answer=str(int(rng.integers(10000000,99999999)))
    record=f'The access code for {item} is {answer}.'
    question=f'What is the access code for {item}? Return only the code.'
    source=None
    if distractors:
        others=np.random.default_rng(seed+123456)
        lines=[]
        for i in range(budget//8+100):
            key='ITEM_'+''.join(others.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'),12))
            code=str(int(others.integers(10000000,99999999)))
            if key!=item and code!=answer:
                lines.append(f'The access code for {key} is {code}.\n')
        source=''.join(lines)
    text,encoded=exact_prompt(tok,budget,128,depth,seed,record,question,source_text=source)
    if text.count(answer)!=1 or text.count(item)!=2:
        raise ValueError('answer collision or target key duplicated')
    start=text.index(record)
    record_span=char_span(encoded['offset_mapping'],start,start+len(record))
    a=text.index(answer)
    answer_span=char_span(encoded['offset_mapping'],a,a+len(answer))
    return {'id':f'{"multikey" if distractors else "niah"}-{budget}-{depth:g}-{seed}','task':'niah','budget':budget,'depth':depth,'seed':seed,
            'tokens':encoded['input_ids'],'answer':answer,'references':[answer],'item':item,'record_span':record_span,
            'answer_span':answer_span,'gold_blocks':list(range(answer_span[0]//128,(answer_span[1]-1)//128+1)),
            'actual_depth':record_span[0]/len(encoded['input_ids']),'max_new_tokens':128}


def make_latency(tok,budget,trial):
    _,encoded=exact_prompt(tok,budget,144,.5,100+trial)
    continuation=tok.encode(' The archive contains numbered cabinets and blue folders.'*40,add_special_tokens=False)[:144]
    if len(continuation)!=144:
        raise ValueError('insufficient forced continuation')
    return {'id':f'latency-{budget}-{trial}','task':'latency','budget':budget,'trial':trial,'tokens':encoded['input_ids'],
            'continuation':continuation,'max_new_tokens':144}


def prepare(profile='pilot'):
    from transformers import AutoTokenizer
    disk_preflight()
    spec=load_eval()[profile]
    if profile=='online-smoke':
        raise ValueError('online smoke uses the frozen pilot conversation fixture')
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,trust_remote_code=False)
    root=Path('artifacts/data')
    root.mkdir(parents=True,exist_ok=True)
    data=root/'gov_report.jsonl'
    if not data.exists():
        with zipfile.ZipFile(DATA_PATH) as archive:
            with data.open('xb') as out:
                out.write(archive.read('data/gov_report.jsonl'))
    # Verify existing extracted data against the pinned archive, never replace it.
    with zipfile.ZipFile(DATA_PATH) as archive:
        expected=hashlib.sha256(archive.read('data/gov_report.jsonl')).hexdigest()
    if file_sha(data)!=expected:
        raise ValueError('existing gov_report differs from pinned archive')
    examples=[make_needle(tok,b,d,s) for b in spec['budgets'] for d in spec['depths'] for s in spec['seeds']]
    bands={'low':[],'high':[]}
    for line in data.read_text().splitlines():
        doc=json.loads(line)
        text=render(tok,GOV_PROMPT.format(context=doc['context']))
        tokens=tok.encode(text,add_special_tokens=False)
        band='low' if 4096<=len(tokens)<=7680 else 'high' if 16384<=len(tokens)<=32256 else None
        if band:
            identity=str(doc.get('_id',doc.get('id',digest(doc))))
            bands[band].append({'id':'gov-'+identity,'source_id':identity,'task':'gov_report','band':band,'tokens':tokens,
                                'references':doc['answers'],'max_new_tokens':512,'budget':8192 if band=='low' else 32768})
    for band in bands:
        selected=sorted(bands[band],key=lambda x:hashlib.sha256((x['source_id']+'11').encode()).hexdigest())[:spec['reports_per_band']]
        if not selected:
            raise ValueError(f'no complete reports in {band} band')
        examples.extend(selected)
    examples += [make_latency(tok,b,t) for b in spec['latency_budgets'] for t in range(spec['latency_trials'])]
    for ex in examples:
        ex['token_hash']=digest(ex['tokens'])
    manifest={'version':1,'profile':profile,'spec':spec,'model_revision':MODEL_REVISION,'data_revision':DATA_REVISION,
              'data_sha256':expected,'archive_sha256':file_sha(DATA_PATH),'upstream_revision':UPSTREAM_REVISION,
              'prompt_sha256':hashlib.sha256(GOV_PROMPT.encode()).hexdigest(),'gov_prompt':GOV_PROMPT,'gov_cap':512,
              'metric':'rouge==1.0.1, English rouge-l F1, max over references',
              'template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
              'upstream_urls':[f'https://github.com/THUDM/LongBench/blob/{UPSTREAM_REVISION}/LongBench/{f}' for f in ('config/dataset2prompt.json','config/dataset2maxlen.json','metrics.py')],
              'band_available':{b:len(rows) for b,rows in bands.items()},'examples':examples}
    manifest['hash']=digest(manifest)
    destination=root/f'{profile}.json'
    if destination.exists() and json.loads(destination.read_text())!=manifest:
        raise ValueError('frozen fixtures changed; preserve existing profile and investigate')
    atomic_json(destination,manifest)
    return manifest


class HitTracker:
    """Bounded opportunity counters, primary query call distinguished from decode."""
    def __init__(self,example,W,k,full=False):
        self.example,self.W,self.k,self.full=example,W,k,full
        self.query_pos=len(example['tokens'])-1
        self.gold=set(range(*example.get('answer_span',[0,0])))
        self.counts={name:0 for name in ('query_opportunities','query_hits','common_opportunities','common_hits','step_opportunities','step_hits','recent_excluded','coverage','gold_fraction','availability','query_layers','overlap_tokens')}
        self.per_layer={}
        self.chance_sum=0.
        self.candidates_sum=0
        self.any_hit=False
        self.before_code_hit=False
        self.hit_steps=set()
        self.code_start_step=None

    def __call__(self,layer,pos,q,view):
        if not self.gold:
            return
        query=pos==self.query_pos
        cutoff=pos-self.W+1
        eligible={i for i in self.gold if i<cutoff}
        fetched=set(int(i) for i,t in zip(view.ids,view.tiers) if t=='retrieved')
        hit=bool(fetched&self.gold)
        if hit:
            self.any_hit=True
            self.hit_steps.add(pos-self.query_pos)
        if query:
            exact=set(int(i) for i,t in zip(view.ids,view.tiers) if t!='compressed')
            self.counts['query_layers']+=1
            self.counts['availability']+=len(exact&self.gold)/len(self.gold)
            if not eligible:
                self.counts['recent_excluded']+=1
            if eligible and self.k and not self.full:
                self.counts['query_opportunities']+=1
                self.counts['query_hits']+=hit
                self.counts['coverage']+=self.gold<=fetched
                self.counts['gold_fraction']+=len(fetched&self.gold)/len(self.gold)
                if max(self.gold)<self.query_pos-2000+1:
                    self.counts['common_opportunities']+=1
                    self.counts['common_hits']+=hit
                self.candidates_sum+=view.eligible_blocks
                self.chance_sum+=min(self.k,view.eligible_blocks)/view.eligible_blocks if view.eligible_blocks else 0
        if eligible and self.k and not self.full:
            self.counts['step_opportunities']+=1
            self.counts['step_hits']+=hit
            opportunities,hits=self.per_layer.get(layer,(0,0))
            self.per_layer[layer]=(opportunities+1,hits+hit)
        intervals=view.intervals[np.asarray(view.tiers)=='compressed']
        if fetched and len(intervals):
            ids=np.array(list(fetched))[:,None]
            self.counts['overlap_tokens']+=int(((ids>=intervals[None,:,0])&(ids<=intervals[None,:,1])).any(-1).sum())

    def finish_output(self,tokens,tokenizer):
        text=tokenizer.decode(tokens,skip_special_tokens=True)
        answer=self.example.get('answer')
        match=re.search(r'(?<!\d)'+re.escape(answer)+r'(?!\d)',text) if answer else None
        if match:
            for step in range(len(tokens)):
                if len(tokenizer.decode(tokens[:step+1],skip_special_tokens=True))>match.start():
                    self.code_start_step=step
                    break
        self.before_code_hit=self.code_start_step is not None and any(s<=self.code_start_step for s in self.hit_steps)

    def result(self):
        c=self.counts
        ratio=lambda a,b:c[a]/c[b] if c[b] else None
        return {**c,'query_hit':ratio('query_hits','query_opportunities'),'query_hit_common':ratio('common_hits','common_opportunities'),
                'step_hit':ratio('step_hits','step_opportunities'),'full_span_coverage':ratio('coverage','query_opportunities'),
                'gold_fraction':ratio('gold_fraction','query_opportunities'),'exact_availability':ratio('availability','query_layers'),
                'candidate_mean':self.candidates_sum/c['query_opportunities'] if c['query_opportunities'] else None,
                'chance_mean':self.chance_sum/c['query_opportunities'] if c['query_opportunities'] else None,
                'any_hit_before_code':self.before_code_hit,'code_start_step':self.code_start_step,'layer_opportunities_hits':self.per_layer}

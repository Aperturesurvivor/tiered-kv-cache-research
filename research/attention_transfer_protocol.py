"""Predeclared new-seed transfer fixtures; no model inference or answer routing."""
import random
from tiered_kv.evals import char_span,render
from tiered_kv.types import digest
from research.reasoning_protocol import _task,INTRO,RESERVE,WORKING_KV

SEEDS=(281,283)
TASKS=('literal','paraphrase','linked_fact')
BUDGETS={'Qwen/Qwen2.5-1.5B-Instruct':(8192,32768),
         'HuggingFaceTB/SmolLM2-1.7B-Instruct':(8192,)}
REVISIONS={'Qwen/Qwen2.5-1.5B-Instruct':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
           'HuggingFaceTB/SmolLM2-1.7B-Instruct':'31b70e2e869a7173562077fd711b654946d38674'}
POLICY={'W':2000,'C':0,'R':128,'sink_tokens':128,'candidate_ceiling':32,'final_ceiling':15,'rounds':2,'cap_per_round':128}


def make_fixture(tok,budget,seed,kind):
    if budget not in (8192,32768) or seed not in SEEDS or kind not in TASKS:
        raise ValueError('outside predeclared transfer cohort')
    records,query,answer,required=_task(seed,kind)
    if seed==SEEDS[1]:
        records.reverse();required=[1-i for i in required]
    rng=random.Random(f'tiered-kv-reasoning-filler-{seed}-{kind}')
    filler=[]
    for i in range(budget//8):
        code=str(rng.randrange(10000000,100000000))
        while code==answer:code=str(rng.randrange(10000000,100000000))
        filler.append(f'The access code for CABINET_{i:06d} is {code}. Its report is stored with blue folders.\n')
    def assemble(n):
        first,second=n*15//100,n*60//100
        archive=(INTRO+''.join(filler[:first])+records[0]+'\n'+''.join(filler[first:second])+records[1]+'\n'+''.join(filler[second:n])+'\n')
        rendered=render(tok,archive+query);encoded=tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
        qstart=rendered.rindex(query);prefix=char_span(encoded['offset_mapping'],qstart,qstart+len(query))[0]
        suffix=rendered[qstart+len(query):]
        if encoded['input_ids'][:prefix]+tok.encode(query+suffix,add_special_tokens=False)!=encoded['input_ids']:
            raise ValueError('query must be exactly token-concatenable')
        return rendered,encoded,prefix,suffix
    target=budget-RESERVE;lo,hi=0,len(filler)
    while lo<hi:
        mid=(lo+hi+1)//2
        if assemble(mid)[2]<=target:lo=mid
        else:hi=mid-1
    text,encoded,prefix,suffix=assemble(lo)
    if not target-128<=prefix<=target:raise ValueError('archive length outside band')
    spans=[]
    for record in records:
        if text.count(record)!=1:raise ValueError('ambiguous fixture fact')
        start=text.index(record);spans.append(char_span(encoded['offset_mapping'],start,start+len(record)))
    if not all(128<=a<b<prefix-WORKING_KV['W'] for a,b in spans):raise ValueError('fact outside eligible cold archive')
    return {'id':f'transfer-{kind}-{budget}-{seed}','task':kind,'budget':budget,'seed':seed,
            'orientation':'reversed' if seed==SEEDS[1] else 'original','query':query,'answer':answer,
            'records':records,'record_spans':spans,'required_record_indices':required,
            'tokens':encoded['input_ids'],'prefix_tokens_count':prefix,'first_suffix':suffix,'token_hash':digest(encoded['input_ids'])}

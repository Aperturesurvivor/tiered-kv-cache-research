"""Predeclared cross-question memory screen, separate from the frozen run.

The four linked-fact archives are reused for a paired mechanism screen. These
are four correlated episodes, not twenty independent benchmark questions.
"""
import re
import json

from research.reasoning_protocol import SEEDS


def policies():
    result = []
    for attention, modes in (('stock', ('none','notes')), ('sparse', ('none','notes','addresses','both'))):
        for mode in modes:
            result.append({'id':f'{attention}-memory-{mode}-r2-cap128',
                           'attention':attention, 'memory':mode, 'rounds':2,
                           'cap_per_round':128, 'format':'json', 'memory_seed':'empty'})
    for attention, mode in (('stock','notes'), ('sparse','both')):
        result.append({'id':f'{attention}-memory-{mode}-stale-r2-cap128',
                       'attention':attention, 'memory':mode, 'rounds':2,
                       'cap_per_round':128, 'format':'json', 'memory_seed':'synthetic_stale'})
    return result


def initialize_memory(memory, episode, policy):
    if policy['memory_seed'] == 'synthetic_stale':
        # Fixed fault injection, not a claimed earlier model observation. It
        # does not consult the real station, answer, record span or retrieval.
        memory.observe(episode['queries'][0]['query'], json.dumps({
            'answer':'STATION_OLD_MEMORY', 'search':'',
            'notes':'An older note says the assigned station was STATION_OLD_MEMORY.'}), [])
    elif policy['memory_seed'] != 'empty':
        raise ValueError('unknown memory seed')


def episode(tok, original):
    if original['task'] != 'linked_fact' or original['seed'] not in SEEDS:
        raise ValueError('requires declared linked-fact archive')
    relation = next(re.fullmatch(r'The designated station for (PROJECT_[A-Z]+) is (STATION_[A-Z]+)\.', r)
                    for r in original['records'] if r.startswith('The designated station'))
    project, station = relation.groups()
    text = tok.decode(original['tokens'][:original['prefix_tokens_count']], skip_special_tokens=False)
    cabinets = list(re.finditer(r'The access code for (CABINET_[0-9]+) is ([0-9]{8})\.', text))
    if len(cabinets) < 10:
        raise ValueError('missing distractor records')
    unrelated = cabinets[len(cabinets)//3]
    queries = [
        {'kind':'bridge_learning', 'query':f'Which station is designated for {project}?', 'answer':station},
        {'kind':'new_entity', 'query':f'What is the access code for {unrelated[1]}?', 'answer':unrelated[2]},
        {'kind':'linked_reuse', 'query':original['query'], 'answer':original['answer']},
        {'kind':'paraphrased_reuse', 'query':f'Give the eight-digit entry credential for the station assigned to {project}.', 'answer':original['answer']},
        {'kind':'bridge_reuse', 'query':f'Name the station assigned to {project}.', 'answer':station},
    ]
    return {**original, 'queries':queries}


def question_fixture(episode, index):
    q = episode['queries'][index]
    # The immutable prefix remains byte-identical. Query, scoring answer and
    # output identity change; retrieval can access only query and model output.
    return {**episode, 'id':episode['id']+f'-memory-q{index}',
            'task':q['kind'], 'query':q['query'], 'answer':q['answer']}

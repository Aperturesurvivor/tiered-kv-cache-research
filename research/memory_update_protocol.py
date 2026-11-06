"""Append-only archive versions for the separately declared memory update test.

Fixture construction only: no generator or claim of bounded online ingestion.
Old token addresses remain stable; evaluator spans never select retrievals.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import random

from tiered_kv.evals import char_span, render
from tiered_kv.store import file_sha
from tiered_kv.types import digest


SEEDS = (211, 223)
BUDGETS = (8192, 32768)
RESERVE = 2048
APPEND_BUDGET = 3072
RECENT = 2000
INTRO = ('Read the following revisioned archive. For a current value, use the '
         'highest numbered revision for that entity. Earlier revisions remain '
         'available for historical questions.\n\n')


def policies():
    return [dict(id=f'{attention}-update-{mode}-r2-cap128', attention=attention,
                 memory=mode, rounds=2, cap_per_round=128, format='json',
                 memory_seed='empty')
            for attention, mode in (('stock', 'none'), ('stock', 'notes'),
                                    ('sparse', 'none'), ('sparse', 'both'))]


def make_episode(tok, budget, seed):
    if budget not in BUDGETS or seed not in SEEDS:
        raise ValueError('undeclared archive-update case')
    rng = random.Random(f'tiered-memory-update-{seed}')
    entity = 'VAULT_' + ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(10))
    added = 'LOCKER_' + ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(10))
    old, new, extra = map(str, rng.sample(range(10000000, 100000000), 3))
    records = [f'Revision 0: The access code for {entity} is {old}.',
               f'Revision 1: The access code for {entity} is {new}.',
               f'Revision 1: The access code for {added} is {extra}.']
    query = f'What is the current access code for {entity}?'
    filler = []
    for i in range(budget // 8):
        code = str(rng.randrange(10000000, 100000000))
        while code in (old, new, extra):
            code = str(rng.randrange(10000000, 100000000))
        filler.append(f'The access code for CABINET_{i:06d} is {code}. Its report is stored with blue folders.\n')

    def encode(archive, visible, version):
        text = render(tok, archive + query)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        qstart = text.rindex(query)
        count = char_span(enc['offset_mapping'], qstart, qstart + len(query))[0]
        suffix = text[qstart + len(query):]
        if enc['input_ids'][:count] + tok.encode(query + suffix, add_special_tokens=False) != enc['input_ids']:
            raise ValueError('archive/query token boundary is not exact')
        spans = []
        for record in visible:
            if text.count(record) != 1:
                raise ValueError('ambiguous versioned record')
            start = text.index(record)
            spans.append(char_span(enc['offset_mapping'], start, start + len(record)))
        return dict(id=f'memory-update-{budget}-{seed}-v{version}', task='archive_update',
                    budget=budget, seed=seed, archive_version=version, query=query,
                    answer=old if version == 0 else new, records=visible,
                    record_spans=spans, tokens=enc['input_ids'], prefix_tokens_count=count,
                    first_suffix=suffix, token_hash=digest(enc['input_ids']))

    def initial(n):
        split = n // 10
        return INTRO + ''.join(filler[:split]) + records[0] + '\n' + ''.join(filler[split:n]) + '\n'

    def fit(build, target, version, visible):
        lo, hi = 0, len(filler)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if encode(build(mid), visible, version)['prefix_tokens_count'] <= target:
                lo = mid
            else:
                hi = mid - 1
        archive = build(lo)
        result = encode(archive, visible, version)
        if not target - 128 <= result['prefix_tokens_count'] <= target:
            raise ValueError('archive does not fill the declared length band')
        return archive, result, lo

    initial_text, v0, used = fit(initial, budget - RESERVE - APPEND_BUDGET, 0, records[:1])
    hot_text = initial_text + '\n'.join(records[1:]) + '\n\n'
    v1 = encode(hot_text, records, 1)
    # Distinct distractors, with no repeated cabinet IDs across the append.
    def aged(n):
        return hot_text + ''.join(filler[used:used+n]) + '\n'
    _, v2, _ = fit(aged, budget - RESERVE, 2, records)
    questions = [
        dict(version=0, kind='learn_old', query=query, answer=old, record=0),
        dict(version=1, kind='hot_update', query=query, answer=new, record=1),
        dict(version=2, kind='aged_update', query=query, answer=new, record=1),
        dict(version=2, kind='historical', query=f'According to revision 0, what was the access code for {entity}?', answer=old, record=0),
        dict(version=2, kind='added_entity', query=f'What is the current access code for {added}?', answer=extra, record=2),
    ]
    episode = dict(id=f'memory-update-{budget}-{seed}', budget=budget, seed=seed,
                   versions=[v0, v1, v2], queries=questions)
    validate_episode(episode)
    return episode


def validate_episode(ep):
    versions = ep['versions']
    if len(versions) != 3 or [q['version'] for q in ep['queries']] != [0, 1, 2, 2, 2]:
        raise ValueError('version/query schedule differs')
    for earlier, later in zip(versions, versions[1:]):
        n = earlier['prefix_tokens_count']
        if not n < later['prefix_tokens_count'] or earlier['tokens'][:n] != later['tokens'][:n]:
            raise ValueError('archive update changed historical token addresses')
    v0, v1, v2 = versions
    a, b = v0['record_spans'][0]
    if not 128 <= a < b <= v0['prefix_tokens_count'] - RECENT:
        raise ValueError('old fact must start outside sink and recent memory')
    if v0['records'] != v1['records'][:1] or v1['records'] != v2['records']:
        raise ValueError('unexpected record insertion or replacement')
    for i in (1, 2):
        a, b = v1['record_spans'][i]
        if not v0['prefix_tokens_count'] <= a < b <= v1['prefix_tokens_count']:
            raise ValueError('new facts leaked into the initial archive')
        if a < v1['prefix_tokens_count'] - RECENT or v2['record_spans'][i] != [a, b]:
            raise ValueError('hot update position differs')
        if b > v2['prefix_tokens_count'] - RECENT:
            raise ValueError('aged correction is still in recent memory')
    if v2['prefix_tokens_count'] > ep['budget'] - RESERVE:
        raise ValueError('no room for counted reasoning')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    expected = json.loads(Path('research/memory-update-fixture-expectations.json').read_text())
    pinned = next(m for m in expected['models'] if m['model_id'] == args.model)
    tok = AutoTokenizer.from_pretrained(args.model, revision=pinned['revision'],
        local_files_only=True, trust_remote_code=False)
    tok.apply_chat_template = partial(tok.apply_chat_template, enable_thinking=False)
    episodes = [make_episode(tok, n, s) for n in BUDGETS for s in SEEDS]
    if (file_sha(__file__) != expected['protocol_sha256']
            or digest(episodes) != pinned['fixture_sha256']
            or digest(tok.chat_template) != pinned['chat_template_sha256']):
        raise ValueError('archive-update fixture differs from declaration')
    print(json.dumps(dict(model=args.model, fixture_sha256=digest(episodes),
        episodes=len(episodes), rows=len(episodes)*len(policies())*5,
        prefix_lengths=[[v['prefix_tokens_count'] for v in ep['versions']] for ep in episodes],
        scope='tokenizer-only plan; generator and incremental-KV validation pending')))


if __name__ == '__main__':
    main()

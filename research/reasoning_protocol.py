"""Frozen task construction and public control messages for multi-round KV tests.

No generator inference here. Ground truth belongs to fixture/evaluation data;
query selection uses only the question and the model's preceding response.
"""
import json
import math
import random

from tiered_kv.evals import char_span, render
from tiered_kv.types import digest


TASKS = ('literal', 'paraphrase', 'linked_fact', 'sum_two_facts')
SEEDS = (157, 163)
ROUND_COUNTS = (1, 2, 4)
RESERVE = 2048
WORKING_KV = {'W': 2000, 'C': 0, 'R': 128, 'k': 16, 'sink_tokens': 128}
INTRO = 'Read the following archive. Use its records to answer the question.\n\n'


def round_request(question, turn, rounds, direct=False):
    if rounds not in ROUND_COUNTS or not 0 <= turn < rounds:
        raise ValueError('invalid declared round')
    if direct:
        if rounds != 1:
            raise ValueError('direct answer has exactly one round')
        return question + '\nReturn only the answer, without explanation.'
    final = turn == rounds - 1
    return (question + '\n'
            + (f'Review round {turn + 1} of {rounds}. Review your preceding draft using the available archive memory. '
               if turn else f'Round 1 of {rounds}. Use the available archive memory. ')
            + 'Return one JSON object with exactly three string fields: "answer", "search", "notes". '
            + 'Put your current answer in "answer" and a short evidence-based explanation in "notes". '
            + ('This is the final round; set "search" to an empty string.' if final else
               'In "search", name the entity or fact whose archived memory you need to inspect next. '
               'Do not invent missing facts. Another round will follow.'))


def parse_response(text):
    """No gold-dependent repair, substring matching, or best-answer selection."""
    text = text.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4].strip()
    elif text.startswith('```\n') and text.endswith('\n```'):
        text = text[4:-4].strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if (not isinstance(value, dict) or set(value) != {'answer', 'search', 'notes'}
            or not all(isinstance(v, str) for v in value.values())):
        return None
    return {k: v.strip() for k, v in value.items()}


def next_search(question, response, tokenizer, cap=64):
    """Bound query-encoder input; invalid/empty requests fall back to the question."""
    parsed = parse_response(response)
    proposed = parsed['search'] if parsed and parsed['search'] else question
    ids = tokenizer.encode(proposed, add_special_tokens=False)
    return tokenizer.decode(ids[:cap], skip_special_tokens=True), {
        'response_valid': parsed is not None,
        'used_model_search': bool(parsed and parsed['search']),
        'original_query_tokens': len(ids), 'query_tokens': min(len(ids), cap),
        'query_truncated': len(ids) > cap,
    }


def policies():
    """Primary per-round budget plus an equal-total-output-budget control.

4 x 128 is shared between schedules and runs once per attention policy.
No token budget is claimed to equal compute across different attention lengths.
"""
    rows = [{'attention': 'stock', 'rounds': 1, 'cap_per_round': 128,
             'format': 'direct', 'schedule_labels': ['direct_vanilla']}]
    for attention in ('stock', 'sparse'):
        for rounds, cap, labels in (
                (1, 128, ['128_per_round']), (2, 128, ['128_per_round']),
                (4, 128, ['128_per_round', '512_total']),
                (1, 512, ['512_total']), (2, 256, ['512_total'])):
            rows.append({'attention': attention, 'rounds': rounds,
                         'cap_per_round': cap, 'format': 'json', 'schedule_labels': labels})
    for row in rows:
        row['id'] = f"{row['attention']}-{row['format']}-r{row['rounds']}-cap{row['cap_per_round']}"
    return rows


def _task(seed, kind):
    rng = random.Random(f'tiered-kv-reasoning-{seed}-{kind}')
    name = lambda prefix: prefix + ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(10))
    item, station, other = name('PROJECT_'), name('STATION_'), name('OTHER_')
    code = str(rng.randrange(10000000, 100000000))
    if kind == 'literal':
        records = [f'The access code for {item} is {code}.',
                   f'The archive entry for {other} concerns blue folders.']
        query = f'What is the access code for {item}?'
        required, answer = [0], code
    elif kind == 'paraphrase':
        records = [f'The emergency radio channel at the seaside watchpost is {code}.',
                   f'The archive entry for {other} concerns blue folders.']
        query = 'Which eight-digit frequency should rescuers use to contact the coastal observation station?'
        required, answer = [0], code
    elif kind == 'linked_fact':
        records = [f'The designated station for {item} is {station}.',
                   f'The access code for {station} is {code}.']
        query = f'What is the access code for the designated station of {item}?'
        required, answer = [0, 1], code
    elif kind == 'sum_two_facts':
        a, b = rng.randrange(100, 1000), rng.randrange(100, 1000)
        records = [f'The eastern depot holds {a} crates for shipment {item}.',
                   f'The western depot holds {b} crates for shipment {item}.']
        query = f'How many crates for shipment {item} are held by the eastern and western depots in total?'
        required, answer = [0, 1], str(a + b)
    else:
        raise ValueError(kind)
    # Swap the two widely separated locations for the second seed, before any
    # model execution. A linked lookup must work in either archive direction.
    if seed == SEEDS[1]:
        records.reverse()
        required = [1 - i for i in required]
    return records, query, answer, required


def make_fixture(tok, budget, seed, kind):
    """Whole records at separated archive depths; no truncation of needed facts.

Context budgets reserve 2048 tokens for all question/review/generation steps.
The archive fills the remaining budget to within one distractor line. Actual
lengths and positions are recorded; nominal budgets are not exact input lengths.
"""
    if budget not in (8192, 32768):
        raise ValueError('initial multi-round cohort uses 8K and 32K budgets')
    records, query, answer, required = _task(seed, kind)
    rng = random.Random(f'tiered-kv-reasoning-filler-{seed}-{kind}')
    filler = []
    for i in range(budget // 8):
        code = str(rng.randrange(10000000, 100000000))
        while code == answer:
            code = str(rng.randrange(10000000, 100000000))
        filler.append(f'The access code for CABINET_{i:06d} is {code}. Its report is stored with blue folders.\n')

    def assemble(n):
        first, second = n * 15 // 100, n * 60 // 100
        archive = (INTRO + ''.join(filler[:first]) + records[0] + '\n'
                   + ''.join(filler[first:second]) + records[1] + '\n'
                   + ''.join(filler[second:n]) + '\n')
        # A real query follows this prefix when serving; no answer or reasoning
        # transcript is present during immutable archive construction.
        rendered = render(tok, archive + query)
        encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        qstart = rendered.rindex(query)
        prefix_count = char_span(encoded['offset_mapping'], qstart, qstart + len(query))[0]
        suffix = rendered[qstart + len(query):]
        if (encoded['input_ids'][:prefix_count]
                + tok.encode(query + suffix, add_special_tokens=False) != encoded['input_ids']):
            raise ValueError('query must be exactly token-concatenable')
        return rendered, encoded, prefix_count, suffix

    target = budget - RESERVE
    lo, hi = 0, len(filler)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if assemble(mid)[2] <= target:
            lo = mid
        else:
            hi = mid - 1
    text, encoded, prefix_count, suffix = assemble(lo)
    if not target - 128 <= prefix_count <= target:
        raise ValueError('archive length does not fill the declared band')
    spans = []
    for record in records:
        if text.count(record) != 1:
            raise ValueError('ambiguous fixture fact')
        start = text.index(record)
        spans.append(char_span(encoded['offset_mapping'], start, start + len(record)))
    if not all(128 <= a < b < prefix_count - WORKING_KV['W'] for a, b in spans):
        raise ValueError('all fact records must be outside sinks and initial recent memory')
    ex = {'id': f'reasoning-{kind}-{budget}-{seed}', 'task': kind, 'budget': budget,
          'seed': seed, 'query': query, 'answer': answer, 'records': records,
          'record_spans': spans, 'required_record_indices': required,
          'tokens': encoded['input_ids'], 'prefix_tokens_count': prefix_count,
          'first_suffix': suffix, 'token_hash': digest(encoded['input_ids'])}
    return ex


def latest_completed_at(rounds, seconds=None, generated_tokens=None):
    """Retrospective budget curve: latest finished round, never best correctness.

These curves are censored observations, not timed interrupted-execution runs.
An unfinished first round yields no completed answer at that budget.
"""
    eligible = [r for r in rounds
                if (seconds is None or r['cumulative_request_s'] <= seconds)
                and (generated_tokens is None or r['cumulative_generated_tokens'] <= generated_tokens)]
    return eligible[-1] if eligible else None


def zero_loss_upper_bound(n, alpha=.05):
    """One-sided exact binomial bound, only for zero losses in n independent pairs."""
    if not isinstance(n, int) or n < 1 or not 0 < alpha < 1:
        raise ValueError('positive independent case count and valid alpha required')
    return -math.expm1(math.log(alpha) / n)

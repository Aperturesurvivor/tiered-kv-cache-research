import sqlite3

import numpy as np
import pytest

from research.reasoning_protocol import SEEDS, TASKS
from research.reasoning_run import trial_plan
from research.semantic_reasoning import diagnostic_policies, semantic_metadata_index


def test_semantic_diagnostic_has_matched_controls_and_separate_hundred_row_plan():
    examples = [{'id': f'{n}-{s}-{t}', 'task': t, 'seed': s}
                for n in (8192, 32768) for s in SEEDS for t in TASKS]
    plan = trial_plan(examples, 'semantic_exact')
    assert len(plan) == 100
    assert sum(r['purpose'] == 'quality' for r in plan) == 80
    assert sum(r['purpose'] == 'repeated_timing' for r in plan) == 20
    assert len({(r['example_id'], r['policy']['id'], r['trial']) for r in plan}) == 100
    assert trial_plan(examples, 'semantic_exact') == plan
    policies = diagnostic_policies()
    for sparse in (p for p in policies if p['attention'] == 'sparse'):
        stock = next(p for p in policies if p['id'] == sparse['id'].replace('sparse-semantic', 'stock'))
        assert all(sparse[k] == stock[k] for k in ('rounds', 'cap_per_round', 'format'))
    assert len(trial_plan(examples)) == 220
    with pytest.raises(ValueError):
        trial_plan(examples, 'invented')


def test_semantic_index_excludes_recent_future_text_and_checks_resume_identity(tmp_path):
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ''.join(map(chr, ids))

    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.extend(texts)
            return np.array([[0., 1.] if 'TARGET' in t else [1., 0.] for t in texts], np.float32)

    text = list('A'*2300)
    text[160:166] = 'TARGET'
    text[301:314] = 'FUTURE_SECRET'
    ex = {'id': 'test', 'tokens': list(map(ord, text)), 'prefix_tokens_count': 2300}
    encoder = Encoder()
    index = semantic_metadata_index(Tokenizer(), ex, tmp_path, encoder, 'encoder-frozen')
    try:
        assert index.search('TARGET', None, 4, 'semantic_exact') == [1, 0, 2]
        assert not any('FUTURE_SECRET' in t for t in encoder.calls)
        assert index.last_search['query_vector'] == [0., 1.]
        assert index.last_search['scanned_blocks'] == 3
        assert index.index.db.execute('SELECT id,start,end FROM chunks ORDER BY id').fetchall() == [(0, 0, 128), (1, 128, 256), (2, 256, 301)]
        with pytest.raises(ValueError):
            index.search('TARGET', None, 4, 'lexical_rare')
    finally:
        index.close()
    before = len(encoder.calls)
    reopened = semantic_metadata_index(Tokenizer(), ex, tmp_path, encoder, 'encoder-frozen')
    assert len(encoder.calls) == before  # index payload is reused, not re-encoded
    reopened.close()
    with pytest.raises(ValueError, match='identity/checksum'):
        semantic_metadata_index(Tokenizer(), ex, tmp_path, encoder, 'different-weights')
    path = tmp_path/'test-semantic-cold.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('UPDATE chunks SET text=? WHERE id=0', ('altered source text',))
    with pytest.raises(ValueError, match='identity/checksum'):
        semantic_metadata_index(Tokenizer(), ex, tmp_path, encoder, 'encoder-frozen')


def test_semantic_index_refuses_unfinished_payload(tmp_path):
    (tmp_path/'test-semantic-cold.sqlite3').write_bytes(b'unfinished')
    with pytest.raises(RuntimeError, match='incomplete semantic index'):
        semantic_metadata_index(None, {'id':'test', 'prefix_tokens_count':2300}, tmp_path, None, 'encoder')

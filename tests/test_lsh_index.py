import sqlite3

import numpy as np
import pytest

from research.lsh_index import DiskLSHIndex, LSHConfig, priority


def test_persistent_append_and_exact_reranking_of_visited_candidates(tmp_path):
    path = tmp_path/'index.sqlite3'
    config = LSHConfig(dimensions=3, tables=4, bits=2, posting_cap=16, rerank_cap=16)
    index = DiskLSHIndex(path, create=True, config=config)
    vectors = np.array([[1., 0, 0], [1, .1, 0], [1, .2, 0]])
    index.add([(i, i*128, (i+1)*128) for i in range(2)], vectors[:2])
    index.close()
    index = DiskLSHIndex(path)
    index.add([(2, 256, 384)], vectors[2:])
    selected = index.search(vectors[2], 3)
    assert selected == [2, 1, 0]
    assert index.last_search['reranked_vectors'] == 3
    index.close()
    reopened = DiskLSHIndex(path, readonly=True, config=config)
    assert reopened.search(vectors[2], 3) == selected
    assert reopened.db.execute('SELECT id,start,end FROM blocks ORDER BY id').fetchall() == [(0, 0, 128), (1, 128, 256), (2, 256, 384)]
    with pytest.raises(sqlite3.OperationalError):
        reopened.add([(3, 384, 512)], vectors[:1])
    reopened.close()


def test_overfull_buckets_bound_work_and_expose_loss(tmp_path):
    c = LSHConfig(dimensions=2, tables=3, bits=3, posting_cap=4, rerank_cap=3)
    index = DiskLSHIndex(tmp_path/'index', create=True, config=c)
    ids = list(reversed(range(100)))
    index.add([(i, i, i+1) for i in ids], np.tile([1., 0], (100, 1)))
    selected = index.search(np.array([1., 0]), 3)
    assert selected == sorted(sorted(ids, key=lambda i: (priority(i), i))[:3])
    d = index.last_search
    assert d['truncated_buckets'] == c.tables
    assert d['posting_rows_read'] == c.tables*(c.posting_cap+1)
    assert d['posting_rows_read'] <= d['posting_rows_bound']
    assert d['unique_candidates'] == 4 and d['reranked_vectors'] == 3
    assert not d['scan_fallback']
    plan = index.db.execute('EXPLAIN QUERY PLAN SELECT block_id FROM buckets WHERE table_id=? AND signature=? ORDER BY priority,block_id LIMIT ?', (0, 0, 5)).fetchall()
    assert any('SEARCH buckets USING PRIMARY KEY' in row[-1] for row in plan)
    assert all('SCAN' not in row[-1] and 'TEMP B-TREE' not in row[-1] for row in plan)
    index.close()


def test_invalid_append_rolls_back_entire_batch(tmp_path):
    index = DiskLSHIndex(tmp_path/'index', create=True, config=LSHConfig(dimensions=2))
    index.add([(0, 0, 1)], [[1, 0]])
    for vectors in ([[0, 0]], [[float('nan'), 1]], [[float('inf'), 1]], [[1, 0, 0]], []):
        with pytest.raises(ValueError):
            index.add([(1, 1, 2)], vectors)
    with pytest.raises(ValueError):
        index.add([(1, 1, 2), (2, 2, 3)], [[1, 0]])
    with pytest.raises(sqlite3.IntegrityError):
        index.add([(1, 1, 2), (0, 2, 3)], [[1, 0], [1, 0]])
    assert index.db.execute('SELECT count(*) FROM blocks').fetchone()[0] == 1
    assert index.db.execute('SELECT count(*) FROM buckets').fetchone()[0] == 16
    with pytest.raises(ValueError):
        index.search([0, 0], 1)
    index.close()


def test_streamed_reference_matches_independent_dense_float64_ranking(tmp_path):
    from research.lsh_scaling import exact
    rng = np.random.default_rng(269)
    vectors = rng.normal(size=(173, 12))
    query = rng.normal(size=12)
    index = DiskLSHIndex(tmp_path/'index', create=True, config=LSHConfig(dimensions=12))
    index.add([(i, i, i+1) for i in range(len(vectors))], vectors)
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = np.einsum('ij,j->i', normalized, query / np.linalg.norm(query), optimize=False)
    assert np.isfinite(scores).all()
    expected = sorted(range(len(vectors)), key=lambda i: (-scores[i], i))[:16]
    selected, work = exact(index, query, 16)
    assert selected == expected
    assert work['vectors_read'] == len(vectors)
    index.close()

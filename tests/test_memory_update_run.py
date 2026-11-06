import pytest

from research.memory_update_run import audit_index, question_fixture, schedule
from research.semantic_index import metadata_index
from research.resources import logical_store_reads
from types import SimpleNamespace


class Tokenizer:
    def decode(self, ids, **kwargs): return ''.join(map(chr,ids))


def test_request_read_counter_includes_every_ancestor_once():
    initial = SimpleNamespace(read_bytes=100,prefix=None)
    hot = SimpleNamespace(read_bytes=20,prefix=initial)
    aged = SimpleNamespace(read_bytes=30,prefix=hot)
    request = SimpleNamespace(read_bytes=4,prefix=aged)
    assert logical_store_reads(request) == 154
    assert logical_store_reads(aged) == 150


def test_version_index_cannot_include_appended_future_text(tmp_path):
    tok = Tokenizer()
    ex = dict(id='v0',tokens=list(map(ord,'A'*129+'SECRET_FUTURE_FACT')),prefix_tokens_count=129)
    index = metadata_index(tok,ex,tmp_path,32)
    try:
        assert audit_index(index,tok,ex) == dict(blocks=2,visible_positions=129,exact_version_metadata=True)
        assert all('SECRET' not in r[0] for r in index.db.execute('SELECT text FROM chunks'))
        index.db.execute("UPDATE chunks SET text=text||'SECRET_FUTURE_FACT' WHERE id=1")
        with pytest.raises(ValueError,match='future version metadata'):
            audit_index(index,tok,ex)
    finally:
        index.close()


def test_questions_bind_to_current_version_with_fixed_policy_opportunities():
    ep = dict(id='episode',versions=[dict(id=f'v{i}',tokens=[i],record_spans=[[i,i+1]]) for i in range(3)],
        queries=[dict(version=v,kind=f'q{i}',query='question',answer=str(i),record=0)
                 for i,v in enumerate((0,1,2,2,2))])
    for qi, vi in enumerate((0,1,2,2,2)):
        ex = question_fixture(ep,qi)
        assert ex['tokens'] == [vi] and ex['answer'] == str(qi)
        assert ex['id'] == f'episode-q{qi}' and ex['required_record_indices'] == [0]
    plan = schedule([ep])
    assert plan == schedule([ep])
    assert len(plan) == 4 and sum(i['queries'] for i in plan) == 20
    assert len({i['policy']['id'] for i in plan}) == 4

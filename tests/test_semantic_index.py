import numpy as np


def test_disk_metadata_returns_addresses_without_changing_payload(tmp_path):
    from research.semantic_index import DiskTextIndex
    path=tmp_path/'metadata.sqlite3'
    index=DiskTextIndex(path,create=True)
    rows=[(0,0,128,'The code for ITEM_ABC is 123.'),
          (4,512,640,'The code for ITEM_XYZ is 987.'),
          (9,1152,1280,'A river flows through the town.')]
    index.add(rows,np.array([[1,0],[0,1],[-1,0]],dtype=np.float32))
    index.close()
    reopened=DiskTextIndex(path)
    try:
        assert reopened.search('ITEM_ABC',None,1,'lexical')==[0]
        assert reopened.search('ITEM_XYZ',None,1,'lexical')==[4]
        assert reopened.search('What is the code for ITEM_XYZ?',None,4,'lexical_rare')==[4]
        assert reopened.last_search['posting_documents']==1
        assert reopened.search('unseen',np.array([0,1],np.float32),1,'semantic')==[4]
        assert reopened.search('ITEM_ABC',np.array([1,0],np.float32),1,'hybrid_bounded')==[0]
        assert reopened.search('no_match',np.array([1,0],np.float32),1,'hybrid_bounded')==[]
        assert reopened.db.execute('SELECT id,start,end,text FROM chunks ORDER BY id').fetchall()==rows
    finally:
        reopened.close()


def test_rare_posting_candidate_cap_is_explicit_and_does_not_scan_common_postings(tmp_path):
    from research.semantic_index import DiskTextIndex
    index=DiskTextIndex(tmp_path/'bounded.sqlite3',create=True)
    index.add([(i,i*128,(i+1)*128,f'The access code for cabinet_{i} is recorded here.') for i in range(100)],np.ones((100,2),np.float32))
    try:
        assert index.search('What is the access code for cabinet_99?',None,4,'lexical_rare')==[99]
        assert index.last_search['candidates']==1 and not index.last_search['truncated']
        result=index.search('access code',None,4,'lexical_rare')
        assert result==[0,1,2,3]
        assert index.last_search['candidates']==64 and index.last_search['truncated']
    finally:
        index.close()


def test_overlapping_metadata_covers_boundary_without_indexing_future_question(tmp_path):
    from research.semantic_index import metadata_index,with_neighbors
    class Tokens:
        def decode(self,ids,**kwargs):
            return ''.join(chr(i) for i in ids)
    text='x '*60+'ITEM_BOUNDARY is 12345678. '+'x '*60
    cutoff=200
    ex={'id':'boundary','tokens':list(map(ord,text[:cutoff]+' FUTURE_QUESTION_SECRET')),'prefix_tokens_count':cutoff}
    index=metadata_index(Tokens(),ex,tmp_path,32)
    try:
        selected=index.search('ITEM_BOUNDARY',None,4,'lexical_rare')
        expanded=with_neighbors(selected,(cutoff+127)//128)
        assert {0,1}<=set(expanded) and len(expanded)<=4
        assert index.search('FUTURE_QUESTION_SECRET',None,4,'lexical_rare')==[]
        assert index.db.execute('SELECT max(end) FROM chunks').fetchone()[0]==cutoff
    finally:
        index.close()
    reopened=metadata_index(Tokens(),ex,tmp_path,32)
    try:
        assert reopened.search('ITEM_BOUNDARY',None,4,'lexical_rare')==selected
    finally:
        reopened.close()

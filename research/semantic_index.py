"""RAG-style text metadata that returns exact disk KV block addresses.

No retrieved text is inserted into the generation prompt. Semantic vectors and
lexical postings live in SQLite; semantic search streams bounded batches from
disk (exact scan, not ANN). The frozen encoder runs on CPU.
"""
import heapq
from collections import Counter
import math
import re
import sqlite3
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,AutoModel

ENCODER_ID='sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


class Encoder:
    def __init__(self):
        snapshot=snapshot_download(ENCODER_ID,revision=ENCODER_REVISION,
            allow_patterns=['model.safetensors','config.json','tokenizer.json',
                            'tokenizer_config.json','special_tokens_map.json','vocab.txt'])
        self.tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
        self.model=AutoModel.from_pretrained(snapshot,local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='sdpa').eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self,texts):
        # Split, rather than silently truncate, blocks over the encoder's
        # published 256-wordpiece limit. Aggregate normalized segment vectors.
        pieces=[]
        owners=[]
        for owner,text in enumerate(texts):
            ids=self.tokenizer.encode(text,add_special_tokens=False)
            for start in range(0,max(1,len(ids)),222):
                pieces.append(self.tokenizer.prepare_for_model(ids[start:start+254],add_special_tokens=True))
                owners.append(owner)
        vectors=[]
        for start in range(0,len(pieces),16):
            inputs=self.tokenizer.pad(pieces[start:start+16],padding=True,return_tensors='pt')
            hidden=self.model(**inputs).last_hidden_state
            mask=inputs['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            vectors.extend(torch.nn.functional.normalize(pooled,dim=1).cpu().numpy())
        result=np.zeros((len(texts),self.model.config.hidden_size),np.float32)
        for owner,vector in zip(owners,vectors):
            result[owner]+=vector
        result/=np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)
        return result


class DiskTextIndex:
    def __init__(self,path,create=False):
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA cache_size=-4096')
        self.db.execute('PRAGMA mmap_size=0')
        self.db.execute('PRAGMA temp_store=FILE')
        if create:
            self.db.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY, start INTEGER, end INTEGER, text TEXT, vector BLOB, term_count INTEGER)')
            self.db.execute('CREATE TABLE postings(term TEXT, chunk_id INTEGER, frequency INTEGER, PRIMARY KEY(term,chunk_id)) WITHOUT ROWID')
            self.db.execute('CREATE TABLE statistics(term TEXT PRIMARY KEY, documents INTEGER) WITHOUT ROWID')
            self.db.execute('CREATE TABLE totals(documents INTEGER, terms INTEGER)')
            self.db.execute('INSERT INTO totals VALUES(0,0)')

    @staticmethod
    def words(text):
        return re.findall(r'\w+',text.casefold(),flags=re.UNICODE)

    def add(self,rows,vectors):
        for (block,start,end,text),vector in zip(rows,vectors):
            terms=Counter(self.words(text))
            length=sum(terms.values())
            self.db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?)',(block,start,end,text,np.asarray(vector,dtype='<f4').tobytes(),length))
            self.db.executemany('INSERT INTO postings VALUES(?,?,?)',[(term,block,count) for term,count in terms.items()])
            self.db.executemany('INSERT OR IGNORE INTO statistics VALUES(?,0)',[(term,) for term in terms])
            self.db.executemany('UPDATE statistics SET documents=documents+1 WHERE term=?',[(term,) for term in terms])
            self.db.execute('UPDATE totals SET documents=documents+1,terms=terms+?',(length,))
        self.db.commit()

    def search(self,query,vector,k,method='hybrid'):
        self.last_search={}
        if method not in ('semantic','lexical','lexical_rare','hybrid','hybrid_bounded') or k<=0:
            raise ValueError('invalid search method/budget')
        lexical=[]
        semantic=[]
        if method in ('lexical','lexical_rare','hybrid','hybrid_bounded'):
            words=sorted(set(self.words(query)))
            documents,terms=self.db.execute('SELECT documents,terms FROM totals').fetchone()
            if words and documents:
                stats=self.db.execute('SELECT term,documents FROM statistics WHERE term IN ('+','.join('?' for _ in words)+')',words).fetchall()
                candidates=None
                if method=='lexical_rare' and stats:
                    # Question-triggered entity lookup. The rarest indexed query
                    # term supplies at most 64 candidates; common terms cannot
                    # force a scan of every archived block. This is approximate
                    # when that posting list exceeds the cap, not general ANN.
                    rare,min_count=min(stats,key=lambda pair:(pair[1],pair[0]))
                    candidates=[r[0] for r in self.db.execute('SELECT chunk_id FROM postings WHERE term=? ORDER BY chunk_id LIMIT 64',(rare,))]
                    self.last_search={'candidate_term':rare,'posting_documents':min_count,'candidates':len(candidates),
                                      'truncated':min_count>len(candidates),'candidate_cap':64}
                weighted=[(term,max(1e-6,math.log((documents-count+.5)/(count+.5)))) for term,count in stats]
                if weighted:
                    # BM25 k1=1.2,b=.75, positive IDF floor as documented by
                    # SQLite FTS5, implemented with portable indexed postings.
                    sql=('WITH q(term,weight) AS (VALUES '+','.join('(?,?)' for _ in weighted)+') '
                         'SELECT p.chunk_id,SUM(q.weight*p.frequency*2.2/'
                         '(p.frequency+1.2*(.25+.75*c.term_count/?))) AS score '
                         'FROM q JOIN postings p ON p.term=q.term JOIN chunks c ON c.id=p.chunk_id ')
                    bindings=[v for pair in weighted for v in pair]+[max(terms/documents,1)]
                    if candidates is not None:
                        sql+='WHERE p.chunk_id IN ('+','.join('?' for _ in candidates)+') '
                        bindings+=candidates
                    sql+='GROUP BY p.chunk_id ORDER BY score DESC,p.chunk_id LIMIT ?'
                    bindings+=[max(k,64)]
                    lexical=[r[0] for r in self.db.execute(sql,bindings)]
        if method=='hybrid_bounded' and lexical:
            ranked=[]
            for block in lexical:
                raw=self.db.execute('SELECT vector FROM chunks WHERE id=?',(block,)).fetchone()[0]
                ranked.append((float(np.frombuffer(raw,dtype='<f4')@vector),-block))
            semantic=[-block for _,block in sorted(ranked,reverse=True)]
        if method in ('semantic','hybrid'):
            best=[]
            cursor=self.db.execute('SELECT id,vector FROM chunks ORDER BY id')
            while rows:=cursor.fetchmany(64):
                values=np.stack([np.frombuffer(r[1],dtype='<f4') for r in rows])
                scores=values@vector
                if not np.isfinite(scores).all():
                    raise ValueError('nonfinite semantic similarity')
                for (block,_),score in zip(rows,scores):
                    heapq.heappush(best,(float(score),-block))
                    if len(best)>max(k,64):
                        heapq.heappop(best)
            semantic=[-block for _,block in sorted(best,reverse=True)]
        if method=='semantic':
            return semantic[:k]
        if method in ('lexical','lexical_rare'):
            return lexical[:k]
        scores={}
        for ranking in (semantic,lexical):
            for rank,block in enumerate(ranking,1):
                scores[block]=scores.get(block,0)+1/(60+rank)
        return sorted(scores,key=lambda block:(-scores[block],block))[:k]

    def close(self):
        self.db.close()


def metadata_index(tok,ex,root,padding):
    from tiered_kv.store import atomic_json,file_sha
    from tiered_kv.types import digest
    import json
    path=root/(ex['id']+f'-metadata-{padding}.sqlite3')
    marker=path.with_suffix('.complete.json')
    if path.exists() and not marker.exists():
        raise RuntimeError('incomplete persistent index retained: '+str(path))
    if not path.exists():
        index=DiskTextIndex(path,create=True)
        for offset in range(0,ex['prefix_tokens_count'],128):
            start=max(0,offset-padding)
            end=min(offset+128+padding,ex['prefix_tokens_count'])
            index.add([(offset//128,start,end,tok.decode(ex['tokens'][start:end],skip_special_tokens=False))],np.zeros((1,1),np.float32))
        index.close()
        atomic_json(marker,{'fixture_sha256':digest(ex),'padding':padding,'database_sha256':file_sha(path)})
    else:
        meta=json.loads(marker.read_text())
        if meta['fixture_sha256']!=digest(ex) or meta['padding']!=padding or meta['database_sha256']!=file_sha(path):
            raise ValueError('persistent index identity/checksum mismatch')
    return DiskTextIndex(path)


def with_neighbors(selected,block_count,budget=4):
    neighbors=[b+delta for b in selected for delta in (1,-1) if 0<=b+delta<block_count]
    return list(dict.fromkeys(selected+neighbors))[:budget]

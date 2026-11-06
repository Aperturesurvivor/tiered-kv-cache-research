"""32-page bounded candidate reranking for one diagnostic; linear metadata scan."""
import math
import time
import numpy as np
import torch
from tiered_kv.types import KVBatch
from research.attention_page_cache import CompletePageCacheStore
from research.attention_quest_reference import QuestBoundIndex, QuestBoundCache
from research.attention_representations import _lse, _shared_rank


class CandidatePageStore(CompletePageCacheStore):
    """Keep the 32 candidates while serving a final subset, without rereading."""
    def candidate_read(self, layer, ids, cutoff):
        return super().read(layer, ids, cutoff)

    def read(self, layer, block_ids, before_pos):
        ids=tuple(int(x) for x in block_ids)
        pages=self.pages[layer]
        if len(set(ids))!=len(ids) or any(b not in pages or b*self.R<self.sink_tokens or (b+1)*self.R>before_pos for b in ids):
            raise ValueError('final pages must be an eligible cached candidate subset')
        if not ids:return KVBatch.empty(self.h,self.d,device=self.cache_device)
        ordered=[pages[b] for b in sorted(ids)]
        return KVBatch(np.concatenate([b.ids for b in ordered]),torch.cat([b.k for b in ordered],2),torch.cat([b.v for b in ordered],2))


class CandidateRerankIndex(QuestBoundIndex):
    def __init__(self,layers,h,d,page,sink,store,pool=32):
        super().__init__(layers,h,d,page,sink)
        self.store=store;self.pool=pool
        self.rerank_stats=[dict(calls=0,candidate_key_gpu_to_cpu_bytes=0,rerank_query_gpu_to_cpu_bytes=0,candidate_read_s=0.,candidate_qk_and_rank_s=0.,candidate_counts={},final_counts={}) for _ in range(layers)]
        self.last_candidates=[None]*layers

    def search(self,layer,Q_rot,k):
        fixed=self.pending[layer]
        candidates,_=super().search(layer,Q_rot,self.pool)
        # Use a single shared per-layer pool; page IDs are canonicalized by store.read.
        stats=self.rerank_stats[layer];stats['calls']+=1
        start=time.perf_counter();batch=self.store.candidate_read(layer,candidates,self.next_pos[layer])
        stats['candidate_read_s']+=time.perf_counter()-start
        self.last_candidates[layer]=candidates.tolist()
        if not len(candidates):return candidates,np.array([],np.float32)
        start=time.perf_counter()
        if Q_rot.device.type!='cpu':stats['rerank_query_gpu_to_cpu_bytes']+=Q_rot.numel()*4
        q=Q_rot.detach().float().cpu()[0,:,0,:]
        if batch.k.device.type!='cpu':stats['candidate_key_gpu_to_cpu_bytes']+=batch.k.numel()*batch.k.element_size()
        keys=batch.k[0].detach().cpu().float();heads=q.shape[0]
        logits=torch.bmm(q.reshape(self.h,heads//self.h,self.d),keys.transpose(1,2)).reshape(heads,len(candidates),self.R).numpy()/np.float32(math.sqrt(self.d))
        ids=np.sort(candidates)
        fixed_logits=torch.bmm(q.reshape(self.h,heads//self.h,self.d),fixed.float().transpose(1,2)).reshape(heads,-1).numpy()/np.float32(math.sqrt(self.d))
        chosen=_shared_rank(_lse(logits,axis=2),fixed_logits,ids,min(k,len(ids)))
        stats['candidate_qk_and_rank_s']+=time.perf_counter()-start
        for key,count in [('candidate_counts',len(ids)),('final_counts',len(chosen))]:
            stats[key][str(count)]=stats[key].get(str(count),0)+1
        return np.array(chosen,dtype=np.int64),np.zeros(len(chosen),np.float32)

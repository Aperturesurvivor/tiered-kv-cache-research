"""Separate device-summary integration; not a change to the frozen CPU policy.

Query-dependent Quest metadata scan stays on CPU. Exact candidate QK and the
fixed-key normalizer run on the input device and transfer only summaries back.
Numerical/routing equivalence on CUDA requires the separate recorded-Q replay.
"""
import math
import time
import numpy as np
import torch
from tiered_kv.cache import TieredCache
from research.attention_device_summaries import score_summaries
from research.attention_quest_reference import QuestBoundCache, QuestBoundIndex
from research.attention_rerank_live import CandidatePageStore
from research.attention_representations import _lse


class DeviceRerankIndex(QuestBoundIndex):
    """Same bound scan and shared final 32-to-15 routing, device QK summaries."""
    def __init__(self,layers,h,d,page,sink,store,pool=32):
        if not isinstance(store,CandidatePageStore) or page!=128 or sink!=128 or pool!=32 or store.cache_pages!=32:
            raise ValueError('device integration requires 128-token pages/sinks and a 32-page cache')
        super().__init__(layers,h,d,page,sink)
        self.store=store;self.pool=pool;self.last_candidates=[None]*layers
        self.device_stats=[dict(calls=0,query_d2h_bytes=0,query_d2h_copies=0,summary_d2h_bytes=0,summary_d2h_copies=0,
            helper_calls=0,validation_scalar_syncs=0,validation_scalar_d2h_logical_bytes=0,
            candidate_read_s=0.,bound_and_rank_s=0.,helper_s=0.,candidate_counts={},final_counts={},
            max_candidate_concat_bytes=0,max_fixed_input_bytes=0,max_helper_logits_bytes=0)
            for _ in range(layers)]

    def search(self,layer,Q_rot,k):
        if k!=15:raise ValueError('final page ceiling must be 15')
        fixed=self.pending.pop(layer)
        if Q_rot.ndim!=4 or Q_rot.shape[0]!=1 or Q_rot.shape[2]!=1 or Q_rot.shape[3]!=self.d or Q_rot.device!=fixed.device:
            raise ValueError('query/fixed geometry or device mismatch')
        self.searches[layer]+=1;stats=self.device_stats[layer];stats['calls']+=1
        ids=np.flatnonzero((self.counts[layer]==self.R)&(np.arange(len(self.counts[layer]))*self.R>=self.sink))
        self.stats[layer]['candidate_peak']=max(self.stats[layer]['candidate_peak'],len(ids))
        if not len(ids):
            self.last_candidates[layer]=[]
            # Empty pool still invalidates retained pages from a previous search.
            self.store.candidate_read(layer,[],self.next_pos[layer])
            for key in ('candidate_counts','final_counts'):stats[key]['0']=stats[key].get('0',0)+1
            return ids,np.array([],np.float32)
        q=Q_rot.detach()[0,:,0,:];heads=q.shape[0]
        if heads%self.h:raise ValueError('GQA mismatch')
        if q.device.type!='cpu':
            stats['query_d2h_bytes']+=q.numel()*4;stats['query_d2h_copies']+=1
        begin=time.perf_counter();grouped=q.float().cpu().numpy().reshape(self.h,heads//self.h,self.d)
        lo=self.sums[layer][ids];hi=self.maxima[layer][ids]
        upper=(np.einsum('kgd,kpd->kgp',np.maximum(grouped,0),hi.transpose(1,0,2))+
               np.einsum('kgd,kpd->kgp',np.minimum(grouped,0),lo.transpose(1,0,2))).reshape(heads,len(ids))/np.float32(math.sqrt(self.d))
        bound=upper+np.float32(math.log(self.R))
        stats['bound_and_rank_s']+=time.perf_counter()-begin
        def summary(keys):
            start=time.perf_counter();masses,normal,account=score_summaries(q,keys,fixed)
            stats['helper_s']+=time.perf_counter()-start;stats['helper_calls']+=1
            stats['validation_scalar_syncs']+=account['validation_scalar_syncs']
            stats['validation_scalar_d2h_logical_bytes']+=account['validation_scalar_host_logical_bytes']
            stats['max_helper_logits_bytes']=max(stats['max_helper_logits_bytes'],account['float32_logit_bytes'])
            stats['max_fixed_input_bytes']=max(stats['max_fixed_input_bytes'],account['fixed_key_input_bytes'])
            if q.device.type!='cpu':
                stats['summary_d2h_bytes']+=account['output_summary_bytes']
                stats['summary_d2h_copies']+=int(masses.numel()>0)+int(normal.numel()>0)
            return masses.detach().cpu().numpy(),normal.detach().cpu().numpy()
        empty=torch.empty((self.h,0,self.R,self.d),dtype=fixed.dtype,device=q.device)
        _,fixed_norm=summary(empty)
        begin=time.perf_counter();denom=np.logaddexp(fixed_norm,_lse(bound,axis=1))
        scores=np.exp(bound-denom[:,None]).sum(axis=0)
        candidates=ids[np.lexsort((ids,-scores))[:self.pool]]
        stats['bound_and_rank_s']+=time.perf_counter()-begin
        start=time.perf_counter();batch=self.store.candidate_read(layer,candidates,self.next_pos[layer])
        stats['candidate_read_s']+=time.perf_counter()-start;self.last_candidates[layer]=candidates.tolist()
        ordered=np.sort(candidates);keys=batch.k[0].reshape(self.h,len(ordered),self.R,self.d)
        stats['max_candidate_concat_bytes']=max(stats['max_candidate_concat_bytes'],batch.k.numel()*batch.k.element_size()+batch.v.numel()*batch.v.element_size())
        mass,fixed_norm_again=summary(keys)
        begin=time.perf_counter();denom=np.logaddexp(fixed_norm_again,_lse(mass,axis=1))
        selected_scores=np.exp(mass-denom[:,None]).sum(axis=0)
        selected=ordered[np.lexsort((ordered,-selected_scores))[:min(k,len(ordered))]]
        stats['bound_and_rank_s']+=time.perf_counter()-begin
        for key,count in (('candidate_counts',len(ordered)),('final_counts',len(selected))):
            stats[key][str(count)]=stats[key].get(str(count),0)+1
        return selected,np.zeros(len(selected),np.float32)


class DeviceQuestCache(QuestBoundCache):
    """Stage exact sink/boundary/recent/current keys on their query device."""
    def __init__(self,*args,**kwargs):
        if kwargs.get('initialize',True):raise ValueError('construct with initialize=False, install DeviceRerankIndex, then call _initialize')
        super().__init__(*args,**kwargs)
        if not isinstance(self.store,CandidatePageStore) or self.store.cache_pages!=32 or torch.device(self.store.cache_device)!=torch.device(self.device):
            raise ValueError('requires a 32-page CandidatePageStore on the attention device')
        self.recent_initialization_h2d_source_bytes=0
        self._read_baseline=[];store=self.store
        while store is not None:
            self._read_baseline.append(store.read_bytes);store=store.prefix
        c=self.config
        if c.W!=2000 or c.sink_tokens!=128 or c.R!=128 or c.C!=0 or c.k!=15 or c.retrieval=='fixed':
            raise ValueError('device integration requires W2000/sink128/R128/C0/k15 dynamic retrieval')
        self.transfer_stats=[dict(sink_h2d_bytes=0,boundary_h2d_bytes=0,boundary_logical_bytes=0,
            boundary_reads=0,attention_sink_kv_h2d_bytes=0,stage_s=0.,max_fixed_concat_bytes=0,max_attention_view_bytes=0) for _ in range(self.store.layers)]

    def _initialize(self):
        super()._initialize()
        if torch.device(self.device).type!='cpu':
            self.recent_initialization_h2d_source_bytes+=self.store.layers*min(self.position,self.config.W)*self.store.record_bytes

    def present(self,layer,q,k,v,pos):
        start=time.perf_counter();cutoff=max(0,pos-self.config.W+1)
        if cutoff<self.config.sink_tokens:raise ValueError('requires disjoint sink and recent keys')
        recent=self.recent.view(layer);keep=np.flatnonzero(recent.ids>=cutoff)
        if not np.array_equal(recent.ids[keep],np.arange(cutoff,pos)):raise ValueError('recent positions differ')
        stats=self.transfer_stats[layer];sink=self.normalizer_sinks[layer]
        if sink is None:
            raw=self.store.read_range(layer,0,self.config.sink_tokens).k[0]
            if raw.device!=q.device:stats['sink_h2d_bytes']+=raw.numel()*raw.element_size()
            sink=raw.to(q.device);self.normalizer_sinks[layer]=sink
        boundary_start=(cutoff//self.config.R)*self.config.R
        boundary=self.store.read_range(layer,boundary_start,cutoff).k[0]
        stats['boundary_reads']+=int(cutoff>boundary_start)
        stats['boundary_logical_bytes']+=(cutoff-boundary_start)*self.store.record_bytes
        if boundary.device!=q.device:stats['boundary_h2d_bytes']+=boundary.numel()*boundary.element_size()
        boundary=boundary.to(q.device);recent_k=recent.k[0,:,keep,:]
        if recent_k.device!=q.device or k.device!=q.device:raise ValueError('recent/current must remain on query device')
        fixed=torch.cat((sink,boundary,recent_k,k[0]),dim=1)
        stats['max_fixed_concat_bytes']=max(stats['max_fixed_concat_bytes'],fixed.numel()*fixed.element_size())
        self.index.stage(layer,fixed);stats['stage_s']+=time.perf_counter()-start
        prior_sink=self.sink_cache[layer]
        view=TieredCache.present(self,layer,q,k,v,pos)
        if prior_sink is not self.sink_cache[layer] and q.device.type!='cpu':
            stats['attention_sink_kv_h2d_bytes']+=len(self.sink_cache[layer].ids)*self.store.record_bytes
        stats['max_attention_view_bytes']=max(stats['max_attention_view_bytes'],view.k.numel()*view.k.element_size()+view.v.numel()*view.v.element_size())
        return view

    def device_accounting(self):
        """Logical payload accounting; allocator peaks and async timing require measurement."""
        chain=[];store=self.store
        while store is not None:
            chain.append(store.read_bytes);store=store.prefix
        return dict(layers=self.transfer_stats,index=self.index.device_stats,index_bytes=self.index.nbytes,
            sink_persistent_bytes=sum(x.numel()*x.element_size() for x in self.normalizer_sinks if x is not None),
            cached_kv_persistent_bytes=sum(b.k.numel()*b.k.element_size()+b.v.numel()*b.v.element_size()
                for layer in self.store.pages for b in layer.values()),
            cached_kv_page_capacity=self.store.cache_pages*self.store.layers,
            logical_store_read_bytes=sum(a-b for a,b in zip(chain,self._read_baseline)),logical_store_read_bytes_by_store=[a-b for a,b in zip(chain,self._read_baseline)],
            candidate_kv_h2d_bytes=(sum(x['cache_misses'] for x in self.store.stats)*self.store.R*self.store.record_bytes if torch.device(self.store.cache_device).type!='cpu' else 0),
            recent_persistent_bytes=sum(x.numel()*x.element_size() for x in self.recent.buffers),
            recent_initialization_h2d_source_bytes=self.recent_initialization_h2d_source_bytes,
            attention_sink_persistent_bytes=sum(x.k.numel()*x.k.element_size()+x.v.numel()*x.v.element_size() for x in self.sink_cache if x is not None),
            peak_cached_pages_total=self.store.peak_resident_pages_total,cache_stats=self.store.stats,
            scope='logical requested tensor payloads, not measured allocator peaks or PCIe traffic; successful cache misses imply H2D K/V bytes; recent initialization is logical source payload, not observed driver traffic; helper timings can be asynchronous; absorb counters remain separate')

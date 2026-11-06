"""Online sparse-Q page-bound reference; linear metadata scan, not ANN."""
import math
import time
import numpy as np
import torch
from tiered_kv.retrieval import BlockIndex
from tiered_kv.cache import TieredCache
from research.attention_representations import _lse


class QuestBoundIndex(BlockIndex):
    """Per-page fp32 coordinate min/max, populated only as keys are evicted."""
    def __init__(self,layers,h,d,page,sink):
        super().__init__(layers,h,d,page,capacity=0)
        self.sums=[np.empty((0,h,d),np.float32) for _ in range(layers)] # minima; frozen checkpoint alone is not resumable
        self.maxima=[np.empty((0,h,d),np.float32) for _ in range(layers)]
        self.sink=sink;self.pending={}
        self.stats=[dict(boundary_calls=0,boundary_logical_bytes=0,normalizer_key_copy_bytes=0,query_copy_bytes=0,evicted_key_copy_bytes=0,evicted_fp32_materialized_bytes=0,
                         stage_s=0.0,boundary_read_s=0.0,host_key_copy_s=0.0,selection_s=0.0,absorb_s=0.0,candidate_peak=0) for _ in range(layers)]

    def absorb(self,layer,evicted):
        begin=time.perf_counter();n=len(evicted.ids)
        if not n:return
        if not np.array_equal(evicted.ids,np.arange(self.next_pos[layer],self.next_pos[layer]+n)):
            raise ValueError('eviction ordering changed')
        need=int(evicted.ids[-1]//self.R)+1;old=len(self.counts[layer])
        if need>old:
            length=max(need,old*2,1); pad=length-old
            self.sums[layer]=np.pad(self.sums[layer],((0,pad),(0,0),(0,0)),constant_values=np.inf)
            self.maxima[layer]=np.pad(self.maxima[layer],((0,pad),(0,0),(0,0)),constant_values=-np.inf)
            self.counts[layer]=np.pad(self.counts[layer],(0,pad))
        self.stats[layer]['evicted_fp32_materialized_bytes']+=evicted.k.numel()*4
        if evicted.k.device.type!='cpu':
            self.stats[layer]['evicted_key_copy_bytes']+=evicted.k.numel()*4  # .float() precedes CPU transfer
        keys=evicted.k[0].detach().float().cpu().numpy().transpose(1,0,2)
        blocks=evicted.ids//self.R
        np.minimum.at(self.sums[layer],blocks,keys);np.maximum.at(self.maxima[layer],blocks,keys)
        np.add.at(self.counts[layer],blocks,1);self.next_pos[layer]+=n
        self.stats[layer]['absorb_s']+=time.perf_counter()-begin

    def stage(self,layer,fixed_keys):
        if layer in self.pending:raise ValueError('staged normalizer was not consumed')
        if fixed_keys.ndim!=3 or fixed_keys.shape[0]!=self.h or fixed_keys.shape[2]!=self.d or not fixed_keys.shape[1]:
            raise ValueError('normalizer shape mismatch')
        self.pending[layer]=fixed_keys

    def search(self,layer,Q_rot,k):
        begin=time.perf_counter();self.searches[layer]+=1
        fixed=self.pending.pop(layer)
        ids=np.flatnonzero((self.counts[layer]==self.R)&(np.arange(len(self.counts[layer]))*self.R>=self.sink))
        self.stats[layer]['candidate_peak']=max(self.stats[layer]['candidate_peak'],len(ids))
        if not len(ids):return ids,np.array([],np.float32)
        if Q_rot.device.type!='cpu':
            self.stats[layer]['query_copy_bytes']+=Q_rot.numel()*4  # .float() precedes CPU transfer
        q=Q_rot.detach().float().cpu()[0,:,0,:];heads=q.shape[0]
        if heads%self.h:raise ValueError('GQA mismatch')
        grouped=q.numpy().reshape(self.h,heads//self.h,self.d)
        lo=self.sums[layer][ids];hi=self.maxima[layer][ids]
        upper=(np.einsum('kgd,kpd->kgp',np.maximum(grouped,0),hi.transpose(1,0,2))+
               np.einsum('kgd,kpd->kgp',np.minimum(grouped,0),lo.transpose(1,0,2))).reshape(heads,len(ids))/np.float32(math.sqrt(self.d))
        logmass=upper+np.float32(math.log(self.R))
        fixed_logits=torch.bmm(q.reshape(self.h,heads//self.h,self.d),fixed.float().transpose(1,2)).reshape(heads,-1).numpy()/np.float32(math.sqrt(self.d))
        denominator=np.logaddexp(_lse(fixed_logits,axis=1),_lse(logmass,axis=1))
        scores=np.exp(logmass-denominator[:,None]).sum(axis=0)
        order=np.lexsort((ids,-scores))[:k]
        self.stats[layer]['selection_s']+=time.perf_counter()-begin
        return ids[order],scores[order]

    @property
    def nbytes(self):
        return sum(a.nbytes+b.nbytes+c.nbytes for a,b,c in zip(self.sums,self.maxima,self.counts))


class QuestBoundCache(TieredCache):
    """Stages exact bounded normalization-only keys before frozen present()."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.normalizer_sinks=[None]*self.store.layers

    def present(self,layer,q,k,v,pos):
        start=time.perf_counter();cutoff=max(0,pos-self.config.W+1)
        if cutoff<self.config.sink_tokens:raise ValueError('this reference requires disjoint sink/recent')
        recent=self.recent.view(layer)
        keep=np.flatnonzero(recent.ids>=cutoff)
        assert np.array_equal(recent.ids[keep],np.arange(cutoff,pos))
        sink=self.normalizer_sinks[layer]
        if sink is None:
            sink=self.store.read_range(layer,0,self.config.sink_tokens).k[0]
            self.normalizer_sinks[layer]=sink
        boundary_start=(cutoff//self.config.R)*self.config.R
        boundary_started=time.perf_counter()
        boundary=self.store.read_range(layer,boundary_start,cutoff).k[0]
        stat=self.index.stats[layer]
        stat['boundary_read_s']+=time.perf_counter()-boundary_started
        stat['boundary_calls']+=int(cutoff>boundary_start)
        stat['boundary_logical_bytes']+=(cutoff-boundary_start)*self.store.record_bytes
        recent_k=recent.k[0,:,keep,:]
        if recent_k.device.type!='cpu':stat['normalizer_key_copy_bytes']+=recent_k.numel()*recent_k.element_size()
        if k.device.type!='cpu':stat['normalizer_key_copy_bytes']+=k.numel()*k.element_size()
        # CPU float32 bmm matches screen's arithmetic and includes current key.
        copy_started=time.perf_counter()
        fixed=torch.cat((sink,boundary,recent_k.detach().cpu(),k[0].detach().cpu()),dim=1)
        stat['host_key_copy_s']+=time.perf_counter()-copy_started
        self.index.stage(layer,fixed)
        stat['stage_s']+=time.perf_counter()-start
        return super().present(layer,q,k,v,pos)

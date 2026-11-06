"""Full-key, sparse-trajectory oracle for diagnosis; O(history), not serving."""
import math
import numpy as np
import torch
from tiered_kv.retrieval import BlockIndex
from tiered_kv.cache import TieredCache


def fullkey_rank(q, keys, current_key, position, *, sink=128, window=2000, page=128, limit=15):
    """Float32 full causal softmax, individually normalized Q heads, shared pages.

    keys are original-position fp16 *past* keys [KV-head, position, dim];
    current_key is [KV-head,1,dim]. Dense scans and GPU workspace are unbounded
    in history and only for this diagnostic.
    """
    assert q.ndim==2 and keys.ndim==3 and current_key.ndim==3
    kv,n,d=keys.shape;heads,qd=q.shape
    assert n==position and position<=32768 and qd==d and current_key.shape==(kv,1,d) and heads%kv==0
    assert 0<heads<=128 and 0<d<=256
    assert 0<limit<=15 and position>=0
    cutoff=max(0,position-window+1)
    candidates=[b for b in range(cutoff//page) if b*page>=sink]
    if not candidates:return np.array([],dtype=np.int64),np.array([],dtype=np.float32)
    query=q.float().reshape(kv,heads//kv,d)
    # Current K joins the full denominator exactly once, before its commit.
    full=torch.cat((keys,current_key),dim=1).float()
    logits=torch.bmm(query,full.transpose(1,2)).reshape(heads,n+1)/math.sqrt(d)
    probs=torch.softmax(logits,dim=1)
    full_pages=cutoff//page
    mass=probs[:,:full_pages*page].reshape(heads,full_pages,page).sum(dim=(0,2))
    scores=mass[candidates[0]:candidates[-1]+1]
    score_cpu=scores.detach().cpu().numpy();ids=np.asarray(candidates,dtype=np.int64)
    order=np.lexsort((ids,-score_cpu))[:limit]
    return ids[order],score_cpu[order]


class FullKeyOracleIndex(BlockIndex):
    def __init__(self,store,device,capacity,*,sink=128,window=2000,page=128):
        super().__init__(store.layers,store.h,store.d,page)
        if capacity<store.count:raise ValueError('history capacity shorter than prefix')
        self.store=store;self.sink=sink;self.window=window;self.capacity=capacity
        self.keys=[];self.history_counts=[store.count]*store.layers;self.pending={}
        for layer in range(store.layers):
            target=torch.empty((store.h,capacity,store.d),dtype=torch.float16,device=device)
            if store.count:
                target[:,:store.count,:].copy_(store.read_range(layer,0,store.count).k[0].to(device))
            self.keys.append(target)

    def stage_current(self,layer,current,position):
        if layer in self.pending or position!=self.history_counts[layer] or position>=self.capacity:
            raise ValueError('oracle current-key position/bound mismatch')
        assert tuple(current.shape)==(1,self.h,1,self.d)
        self.pending[layer]=(position,current[0].detach())

    def search(self,layer,q,k):
        self.searches[layer]+=1
        position,current=self.pending[layer]
        return fullkey_rank(q.detach()[0,:,0,:],self.keys[layer][:,:position,:],current,position,
                            sink=self.sink,window=self.window,page=self.R,limit=k)

    def commit_current(self,layer):
        position,current=self.pending.pop(layer)
        self.keys[layer][:,position:position+1,:].copy_(current.to(dtype=torch.float16))
        self.history_counts[layer]+=1


class FullKeyOracleCache(TieredCache):
    def present(self,layer,q,k,v,pos):
        self.index.stage_current(layer,k,pos)
        return super().present(layer,q,k,v,pos)

    def commit_layer(self,layer,record):
        super().commit_layer(layer,record)
        self.index.commit_current(layer)

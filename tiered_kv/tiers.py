from dataclasses import dataclass
import numpy as np
import torch
from .types import KVBatch


class RecentTier:
    """Fixed-capacity ring; current token takes one of W slots."""
    def __init__(self,layers,W,h,d,device,dtype=torch.float16):
        self.W,self.h,self.d=W,h,d
        self.buffers=[torch.empty(2,h,W,d,device=device,dtype=dtype) for _ in range(layers)]
        self.ids=[np.full(W,-1,np.int64) for _ in range(layers)]
        self.starts=[0]*layers
        self.sizes=[0]*layers

    def view(self,layer):
        n,start=self.sizes[layer],self.starts[layer]
        order=(np.arange(n)+start)%self.W
        buf=self.buffers[layer]
        # index_select produces a bounded contiguous view, including at wrap.
        indices=torch.as_tensor(order,device=buf.device)
        packed=buf.index_select(2,indices)
        return KVBatch(self.ids[layer][order].copy(),packed[0:1],packed[1:2])

    def prepare(self,layer,query_pos):
        view=self.view(layer)
        n=int(np.searchsorted(view.ids,query_pos-self.W+1))
        # Consumers that need CPU statistics perform their own transfer. Fixed
        # retrieval needs only IDs; retain tensors on device for partial blocks.
        old=view.slice(0,n)
        self.starts[layer]=(self.starts[layer]+n)%self.W
        self.sizes[layer]-=n
        return old,view.slice(n,len(view.ids))

    def commit(self,layer,record):
        n=len(record.ids)
        if n>1:
            # Prefix/checkpoint initialization should transfer a block, not launch
            # two device copies for every historical token. Single-token decode
            # retains the same path below. This preserves exact stored bits.
            if self.sizes[layer]+n>self.W:
                raise ValueError('prepare must evict before commit')
            if np.any(np.diff(record.ids)<=0) or (self.sizes[layer] and record.ids[0]<=self.ids[layer][(self.starts[layer]+self.sizes[layer]-1)%self.W]):
                raise ValueError('recent positions must increase')
            slots=(self.starts[layer]+self.sizes[layer]+np.arange(n))%self.W
            buf=self.buffers[layer]
            values=torch.cat((record.k,record.v),0).to(device=buf.device,dtype=buf.dtype)
            buf.index_copy_(2,torch.as_tensor(slots,device=buf.device),values)
            self.ids[layer][slots]=record.ids
            self.sizes[layer]+=n
            return
        for j,pos in enumerate(record.ids):
            if self.sizes[layer]>=self.W:
                raise ValueError('prepare must evict before commit')
            if self.sizes[layer] and pos<=self.ids[layer][(self.starts[layer]+self.sizes[layer]-1)%self.W]:
                raise ValueError('recent positions must increase')
            slot=(self.starts[layer]+self.sizes[layer])%self.W
            self.buffers[layer][0,:,slot].copy_(record.k[0,:,j])
            self.buffers[layer][1,:,slot].copy_(record.v[0,:,j])
            self.ids[layer][slot]=pos
            self.sizes[layer]+=1


@dataclass
class Summary:
    start: int
    end: int
    count: int
    rep: int
    ks: np.ndarray
    vs: np.ndarray


class CompressedTier:
    def __init__(self,layers,C,B,policy,rotary):
        self.C,self.B,self.policy,self.rotary=C,B,policy,rotary
        self.sealed=[[] for _ in range(layers)]
        self.pending=[None]*layers

    def absorb(self,layer,evicted):
        if self.C==0 or len(evicted.ids)==0:
            return
        keys=evicted.k[0].float().cpu().numpy().transpose(1,0,2)
        values=evicted.v[0].float().cpu().numpy().transpose(1,0,2)
        if self.policy=='mean_pre':
            keys=self.rotary.apply(keys,evicted.ids,inverse=True)
        for pos,k,v in zip(evicted.ids,keys,values):
            pending=self.pending[layer]
            if pending is None:
                if self.sealed[layer] and pos!=self.sealed[layer][-1].end+1:
                    raise ValueError('noncontiguous summary')
                pending=Summary(int(pos),int(pos),1,int(pos),k.copy(),v.copy())
                self.pending[layer]=pending
            else:
                if pos!=pending.end+1:
                    raise ValueError('noncontiguous summary')
                pending.end=int(pos)
                pending.count+=1
                if self.policy!='first':
                    pending.ks+=k
                    pending.vs+=v
                    pending.rep=int(pos)
            if pending.count==self.B:
                self.sealed[layer].append(pending)
                self.pending[layer]=None
                if len(self.sealed[layer])>self.C:
                    a,b=self.sealed[layer][:2]
                    merged=Summary(a.start,b.end,a.count+b.count,a.rep if self.policy=='first' else b.end,
                                   a.ks if self.policy=='first' else a.ks+b.ks,
                                   a.vs if self.policy=='first' else a.vs+b.vs)
                    self.sealed[layer][:2]=[merged]

    def entries(self,layer):
        return self.sealed[layer]+([self.pending[layer]] if self.pending[layer] else [])

    def materialize(self,s):
        if self.policy=='first':
            return s.ks,s.vs
        k=s.ks/s.count
        if self.policy=='mean_pre':
            k=self.rotary.apply(k[None],np.array([s.end]))[0]
        return k,s.vs/s.count

    def view(self,layer,h,d,device,dtype):
        summaries=self.entries(layer)
        if not summaries:
            return KVBatch.empty(h,d,device,dtype),np.empty((0,2),np.int64)
        arrays=[self.materialize(s) for s in summaries]
        k,v=(torch.from_numpy(np.stack([a[i] for a in arrays])).permute(1,0,2)[None].to(device=device,dtype=dtype) for i in (0,1))
        return KVBatch(np.array([s.rep for s in summaries]),k,v),np.array([(s.start,s.end) for s in summaries])

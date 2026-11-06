from dataclasses import dataclass, field
import hashlib
import json
import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


@dataclass
class KVBatch:
    ids: np.ndarray
    k: torch.Tensor
    v: torch.Tensor

    def __post_init__(self):
        self.ids = np.asarray(self.ids,dtype=np.int64)
        if self.k.shape != self.v.shape or self.k.ndim!=4 or self.k.shape[0]!=1 or self.k.shape[2]!=len(self.ids):
            raise ValueError('KV must be [1,Hkv,S,D], matching IDs')

    def slice(self,a,b):
        return KVBatch(self.ids[a:b],self.k[:,:,a:b],self.v[:,:,a:b])

    def cpu(self):
        return KVBatch(self.ids.copy(),self.k.detach().cpu(),self.v.detach().cpu())

    @classmethod
    def empty(cls,h,d,device='cpu',dtype=torch.float16):
        x=torch.empty(1,h,0,d,device=device,dtype=dtype)
        return cls(np.array([],dtype=np.int64),x,x)


@dataclass
class KVView(KVBatch):
    intervals: np.ndarray
    tiers: list[str]
    scores: dict = field(default_factory=dict)
    eligible_blocks: int = 0
    selected_blocks: list = field(default_factory=list)

    def valid(self,pos):
        return torch.as_tensor(self.intervals[:,1]<=pos,device=self.k.device)[None,None,None,:]


class Rotary:
    """Default Qwen split-half RoPE, FP32 CPU sufficient statistics only."""
    def __init__(self,inv_freq):
        self.inv_freq=np.asarray(inv_freq,dtype=np.float32)

    def apply(self,x,positions,inverse=False):
        angles=np.asarray(positions,dtype=np.float32)[:,None]*self.inv_freq[None,:]
        angles=np.concatenate((angles,angles),axis=-1)[:,None,:]
        half=x.shape[-1]//2
        rotated=np.concatenate((-x[...,half:],x[...,:half]),axis=-1)
        return x*np.cos(angles)+rotated*np.sin(angles)*(-1 if inverse else 1)

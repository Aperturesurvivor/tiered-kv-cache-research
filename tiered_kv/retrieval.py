import numpy as np


class BlockIndex:
    """Only evicted keys; centroid statistics grow O(N/R)."""
    def __init__(self,layers,h,d,R,capacity=16):
        self.h,self.d,self.R=h,d,R
        self.sums=[np.zeros((capacity,h,d),np.float32) for _ in range(layers)]
        self.counts=[np.zeros(capacity,np.int64) for _ in range(layers)]
        self.next_pos=[0]*layers
        self.searches=[0]*layers

    def absorb(self,layer,evicted):
        n=len(evicted.ids)
        if n==0:
            return
        if not np.array_equal(evicted.ids,np.arange(self.next_pos[layer],self.next_pos[layer]+n)):
            raise ValueError('index eviction order violated')
        need=int(evicted.ids[-1]//self.R)+1
        if need>len(self.counts[layer]):
            capacity=max(need,len(self.counts[layer])*2)
            self.sums[layer]=np.pad(self.sums[layer],((0,capacity-len(self.counts[layer])),(0,0),(0,0)))
            self.counts[layer]=np.pad(self.counts[layer],(0,capacity-len(self.counts[layer])))
        keys=evicted.k[0].detach().float().cpu().numpy().transpose(1,0,2)
        blocks=evicted.ids//self.R
        np.add.at(self.sums[layer],blocks,keys)
        np.add.at(self.counts[layer],blocks,1)
        self.next_pos[layer]+=n

    def search(self,layer,Q_rot,k):
        if k<=0:
            raise ValueError('disabled retrieval must not call search')
        self.searches[layer]+=1
        ids=np.flatnonzero(self.counts[layer])
        if not len(ids):
            return ids,np.array([],np.float32)
        q=Q_rot.detach().float().cpu().numpy()[0,:,0,:]
        if q.shape[0]%self.h:
            raise ValueError('GQA grouping mismatch')
        q=q.reshape(self.h,-1,self.d).mean(1)
        centroids=self.sums[layer][ids]/self.counts[layer][ids,None,None].astype(np.float32)
        numerator=(centroids*q[None]).sum(-1)
        denominator=np.maximum(np.linalg.norm(centroids,axis=-1)*np.linalg.norm(q,axis=-1)[None],1e-8)
        scores=(numerator/denominator).mean(-1)
        order=np.lexsort((ids,-scores))[:k]
        return ids[order],scores[order]

    def eligible(self,layer):
        return int(np.count_nonzero(self.counts[layer]))

    @property
    def nbytes(self):
        return sum(a.nbytes+b.nbytes for a,b in zip(self.sums,self.counts))


class FixedBlockIndex(BlockIndex):
    """Externally retrieved disk addresses; no per-layer centroid archive in RAM."""
    def __init__(self,layers,h,d,R,blocks):
        super().__init__(layers,h,d,R,capacity=0)
        self.blocks=np.asarray(blocks,dtype=np.int64)

    def absorb(self,layer,evicted):
        n=len(evicted.ids)
        if not np.array_equal(evicted.ids,np.arange(self.next_pos[layer],self.next_pos[layer]+n)):
            raise ValueError('index eviction order violated')
        self.next_pos[layer]+=n

    def eligible(self,layer):
        return (self.next_pos[layer]+self.R-1)//self.R

    def search(self,layer,Q_rot,k):
        if k<=0:
            raise ValueError('disabled retrieval must not call search')
        self.searches[layer]+=1
        selected=self.blocks[self.blocks<self.eligible(layer)][:k]
        return selected,np.zeros(len(selected),np.float32)

    @property
    def nbytes(self):
        return self.blocks.nbytes

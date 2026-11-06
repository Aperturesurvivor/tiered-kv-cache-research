import json
import os
import uuid
import numpy as np
import torch
from .types import KVBatch, KVView, Rotary
from .tiers import RecentTier, CompressedTier, Summary
from .retrieval import BlockIndex,FixedBlockIndex
from .store import ColdStore, file_sha


class TieredCache:
    def __init__(self,config,store,rotary,device,dtype=torch.float16,initialize=True):
        self.config,self.store,self.rotary,self.device,self.dtype=config,store,rotary,device,dtype
        self.position=store.count
        self.pending_token=None
        self.recent=RecentTier(store.layers,config.W,store.h,store.d,device,dtype)
        self.compressed=CompressedTier(store.layers,config.C,config.B,config.merge,rotary)
        self.index=(FixedBlockIndex(store.layers,store.h,store.d,config.R,config.fixed_blocks)
                    if config.retrieval=='fixed' else
                    BlockIndex(store.layers,store.h,store.d,config.R,capacity=16 if config.k else 0))
        self.retrieved_cache=[None]*store.layers
        self.sink_cache=[None]*store.layers
        self.committed_layers=set()
        self.max_entries=0
        self.max_entries_by_layer=[0]*store.layers
        self.present_calls=[0]*store.layers
        self.trace=None
        self.oracle_blocks=None
        if initialize:
            self._initialize()

    def _absorb(self,layer,batch):
        if self.config.C:
            self.compressed.absorb(layer,batch)
        if self.config.k:
            self.index.absorb(layer,batch)

    def select_blocks(self,blocks):
        """Replace bounded disk selections at a complete-token query boundary."""
        from dataclasses import replace
        if self.config.retrieval!='fixed' or self.committed_layers:
            raise ValueError('selection changes require fixed retrieval at a complete-token boundary')
        updated=replace(self.config,fixed_blocks=tuple(blocks))
        if updated.fixed_blocks!=self.config.fixed_blocks:
            self.index.blocks=np.asarray(updated.fixed_blocks,dtype=np.int64)
            self.retrieved_cache=[None]*self.store.layers
        self.config=updated

    def _initialize(self):
        # State is after the preceding token. The next present() evicts one
        # more record to reserve the current token's slot.
        cutoff=max(0,self.position-self.config.W)
        for layer in range(self.store.layers):
            if not self.config.C and self.config.retrieval=='fixed':
                # Fixed address selection needs only the eligible position, not
                # a scan of every archived KV record to rebuild centroids.
                if self.config.k:
                    self.index.next_pos[layer]=cutoff
            elif self.config.C or self.config.k:
                for start in range(0,cutoff,self.config.R):
                    self._absorb(layer,self.store.read_range(layer,start,min(start+self.config.R,cutoff)))
            tail=self.store.read_range(layer,cutoff,self.position)
            self.recent.commit(layer,tail)

    def present(self,layer,Q_rot,K_current,V_current,pos):
        if pos!=self.position or layer in self.committed_layers or K_current.shape[2]!=1:
            raise ValueError('tiered presentation requires current single absolute position')
        self.present_calls[layer]+=1
        old,recent=self.recent.prepare(layer,pos)
        self._absorb(layer,old)
        c=self.config
        blocks,scores=(self.index.search(layer,Q_rot,c.k) if c.k else (np.array([],np.int64),np.array([])))
        if c.k and self.oracle_blocks:
            gold=[b for b in self.oracle_blocks if b<self.index.eligible(layer)][:c.k]
            original_scores=dict(zip(blocks.tolist(),scores.tolist()))
            selected=gold+[int(b) for b in blocks if b not in gold]
            blocks=np.array(selected[:c.k],dtype=np.int64)
            scores=np.array([original_scores.get(int(b)) for b in blocks],dtype=object)
        cutoff=max(0,pos-c.W+1)
        if c.retrieval=='fixed' and c.k and len(blocks):
            # Pin selected exact blocks. Extend a partial block using the exact
            # record already leaving recent memory; do not reread all selected
            # disk blocks for every newly eligible token.
            cache_key=(tuple(blocks),min(cutoff,(int(max(blocks))+1)*c.R))
            saved=self.retrieved_cache[layer]
            if saved is None or saved[0]!=cache_key:
                extends=(saved is not None and saved[0][0]==cache_key[0] and len(old.ids)>0
                         and cache_key[1]-saved[0][1]==len(old.ids) and old.ids[0]==saved[0][1]
                         and np.isin(old.ids//c.R,blocks).all())
                if extends:
                    # Match the archive's fp16 representation even in fp32 tiny
                    # tests; production records already have this dtype.
                    k=old.k.to(device=self.device,dtype=torch.float16).to(dtype=self.dtype)
                    v=old.v.to(device=self.device,dtype=torch.float16).to(dtype=self.dtype)
                    source=KVBatch(np.concatenate((saved[1].ids,old.ids)),torch.cat((saved[1].k,k),2),torch.cat((saved[1].v,v),2))
                else:
                    source=self.store.read(layer,blocks,cutoff)
                saved=(cache_key,KVBatch(source.ids,source.k.to(device=self.device,dtype=self.dtype),source.v.to(device=self.device,dtype=self.dtype)))
                self.retrieved_cache[layer]=saved
            retrieved=saved[1]
        else:
            retrieved=self.store.read(layer,blocks,cutoff) if c.k else KVBatch.empty(self.store.h,self.store.d)
        sinks=KVBatch.empty(self.store.h,self.store.d)
        sink_count=min(c.sink_tokens,cutoff)
        if sink_count:
            saved=self.sink_cache[layer]
            if saved is None or len(saved.ids)!=sink_count:
                source=self.store.read_range(layer,0,sink_count)
                saved=KVBatch(source.ids,source.k.to(device=self.device,dtype=self.dtype),source.v.to(device=self.device,dtype=self.dtype))
                self.sink_cache[layer]=saved
            keep=~np.isin(saved.ids,retrieved.ids)
            if keep.all():
                sinks=saved
            elif keep.any():
                indices=torch.as_tensor(np.flatnonzero(keep),device=self.device)
                sinks=KVBatch(saved.ids[keep],saved.k.index_select(2,indices),saved.v.index_select(2,indices))
        compressed,intervals=self.compressed.view(layer,self.store.h,self.store.d,self.device,self.dtype)
        current=KVBatch(np.array([pos]),K_current,V_current)
        batches=[compressed,sinks,retrieved,recent,current]
        labels=['compressed','sink','retrieved','recent','recent']
        ids=np.concatenate([b.ids for b in batches])
        exact=ids[len(compressed.ids):]
        if len(np.unique(exact))!=len(exact):
            raise ValueError('duplicate exact positions')
        spans=np.concatenate([intervals]+[np.stack((b.ids,b.ids),1) for b in batches[1:]])
        view=KVView(ids,torch.cat([b.k.to(device=self.device,dtype=self.dtype) for b in batches],2),
                    torch.cat([b.v.to(device=self.device,dtype=self.dtype) for b in batches],2),spans,
                    [label for label,b in zip(labels,batches) for _ in b.ids],
                    dict(zip(blocks.tolist(),scores.tolist())),self.index.eligible(layer) if c.k else 0,blocks.tolist())
        if len(ids)>c.W+c.k*c.R+c.C+1+c.sink_tokens or not np.all(spans[:,1]<=pos):
            raise ValueError('capacity/future visibility invariant failed')
        self.max_entries=max(self.max_entries,len(ids))
        self.max_entries_by_layer[layer]=max(self.max_entries_by_layer[layer],len(ids))
        if self.trace:
            self.trace(layer,pos,Q_rot,view)
        return view

    def commit_layer(self,layer,record):
        if layer in self.committed_layers or record.ids.tolist()!=[self.position]:
            raise ValueError('layer commit duplicated / wrong position')
        self.store.append(layer,record)
        self.recent.commit(layer,record)
        self.committed_layers.add(layer)

    def finish_token(self):
        if self.committed_layers!=set(range(self.store.layers)):
            raise ValueError('cannot advance before all layers commit')
        self.position+=1
        self.committed_layers.clear()

    def checkpoint(self,pending_token=None,token_hash=None):
        if self.committed_layers:
            raise ValueError('checkpoint only at complete token boundary')
        state={}
        metadata=[]
        for layer in range(self.store.layers):
            recent=self.recent.view(layer).cpu()
            state[f'rk{layer}']=recent.k.numpy()
            state[f'rv{layer}']=recent.v.numpy()
            state[f'ri{layer}']=recent.ids
            state[f'is{layer}']=self.index.sums[layer]
            state[f'ic{layer}']=self.index.counts[layer]
            meta=[]
            for j,s in enumerate(self.compressed.entries(layer)):
                state[f'sk{layer}_{j}']=s.ks
                state[f'sv{layer}_{j}']=s.vs
                meta.append([s.start,s.end,s.count,s.rep])
            metadata.append({'summaries':meta,'sealed':len(self.compressed.sealed[layer]),'index_next':self.index.next_pos[layer]})
        state['meta']=np.array(json.dumps(metadata))
        name='state-'+uuid.uuid4().hex+'.npz'
        with (self.store.root/name).open('xb') as out:
            np.savez(out,**state)
            out.flush()
            os.fsync(out.fileno())
        self.pending_token=pending_token
        return self.store.checkpoint({'config_hash':self.config.digest,'model_revision':self.config.model_revision,
             'token_hash':token_hash,'next_position':self.position,'pending_token':pending_token,
             'state':name,'state_sha256':file_sha(self.store.root/name)})

    @classmethod
    def reopen(cls,root,config,rotary,device,dtype=torch.float16):
        store=ColdStore.open(root)
        meta=store.metadata
        if meta['config_hash']!=config.digest or meta['model_revision']!=config.model_revision or meta['next_position']!=store.count:
            store.close()
            raise ValueError('checkpoint identity mismatch')
        if file_sha(store.root/meta['state'])!=meta['state_sha256']:
            store.close()
            raise ValueError('state checksum mismatch')
        obj=cls(config,store,rotary,device,dtype,initialize=False)
        with np.load(store.root/meta['state'],allow_pickle=False) as state:
            layers=json.loads(str(state['meta']))
            for layer,entry in enumerate(layers):
                obj.recent.commit(layer,KVBatch(state[f'ri{layer}'],torch.from_numpy(state[f'rk{layer}']),torch.from_numpy(state[f'rv{layer}'])))
                obj.index.sums[layer]=state[f'is{layer}'].copy()
                obj.index.counts[layer]=state[f'ic{layer}'].copy()
                obj.index.next_pos[layer]=entry['index_next']
                summaries=[Summary(*s,state[f'sk{layer}_{j}'].copy(),state[f'sv{layer}_{j}'].copy()) for j,s in enumerate(entry['summaries'])]
                obj.compressed.sealed[layer]=summaries[:entry['sealed']]
                obj.compressed.pending[layer]=summaries[-1] if len(summaries)>entry['sealed'] else None
        obj.pending_token=meta['pending_token']
        return obj

    def verify_rebuild(self):
        other=TieredCache(self.config,self.store,self.rotary,self.device,self.dtype)
        for layer in range(self.store.layers):
            if self.index.next_pos[layer]!=other.index.next_pos[layer] or not np.array_equal(self.index.counts[layer],other.index.counts[layer]) or not np.allclose(self.index.sums[layer],other.index.sums[layer],atol=1e-5,rtol=1e-5):
                raise ValueError('index cold rebuild mismatch')
        return True

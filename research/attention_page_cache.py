"""Diagnostic-only bounded complete-page GPU cache over the frozen ColdStore.

Only read() is intercepted; index construction, sinks and recent reads still
use the original read_range. This cache does not bound centroid index search.
"""
import numpy as np
import torch
from tiered_kv.store import ColdStore
from tiered_kv.types import KVBatch


class CompletePageCacheStore(ColdStore):
    def __init__(self,*args,cache_device,cache_pages=15,sink_tokens=128,**kwargs):
        if cache_pages<1 or sink_tokens<0:
            raise ValueError('invalid cache bound')
        super().__init__(*args,**kwargs)
        self.cache_device=cache_device
        self.cache_pages=cache_pages
        self.sink_tokens=sink_tokens
        self.pages=[{} for _ in range(self.layers)]
        self.previous_order=[None]*self.layers
        self.peak_resident_pages_total=0
        self.stats=[dict(calls=0,unchanged_set=0,reordered=0,added_pages=0,dropped_pages=0,
                         cache_hits=0,cache_misses=0,peak_resident_pages=0) for _ in range(self.layers)]

    def read(self,layer,block_ids,before_pos):
        order=tuple(int(b) for b in block_ids)
        if len(set(order))!=len(order) or len(order)>self.cache_pages:
            raise ValueError('duplicate pages or cache bound exceeded')
        if any(b<0 or b*self.R<self.sink_tokens or (b+1)*self.R>before_pos or
               (b+1)*self.R>self.counts[layer] for b in order):
            raise ValueError('only complete non-sink pages are cacheable')
        selected=set(order); pages=self.pages[layer]; previous=self.previous_order[layer]
        current=set(pages); dropped=current-selected; missing=selected-current
        stats=self.stats[layer]; stats['calls']+=1
        stats['unchanged_set']+=int(previous is not None and set(previous)==selected)
        stats['reordered']+=int(previous is not None and set(previous)==selected and previous!=order)
        stats['added_pages']+=len(missing);stats['dropped_pages']+=len(dropped)
        stats['cache_hits']+=len(selected & current);stats['cache_misses']+=len(missing)
        try:
            for block in dropped: del pages[block]
            # Evict first, before loading replacements; live persistent entries never
            # exceed the bound (temporary concatenation is separately measured).
            for block in sorted(missing):
                batch=super().read(layer,[block],before_pos)
                assert batch.ids.tolist()==list(range(block*self.R,(block+1)*self.R))
                assert batch.k.dtype==torch.float16 and batch.v.dtype==torch.float16
                pages[block]=KVBatch(batch.ids,batch.k.to(self.cache_device),batch.v.to(self.cache_device))
        except BaseException:
            pages.clear()
            self.previous_order[layer]=None
            raise
        assert len(pages)<=self.cache_pages
        stats['peak_resident_pages']=max(stats['peak_resident_pages'],len(pages))
        self.peak_resident_pages_total=max(self.peak_resident_pages_total,sum(map(len,self.pages)))
        self.previous_order[layer]=order
        if not order: return KVBatch.empty(self.h,self.d,device=self.cache_device)
        ordered=[pages[b] for b in sorted(selected)]
        return KVBatch(np.concatenate([b.ids for b in ordered]),torch.cat([b.k for b in ordered],2),torch.cat([b.v for b in ordered],2))

    def close(self):
        self.pages=[{} for _ in range(self.layers)]
        super().close()

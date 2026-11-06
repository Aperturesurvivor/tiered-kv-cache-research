"""Append-only little-endian fp16 records. Recovery ignores tails; never deletes."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from .types import KVBatch


def file_sha(path,limit=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as src:
        left=limit
        while left is None or left>0:
            block=src.read(1024*1024 if left is None else min(left,1024*1024))
            if not block:
                break
            h.update(block)
            if left is not None:
                left-=len(block)
        if left is not None and left:
            raise ValueError('short file')
    return h.hexdigest()


def atomic_json(path,data):
    path=Path(path)
    # Unique durable temporary files are retained if interrupted; no cleanup.
    import uuid
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as out:
        json.dump(data,out,sort_keys=True,indent=2)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ColdStore:
    def __init__(self,root,layers,h,d,block_size=128,prefix=None,staged_tensor_writes=False):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=False)
        self.layers,self.h,self.d,self.R=layers,h,d,block_size
        self.prefix=prefix
        if prefix and not prefix.readonly:
            raise ValueError('shared prefix must be immutable')
        if prefix and (prefix.layers,prefix.h,prefix.d,prefix.R)!=(layers,h,d,block_size):
            raise ValueError('prefix shape mismatch')
        self.base=prefix.count if prefix else 0
        self.counts=[self.base]*layers
        self.buffers=[bytearray() for _ in range(layers)]
        self.staged_tensor_writes=staged_tensor_writes
        self.tensor_buffers=[None]*layers
        self.tensor_counts=[0]*layers
        self.files=[self.path(i).open('x+b',buffering=0) for i in range(layers)]
        self.readonly=False
        self.metadata={}
        self.read_bytes=self.write_bytes=0
        self.committed=0

    def path(self,layer):
        return self.root/f'layer-{layer:02d}.kv'

    @property
    def record_bytes(self):
        return 2*self.h*self.d*2

    @property
    def count(self):
        if len(set(self.counts))!=1:
            raise ValueError('layers not committed at same position')
        return self.counts[0]

    def append(self,layer,positions,K_rot=None,V=None):
        if self.readonly:
            raise ValueError('immutable store')
        batch=positions if isinstance(positions,KVBatch) else KVBatch(positions,K_rot,V)
        n=len(batch.ids)
        if not np.array_equal(batch.ids,np.arange(self.counts[layer],self.counts[layer]+n)):
            raise ValueError('positions must be consecutive, once only')
        if tuple(batch.k.shape[1::2])!=(self.h,self.d):
            raise ValueError('store shape mismatch')
        if self.staged_tensor_writes:
            packed=torch.stack((batch.k[0],batch.v[0]),dim=1).permute(2,1,0,3).detach().to(dtype=torch.float16)
            if self.tensor_buffers[layer] is None:
                self.tensor_buffers[layer]=torch.empty((self.R,2,self.h,self.d),dtype=torch.float16,device=packed.device)
            if self.tensor_buffers[layer].device!=packed.device:
                raise ValueError('write staging device changed')
            cursor=0
            while cursor<n:
                begin=self.tensor_counts[layer]
                take=min(n-cursor,self.R-begin)
                self.tensor_buffers[layer][begin:begin+take].copy_(packed[cursor:cursor+take])
                self.tensor_counts[layer]+=take
                cursor+=take
                if self.tensor_counts[layer]==self.R:
                    self._flush(layer)
            self.counts[layer]+=n
            self.write_bytes+=n*self.record_bytes
            return
        packed=torch.stack((batch.k[0],batch.v[0]),dim=1).permute(2,1,0,3).detach().cpu().numpy().astype('<f2').tobytes()
        # Flush each R-record boundary. Buffer never exceeds R records.
        cursor=0
        while cursor<len(packed):
            take=min(len(packed)-cursor,self.R*self.record_bytes-len(self.buffers[layer]))
            self.buffers[layer].extend(packed[cursor:cursor+take])
            cursor+=take
            if len(self.buffers[layer])==self.R*self.record_bytes:
                self._flush(layer)
        self.counts[layer]+=n
        self.write_bytes+=len(packed)

    def _flush(self,layer):
        if self.tensor_counts[layer]:
            data=self.tensor_buffers[layer][:self.tensor_counts[layer]].cpu().numpy().astype('<f2',copy=False).tobytes()
            count=self.files[layer].write(data)
            if count!=len(data):
                raise OSError(f'short write: {count}/{len(data)}')
            self.tensor_counts[layer]=0
        data=self.buffers[layer]
        if data:
            count=self.files[layer].write(data)
            if count!=len(data):
                raise OSError(f'short write: {count}/{len(data)}')
            data.clear()

    def read_range(self,layer,start,end):
        if not 0<=start<=end<=self.counts[layer]:
            raise ValueError('read outside committed/buffered extent')
        chunks=[]
        if start<self.base:
            chunks.append(self.prefix.read_range(layer,start,min(end,self.base)))
            start=min(end,self.base)
        if start<end:
            offset=(start-self.base)*self.record_bytes
            length=(end-start)*self.record_bytes
            written=(self.counts[layer]-self.base-self.tensor_counts[layer])*self.record_bytes-len(self.buffers[layer])
            raw=b''
            if offset<written:
                size=min(length,written-offset)
                raw=os.pread(self.files[layer].fileno(),size,offset)
                if len(raw)!=size:
                    raise ValueError('short read / corrupt file')
            if len(raw)<length:
                buf_offset=max(0,offset-written)
                if self.tensor_counts[layer]:
                    first=buf_offset//self.record_bytes
                    records=(length-len(raw))//self.record_bytes
                    raw+=self.tensor_buffers[layer][first:first+records].cpu().numpy().astype('<f2',copy=False).tobytes()
                else:
                    raw+=bytes(self.buffers[layer][buf_offset:buf_offset+length-len(raw)])
            if len(raw)!=length:
                raise ValueError('short buffer read')
            self.read_bytes+=length
            arr=np.frombuffer(raw,dtype='<f2').reshape(-1,2,self.h,self.d).copy()
            chunks.append(KVBatch(np.arange(start,end),torch.from_numpy(arr[:,0]).permute(1,0,2)[None],torch.from_numpy(arr[:,1]).permute(1,0,2)[None]))
        if not chunks:
            return KVBatch.empty(self.h,self.d)
        return KVBatch(np.concatenate([b.ids for b in chunks]),torch.cat([b.k for b in chunks],2),torch.cat([b.v for b in chunks],2))

    def read(self,layer,block_ids,before_pos):
        chunks=[]
        for block in sorted(set(int(b) for b in block_ids)):
            start=block*self.R
            end=min(start+self.R,before_pos,self.counts[layer])
            if start<0:
                raise ValueError('negative block')
            if start<end:
                chunks.append(self.read_range(layer,start,end))
        if not chunks:
            return KVBatch.empty(self.h,self.d)
        return KVBatch(np.concatenate([b.ids for b in chunks]),torch.cat([b.k for b in chunks],2),torch.cat([b.v for b in chunks],2))

    def checkpoint(self,metadata):
        if self.readonly:
            raise ValueError('immutable store')
        count=self.count
        for layer,f in enumerate(self.files):
            self._flush(layer)
            f.flush()
            os.fsync(f.fileno())
        sizes=[(count-self.base)*self.record_bytes]*self.layers
        doc={'version':1,'dtype':'<f2','layout':'token,KV,head,dim','layers':self.layers,
             'h':self.h,'d':self.d,'R':self.R,'base':self.base,'count':count,
             'prefix':str(self.prefix.root) if self.prefix else None,
             'prefix_manifest_sha256':file_sha(self.prefix.root/'checkpoint.json') if self.prefix else None,
             'sizes':sizes,'sha256':[file_sha(self.path(i),n) for i,n in enumerate(sizes)],'metadata':metadata,
             'staged_tensor_writes':self.staged_tensor_writes}
        atomic_json(self.root/'checkpoint.json',doc)
        self.committed=count
        self.metadata=metadata
        return doc

    @classmethod
    def open(cls,root,readonly=False):
        root=Path(root).resolve()
        doc=json.loads((root/'checkpoint.json').read_text())
        if doc['version']!=1 or doc['dtype']!='<f2' or doc['layout']!='token,KV,head,dim':
            raise ValueError('unsupported store manifest')
        obj=cls.__new__(cls)
        obj.root,obj.layers,obj.h,obj.d,obj.R=root,doc['layers'],doc['h'],doc['d'],doc['R']
        obj.base=doc['base']
        obj.prefix=cls.open(doc['prefix'],readonly=True) if doc['prefix'] else None
        if obj.prefix and (obj.prefix.count!=obj.base or file_sha(obj.prefix.root/'checkpoint.json')!=doc['prefix_manifest_sha256']):
            raise ValueError('prefix identity mismatch')
        obj.counts=[doc['count']]*obj.layers
        obj.buffers=[bytearray() for _ in range(obj.layers)]
        obj.staged_tensor_writes=doc.get('staged_tensor_writes',False)
        obj.tensor_buffers=[None]*obj.layers
        obj.tensor_counts=[0]*obj.layers
        obj.files=[]
        obj.readonly=readonly
        obj.metadata=doc['metadata']
        obj.committed=doc['count']
        obj.read_bytes=obj.write_bytes=0
        for layer in range(obj.layers):
            size=doc['sizes'][layer]
            if size!=(doc['count']-obj.base)*obj.record_bytes or obj.path(layer).stat().st_size<size:
                raise ValueError('short file / invalid extent')
            if file_sha(obj.path(layer),size)!=doc['sha256'][layer]:
                raise ValueError('checksum mismatch')
            f=obj.path(layer).open('rb' if readonly else 'r+b',buffering=0)
            # Ignore owned uncommitted tails; the next append overwrites from commit.
            f.seek(size)
            obj.files.append(f)
        return obj

    @property
    def allocated_bytes(self):
        return sum(self.path(i).stat().st_blocks*512 for i in range(self.layers))

    def close(self):
        for f in self.files:
            f.close()
        self.tensor_buffers=[None]*self.layers
        self.tensor_counts=[0]*self.layers
        if self.prefix:
            self.prefix.close()

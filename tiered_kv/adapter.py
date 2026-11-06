"""Narrow forward/mask seams for transformers 4.51.3 Qwen2, Qwen3 and Llama."""
import types
import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from .types import KVBatch, KVView


def attend(q,k,v,valid,check=True):
    if check and not bool(valid.any(-1).all()):
        raise ValueError('all-masked attention row')
    groups=q.shape[1]//k.shape[1]
    return scaled_dot_product_attention(q,repeat_kv(k,groups),repeat_kv(v,groups),attn_mask=valid,dropout_p=0.,is_causal=False,scale=q.shape[-1]**-.5)


class DenseController:
    def __init__(self,layers,store=None):
        self.keys=[None]*layers
        self.values=[None]*layers
        self.position=0
        self.store=store
        self.pending_token=None
        self.max_entries=0
        self.trace=None

    def initialize(self,store,device,dtype):
        for layer in range(store.layers):
            batch=store.read_range(layer,0,store.count)
            self.keys[layer]=batch.k.to(device=device,dtype=dtype)
            self.values[layer]=batch.v.to(device=device,dtype=dtype)
        self.position=store.count
        self.max_entries=self.position

    def dense(self,layer,q,k,v,positions):
        if int(positions[0])!=self.position:
            raise ValueError('dense position mismatch')
        if self.store:
            self.store.append(layer,KVBatch(positions,k,v))
        self.keys[layer]=k if self.keys[layer] is None else torch.cat((self.keys[layer],k),2)
        self.values[layer]=v if self.values[layer] is None else torch.cat((self.values[layer],v),2)
        n=self.keys[layer].shape[2]
        self.max_entries=max(self.max_entries,n)
        valid=torch.arange(n,device=q.device)[None,None,None,:]<=torch.as_tensor(positions,device=q.device)[None,None,:,None]
        if self.trace and len(positions)==1:
            ids=np.arange(n)
            self.trace(layer,int(positions[0]),q,KVView(ids,self.keys[layer],self.values[layer],np.stack((ids,ids),1),['recent']*n))
        return self.keys[layer],self.values[layer],valid


class ModelAdapter:
    @classmethod
    def install(cls,model,controller):
        if model.config.model_type not in ('qwen2','qwen3','llama') or hasattr(model,'_tiered_adapter'):
            raise ValueError('expected unpatched Qwen2, Qwen3 or Llama')
        if getattr(model.config,'use_sliding_window',False):
            raise ValueError('sliding-window models need a separate adapter')
        obj=cls()
        obj.model=model.eval().requires_grad_(False)
        obj.controller=controller
        obj.originals=[]
        obj.original_mask=model.model._update_causal_mask
        # Caller supplies absolute positions. Attention constructs bounded masks.
        model.model._update_causal_mask=types.MethodType(lambda self,*a,**kw:None,model.model)
        for layer in model.model.layers:
            attention=layer.self_attn
            obj.originals.append(attention.forward)
            attention.forward=types.MethodType(obj._make_forward(),attention)
        model._tiered_adapter=obj
        return obj

    def _make_forward(self):
        adapter=self
        def forward(module,hidden_states,position_embeddings,attention_mask=None,past_key_value=None,cache_position=None,**kwargs):
            shape=hidden_states.shape[:-1]
            hidden_shape=(*shape,-1,module.head_dim)
            q=module.q_proj(hidden_states).view(hidden_shape)
            k=module.k_proj(hidden_states).view(hidden_shape)
            if module.config.model_type=='qwen3':
                q=module.q_norm(q)
                k=module.k_norm(k)
            q=q.transpose(1,2)
            k=k.transpose(1,2)
            v=module.v_proj(hidden_states).view(hidden_shape).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
            controller=adapter.controller
            positions=adapter.call_positions
            if isinstance(controller,DenseController):
                keys,values,valid=controller.dense(module.layer_idx,q,k,v,positions)
            else:
                view=controller.present(module.layer_idx,q,k,v,int(positions[0]))
                keys,values,valid=view.k,view.v,view.valid(int(positions[0]))
            # Invariants guarantee nonempty current and no future-only row; avoid
            # per-layer GPU synchronization in headline timings.
            output=attend(q,keys,values,valid,check=False).transpose(1,2).reshape(*shape,-1).contiguous()
            output=module.o_proj(output)
            if not isinstance(controller,DenseController):
                controller.commit_layer(module.layer_idx,KVBatch(positions,k,v))
            return output,None
        return forward

    @property
    def position(self):
        return self.controller.position

    def set_controller(self,controller):
        self.controller=controller

    @torch.inference_mode()
    def forward_tokens(self,tokens,absolute_pos):
        if absolute_pos!=self.position or not tokens:
            raise ValueError('explicit next position required')
        if absolute_pos+len(tokens)>self.model.config.max_position_embeddings:
            raise ValueError('native context budget exceeded')
        device=next(self.model.parameters()).device
        self.call_positions=np.arange(absolute_pos,absolute_pos+len(tokens))
        positions=torch.as_tensor(self.call_positions,device=device)
        out=self.model(input_ids=torch.as_tensor(tokens,device=device)[None],position_ids=positions[None],cache_position=positions,
                       use_cache=False,output_attentions=False,output_hidden_states=False,logits_to_keep=1)
        if isinstance(self.controller,DenseController):
            self.controller.position+=len(tokens)
        else:
            self.controller.finish_token()
        return out.logits[0,-1]

    def step(self,token_id,absolute_pos):
        return self.forward_tokens([int(token_id)],absolute_pos)

    def consume_pending(self):
        token=self.controller.pending_token
        if token is None:
            raise ValueError('no pending token')
        logits=self.step(token,self.position)
        self.controller.pending_token=None
        return logits

    def uninstall(self):
        for layer,original in zip(self.model.model.layers,self.originals):
            layer.self_attn.forward=original
        self.model.model._update_causal_mask=self.original_mask
        del self.model._tiered_adapter

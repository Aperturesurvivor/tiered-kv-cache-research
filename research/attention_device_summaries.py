"""Bounded device-side QK summaries, independent of the frozen selector.

This helper computes float32 page log masses and one fixed-key log normalizer
per query head on the input device. It performs no ranking or key selection.
The caller must provide complete pages in its chosen ID order and include each
sink, boundary, recent and current key exactly once in ``fixed_keys``.
"""
import math
import torch

MAX_PAGES=32
MAX_PAGE_TOKENS=128
MAX_FIXED_TOKENS=2255  # sink 128 + boundary at most 127 + recent/current 2000
MAX_HEADS=128
MAX_DIM=256


def score_summaries(query,candidate_keys,fixed_keys):
    """Return (page_logmass, fixed_logsumexp, logical byte accounting).

    Shapes are [query_heads, dim], [kv_heads, pages, tokens_per_page, dim],
    [kv_heads, fixed_tokens, dim]. GQA uses contiguous query-head groups as in
    transformers repeat_kv. Empty/underfilled pages are valid; fixed keys must
    include at least the current causal key. No host copies of results occur.
    Finite checks intentionally synchronize one combined device scalar.
    """
    if not all(isinstance(x,torch.Tensor) for x in (query,candidate_keys,fixed_keys)):
        raise TypeError('query and keys must be tensors')
    if query.ndim!=2 or candidate_keys.ndim!=4 or fixed_keys.ndim!=3:
        raise ValueError('invalid tensor ranks')
    heads,dim=query.shape;kv,pages,page_tokens,key_dim=candidate_keys.shape
    if (not 0<heads<=MAX_HEADS or not 0<dim<=MAX_DIM or not 0<kv<=heads or heads%kv or
            key_dim!=dim or not 0<=pages<=MAX_PAGES or not 0<page_tokens<=MAX_PAGE_TOKENS or
            fixed_keys.shape[0]!=kv or fixed_keys.shape[2]!=dim or not 0<fixed_keys.shape[1]<=MAX_FIXED_TOKENS):
        raise ValueError('geometry or workspace exceeds declared bounds')
    tensors=(query,candidate_keys,fixed_keys)
    if any(x.device!=query.device for x in tensors) or any(x.dtype not in (torch.float16,torch.bfloat16,torch.float32) for x in tensors):
        raise ValueError('inputs must share a device and use floating-point keys and queries')
    q=query.float().reshape(kv,heads//kv,dim)
    fixed=fixed_keys.float()
    scale=math.sqrt(dim)
    fixed_logits=torch.bmm(q,fixed.transpose(1,2)).reshape(heads,-1)/scale
    fixed_lse=torch.logsumexp(fixed_logits,dim=1)
    if pages:
        candidates=candidate_keys.float().reshape(kv,pages*page_tokens,dim)
        logits=torch.bmm(q,candidates.transpose(1,2)).reshape(heads,pages,page_tokens)/scale
        page_lse=torch.logsumexp(logits,dim=2)
    else:
        page_lse=torch.empty((heads,0),dtype=torch.float32,device=query.device)
    # One synchronization permits a fail-closed response for invalid input or
    # overflow. It does not transfer candidate-key payload to the host.
    finite=torch.isfinite(query).all() & torch.isfinite(candidate_keys).all() & torch.isfinite(fixed_keys).all()
    finite=finite & torch.isfinite(fixed_lse).all() & torch.isfinite(page_lse).all()
    if not bool(finite.item()):raise ValueError('non-finite input or score')
    def size(x):return x.numel()*x.element_size()
    accounting={
        'query_input_bytes':size(query),'candidate_key_input_bytes':size(candidate_keys),'fixed_key_input_bytes':size(fixed_keys),
        'output_summary_bytes':size(page_lse)+size(fixed_lse),
        'float32_key_query_cast_bytes_upper_bound':4*(query.numel()+candidate_keys.numel()+fixed_keys.numel()),
        'float32_logit_bytes':4*heads*(pages*page_tokens+fixed_keys.shape[1]),
        'validation_scalar_syncs':1 if query.device.type!='cpu' else 0,
        'validation_scalar_host_logical_bytes':1 if query.device.type!='cpu' else 0,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'cuda_matmul_tf32_allowed':bool(torch.backends.cuda.matmul.allow_tf32) if query.device.type=='cuda' else None,
        'scope':'logical tensor payloads, not measured allocations or peak workspace; finite checks and library temporaries add memory; no output is copied to host',
    }
    return page_lse,fixed_lse,accounting

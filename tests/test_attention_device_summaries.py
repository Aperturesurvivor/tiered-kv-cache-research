import math
import numpy as np
import pytest
import torch
from research.attention_device_summaries import score_summaries
from research.attention_representations import _lse,_shared_rank


def reference(q,candidates,fixed):
    heads,d=q.shape;kv,pages,page,_=candidates.shape
    grouped=q.float().reshape(kv,heads//kv,d)
    fixed_logits=torch.bmm(grouped,fixed.float().transpose(1,2)).reshape(heads,-1).numpy()/np.float32(math.sqrt(d))
    if pages:
        logits=torch.bmm(grouped,candidates.float().reshape(kv,pages*page,d).transpose(1,2)).reshape(heads,pages,page).numpy()/np.float32(math.sqrt(d))
        masses=_lse(logits,axis=2)
    else:masses=np.empty((heads,0),np.float32)
    return masses,_lse(fixed_logits,axis=1),fixed_logits


@pytest.mark.parametrize('heads,kv,pages',[(6,2,3),(4,4,1),(6,2,0)])
def test_cpu_summaries_match_frozen_formula_and_accounting(heads,kv,pages):
    torch.manual_seed(17);q=torch.randn(heads,8).half();candidate=torch.randn(kv,pages,4,8).half();fixed=torch.randn(kv,11,8).half()
    expected,normalizer,fixed_logits=reference(q,candidate,fixed)
    got,fixed_got,meta=score_summaries(q,candidate,fixed)
    np.testing.assert_allclose(got.numpy(),expected,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(fixed_got.numpy(),normalizer,atol=2e-6,rtol=2e-6)
    assert got.device==q.device and got.shape==(heads,pages) and fixed_got.shape==(heads,)
    assert meta['candidate_key_input_bytes']==candidate.numel()*2
    assert meta['fixed_key_input_bytes']==fixed.numel()*2
    assert meta['output_summary_bytes']==heads*(pages+1)*4
    assert meta['float32_logit_bytes']==heads*(pages*4+11)*4 and meta['validation_scalar_syncs']==0
    if pages:
        ids=np.arange(11,11+pages)
        old=_shared_rank(expected,fixed_logits,ids,min(pages,2))
        # Compare ranking using only the returned summaries with the reference.
        cold=_lse(got.numpy(),axis=1);denom=np.logaddexp(fixed_got.numpy(),cold)
        score=np.exp(got.numpy()-denom[:,None]).sum(axis=0)
        assert ids[np.lexsort((ids,-score))[:min(pages,2)]].tolist()==old


def test_nonfinite_shape_and_capacity_fail_closed():
    q=torch.ones(4,8);keys=torch.ones(2,1,4,8);fixed=torch.ones(2,7,8)
    for tensor in (q,keys,fixed):
        bad=tensor.clone();bad.view(-1)[0]=float('nan')
        args=[q,keys,fixed];args[next(i for i,t in enumerate((q,keys,fixed)) if t is tensor)]=bad
        with pytest.raises(ValueError,match='non-finite'):score_summaries(*args)
    with pytest.raises(ValueError,match='geometry'):score_summaries(q,torch.ones(2,33,4,8),fixed)
    with pytest.raises(ValueError,match='geometry'):score_summaries(torch.ones(3,8),keys,fixed)
    with pytest.raises(ValueError,match='geometry'):score_summaries(q,keys,torch.empty(2,0,8))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA device unavailable')
def test_cuda_values_against_independent_cpu_reference():
    torch.manual_seed(20);q=torch.randn(12,32).half();keys=torch.randn(2,7,8,32).half();fixed=torch.randn(2,21,32).half()
    expected,normalizer,_=reference(q,keys,fixed)
    got,actual,meta=score_summaries(q.cuda(),keys.cuda(),fixed.cuda())
    np.testing.assert_allclose(got.cpu().numpy(),expected,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(actual.cpu().numpy(),normalizer,atol=2e-5,rtol=2e-5)
    assert meta['validation_scalar_syncs']==1


def test_extreme_finite_logits_keep_stable_logsumexp():
    q=torch.full((4,8),128.,dtype=torch.float16)
    pages=torch.empty(2,2,4,8,dtype=torch.float16);pages[:,0].fill_(128.);pages[:,1].fill_(-128.)
    fixed=torch.full((2,9,8),-128.,dtype=torch.float16)
    expected,normalizer,_=reference(q,pages,fixed)
    got,actual,_=score_summaries(q,pages,fixed)
    assert torch.isfinite(got).all() and torch.isfinite(actual).all()
    np.testing.assert_allclose(got.numpy(),expected,atol=.02,rtol=2e-6)
    np.testing.assert_allclose(actual.numpy(),normalizer,atol=.02,rtol=2e-6)

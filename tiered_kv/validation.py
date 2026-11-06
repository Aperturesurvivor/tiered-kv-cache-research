"""Ordered acceptance gates. Unit tests alone never certify the MPS model."""
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .adapter import ModelAdapter, DenseController
from .generate import sync, load_model, weights_digest, online_smoke
from .store import atomic_json, file_sha
from .telemetry import doctor, ResourceGuard, disk_preflight, runtime_identity
from .types import digest


def implementation_hash():
    files=sorted(Path('tiered_kv').glob('*.py'))+sorted(Path('tests').glob('*.py'))+[Path('configs/matrix.yaml'),Path('configs/eval.yaml'),Path('requirements.lock')]
    return digest({str(f):file_sha(f) for f in files})


def parity(model,length,tokens=None,probe='synthetic'):
    """Sequential stock and patched execution on the very same weights."""
    device=next(model.parameters()).device
    tolerance=1e-2 if device.type=='mps' else 1e-4
    tokens=tokens if tokens is not None else [(i*17+11)%min(model.config.vocab_size,10000) for i in range(length)]
    if len(tokens)!=length:
        raise ValueError('parity probe length mismatch')
    expected=[]
    generated=[]
    with torch.inference_mode():
        out=model(torch.tensor([tokens],device=device),use_cache=True,logits_to_keep=1)
        past=out.past_key_values
        expected.append(out.logits[0,-1].detach().cpu())
        for _ in range(16):
            token=int(out.logits[0,-1].argmax())
            generated.append(token)
            out=model(torch.tensor([[token]],device=device),past_key_values=past,use_cache=True,logits_to_keep=1)
            past=out.past_key_values
            expected.append(out.logits[0,-1].detach().cpu())
    del past,out
    # Stock with the production chunk schedule is the logic reference. Comparing
    # different GEMM/attention shapes confounds implementation error with fp16
    # rounding: the saved T4 diagnostic found identical stock/adapter drift.
    chunk_expected=[]
    chunk_decode=[]
    chunk_generated=[]
    with torch.inference_mode():
        past=None
        for offset in range(0,length,128):
            out=model(torch.tensor([tokens[offset:offset+128]],device=device),past_key_values=past,use_cache=True,logits_to_keep=1)
            past=out.past_key_values
            chunk_expected.append(out.logits[0,-1].detach().float().cpu())
        for _ in range(16):
            token=int(out.logits[0,-1].argmax())
            chunk_generated.append(token)
            out=model(torch.tensor([[token]],device=device),past_key_values=past,use_cache=True,logits_to_keep=1)
            past=out.past_key_values
            chunk_decode.append(out.logits[0,-1].detach().float().cpu())
    del past,out
    adapter=ModelAdapter.install(model,DenseController(model.config.num_hidden_layers))
    errors=[]
    margins=[]
    greedy=[]
    try:
        got=adapter.forward_tokens(tokens,0)
        for step,want in enumerate(expected):
            actual=got.detach().cpu()
            errors.append(float((actual-want).abs().max()))
            margin=float(torch.topk(want.float(),2).values.diff().abs()[0])
            margins.append(margin)
            agreement=int(actual.argmax())==int(want.argmax())
            greedy.append(agreement)
            if not torch.allclose(actual,want,atol=tolerance,rtol=tolerance):
                raise ValueError(f'parity length {length} step {step}: max error {errors[-1]}')
            # Non-near-tie: actual error is smaller than half the reference margin.
            if margin>2*errors[-1] and not agreement:
                raise ValueError('non-near-tie greedy mismatch')
            if step<16:
                got=adapter.step(generated[step],adapter.position)
        adapter.set_controller(DenseController(model.config.num_hidden_layers))
        chunk_metrics=[]
        def check_chunk(got,want,label):
            actual=got.detach().float().cpu()
            error=float((actual-want).abs().max())
            if not bool(torch.isfinite(actual).all()) or not torch.allclose(actual,want,atol=tolerance,rtol=tolerance):
                raise ValueError(f'same-chunk stock parity {label}: max error {error}')
            chunk_metrics.append({'label':label,'max_abs':error,'greedy_agreement':int(actual.argmax())==int(want.argmax()),
                                  'stock_margin':float(want.topk(2).values.diff().abs()[0])})
        for block,offset in enumerate(range(0,length,128)):
            got=adapter.forward_tokens(tokens[offset:offset+128],offset)
            check_chunk(got,chunk_expected[block],f'prefill-{offset+len(tokens[offset:offset+128])}')
        actual=got.detach().float().cpu()
        one_shot=expected[0].float()
        cross_shape={'adapter_max_abs':float((actual-one_shot).abs().max()),
                     'stock_max_abs':float((chunk_expected[-1]-one_shot).abs().max()),
                     'stock_greedy_agreement':int(chunk_expected[-1].argmax())==int(one_shot.argmax()),
                     'scope':'diagnostic only: chunked vs one-shot floating-point computation'}
        for step,(token,want) in enumerate(zip(chunk_generated,chunk_decode)):
            got=adapter.step(token,adapter.position)
            check_chunk(got,want,f'decode-{step+1}')
        return {'length':length,'probe':probe,'max_errors':errors,'argmax_margins':margins,'greedy_agreement':greedy,
                'same_chunk_stock':chunk_metrics,'cross_shape':cross_shape,'atol':tolerance,'rtol':tolerance}
    finally:
        adapter.uninstall()


def validate(through=5):
    disk_preflight()
    root=Path('results')/'validation'
    root.mkdir(parents=True,exist_ok=True)
    state={'implementation_hash':implementation_hash(),'runtime':runtime_identity(),'gates':[]}
    env=doctor(root/'doctor.json')
    state['gates'].append({'gate':1,'status':env['status'],'evidence':'doctor.json'})
    atomic_json(root/'validation.json',state)
    if env['status']!='passed' or through==1:
        return state
    def tests(gate,arguments):
        log=root/f'gate-{gate}.txt'
        proc=subprocess.run([sys.executable,'-m','pytest','-q',*arguments],capture_output=True,text=True)
        log.write_text(proc.stdout+proc.stderr)
        if proc.returncode:
            raise RuntimeError(f'gate {gate} tests failed; see {log}')
        if 'skipped' in proc.stdout:
            raise RuntimeError(f'gate {gate}: skipped required test')
        return str(log)
    model=None
    try:
        with ResourceGuard() as guard:
            if through>=2:
                tests(2,['tests/test_attention.py','-k','dense_adapter'])
                model,tok=load_model()
                before=weights_digest(model)
                measurements=[parity(model,n) for n in (64,512,1025)]
                natural=tok.encode('The archive contains records about cities, rivers, engineering projects, and public reports. Read each record carefully and answer using the supplied evidence. '*150)
                measurements.extend(parity(model,n,natural[:n],probe='natural_text') for n in (512,1025))
                if weights_digest(model)!=before:
                    raise ValueError('frozen model weights changed')
                state['gates'].append({'gate':2,'status':'passed','measurements':measurements,'weights_sha256':before})
                atomic_json(root/'validation.json',state)
            if through>=3:
                log=tests(3,['tests/test_attention.py','-k','sparse_gather'])
                state['gates'].append({'gate':3,'status':'passed','evidence':log})
                atomic_json(root/'validation.json',state)
            if through>=4:
                log=tests(4,['tests/test_cache.py','-k','compression or recent or rotary or bounds'])
                state['gates'].append({'gate':4,'status':'passed','evidence':log})
                atomic_json(root/'validation.json',state)
            if through>=5:
                log=tests(5,['tests/test_cache.py','-k','store or index or online'])
                from .config import load_matrix
                configs=[c for c in load_matrix('configs/matrix.yaml') if c.id in ('full','default')]
                rows=online_smoke(model,tok,root,configs,guard)
                if len(rows)!=2 or any(r['status']!='ok' for r in rows):
                    raise RuntimeError('2K full/default online/resume validation failed')
                if weights_digest(model)!=before:
                    raise ValueError('model weights changed during online validation')
                state['gates'].append({'gate':5,'status':'passed','evidence':log,'online':'online.jsonl','weights_unchanged':True})
    except Exception as exc:
        next_gate=max((g['gate'] for g in state['gates']),default=0)+1
        state['gates'].append({'gate':next_gate,'status':'failed','failure_reason':f'{type(exc).__name__}: {exc}'})
    state['implementation_hash']=implementation_hash()
    atomic_json(root/'validation.json',state)
    return state


def require_gates():
    path=Path('results/validation/validation.json')
    if not path.exists():
        raise RuntimeError('run validate --through 5 before the matrix')
    state=json.loads(path.read_text())
    if state['implementation_hash']!=implementation_hash():
        raise RuntimeError('implementation changed since validation; rerun ordered gates')
    if state.get('runtime')!=runtime_identity():
        raise RuntimeError('runtime changed or absent since validation; rerun ordered gates')
    if [(g['gate'],g['status']) for g in state['gates']]!=[(i,'passed') for i in range(1,6)]:
        raise RuntimeError('ordered gates 1–5 have not all passed')
    return state

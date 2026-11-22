"""Real-weight interface gates before any Dream experiments."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from .official_reference import load_official_model, official_steps, namespace
from .pretrained_model import PretrainedMamba2, ssd_chunk

ROOT=Path('artifacts/pretrained-v1')


def error(a,b):
    return {'max_absolute':float((a-b).abs().max()),'rms':float((a-b).square().mean().sqrt())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=PretrainedMamba2.load(device=args.device)
    tokenizer=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    docs=list((ROOT/'data/wikitext103/train').glob('*.pt'))
    if not docs: raise RuntimeError('Prepare real training documents before interface validation')
    record=torch.load(docs[0],weights_only=True)
    tokens=record['tokens'][:600][None].to(args.device)
    result={'model':model.asset_manifest,'torch_version':str(torch.__version__),
            'device':str(args.device),'dtype':'float32','document':record['metadata'],
            'selected_layer':12,'official_sources':'state-spaces/mamba v2.2.5',
            'backend':'functional PyTorch SSD, compared to unmodified official Python definitions; native CUDA kernels not used'}
    with torch.no_grad():
        full,cache=model.run(tokens[:,:95])
        state=None;pieces=[]
        for a,b in [(0,17),(17,40),(40,95)]:
            logits,state=model.run(tokens[:,a:b],state);pieces.append(logits)
        split=torch.cat(pieces,1)
        result['continuous_vs_split_logits']=error(full,split)
        result['continuous_vs_split_ssm_max']=max(float((x.ssm-y.ssm).abs().max()) for x,y in zip(cache.layers,state.layers))
        torch.testing.assert_close(full,split,atol=1e-3,rtol=1e-4)
        for x,y in zip(cache.layers,state.layers):
            torch.testing.assert_close(x.ssm,y.ssm,atol=2e-4,rtol=2e-4)
            torch.testing.assert_close(x.conv,y.conv,atol=2e-4,rtol=2e-4)
    print('Segmented equivalence passed',result['continuous_vs_split_logits'],flush=True)
    weights=torch.load(ROOT/'model/pytorch_model.bin',map_location='cpu',mmap=True,weights_only=True)
    config=json.loads((ROOT/'model/config.json').read_text())
    reference=load_official_model(weights,config,args.device)
    with torch.no_grad():
        actual,actual_state=model.run(tokens[:,:16])
        expected,expected_state=official_steps(reference,tokens[:,:16])
        result['official_step_logits']=error(actual,expected)
        result['official_step_ssm_max']=max(float((s.ssm-expected_state[i][1]).abs().max()) for i,s in enumerate(actual_state.layers))
        torch.testing.assert_close(actual,expected,atol=1e-3,rtol=1e-4)
        for i,s in enumerate(actual_state.layers):
            torch.testing.assert_close(s.ssm,expected_state[i][1],atol=2e-4,rtol=2e-4)
            torch.testing.assert_close(s.conv,expected_state[i][0],atol=2e-4,rtol=2e-4)
    del reference,weights,actual_state,expected_state,actual,expected,full,cache,state,split,pieces
    gc.collect();torch.cuda.empty_cache()
    print('Official full-model step equivalence passed',result['official_step_logits'],flush=True)
    # Independent official SSD reference and a double-precision initial-state gradcheck.
    ns=namespace();torch.manual_seed(1)
    x=torch.randn(1,8,2,3,dtype=torch.float64)
    dt=torch.rand(1,8,2,dtype=torch.float64)*.1
    a=-torch.tensor([.5,2.],dtype=torch.float64)
    b=torch.randn(1,8,5,dtype=torch.float64);c=torch.randn_like(b)
    h=torch.randn(1,2,3,5,dtype=torch.float64,requires_grad=True)
    y,end=ssd_chunk(x,dt,a,b,c,h)
    expected,expected_end=ns['ssd_minimal_discrete'](x*dt[...,None],dt*a,
        b[:,:,None].expand(-1,-1,2,-1),c[:,:,None].expand(-1,-1,2,-1),8,initial_states=h[:,None])
    torch.testing.assert_close(y,expected);torch.testing.assert_close(end,expected_end)
    result['ssd_reference_and_gradcheck']=bool(torch.autograd.gradcheck(lambda initial:ssd_chunk(x,dt,a,b,c,initial),(h,)))
    # Anchor is observed, but has not yet been processed by the LM.
    boundary=512;length=32
    start=time.monotonic()
    cache=model.wake(tokens[:,:boundary-1])
    selected=cache.layers[12].ssm.detach().requires_grad_(True)
    edited=cache.edit(12,selected)
    assert edited.layers[12].conv is cache.layers[12].conv
    assert all(edited.layers[i] is cache.layers[i] for i in range(24) if i!=12)
    inputs=tokens[:,boundary-1:boundary+length-1]
    targets=tokens[:,boundary:boundary+length]
    logits,_=model.run(inputs,edited)
    loss=F.cross_entropy(logits.flatten(0,1),targets.flatten())
    loss.backward()
    result['base_nll_32']=float(loss.detach())
    result['injected_state_gradient_norm']=float(selected.grad.norm())
    result['all_base_parameters_frozen']=all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert result['injected_state_gradient_norm']>0 and result['all_base_parameters_frozen']
    with torch.no_grad():
        joined,_=model.run(tokens[:,:boundary+length-1])
        result['anchor_alignment']=error(joined[:,boundary-1:],logits)
        torch.testing.assert_close(joined[:,boundary-1:],logits,atol=1e-3,rtol=1e-4)
        unmodified=logits.detach()
        torch.manual_seed(2026)
        for name,state in [('zero',torch.zeros_like(selected)),
                           ('mask10',selected.detach()*(torch.rand_like(selected)>.1))]:
            changed,_=model.run(inputs,cache.edit(12,state))
            difference=error(unmodified,changed)
            difference['first_future_token_max_difference']=float((unmodified[:,0]-changed[:,0]).abs().max())
            difference['nll']=float(F.cross_entropy(changed.flatten(0,1),targets.flatten()))
            assert difference['first_future_token_max_difference']>1e-5
            result[name+'_intervention']=difference
    result['seconds_state_gradient_check']=time.monotonic()-start
    result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None
    result['selected_state_shape']=list(selected.shape)
    result['passed']=True
    (ROOT/'interface-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['model','document']},indent=2),flush=True)

if __name__=='__main__':main()

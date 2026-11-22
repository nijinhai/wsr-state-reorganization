"""Functional PyTorch execution of the official Mamba-2 weights and equations.

The SSD contraction follows the official minimal SSD and Mamba2.step references.
No state compression, SSM simplification, or changes to the pretrained architecture.
Cache updates are functional so gradients can reach an injected SSM state.
"""
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LayerCache:
    conv: torch.Tensor
    ssm: torch.Tensor


@dataclass(frozen=True)
class Cache:
    layers: tuple
    processed: int = 0

    def detach(self):
        return Cache(tuple(LayerCache(x.conv.detach(),x.ssm.detach()) for x in self.layers),self.processed)

    def edit(self,layer,ssm):
        old=self.layers[layer]
        if ssm.shape!=old.ssm.shape or ssm.dtype!=old.ssm.dtype or ssm.device!=old.ssm.device:
            raise ValueError('Injected SSM shape/dtype/device must match the real cache')
        layers=list(self.layers)
        layers[layer]=LayerCache(old.conv,ssm)
        return Cache(tuple(layers),self.processed)

    def edit_many(self, edits):
        """Functionally replace selected SSMs, validating every edit before injection."""
        for layer, ssm in edits.items():
            if type(layer) is not int or not 0 <= layer < len(self.layers):
                raise ValueError('Edit layer must be a valid zero-based integer index')
            old = self.layers[layer].ssm
            if (ssm.shape, ssm.dtype, ssm.device) != (old.shape, old.dtype, old.device):
                raise ValueError(f'Injected SSM shape/dtype/device mismatch at layer {layer}')
        layers = tuple(LayerCache(old.conv, edits[i]) if i in edits else old
                       for i, old in enumerate(self.layers))
        return Cache(layers, self.processed)


class RMSNorm(nn.Module):
    def __init__(self,width,eps=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(width))
        self.eps=eps

    def forward(self,x,z=None):
        dtype=x.dtype
        x=x.float()
        # Official Mamba2 defaults to norm_before_gate=False.
        if z is not None: x=x*F.silu(z.float())
        return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.eps)*self.weight.float()).to(dtype)


def segment_decay(log_decay):
    """Stable exp(sum_{k=j+1..i} log_decay[k]); never exponentiate masked upper values."""
    size=log_decay.shape[-1]
    values=log_decay[..., :,None].expand(*log_decay.shape,size)
    strict=torch.ones(size,size,dtype=torch.bool,device=values.device).tril(-1)
    sums=values.masked_fill(~strict,0).cumsum(-2)
    lower=torch.ones(size,size,dtype=torch.bool,device=values.device).tril()
    return sums.masked_fill(~lower,-torch.inf).exp()


def ssd_chunk(x,dt,a,b,c,initial):
    """One exact SSD block, shared B/C group. x=(B,T,H,P), b/c=(B,T,N)."""
    log_decay=(dt*a).transpose(1,2)
    decay=segment_decay(log_decay)
    cross=torch.einsum('btn,bsn->bts',c,b)
    inputs=x*dt[...,None]
    within=torch.einsum('bhts,bshp->bthp',decay*cross[:,None],inputs)
    from_initial=torch.einsum('btn,bhpn->bthp',c,initial)*log_decay.cumsum(-1).exp().transpose(1,2)[...,None]
    final=initial*log_decay.sum(-1).exp()[...,None,None]+torch.einsum('bhs,bshp,bsn->bhpn',decay[:,:,-1],inputs,b)
    return within+from_initial,final


class Mixer(nn.Module):
    def __init__(self,width,state=128,headdim=64,conv=4,block=64):
        super().__init__()
        self.d_model=width
        self.d_inner=self.d_ssm=2*width
        self.d_state=state
        self.d_conv=conv
        self.headdim=headdim
        self.nheads=self.d_ssm//headdim
        self.ngroups=1
        self.rmsnorm=True
        self.process_group=None
        self.activation='silu'
        self.act=nn.SiLU()
        self.block=block
        self.in_proj=nn.Linear(width,2*self.d_ssm+2*state+self.nheads,bias=False)
        self.conv1d=nn.Conv1d(self.d_ssm+2*state,self.d_ssm+2*state,conv,
                            groups=self.d_ssm+2*state,padding=conv-1,bias=True)
        self.dt_bias=nn.Parameter(torch.zeros(self.nheads))
        self.A_log=nn.Parameter(torch.zeros(self.nheads))
        self.D=nn.Parameter(torch.ones(self.nheads))
        self.norm=RMSNorm(self.d_ssm)
        self.out_proj=nn.Linear(self.d_ssm,width,bias=False)

    def initial(self,batch,device,dtype):
        return LayerCache(torch.zeros(batch,self.conv1d.in_channels,self.d_conv,device=device,dtype=dtype),
                          torch.zeros(batch,self.nheads,self.headdim,self.d_state,device=device,dtype=dtype))

    def forward(self,u,state):
        projected=self.in_proj(u)
        z,xbc,raw_dt=projected.split([self.d_ssm,self.d_ssm+2*self.d_state,self.nheads],-1)
        # Official cache holds the latest d_conv pre-convolution inputs, including the newest one.
        joined=torch.cat([state.conv[:,:,1:],xbc.transpose(1,2)],-1)
        convolved=F.conv1d(joined,self.conv1d.weight,self.conv1d.bias,groups=self.conv1d.groups)
        convolved=F.silu(convolved.transpose(1,2))
        x,b,c=convolved.split([self.d_ssm,self.d_state,self.d_state],-1)
        x=x.reshape(u.shape[0],u.shape[1],self.nheads,self.headdim)
        dt=F.softplus(raw_dt+self.dt_bias)
        a=-self.A_log.float().exp()
        h=state.ssm
        outputs=[]
        for start in range(0,u.shape[1],self.block):
            end=start+self.block
            y,h=ssd_chunk(x[:,start:end],dt[:,start:end],a,b[:,start:end],c[:,start:end],h)
            outputs.append(y)
        y=torch.cat(outputs,1)+self.D[None,None,:,None]*x
        y=self.norm(y.flatten(2),z)
        conv=torch.cat([state.conv,xbc.transpose(1,2)],-1)[:,:,-self.d_conv:]
        return self.out_proj(y),LayerCache(conv,h)


class Block(nn.Module):
    def __init__(self,width,**kwargs):
        super().__init__()
        self.norm=RMSNorm(width)
        self.mixer=Mixer(width,**kwargs)


class Body(nn.Module):
    def __init__(self,width,layers,vocab,**kwargs):
        super().__init__()
        self.embedding=nn.Embedding(vocab,width)
        self.layers=nn.ModuleList([Block(width,**kwargs) for _ in range(layers)])
        self.norm_f=RMSNorm(width)


class PretrainedMamba2(nn.Module):
    def __init__(self,width=768,layers=24,vocab=50288,state=128,headdim=64,conv=4,block=64):
        super().__init__()
        self.backbone=Body(width,layers,vocab,state=state,headdim=headdim,conv=conv,block=block)
        self.lm_head=nn.Linear(width,vocab,bias=False)
        self.lm_head.weight=self.backbone.embedding.weight
        self.width=width

    @classmethod
    def load(cls,root='artifacts/pretrained-v1/model',device='cpu',block=64):
        root=Path(root)
        config=json.loads((root/'config.json').read_text())
        if config['ssm_cfg']!={'layer':'Mamba2'} or config['d_intermediate']!=0 or config['attn_layer_idx']:
            raise ValueError('Unsupported official configuration; do not silently approximate it')
        vocab=math.ceil(config['vocab_size']/config['pad_vocab_size_multiple'])*config['pad_vocab_size_multiple']
        model=cls(width=config['d_model'],layers=config['n_layer'],vocab=vocab,block=block)
        weights=torch.load(root/'pytorch_model.bin',map_location='cpu',mmap=True,weights_only=True)
        model.load_state_dict(weights,strict=True)
        if not torch.equal(weights['lm_head.weight'],weights['backbone.embedding.weight']):
            raise ValueError('Expected tied official embeddings')
        model.requires_grad_(False).eval().to(device)
        model.asset_manifest=json.loads((root/'manifest.json').read_text())
        return model

    def initial(self,batch=1):
        p=self.backbone.embedding.weight
        return Cache(tuple(layer.mixer.initial(batch,p.device,p.dtype) for layer in self.backbone.layers))

    def run(self,tokens,cache=None,num_last_tokens=0,return_hidden=False):
        if tokens.ndim!=2 or not tokens.shape[1]: raise ValueError('Expected nonempty B,T token tensor')
        cache=self.initial(tokens.shape[0]) if cache is None else cache
        x=self.backbone.embedding(tokens)
        residual=None
        states=[]
        for layer,old in zip(self.backbone.layers,cache.layers):
            residual=x if residual is None else residual+x
            x=layer.norm(residual)
            x,new=layer.mixer(x,old)
            states.append(new)
        x=self.backbone.norm_f(x+residual)
        if num_last_tokens: x=x[:,-num_last_tokens:]
        return (x if return_hidden else self.lm_head(x)),Cache(tuple(states),cache.processed+tokens.shape[1])

    @torch.no_grad()
    def wake(self,tokens,cache=None,chunk=256):
        for start in range(0,tokens.shape[1],chunk):
            _,cache=self.run(tokens[:,start:start+chunk],cache,num_last_tokens=1)
        return cache

"""Execute pinned official Python references, independently of the functional adapter.

Original files and Apache-2.0 license are retained under artifacts/pretrained-v1/vendor.
Only official class/function definitions are loaded; CUDA-extension imports are not executed.
The official non-fused normalization reference replaces optional Triton normalization.
"""
import ast
import copy
from functools import partial
from collections import namedtuple
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat

VENDOR=Path('artifacts/pretrained-v1/vendor')
if not VENDOR.exists():
    # Public release: bundled pinned Python references, without downloaded weights.
    VENDOR=Path(__file__).resolve().parents[1]/'third_party/mamba'


def definitions(filename,names,namespace):
    tree=ast.parse((VENDOR/filename).read_text())
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes}!=set(names): raise ValueError('Official source definitions changed')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(VENDOR/filename),'exec'),namespace)


def namespace():
    ns={'torch':torch,'nn':nn,'F':F,'math':math,'rearrange':rearrange,'repeat':repeat,
        'Optional':Optional,'Tensor':Tensor,'namedtuple':namedtuple,'partial':partial,'copy':copy,'MambaConfig':SimpleNamespace,
        'PyTorchModelHubMixin':type('HubMixin',(),{}),'GenerationMixin':type('GenerationMixin',(),{}),
        'causal_conv1d_update':None,'selective_state_update':None,'RMSNorm':None,
        'layer_norm_fn':None,'rms_norm_fn':None}
    definitions('layernorm_gated.py',['rms_norm_ref'],ns)
    ref_norm=ns['rms_norm_ref']
    class ReferenceNorm(nn.Module):
        def __init__(self,hidden_size,eps=1e-5,norm_before_gate=False,group_size=None,device=None,dtype=None):
            super().__init__()
            self.weight=nn.Parameter(torch.ones(hidden_size,device=device,dtype=dtype))
            self.eps=eps
            self.group_size=group_size
            self.norm_before_gate=norm_before_gate
        def forward(self,x,z=None):
            return ref_norm(x,self.weight,None,z=z,eps=self.eps,group_size=self.group_size,
                            norm_before_gate=self.norm_before_gate)
    ns['RMSNorm']=ns['RMSNormGated']=ReferenceNorm
    definitions('mamba2.py',['Mamba2'],ns)
    definitions('block.py',['Block'],ns)
    definitions('mixer_seq_simple.py',['create_block','_init_weights','MixerModel','MambaLMHeadModel'],ns)
    definitions('ssd_minimal.py',['segsum','ssd_minimal_discrete'],ns)
    return ns


def load_official_model(weights,config,device):
    ns=namespace()
    config=SimpleNamespace(**config)
    config.fused_add_norm=False # Official supported path; equations and weight names are unchanged.
    model=ns['MambaLMHeadModel'](config,device=device,dtype=torch.float32)
    model.load_state_dict(weights,strict=True)
    return model.requires_grad_(False).eval()


@torch.no_grad()
def official_steps(model,tokens,cache=None):
    if cache is None:
        cache=model.allocate_inference_cache(tokens.shape[0],tokens.shape[1])
    params=SimpleNamespace(key_value_memory_dict=cache,seqlen_offset=1)
    logits=[]
    for token in tokens.unbind(1):
        logits.append(model(token[:,None],inference_params=params).logits)
        params.seqlen_offset+=1
    return torch.cat(logits,1),cache

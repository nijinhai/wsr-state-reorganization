"""Counterfactuals within the original Direct Dream computation, with no new memory."""
import torch
from torch.nn import functional as F


def norm_matched_first(initial,trace):
    first=trace[0]-initial;total=trace[-1]-initial
    ratio=total.double().norm()/first.double().norm().clamp_min(1e-20)
    return initial+first*ratio.to(initial.dtype)


def rollout(dream,state,history=None,intervention=None,at=4,rounds=8):
    context=state.new_zeros(state.shape[0],dream.hidden)
    if history is not None and history.shape[1]:
        _,encoded=dream.history_gru(F.silu(dream.history_project(history)));context=encoded[0]
    scale=state.detach().square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-4)
    h=state;z=torch.zeros_like(context);trace=[];memories=[]
    head=dream.head_embedding(torch.arange(h.shape[1],device=h.device))[None,:,None]
    for k in range(rounds):
        if k==at:
            if intervention=='reset_state':h=state
            elif intervention=='reverse_state':h=2*state-h
            elif intervention=='reset_memory':z=torch.zeros_like(z)
            elif intervention is not None:raise ValueError(intervention)
        read=F.silu(dream.read(h/scale))
        if dream.mode=='D':
            z=dream.recur(torch.cat([read.mean((1,2)),context],-1),z);control=z
        else:control=context
        control=control[:,None,None].expand(-1,h.shape[1],h.shape[2],-1)
        ids=head.expand(h.shape[0],-1,h.shape[2],-1)
        delta=.1*scale*dream.editor(torch.cat([read,control,ids],-1))
        h=h+delta;trace.append(h);memories.append(z)
    return h,trace,memories


def trajectory(initial,trace):
    # Float64 diagnostics prevent cosines slightly beyond [-1,1].
    previous=initial;deltas=[];rows=[]
    for i,h in enumerate(trace):
        d=(h-previous).detach().double().flatten();deltas.append(d)
        total=(h-initial).detach().double().flatten()
        rows.append({'round':i+1,'delta_norm':float(d.norm()),'total_norm':float(total.norm()),
                     'relative_total':float(total.norm()/initial.double().norm().clamp_min(1e-20)),
                     'state_norm_ratio':float(h.double().norm()/initial.double().norm().clamp_min(1e-20)),
                     'cosine_total_edit_with_initial':float(F.cosine_similarity(total[None],initial.detach().double().flatten()[None])),
                     'projection_onto_initial':float((total*initial.detach().double().flatten()).sum()/initial.double().square().sum().clamp_min(1e-20)),
                     'cosine_with_first_delta':float(F.cosine_similarity(d[None],deltas[0][None])),
                     'cosine_with_previous_delta':None if i==0 else float(F.cosine_similarity(d[None],deltas[-2][None]))})
        previous=h
    return rows

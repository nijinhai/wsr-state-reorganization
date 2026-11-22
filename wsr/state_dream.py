"""Eight shared-parameter edits of one real pretrained SSM cache."""
import torch
from torch import nn
from torch.nn import functional as F


class StateDream(nn.Module):
    def __init__(self,mode='D',memory=128,heads=24,width=768,hidden=64,rounds=8):
        super().__init__()
        if mode not in ['A','D']: raise ValueError(mode)
        self.mode,self.rounds=mode,rounds
        self.history_project=nn.Linear(width,hidden)
        self.history_gru=nn.GRU(hidden,hidden,batch_first=True)
        self.read=nn.Linear(memory,hidden)
        self.head_embedding=nn.Embedding(heads,16)
        self.editor=nn.Sequential(nn.Linear(hidden*2+16,hidden),nn.SiLU(),nn.Linear(hidden,memory))
        nn.init.normal_(self.editor[-1].weight,std=.002)
        nn.init.zeros_(self.editor[-1].bias)
        # Created after shared components, so both modes share the same seeded initialization.
        self.recur=nn.GRUCell(2*hidden,hidden)
        self.hidden=hidden

    def forward(self,state,history=None,rounds=None):
        context=state.new_zeros(state.shape[0],self.hidden)
        if history is not None and history.shape[1]:
            _,z=self.history_gru(F.silu(self.history_project(history)))
            context=z[0]
        scale=state.detach().square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-4)
        h=state
        z=torch.zeros_like(context)
        snapshots=[]
        head=self.head_embedding(torch.arange(h.shape[1],device=h.device))[None,:,None]
        for _ in range(self.rounds if rounds is None else rounds):
            read=F.silu(self.read(h/scale))
            if self.mode=='D':
                z=self.recur(torch.cat([read.mean((1,2)),context],-1),z)
                control=z
            else:
                control=context
            control=control[:,None,None].expand(-1,h.shape[1],h.shape[2],-1)
            ids=head.expand(h.shape[0],-1,h.shape[2],-1)
            delta=.1*scale*self.editor(torch.cat([read,control,ids],-1))
            h=h+delta
            snapshots.append(h)
        return h,snapshots


def damage(state,seed,rate=.1):
    generator=torch.Generator(device=state.device).manual_seed(seed)
    return state*(torch.rand(state.shape,generator=generator,device=state.device,dtype=state.dtype)>=rate)


def history_input(model,tokens,boundary,window):
    if window==0:return None
    # boundary counts visible tokens, and the last visible token is a pending anchor.
    with torch.no_grad():
        return model.backbone.embedding(tokens[:,max(0,boundary-window):boundary]).detach()


def edit_cache(model,dream,cache,tokens,boundary,window,layer,corruption=False,seed=0,rounds=None):
    if cache.processed!=boundary-1: raise ValueError('Anchor/cache boundary misalignment')
    h=cache.layers[layer].ssm
    initial=damage(h,seed) if corruption else h
    if dream is None: return cache.edit(layer,initial),[],initial
    history=history_input(model,tokens,boundary,window)
    edited,trace=dream(initial,history,rounds=rounds)
    return cache.edit(layer,edited),trace,initial


def trace_metrics(initial,snapshots):
    denominator=float(initial.detach().norm().clamp_min(1e-12))
    previous=initial
    rows=[]
    for i,state in enumerate(snapshots):
        delta=state-previous
        rows.append({'round':i+1,'delta_norm':float(delta.detach().norm()),
                     'relative_delta':float(delta.detach().norm())/denominator,
                     'relative_total_edit':float((state-initial).detach().norm())/denominator,
                     'per_head_delta_norm':delta.detach().square().sum((2,3)).sqrt().flatten().tolist()})
        previous=state
    return rows

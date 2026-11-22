from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class State:
    ssm: torch.Tensor
    conv: torch.Tensor

    def detach(self):
        return State(self.ssm.detach(), self.conv.detach())

class Backbone(nn.Module):
    """Simplified selective SSM with scalar decay per head (not official Mamba-2)."""
    def __init__(self, vocab=40, width=32, heads=4, memory=8, kernel=3):
        super().__init__()
        self.width, self.heads, self.memory, self.kernel = width, heads, memory, kernel
        self.embedding = nn.Embedding(vocab, width)
        self.conv_weight = nn.Parameter(torch.randn(kernel, width) / kernel)
        self.select = nn.Linear(width, heads * (2 * memory + 1))
        self.log_a = nn.Parameter(torch.zeros(heads))
        self.skip = nn.Parameter(torch.ones(width))
        self.gate = nn.Linear(width, width)
        self.output = nn.Linear(width, vocab)

    def initial(self, batch, device):
        return State(torch.zeros(batch, self.heads, self.width // self.heads, self.memory, device=device, dtype=self.embedding.weight.dtype),
                     torch.zeros(batch, self.kernel - 1, self.width, device=device, dtype=self.embedding.weight.dtype))

    def run(self, tokens, state=None, observer=None):
        state = self.initial(tokens.shape[0], tokens.device) if state is None else state
        h, cache, outputs = state.ssm, state.conv, []
        for token in tokens.unbind(1):
            e = self.embedding(token)
            window = torch.cat([cache, e[:, None]], dim=1)
            x = F.silu((window * self.conv_weight).sum(1))
            cache = window[:, 1:]
            parameters = self.select(x).reshape(-1, self.heads, 2*self.memory+1)
            b, c = parameters[..., :self.memory], parameters[..., self.memory:2*self.memory]
            dt = F.softplus(parameters[..., -1])
            decay = torch.exp(-dt * self.log_a.exp())
            h = decay[..., None, None]*h + dt[..., None, None]*x.reshape(-1,self.heads,self.width//self.heads,1)*b[...,None,:]
            if observer is not None:
                observer(dt, decay, State(h, cache))
            y = (h*c[...,None,:]).sum(-1).flatten(1) + self.skip*x
            outputs.append(self.output(y * F.silu(self.gate(e))))
        return torch.stack(outputs,1), State(h,cache)

class Dream(nn.Module):
    def __init__(self, model, mode='D', rounds=8, hidden=64):
        super().__init__()
        self.mode, self.rounds = mode, rounds
        size = model.width * model.memory
        self.history = nn.GRU(model.width, hidden, batch_first=True)
        self.read = nn.Linear(size, hidden)
        self.recur = nn.GRUCell(2*hidden, hidden)
        self.edit = nn.Sequential(nn.Linear(size+hidden, hidden), nn.Tanh(), nn.Linear(hidden,size))
        nn.init.normal_(self.edit[-1].weight, std=0.001)
        nn.init.zeros_(self.edit[-1].bias)

    def forward(self, state, history_embeddings):
        _, context = self.history(history_embeddings)
        context = context[0]
        z = torch.zeros_like(context)
        h, changes = state.ssm, []
        for _ in range(self.rounds):
            flat = h.flatten(1)
            if self.mode == 'D':
                z = self.recur(torch.cat([self.read(flat), context],-1), z)
            else:
                z = context
            delta = self.edit(torch.cat([flat,z],-1)).reshape_as(h)
            changes.append(delta)
            h = h + delta
        return State(h,state.conv), torch.stack(changes,1)

def corrupt(state, rate=0.1):
    return State(state.ssm * (torch.rand_like(state.ssm) >= rate),state.conv)

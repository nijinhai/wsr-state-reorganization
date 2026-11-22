"""Query-blind global precomputation; no path executor or per-hop architecture."""
import torch
from torch import nn
from torch.nn import functional as F


class Precompute(nn.Module):
    def __init__(self, mode='C', nodes=6, colors=3, hops=4, width=32,
                 hidden=128, rounds=8):
        super().__init__()
        if mode not in ('A', 'B', 'B-budget', 'C', 'D'):
            raise ValueError(mode)
        if width < max(nodes + colors, hops * colors):
            raise ValueError('Width must support raw encoding and arbitrary answer readout')
        self.mode, self.nodes, self.colors, self.hops = mode, nodes, colors, hops
        self.width, self.hidden, self.rounds = width, hidden, rounds
        self.raw_size, self.state_size = nodes * (nodes + colors), nodes * width
        self.query_size = nodes + hops if mode == 'D' else 0
        # A/B/C use exactly the same gather + h-selected linear classifier.
        self.head_weight = nn.Parameter(torch.empty(hops, width, colors))
        self.head_bias = nn.Parameter(torch.zeros(hops, colors))
        nn.init.normal_(self.head_weight, std=.05)
        if mode != 'A':
            self.initial = nn.Linear(self.raw_size + self.query_size, self.state_size)
            self.update = nn.Sequential(
                nn.Linear(self.state_size + self.raw_size + self.query_size, hidden),
                nn.SiLU(), nn.Linear(hidden, self.state_size))
            self.norm = nn.LayerNorm(width)

    def raw(self, links, colors):
        return torch.cat([F.one_hot(links, self.nodes),
                          F.one_hot(colors, self.colors)], -1).float()

    def prepare(self, links, colors, rounds=None, query=None, intervention=None,
                return_trace=False):
        """A/B/C cannot accept a query. One call serves all subsequent queries."""
        if self.mode != 'D' and query is not None:
            raise ValueError('Queries are prohibited during blind precomputation')
        raw = self.raw(links, colors)
        if self.mode == 'A':
            w = F.pad(raw, (0, self.width - raw.shape[-1]))
            return w, [w] if return_trace else None
        x = raw.flatten(1)
        if self.mode == 'D':
            if query is None:
                raise ValueError('D requires its query before computation')
            q = torch.cat([F.one_hot(query[:, 0], self.nodes),
                           F.one_hot(query[:, 1] - 1, self.hops)], -1).float()
            x = torch.cat([x, q], -1)
        w = self.initial(x).reshape(-1, self.nodes, self.width)
        trace = [w] if return_trace else None
        depth = self.rounds if rounds is None else rounds
        if self.mode in ('B', 'B-budget') and depth != 1:
            raise ValueError('Single-pass models execute exactly one update')
        for k in range(depth):
            delta = self.update(torch.cat([w.flatten(1), x], -1)).reshape_as(w)
            w = self.norm(w + .25 * delta)
            if intervention is not None and k + 1 == intervention['after_round']:
                kind = intervention['kind']
                if kind == 'zero':
                    w = torch.zeros_like(w)
                elif kind == 'swap':
                    w = w.roll(1, 0)
                elif kind == 'replace':
                    w = intervention['value']
                else:
                    raise ValueError(kind)
            if return_trace:
                trace.append(w)
        return w, trace

    def readout(self, w, query):
        if query.ndim != 3:
            raise ValueError('Query must be [world, query, (start, hops)]')
        rows = torch.arange(len(w), device=w.device)[:, None]
        state = w[rows, query[..., 0]]
        weight = self.head_weight[query[..., 1] - 1]
        return torch.einsum('bqd,bqdc->bqc', state, weight) + self.head_bias[query[..., 1] - 1]

    def forward(self, links, colors, query, rounds=None):
        if self.mode == 'D':
            batch, count = query.shape[:2]
            l = links[:, None].expand(-1, count, -1).reshape(-1, self.nodes)
            c = colors[:, None].expand(-1, count, -1).reshape(-1, self.nodes)
            q = query.reshape(-1, 2)
            w, _ = self.prepare(l, c, rounds=rounds, query=q)
            return self.readout(w, q[:, None]).reshape(batch, count, self.colors)
        w, _ = self.prepare(links, colors, rounds=rounds)
        return self.readout(w, query)

    def macs(self, query_count=1, rounds=None):
        """Dense linear multiply-accumulates, excluding norms/activations/indexing."""
        k = self.rounds if rounds is None else rounds
        read = query_count * self.width * self.colors
        if self.mode == 'A':
            return {'prepare': 0, 'readout': read, 'total': read}
        raw, state = self.raw_size + self.query_size, self.state_size
        prep = raw * state + k * ((state + raw) * self.hidden + self.hidden * state)
        if self.mode == 'D':
            prep *= query_count
        return {'prepare': prep, 'readout': read, 'total': prep + read}


def budget_hidden(nodes=6, colors=3, width=32, hidden=128, rounds=8):
    # Matches dense preparation MACs of C8, without changing W or readout.
    return hidden * rounds

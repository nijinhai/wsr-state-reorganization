"""Read-only fact bank plus transient recurrent workspace; no Mamba dependency."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Workspace(nn.Module):
    def __init__(self, mode='A', hidden=16, rounds=8, penalty=1., rho=.9):
        super().__init__()
        if mode not in ['A', 'B', 'C', 'D'] or penalty < 0 or not 0 <= rho <= 1:
            raise ValueError('Invalid workspace configuration')
        self.mode, self.hidden, self.rounds = mode, hidden, rounds
        self.penalty, self.rho = penalty, rho
        self.node = nn.Embedding(16, hidden)
        self.initial = nn.Parameter(torch.zeros(hidden))
        self.value = nn.Sequential(nn.Linear(2, hidden), nn.SiLU())
        self.read_query = nn.Linear(hidden, hidden, bias=False)
        nn.init.normal_(self.read_query.weight, std=.02)
        self.weight_query = nn.Embedding(2, hidden)
        nn.init.normal_(self.weight_query.weight, std=.02)
        self.update = nn.GRUCell(hidden, hidden)
        self.answer = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 13))
        # Extra parameters created last. Common core initialization is exactly paired.
        self.round_embedding = nn.Embedding(rounds, hidden) if mode == 'B' else None
        if self.round_embedding is not None:
            nn.init.zeros_(self.round_embedding.weight)

    def forward(self, memory, start, program, rounds=None, intervention=None,
                penalty=None, disable_mask=False, return_trace=False):
        rounds = self.rounds if rounds is None else rounds
        if not 1 <= rounds <= 16:
            raise ValueError('Diagnostic depth must be between 1 and 16')
        batch, slots = memory.shape[:2]
        facts = F.one_hot(memory[..., 0], 2).float()
        values = self.value(facts)
        keys = self.node.weight
        w = self.initial[None].expand(batch, -1)
        coverage = w.new_zeros(batch, slots)
        visited = torch.zeros(batch, slots, dtype=torch.bool, device=w.device)
        states, weights, coverages = [], [], []
        for k in range(rounds):
            if intervention is not None and k == intervention.get('round', 3) and intervention.get('coverage') == 'zero':
                coverage = torch.zeros_like(coverage)
            signal = 0 if self.round_embedding is None else self.round_embedding.weight[min(k, self.rounds - 1)]
            address = self.node(program[:, min(k, 7)])
            scores = (address + self.read_query(w) + signal) @ keys.T / math.sqrt(self.hidden)
            strength = self.penalty if penalty is None else penalty
            if self.mode == 'C':
                scores = scores - strength * coverage
            if self.mode == 'D' and not disable_mask:
                scores = scores.masked_fill(visited, -torch.inf)
            attention = scores.softmax(-1)
            evidence = torch.einsum('bn,bnd->bd', attention, values)
            if intervention is not None and k == intervention.get('round', 2):
                kind = intervention['kind']
                if kind == 'zero_evidence':
                    evidence = torch.zeros_like(evidence)
                elif kind == 'swap_evidence':
                    evidence = evidence.roll(1, 0)
                elif kind in ['high_value_swap', 'low_value_swap']:
                    index = attention.argmax(-1) if kind == 'high_value_swap' else attention.argmin(-1)
                    row = torch.arange(batch, device=w.device)
                    donor = values.roll(1, 0)[row, index]
                    evidence = evidence + attention[row, index, None] * (donor - values[row, index])
                elif kind == 'none':
                    pass
                else:
                    raise ValueError(kind)
            # The visible query gives the current address; the state transition itself is learned.
            w = self.update(evidence + self.weight_query(program[:, 8 + min(k, 7)]) + signal, w)
            if intervention is not None and k == intervention.get('round', 2):
                if intervention.get('state') == 'zero':
                    w = torch.zeros_like(w)
                elif intervention.get('state') == 'swap':
                    w = w.roll(1, 0)
                elif intervention.get('state') == 'replace':
                    w = intervention['replacement']
            if return_trace:
                states.append(w)
                weights.append(attention)
                coverages.append(coverage)
            coverage = self.rho * coverage + attention
            visited = visited | F.one_hot(attention.argmax(-1), slots).bool()
        logits = self.answer(w)
        trace = None if not return_trace else {'states': torch.stack(states, 1),
                'attention': torch.stack(weights, 1), 'coverage_before': torch.stack(coverages, 1),
                'final_coverage': coverage}
        return logits, trace

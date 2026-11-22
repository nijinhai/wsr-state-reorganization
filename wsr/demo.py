"""CPU overfit/reuse demonstration, explicitly separate from held-out experiments."""
import argparse
import json
import torch
from torch.nn import functional as F

from .precompute_experiment import CONFIGS, new_model
from .precompute_tasks import make_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['A', 'B', 'B-budget', 'C2', 'C4', 'C8'], default='C8')
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2048:
        parser.error('--steps must be between 1 and 2048')
    torch.set_num_threads(1)
    data = make_data(sizes=(16, 8, 8))
    block = {key: val[:32] for key, val in data['train'].items()}
    model = new_model(args.model, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    ids = data['train_query_ids']
    query = data['queries'][ids][None].expand(32, -1, -1)
    target = block['labels'][:, ids]
    print('CPU demonstration: 32-instance overfit check, not the held-out benchmark.', flush=True)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(block['links'], block['colors'], query)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
            print(f'update={step+1:4d} CE={loss.item():.4f}', flush=True)
    model.eval()
    with torch.no_grad():
        # Exactly one query-blind call. Its returned W is reused for both groups.
        w, _ = model.prepare(block['links'], block['colors'])
        all_queries = data['queries'][None].expand(32, -1, -1)
        logits = model.readout(w, all_queries)
        parts = torch.cat([model.readout(w, all_queries[:, :12]),
                           model.readout(w, all_queries[:, 12:])], 1)
        maximum_difference = float((parts - logits).abs().max())
        if maximum_difference > 1e-6:
            raise RuntimeError('Reusing W changed query results')
        predictions = logits.argmax(-1)
        scores = {group: float((predictions[:, data[key]] == block['labels'][:, data[key]]).float().mean())
                  for group, key in [('training_queries', 'train_query_ids'), ('held_query_combinations', 'held_query_ids')]}
    print(json.dumps({'model': args.model, 'parameters': sum(p.numel() for p in model.parameters()),
                      'updates': args.steps, 'worlds': 32, 'evaluation_scope': 'same 32 training instances',
                      'accuracy': scores, 'prepare_calls_for_24_queries_per_world': 1,
                      'shared_workspace_readout_max_difference': maximum_difference,
                      'Mamba_loaded': False}, indent=2))


if __name__ == '__main__':
    main()

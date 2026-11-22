"""Target collapse check on exactly the P final evaluation prefixes, per distance."""
import json
import torch

from .v02_aux import TargetBank, p_key
from .v02_data import ROOT, DISTANCES
from .v02_train import write_json


def main():
    torch.set_num_threads(4)
    bank = TargetBank('validation', 'cpu')
    rows = json.loads((ROOT / 'bank/index.json').read_text())['validation']
    rows = [r for r in rows if all(p_key(r, 0, d) in bank.lookup for d in DISTANCES)]
    result = {}
    for distance in DISTANCES:
        vectors = torch.stack([bank.features[bank.lookup[p_key(r, 0, distance)]] for r in rows]).double()
        singular = torch.linalg.svdvals(vectors - vectors.mean(0))
        probability = singular.square() / singular.square().sum().clamp_min(1e-20)
        result[str(distance)] = {
            'states': len(vectors),
            'coordinate_std_mean': float(vectors.std(0).mean()),
            'effective_rank': float((-probability[probability > 0] * probability[probability > 0].log()).sum().exp()),
            'mean_pairwise_cosine': float((vectors.sum(0).square().sum() - len(vectors)) / (len(vectors) * (len(vectors) - 1))),
        }
    write_json(ROOT / 'matched-P-target-diagnostics.json', result)


if __name__ == '__main__':
    main()

"""Independent latent worlds; a shared token vocabulary and next-answer interface."""
import torch

VOCAB = 40
QUERY, SEP, READ = 32, 33, 35

def world(seed, family, episodes=3, length=12):
    g = torch.Generator().manual_seed(seed)
    mapping = torch.randperm(8, generator=g)
    alias = torch.randperm(16, generator=g).reshape(8,2)
    total = 0
    result = []
    for _ in range(episodes):
        history = []
        if family == 'memory':
            values = torch.randint(0,4,(length,),generator=g).tolist()
            for value in values:
                total = (total+value)%8
                history.append(value)
            answer = total
        elif family == 'redundancy':
            # Context demonstrations expose the world-specific alias-to-label rule.
            for latent in range(8):
                history.extend([int(alias[latent,0])+8, int(mapping[latent]), SEP])
                history.extend([int(alias[latent,1])+8, int(mapping[latent]), SEP])
            latent = int(torch.randint(8,(),generator=g))
            history.append(int(alias[latent,int(torch.randint(2,(),generator=g))])+8)
            answer = int(mapping[latent])
        elif family == 'rules':
            for source,target in enumerate(mapping.tolist()):
                history.extend([source,target,SEP])
            source = int(torch.randint(8,(),generator=g))
            history.extend([source,34])
            answer = int(mapping[int(mapping[source])])
        else:
            raise ValueError(family)
        result.append((torch.tensor([history+[QUERY]]),torch.tensor([answer])))
    return result

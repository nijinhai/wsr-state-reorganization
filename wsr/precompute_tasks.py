"""Random functional graphs; final answers are labels, never model features."""
import random
import torch


def queries(nodes=6, hops=4):
    return torch.tensor([(s, h) for s in range(nodes) for h in range(1, hops + 1)])


def answers(links, colors, query):
    out = []
    for s, h in query.tolist():
        position = torch.full((len(links),), s, dtype=torch.long)
        for _ in range(h):
            position = links[torch.arange(len(links)), position]
        out.append(colors[torch.arange(len(links)), position])
    return torch.stack(out, 1)


def make_data(nodes=6, colors=3, hops=4, sizes=(1024, 128, 128), seed=318031):
    if nodes % colors:
        raise ValueError('Balanced colors require nodes divisible by colors')
    rng = random.Random(seed)
    used_graphs, data = set(), {}
    q = queries(nodes, hops)
    for name, size in zip(('train', 'validation', 'test'), sizes):
        links, values = [], []
        for _ in range(size):
            while True:
                graph = tuple(rng.choice([j for j in range(nodes) if j != i]) for i in range(nodes))
                if graph not in used_graphs:
                    used_graphs.add(graph)
                    break
            color = [i % colors for i in range(nodes)]
            rng.shuffle(color)
            # All rotations of one world stay in the same partition.
            # Exact per-query answer balance, without conditioning features on answers.
            for shift in range(colors):
                links.append(graph)
                values.append([(c + shift) % colors for c in color])
        l, c = torch.tensor(links), torch.tensor(values)
        data[name] = {'links': l, 'colors': c, 'labels': answers(l, c, q)}
    data['queries'] = q
    data['train_query_ids'] = ((q[:, 0] + q[:, 1]) % 3 != 0).nonzero().flatten()
    data['held_query_ids'] = ((q[:, 0] + q[:, 1]) % 3 == 0).nonzero().flatten()
    return data


def audit(data, colors=3):
    q, seen = data['queries'], set()
    result = {}
    for name in ('train', 'validation', 'test'):
        b = data[name]
        graphs = {tuple(x) for x in b['links'].tolist()}
        assert not seen & graphs
        seen |= graphs
        assert torch.equal(b['labels'], answers(b['links'], b['colors'], q))
        assert not (b['links'] == torch.arange(b['links'].shape[1])).any()
        hist = torch.stack([torch.bincount(y, minlength=colors) for y in b['labels'].T])
        assert (hist.max(1).values == hist.min(1).values).all()
        result[name] = {'worlds': len(b['links']), 'unique_graphs': len(graphs),
                        'each_query_color_counts': hist.tolist(), 'symbolic_accuracy': 1.,
                        'uniform_and_majority_accuracy': 1 / colors}
    assert not set(data['train_query_ids'].tolist()) & set(data['held_query_ids'].tolist())
    result['train_queries'] = q[data['train_query_ids']].tolist()
    result['held_queries'] = q[data['held_query_ids']].tolist()
    return result

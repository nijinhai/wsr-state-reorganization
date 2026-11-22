"""Bounded query-blind precomputation study; no Mamba/old Dream imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

import torch
from torch.nn import functional as F

from .precompute_model import Precompute, budget_hidden
from .precompute_tasks import make_data, audit, answers

ROOT = Path('artifacts/wsr-v0.3-precompute')
CONFIGS = {'A': ('A', 0, 128), 'B': ('B', 1, 128),
           'B-budget': ('B-budget', 1, budget_hidden()),
           'C2': ('C', 2, 128), 'C4': ('C', 4, 128), 'C8': ('C', 8, 128),
           'D8': ('D', 8, 128)}
UPDATES = 2048
MILESTONES = (0, 128, 512, 1024, 2048)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {'nodes': 6, 'colors': 3, 'hops': [1, 2, 3, 4], 'width_per_start': 32,
                'update_hidden': 128, 'B_budget_hidden': 1024, 'configs': CONFIGS,
                'seeds': [0, 1, 2], 'updates': UPDATES, 'pilot_updates': 512,
                'optimizer': 'Adam', 'lr': .001, 'clip': 1., 'batch_worlds': 16,
                'queries_per_world': 8, 'answer_labels_per_update': 128,
                'selection': 'one fixed task; pilot gates correctness and learning, not C superiority',
                'task': 'random no-self-link functional graph; balanced three colors; 1-4 hops',
                'query_split': '(start+hops)%3 == 0 held out, other pairs train; all starts/hops trained',
                'data_split': 'disjoint link graphs; all three color rotations remain together',
                'supervision': 'final color CE only; no intermediate, coverage, round encoding, anytime',
                'readout': 'gather W[start], then hop-selected linear color classifier shared across starts',
                'W_prior': 'one 32-dimensional output slot per start, no per-hop state or transition wiring',
                'update': 'global dense MLP sees ALL raw facts and ALL W each round; no enforced one-hop step',
                'C1': 'exactly B: same initialization, forward, optimizer, and training schedule; reuse run',
                'B_budget': 'single wider update, same W/readout, preparation dense MACs matched to C8',
                'D': 'query-conditioned initial/update; separate computation per query, same readout',
                'device': 'CPU FP32 single thread', 'test_policy': 'evaluate models only after all 21 main runs finish',
                'no_new_mechanisms': True, 'Mamba_loaded': False}
    protocol = json.loads(json.dumps(protocol))
    p = ROOT / 'protocol.json'
    if p.exists():
        assert json.loads(p.read_text()) == protocol, 'Do not silently change existing protocol'
    else:
        write(p, protocol)
    before = ROOT / 'preservation-before.json'
    if not before.exists():
        old = {str(p): digest(p) for p in Path('artifacts').rglob('*')
               if p.is_file() and ROOT not in p.parents}
        write(before, old)
        for base in ('wsr', 'tests'):
            for p in Path(base).glob('*.py'):
                if 'precompute' not in p.name:
                    target = ROOT / 'source-before' / p
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
        for p in Path('.').glob('*.md'):
            shutil.copy2(p, ROOT / 'source-before' / p)
    if not (ROOT / 'data.pt').exists():
        data = make_data()
        write(ROOT / 'data-quality.json', audit(data))
        save(ROOT / 'data.pt', data)
    return torch.load(ROOT / 'data.pt', weights_only=True)


def new_model(name, seed):
    torch.manual_seed(seed)
    mode, rounds, hidden = CONFIGS[name]
    return Precompute(mode, rounds=rounds, hidden=hidden)


def metric(logits, labels, ids, q):
    result = {}
    for name, select in [('all', ids), *[(f'h{h}', ids[q[ids, 1] == h]) for h in range(1, 5)]]:
        p, y = logits[:, select], labels[:, select]
        result[name] = {'accuracy': float((p.argmax(-1) == y).float().mean()),
                        'ce': float(F.cross_entropy(p.flatten(0, 1), y.flatten())),
                        'answers': y.numel()}
    return result


@torch.no_grad()
def evaluate(model, block, data, rounds=None):
    q = data['queries']
    chunks = []
    for i in range(0, len(block['links']), 32):
        l, c = block['links'][i:i+32], block['colors'][i:i+32]
        chunks.append(model(l, c, q[None].expand(len(l), -1, -1), rounds=rounds))
    logits = torch.cat(chunks)
    return {name: metric(logits, block['labels'], data[ids], q) for name, ids in
            [('seen_query', 'train_query_ids'), ('held_query', 'held_query_ids')]}


def train(name, seed, data, until=UPDATES, small=False):
    directory = ROOT / ('smoke' if small else 'runs') / f'{name}-seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    model = new_model(name, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    block = data['train']
    if small:
        block = {k: v[:32] for k, v in block.items()}
    gen = torch.Generator().manual_seed(318500 + seed)
    worlds = torch.randint(len(block['links']), (UPDATES, 16), generator=gen)
    # Same 8 distinct training queries for a world in a given update in every model.
    query_order = torch.rand(UPDATES, 16, len(data['train_query_ids']), generator=gen).argsort(-1)[..., :8]
    query_ids = data['train_query_ids'][query_order]
    config = {'name': name, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
              'rounds': model.rounds, 'hidden': model.hidden, 'small32': small,
              'planned_updates': UPDATES, 'labels_per_update': 128}
    initial_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    if (directory / 'config.json').exists():
        assert json.loads((directory / 'config.json').read_text()) == config
    else:
        write(directory / 'config.json', config)
        save(directory / 'schedule.pt', {'worlds': worlds, 'query_ids': query_ids})
    done, curves, records = 0, [], []
    if (directory / 'last.pt').exists():
        saved = torch.load(directory / 'last.pt', weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        done, curves, records = saved['step'], saved['curves'], saved['records']
    def checkpoint(step):
        result = {'step': step, 'train': evaluate(model, block, data)}
        if not small:
            result['validation'] = evaluate(model, data['validation'], data)
        curves.append(result)
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'step': step, 'curves': curves, 'records': records, 'config': config}
        save(directory / f'checkpoint-{step:05d}.pt', payload)
        save(directory / 'last.pt', payload)
        write(directory / 'learning-curves.json', curves)
        write(directory / 'training.json', records)
        brief = {k: {j: round(v['all']['accuracy'], 4) for j, v in result[k].items()}
                 for k in ('train', 'validation') if k in result}
        print(name, seed, step, brief, flush=True)
    if not curves:
        checkpoint(0)
    for i in range(done, until):
        tick = time.perf_counter()
        ids, qi = worlds[i], query_ids[i]
        l, c = block['links'][ids], block['colors'][ids]
        query = data['queries'][qi]
        y = block['labels'][ids[:, None], qi]
        optimizer.zero_grad(set_to_none=True)
        logits = model(l, c, query)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        records.append({'step': i+1, 'ce': float(loss.detach()), 'gradient': float(grad),
                        'seconds': time.perf_counter()-tick, 'answers': 128})
        if i+1 in MILESTONES or i+1 == until:
            checkpoint(i+1)
    write(directory / 'summary.json', {**config, 'updates': max(done, until),
          'training_seconds': sum(x['seconds'] for x in records),
          'parameter_change_l2_from_initialization': float(torch.stack([
              (v.detach()-initial_parameters[k]).square().sum() for k, v in model.named_parameters()]).sum().sqrt()),
          'answer_labels': 128 * max(done, until), 'final': curves[-1]})
    return model, curves[-1]


def loaded(name, seed):
    path = ROOT / 'runs' / f'{name}-seed{seed}' / 'last.pt'
    payload = torch.load(path, weights_only=True)
    assert payload['step'] == UPDATES
    model = new_model(name, seed)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, payload


@torch.no_grad()
def diagnostics(model, block, data):
    q = data['queries']
    l, c = block['links'], block['colors']
    query = q[None].expand(len(l), -1, -1)
    # Relabel node IDs consistently, including start queries. No label changes.
    perm = torch.randperm(model.nodes, generator=torch.Generator().manual_seed(93))
    lp, cp = torch.empty_like(l), torch.empty_like(c)
    lp[:, perm], cp[:, perm] = perm[l], c
    qp = query.clone()
    qp[..., 0] = perm[query[..., 0]]
    logits = model(lp, cp, qp)
    result = {'node_permutation': {name: metric(logits, block['labels'], data[ids], q)
                                  for name, ids in [('seen_query', 'train_query_ids'),
                                                    ('held_query', 'held_query_ids')]}}
    # Original held pairs may become trained pairs under node relabeling.
    # Also exclude relabeled graphs that happen to match a training graph.
    train_graphs = {tuple(x) for x in data['train']['links'].tolist()}
    novel = torch.tensor([tuple(x) not in train_graphs for x in lp.tolist()])
    mapped_held = (qp[0, :, 0] + qp[0, :, 1]) % 3 == 0
    result['node_permutation_mapped_split_novel_rules'] = {
        name: metric(logits[novel], block['labels'][novel], mask.nonzero().flatten(), q)
        for name, mask in [('seen_query', ~mapped_held), ('held_query', mapped_held)]}
    result['node_permutation_metadata'] = {
        'permutation': perm.tolist(), 'novel_worlds': int(novel.sum()),
        'excluded_worlds_matching_training_graph': int((~novel).sum()),
        'original_held_queries_becoming_seen': int((~mapped_held[data['held_query_ids']]).sum()),
        'note': 'original groups and mapped groups are distinct; no model/optimizer changes'}
    shifted = (c + 1) % model.colors
    logits = model(l, shifted, query)
    result['color_rotation'] = metric(logits, (block['labels']+1) % model.colors,
                                      torch.arange(len(q)), q)
    if model.mode == 'C':
        w, trace = model.prepare(l, c, return_trace=True)
        result['dynamics'] = {'update_norm': [float((trace[k+1]-trace[k]).norm(dim=-1).mean())
                                              for k in range(model.rounds)],
                             'consecutive_cosine': [float(F.cosine_similarity(
                                 (trace[k]-trace[k-1]).flatten(1),
                                 (trace[k+1]-trace[k]).flatten(1)).mean())
                                 for k in range(1, model.rounds)]}
    return result


@torch.no_grad()
def benchmark(model, block, data):
    l, c = block['links'][:16], block['colors'][:16]
    result = {}
    for count in (1, 4, 16, 64):
        # Sampling with replacement permits >24 independent query draws.
        gen = torch.Generator().manual_seed(4400+count)
        ids = torch.randint(len(data['queries']), (16, count), generator=gen)
        query = data['queries'][ids]
        for _ in range(3):
            model(l, c, query)
        timings = []
        for _ in range(9):
            tick = time.perf_counter()
            model(l, c, query)
            timings.append(time.perf_counter()-tick)
        elapsed = statistics.median(timings) / len(l)
        result[str(count)] = {**model.macs(count), 'seconds_per_world': elapsed,
                              'seconds_per_query': elapsed/count, 'MACs_per_query': model.macs(count)['total']/count,
                              'timing': 'median9, warmup3, batch16 vectorized CPU, no backward',
                              'accuracy': float((model(l, c, query).argmax(-1) == block['labels'][:16].gather(1, ids)).float().mean())}
    return result


def analyze(data):
    # Gate before ANY model evaluates test.
    for seed in range(3):
        for name in CONFIGS:
            loaded(name, seed)
    write(ROOT / 'test-gate.json', {'all_21_runs_complete': True, 'updates_each': UPDATES,
                                   'selection': 'uniform final checkpoint; no test tuning'})
    result = {'runs': {}, 'shortcuts': {}}
    for split in ('validation', 'test'):
        b, q = data[split], data['queries']
        row = torch.arange(len(b['links']))[:, None]
        start = b['colors'][:, q[:, 0]]
        one = b['colors'][row, b['links'][:, q[:, 0]]]
        hist = F.one_hot(b['colors'], 3).sum(1).argmax(-1)[:, None].expand_as(start)
        result['shortcuts'][split] = {name: metric(F.one_hot(pred, 3).float(), b['labels'],
                                                 torch.arange(len(q)), q)
                                      for name, pred in [('start_color', start), ('always_one_hop', one),
                                                         ('majority_color', hist)]}
    for seed in range(3):
        for name in CONFIGS:
            model, payload = loaded(name, seed)
            item = {'parameters': sum(p.numel() for p in model.parameters()),
                    'updates': payload['step'], 'training_seconds': sum(r['seconds'] for r in payload['records']),
                    'answer_labels': 128*payload['step'], 'test': evaluate(model, data['test'], data),
                    'diagnostics': diagnostics(model, data['test'], data),
                    'cost': benchmark(model, data['test'], data)}
            if name in ('C8', 'D8'):
                item['inference_depth'] = {str(k): evaluate(model, data['test'], data, rounds=k)
                                            for k in (1, 2, 4, 8)}
            result['runs'][f'{name}-seed{seed}'] = item
            print('analyzed', name, seed, flush=True)
    for name in CONFIGS:
        rows = [result['runs'][f'{name}-seed{s}'] for s in range(3)]
        result.setdefault('aggregate', {})[name] = {
            group: {'mean': statistics.mean(r['test'][group]['all']['accuracy'] for r in rows),
                    'sd': statistics.stdev(r['test'][group]['all']['accuracy'] for r in rows)}
            for group in ('seen_query', 'held_query')}
    write(ROOT / 'results.json', result)
    rows = []
    for run, x in result['runs'].items():
        for group in ('seen_query', 'held_query'):
            for depth, values in x['test'][group].items():
                rows.append({'run': run, 'queries': group, 'hops': depth, **values,
                             'parameters': x['parameters'], 'training_seconds': x['training_seconds']})
    with (ROOT / 'main-results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return result


def preservation():
    old = json.loads((ROOT / 'preservation-before.json').read_text())
    changed = [p for p, h in old.items() if not Path(p).exists() or digest(Path(p)) != h]
    write(ROOT / 'preservation-after.json', {'protected_files': len(old), 'changed': changed,
                                            'passed': not changed})
    if changed:
        raise RuntimeError(f'Old files changed: {changed[:5]}')
    target = ROOT / 'source-run'
    for p in [*Path('wsr').glob('precompute*.py'), Path('tests/test_precompute.py')]:
        dest = target / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    print('old files unchanged', len(old), flush=True)


def refresh_diagnostics(data):
    """Clarify permutation group accounting without rerunning training/timings."""
    path = ROOT / 'results.json'
    result = json.loads(path.read_text())
    backup = ROOT / 'results-before-permutation-split-clarification.json'
    if not backup.exists():
        shutil.copy2(path, backup)
    for seed in range(3):
        for name in CONFIGS:
            model, _ = loaded(name, seed)
            result['runs'][f'{name}-seed{seed}']['diagnostics'] = diagnostics(model, data['test'], data)
    write(path, result)
    write(ROOT / 'analysis-correction.json', {
        'change': 'Permutation diagnosis now reports mapped query groups and excludes mapped training graphs',
        'reason': 'A relabeled held query may become a trained pair; relabeled graphs may match training graphs',
        'original_diagnostics_preserved': str(backup), 'model_updates': 0,
        'main_accuracies_changed': False, 'benchmarks_rerun': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'smoke', 'pilot', 'train', 'analyze', 'diagnostics', 'audit'])
    args = parser.parse_args()
    data = prepare()
    if args.action == 'prepare':
        print('prepared', ROOT, flush=True)
    elif args.action == 'smoke':
        for name in ('B', 'C8', 'D8'):
            train(name, 0, data, until=1024, small=True)
    elif args.action == 'pilot':
        for name in CONFIGS:
            train(name, 0, data, until=512)
    elif args.action == 'train':
        assert (ROOT / 'pilot-decision.json').exists(), 'Record pilot decision before seed expansion'
        for seed in range(3):
            for name in CONFIGS:
                train(name, seed, data)
    elif args.action == 'analyze':
        analyze(data)
    elif args.action == 'diagnostics':
        refresh_diagnostics(data)
    elif args.action == 'audit':
        preservation()


if __name__ == '__main__':
    main()

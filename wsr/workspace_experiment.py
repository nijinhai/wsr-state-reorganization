"""Bounded A/B/C/D workspace study with paired schedules and final-answer CE only."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from .workspace_model import Workspace
from .workspace_tasks import make_data, validate_data, execute
from .v02_data import digest
from .v02_train import write_json, save_checkpoint

ROOT = Path('artifacts/wsr-v0.3-pre')
STEPS = 512
MILESTONES = (0, 64, 128, 256, 512, 768, 1024)


def initialize():
    torch.set_num_threads(1)
    ROOT.mkdir(exist_ok=True)
    protocol = {'task': 'weighted sum of eight binary reads at explicit addresses; distinct or a-b-c-d-a-e-f-g reread',
                'models': ['A', 'B', 'C', 'D'], 'workspace_hidden': 16, 'training_rounds': 8,
                'coverage_lambda': 1., 'coverage_rho': .9,
                'hard_mask': 'exclude previous attention argmax slots; original bank not erased',
                'seeds': [0, 1, 2], 'optimizer': 'Adam', 'lr': .001, 'clip': 1.,
                'updates': STEPS, 'allocated_schedule_updates': 1024, 'small32_updates': 1024,
                'budget_selection': 'budget-decision.json; A-only curve before seed expansion/test',
                'batch': 64, 'milestones': [n for n in MILESTONES if n <= STEPS],
                'supervision': 'final count CE only; reference paths never passed to model or optimizer',
                'program_control_prior': 'query provides current address by a fixed external cursor; no intermediate values or transformation execution supplied',
                'trained_read_rounds': 'fixed eight shared reads/updates; no random-depth training',
                'heads': 'answer head sees workspace only; no direct bank/history bypass',
                'device': 'CPU FP32, one intra-op thread', 'Mamba_loaded': False,
                'Mamba_Dream_updates': 0, 'old_test_split_used': False,
                'synthetic_test': 'labels structurally verified, never used for gate or selection; model evaluation after all main runs',
                'round_encoding_extrapolation': 'rounds beyond eight reuse final trained embedding',
                'coverage_selection': 'lambda/rho fixed before pilot, no diversity-based tuning'}
    protocol['train_examples'] = 8008
    target = ROOT / 'protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == protocol
    else:
        write_json(target, protocol)
    data_path = ROOT / 'data.pt'
    if not data_path.exists():
        data = make_data()
        write_json(ROOT / 'data-quality.json', validate_data(data))
        save_checkpoint(data_path, data)
    return torch.load(data_path, weights_only=True)


def inputs(block, indices=None):
    return [block[k] if indices is None else block[k][indices] for k in ['memory', 'start', 'program']]


def classify(logits, labels, family, selected=None):
    selected = torch.ones_like(labels, dtype=torch.bool) if selected is None else selected
    result = {}
    for name, mask in [('all', selected), ('distinct', selected & (family == 0)), ('reread', selected & (family == 1))]:
        if mask.any():
            result[name] = {'accuracy': float((logits[mask].argmax(-1) == labels[mask]).float().mean()),
                            'ce': float(F.cross_entropy(logits[mask], labels[mask])), 'examples': int(mask.sum())}
    return result


def partial_counts(block):
    bits = block['memory'][torch.arange(len(block['label']))[:, None], block['program'][:, :8], 0]
    return (bits * (1 + block['program'][:, 8:])).cumsum(1)


def shortcut_controls(data):
    train = data['train']
    row = torch.arange(len(train['label']))[:, None]
    library = train['memory'].sum((1,2))
    unweighted = train['memory'][row, train['program'][:,:8], 0].sum(1)
    metrics = {}
    for name, values, cap in [('global_bank_ones_ignore_query', library, 16),
                              ('query_unweighted_count_ignore_weights', unweighted, 8)]:
        lookup = torch.zeros(cap+1, dtype=torch.long)
        for value in range(cap+1):
            mask = values == value
            if mask.any():
                lookup[value] = torch.bincount(train['label'][mask], minlength=13).argmax()
        result = {}
        for split in ['validation_new', 'test_new']:
            block = data[split]
            x = block['memory'].sum((1,2)) if cap == 16 else block['memory'][
                torch.arange(len(block['label']))[:,None], block['program'][:,:8], 0].sum(1)
            predictions = lookup[x]
            result[split] = {name: float((predictions[mask] == block['label'][mask]).float().mean())
                            for name, mask in [('all', torch.ones_like(block['label'], dtype=torch.bool)),
                                               ('distinct', block['family']==0), ('reread', block['family']==1)]}
        metrics[name] = result
    write_json(ROOT / 'shortcut-controls.json', metrics)
    train_queries = {tuple(q) for q in train['program'].tolist()}
    write_json(ROOT / 'query-overlap.json', {split:len(train_queries & {tuple(q) for q in block['program'].tolist()})
                                          for split,block in data.items()})


@torch.no_grad()
def intermediate_analysis(model, train_block, block, trace):
    # Explicitly posthoc supervised readout. It never sends labels/gradients to W.
    _, train_trace = model(*inputs(train_block), return_trace=True)
    train_counts, counts = partial_counts(train_block), partial_counts(block)
    probes = []
    for k in range(8):
        x = train_trace['states'][:, k].double(); y = train_counts[:, k].double()
        mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
        x = (x - mean) / scale; ym = y.mean()
        weights = torch.linalg.solve(x.T @ x / len(x) + .001 * torch.eye(x.shape[1], dtype=x.dtype), x.T @ (y-ym) / len(x))
        prediction = (trace['states'][:, k].double() - mean) / scale @ weights + ym
        target = counts[:, k]
        probes.append({'round': k+1, 'heldout_MAE': float((prediction-target).abs().mean()),
                       'heldout_rounded_accuracy': float((prediction.round()==target).float().mean()),
                       'readout': 'separate frozen-model linear ridge per round; train-only fit, fixed lambda0.001'})
    replacements = {}
    values = counts[:, 3]
    first_bits = block['memory'][torch.arange(len(values)), block['program'][:,0], 0]
    for condition in ['preserve_partial_count', 'change_partial_count']:
        donors = list(range(len(values))); eligible = torch.zeros(len(values), dtype=torch.bool)
        for i in range(len(values)):
            for offset in range(1, len(values)):
                j = (i + offset) % len(values)
                if block['family'][j] != block['family'][i] or first_bits[j] != first_bits[i]:
                    continue
                equal = values[j] == values[i]
                expected = block['label'][i] - values[i] + values[j]
                if bool(equal) == (condition == 'preserve_partial_count') and 0 <= expected <= 12:
                    donors[i] = j; eligible[i] = True; break
        donors = torch.tensor(donors)
        logits, _ = model(*inputs(block), intervention={'kind': 'none', 'round': 3, 'state': 'replace',
                                                      'replacement': trace['states'][donors, 3]})
        counterfactual = block['label'] - values + values[donors]
        replacements[condition] = {'original_answer': classify(logits, block['label'], block['family'], eligible),
                                  'counterfactual_answer': classify(logits, counterfactual, block['family'], eligible),
                                  'eligible': int(eligible.sum()),
                                  'selection': 'analysis-only equal/different partial count, matched task family and first bit; no training use'}
    initial = model.initial.expand(len(block['label']), -1)[:, None]
    updates = trace['states'] - torch.cat([initial, trace['states'][:, :-1]], 1)
    dynamics = {'mean_update_norm_by_round': updates.norm(dim=-1).mean(0).tolist(),
                'consecutive_update_cosine': F.cosine_similarity(updates[:, 1:], updates[:, :-1], dim=-1).mean(0).tolist()}
    return {'posthoc_count_probes': probes, 'state_replacements': replacements, 'dynamics': dynamics}


@torch.no_grad()
def evaluate(model, block, **kwargs):
    chunks = []
    for start in range(0, len(block['label']), 128):
        index = slice(start, start + 128)
        logits, _ = model(*inputs(block, index), **kwargs)
        chunks.append(logits)
    logits = torch.cat(chunks)
    result = {}
    for name, mask in [('all', torch.ones_like(block['label'], dtype=torch.bool)),
                       ('distinct', block['family'] == 0), ('reread', block['family'] == 1)]:
        p, y = logits[mask], block['label'][mask]
        result[name] = {'accuracy': float((p.argmax(-1) == y).float().mean()),
                        'ce': float(F.cross_entropy(p, y)), 'examples': int(mask.sum())}
    return result


def train(mode, seed, data, until=STEPS, small=False):
    name = f'{mode}-seed{seed}' if not small else 'A-small32'
    directory = ROOT / 'runs' / name
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = Workspace(mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    block = data['train']
    if small:
        ids = torch.cat([(block['family'] == i).nonzero().flatten()[:16] for i in [0, 1]])
        block = {k: v[ids] for k, v in block.items()}
    schedule = torch.randint(len(block['label']), (1024, 64), generator=torch.Generator().manual_seed(53000 + seed))
    config = {'mode': mode, 'seed': seed, 'hidden': 16, 'rounds': 8, 'penalty': 1., 'rho': .9,
              'lr': .001, 'batch': 64, 'planned_updates': 1024, 'small32': small,
              'parameters': sum(p.numel() for p in model.parameters())}
    if (directory / 'config.json').exists():
        assert json.loads((directory / 'config.json').read_text()) == config
    else:
        write_json(directory / 'config.json', config)
        save_checkpoint(directory / 'schedule.pt', schedule)
    curves, training, done = [], [], 0
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    last = directory / 'last.pt'
    if last.exists():
        saved = torch.load(last, weights_only=True)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        done, curves, training = saved['step'], saved['curves'], saved['training']
    def checkpoint(step):
        validation = {'train': evaluate(model, block),
                      'validation_known': evaluate(model, data['validation_known']),
                      'validation_new': evaluate(model, data['validation_new'])}
        curves.append({'step': step, 'evaluation': validation})
        saved = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                 'curves': curves, 'training': training, 'config': config}
        save_checkpoint(directory / f'checkpoint-{step:05d}.pt', saved)
        save_checkpoint(last, saved)
        write_json(directory / 'learning-curves.json', curves)
        write_json(directory / 'training.json', training)
        print(name, step, 'train', round(validation['train']['all']['accuracy'], 4),
              'val-known', round(validation['validation_known']['all']['accuracy'], 4),
              'val-new', round(validation['validation_new']['all']['accuracy'], 4), flush=True)
    if not curves:
        checkpoint(0)
    if done >= until:
        if not small and done == STEPS:
            summary_path = directory / 'summary.json'
            summary = json.loads(summary_path.read_text())
            summary.update({'complete': True, 'formal_budget_updates': STEPS, 'allocated_schedule_ceiling': 1024})
            write_json(summary_path, summary)
        return curves[-1]
    for step in range(done, until):
        tick = time.monotonic()
        ids = schedule[step]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*inputs(block, ids))
        loss = F.cross_entropy(logits, block['label'][ids])
        assert torch.isfinite(loss)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        assert math.isfinite(float(gradient)) and gradient > 0
        optimizer.step()
        training.append({'step': step + 1, 'ce': float(loss.detach()), 'gradient_norm': float(gradient),
                         'seconds': time.monotonic() - tick, 'answer_labels': 64, 'read_rounds': 8})
        if step + 1 in MILESTONES or step + 1 == until:
            checkpoint(step + 1)
    update = sum((v.detach() - initial[k]).double().square().sum() for k, v in model.state_dict().items()).sqrt()
    summary = {'config': config, 'updates': until, 'complete': until == STEPS or small,
               'formal_budget_updates': STEPS if not small else None, 'allocated_schedule_ceiling': 1024,
               'parameter_update_norm': float(update), 'finite_nonzero_gradients': True,
               'training_seconds': sum(r['seconds'] for r in training), 'answer_draws': until * 64,
               'evaluation': curves[-1]['evaluation']}
    write_json(directory / 'summary.json', summary)
    return curves[-1]


def pilot(data):
    small = train('A', 0, data, until=1024, small=True)
    for mode in 'ABCD':
        train(mode, 0, data, until=512)
    a = json.loads((ROOT / 'runs/A-seed0/summary.json').read_text())
    evaluation = a['evaluation']
    passed = (small['evaluation']['train']['all']['accuracy'] >= .99
              and evaluation['train']['all']['accuracy'] > .35
              and evaluation['validation_known']['all']['accuracy'] > .35
              and a['parameter_update_norm'] > 0)
    result = {'passed': passed, 'small32_accuracy': small['evaluation']['train']['all']['accuracy'],
              'A_train': evaluation['train'], 'A_validation_known': evaluation['validation_known'],
              'criterion': 'small32 >=99%; full train >35%, new-world known-program validation >35%; finite updates',
              'test_evaluation_used': False}
    write_json(ROOT / 'pilot-gate.json', result)
    print('Pilot gate', result, flush=True)


@torch.no_grad()
def analyze(data):
    for seed in range(3):
        for mode in 'ABCD':
            path = ROOT / 'runs' / f'{mode}-seed{seed}' / 'summary.json'
            assert path.exists() and json.loads(path.read_text())['updates'] == STEPS
    result = []
    for seed in range(3):
        for mode in 'ABCD':
            directory = ROOT / 'runs' / f'{mode}-seed{seed}'
            model = Workspace(mode)
            model.load_state_dict(torch.load(directory / 'last.pt', weights_only=True)['model'])
            model.eval()
            metrics = {split: evaluate(model, data[split]) for split in ['train', 'validation_known', 'validation_new', 'test_known', 'test_new']}
            block = data['test_new']
            depth = {str(n): evaluate(model, block, rounds=n) for n in [1, 2, 4, 6, 8, 10, 12]}
            logits, trace = model(*inputs(block), return_trace=True)
            attention = trace['attention']
            top = attention.argmax(-1)
            repeated = torch.stack([(top[:, k:k+1] == top[:, :k]).any(-1) for k in range(1, 8)], 1)
            overlap = sum((attention[:, k] * attention[:, :k].sum(1)).sum(-1) for k in range(1, 8)) / 28
            slot_mass = attention.gather(2, block['path'][..., None]).squeeze(-1)
            stats = {}
            for family, mask in [('all', torch.ones_like(block['label'], dtype=torch.bool)),
                                 ('distinct', block['family'] == 0), ('reread', block['family'] == 1)]:
                reference_repeat = torch.stack([(block['path'][:, k:k+1] == block['path'][:, :k]).any(-1) for k in range(1, 8)], 1)
                needed = reference_repeat[mask]
                values = slot_mass[:, 1:][mask]
                stats[family] = {'top1_repeat_fraction': float(repeated[mask].float().mean()),
                                 'mean_unique_top1_slots': float(torch.tensor([len(x.unique()) for x in top[mask]]).float().mean()),
                                 'soft_pairwise_overlap': float(overlap[mask].mean()),
                                 'attention_entropy': float((-(attention[mask].clamp_min(1e-20).log() * attention[mask]).sum(-1)).mean()),
                                 'reference_slot_attention': float(slot_mass[mask].mean()),
                                 'reference_top1_accuracy': float((top[mask] == block['path'][mask]).float().mean()),
                                 'reference_reread_slot_attention': float(values[needed].mean()) if needed.any() else None}
            interventions = {}
            for kind in ['zero_evidence', 'swap_evidence', 'high_value_swap', 'low_value_swap']:
                interventions[kind] = evaluate(model, block, intervention={'kind': kind, 'round': 3})
            for kind in ['zero', 'swap']:
                interventions['middle_state_' + kind] = evaluate(model, block,
                    intervention={'kind': 'none', 'round': 3, 'state': kind})
            if mode == 'C':
                interventions['coverage_disabled'] = evaluate(model, block, penalty=0.)
                interventions['coverage_reset_after4'] = evaluate(model, block, intervention={'kind':'none', 'round':4, 'coverage':'zero'})
                interventions['lambda_inference_scan'] = {str(v): evaluate(model, block, penalty=v) for v in [.25, 1., 4.]}
            if mode == 'D':
                interventions['hard_mask_disabled'] = evaluate(model, block, disable_mask=True)
            # Flip the unique last queried bit. Exact new labels move by +/-1;
            # values remain in the training distribution.
            counterfactual = {k: v.clone() for k,v in block.items()}
            ids = torch.arange(len(block['label']))
            final_slot = block['program'][:, 7]
            old_value = block['memory'][ids, final_slot, 0]
            counterfactual['memory'][ids, final_slot, 0] = 1 - old_value
            counterfactual['label'] = block['label'] + (1 + block['program'][:,15]) * (1 - 2 * old_value)
            interventions['final_value_counterfactual'] = evaluate(model, counterfactual)
            changed, _ = model(*inputs(counterfactual))
            interventions['counterfactual_prediction_change_fraction'] = float((changed.argmax(-1) != logits.argmax(-1)).float().mean())
            repeated_cf = {k: v.clone() for k,v in block.items()}
            first_slot = block['program'][:, 0]
            old_value = block['memory'][ids, first_slot, 0]
            repeated_cf['memory'][ids, first_slot, 0] = 1 - old_value
            multiplicity = ((block['program'][:,:8] == first_slot[:, None]) * (1 + block['program'][:,8:])).sum(-1)
            repeated_cf['label'] = block['label'] + multiplicity * (1 - 2 * old_value)
            interventions['first_or_repeated_value_counterfactual'] = evaluate(model, repeated_cf)
            weight_cf = {k: v.clone() for k,v in block.items()}
            weight_cf['program'][:,8:] = weight_cf['program'][:,8:].roll(1,1)
            weight_cf['label'] = partial_counts(weight_cf)[:,-1]
            new_logits, _ = model(*inputs(weight_cf))
            changed_mask = weight_cf['label'] != block['label']
            interventions['query_weight_counterfactual_changed_cases'] = classify(new_logits, weight_cf['label'],
                                                                                 block['family'], changed_mask)
            record = {'mode': mode, 'seed': seed, 'metrics': metrics, 'rounds': depth,
                      'attention': stats, 'interventions': interventions,
                      'intermediate': intermediate_analysis(model, data['train'], block, trace),
                      'parameters': sum(p.numel() for p in model.parameters())}
            result.append(record)
            out = ROOT / 'analysis'; out.mkdir(exist_ok=True)
            save_checkpoint(out / f'{mode}-seed{seed}-states.pt', {'states': trace['states'], 'attention': attention,
                            'coverage_before': trace['coverage_before'], 'labels': block['label'],
                            'reference_path': block['path'], 'family': block['family'], 'logits': logits})
            write_json(out / f'{mode}-seed{seed}.json', record)
            print('Analysis', mode, seed, 'test-new', metrics['test_new'], flush=True)
    write_json(ROOT / 'results.json', result)
    shortcut_controls(data)


def report():
    rows = json.loads((ROOT / 'results.json').read_text())
    def moments(values):
        values = torch.tensor(values, dtype=torch.float64)
        return float(values.mean()), float(values.std(unbiased=True))
    lines = ['# WSR-v0.3 前置：只读信息库与多轮工作区', '',
             '三配对种子，A 普通 / B 轮次编码 / C 软 Coverage / D 硬 top-1 读取屏蔽。仅最终计数 CE。', '',
             '| 组 | 参数量 | 训练准确率 | 同查询池新实例 | 留出查询组合 distinct | 留出查询组合 reread | 留出查询组合总准确率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    main = []
    for mode in 'ABCD':
        group = [r for r in rows if r['mode'] == mode]
        cells = []
        for split, family in [('train', 'all'), ('test_known', 'all'), ('test_new', 'distinct'), ('test_new', 'reread'), ('test_new', 'all')]:
            mean, sd = moments([r['metrics'][split][family]['accuracy'] * 100 for r in group])
            cells.append(f'{mean:.2f} ± {sd:.2f}%')
        lines.append(f"| {mode} | {group[0]['parameters']} | " + ' | '.join(cells) + ' |')
        for r in group:
            for split in r['metrics']:
                for family, metric in r['metrics'][split].items():
                    main.append({'mode': mode, 'seed': r['seed'], 'split': split, 'family': family, **metric})
    lines += ['', '均值 ± 三个初始化/训练日程种子的样本 SD，不是总体置信区间。最终检查点统一512步，不按最佳验证点替换。', '',
              '## 读取行为与因果诊断', '',
              '| 组 | top-1 重读率 | 不同读取槽数 / 8 | 参考路径 top-1 一致率 | 重读任务正确槽 attention | 中途 evidence 清零后准确率 |',
              '|---|---:|---:|---:|---:|---:|']
    for mode in 'ABCD':
        group = [r for r in rows if r['mode'] == mode]
        def avg(fn): return sum(fn(r) for r in group) / len(group)
        lines.append(f"| {mode} | {avg(lambda r:r['attention']['all']['top1_repeat_fraction']):.3f} | {avg(lambda r:r['attention']['all']['mean_unique_top1_slots']):.3f} | {avg(lambda r:r['attention']['all']['reference_top1_accuracy']):.3f} | {avg(lambda r:r['attention']['reread']['reference_reread_slot_attention']):.3f} | {100*avg(lambda r:r['interventions']['zero_evidence']['all']['accuracy']):.2f}% |")
    lines += ['', '## 计算轮数', '', '| 组 | 1轮 | 2轮 | 4轮 | 6轮 | 8轮（训练值） | 10轮 | 12轮 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode in 'ABCD':
        group = [r for r in rows if r['mode'] == mode]
        cells = [f"{100*sum(r['rounds'][str(n)]['all']['accuracy'] for r in group)/len(group):.2f}%" for n in [1,2,4,6,8,10,12]]
        lines.append('| ' + mode + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 解释边界', '',
              '- 提供固定八轮和当前查询地址/权重游标；状态更新学习加权计数，没有提供正确部分和或硬编码累加。',
              '- 只读原始库不被擦除；Coverage 改变读取分数，硬屏蔽仅排除历史 attention argmax 槽。软 attention 的其余内容不等于从未读取。',
              '- 参考路径只用于验证标签与事后分析。重复参考访问表示标准执行会重读；网络可能缓存事实，不是证明任何算法都必须物理重读。',
              '- C 的 lambda=1、rho=0.9 预先固定；没有根据读取多样性选择配置。C/D 另有16维机制状态，W容量均16；B额外128参数。',
              '- 所有查询均训练八轮；B在额外轮数复用第八轮编码，地址复用最后一个查询地址。深度曲线不是独立训练的等预算单轮对照。',
              '- 中途干预改变预测只能说明依赖；需结合准确率、反事实计数跟随及高/低读取槽对照判断依赖是否有用。',
              '- 没有加载、更新或连接 Mamba/Direct Dream，没有随机轮数训练或辅助监督退火。',
              '- “同查询池新实例”只表示查询hash bucket在训练允许集合，不保证该具体查询曾出现；留出组合bucket与训练严格不相交。',
              '', '![学习曲线与中间状态](overview.png)']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    with (ROOT / 'main-results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(main[0])); writer.writeheader();writer.writerows(main)
    for name, extractor in [('rounds', lambda r:[{'rounds': n, 'family': family, **m} for n,v in r['rounds'].items() for family,m in v.items()]),
                             ('attention', lambda r:[{'family': family, **m} for family,m in r['attention'].items()])]:
        table = [{'mode': r['mode'], 'seed': r['seed'], **x} for r in rows for x in extractor(r)]
        with (ROOT / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    tables = {'interventions': [], 'partial-count-probes': [], 'state-replacements': [], 'paired-differences': []}
    for r in rows:
        base = {'mode':r['mode'], 'seed':r['seed']}
        for kind, values in r['interventions'].items():
            if isinstance(values, dict) and 'all' in values:
                tables['interventions'].extend(base | {'intervention':kind, 'family':f} | metric for f,metric in values.items())
        tables['partial-count-probes'].extend(base | probe for probe in r['intermediate']['posthoc_count_probes'])
        for condition, value in r['intermediate']['state_replacements'].items():
            for answer in ['original_answer','counterfactual_answer']:
                for family, metric in value[answer].items():
                    tables['state-replacements'].append(base | {'intervention':condition, 'answer':answer, 'family':family} | metric)
        baseline = next(x for x in rows if x['mode']=='A' and x['seed']==r['seed'])
        for family in ['all','distinct','reread']:
            tables['paired-differences'].append(base | {'family':family, 'accuracy_difference_pp':100*(r['metrics']['test_new'][family]['accuracy']-baseline['metrics']['test_new'][family]['accuracy'])})
    for name, table in tables.items():
        with (ROOT / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)


def audit():
    original = json.loads((ROOT / 'preservation-before.json').read_text())
    altered = [p for p,h in original['artifacts'].items() if digest(p) != h]
    assert not altered, altered
    calibrations = json.loads((ROOT / 'calibration-preservation.json').read_text())
    assert all(digest(p) == h for p,h in calibrations.items())
    for p,h in original['sources'].items():
        assert digest(ROOT / 'source-before' / p) == h
    rows = json.loads((ROOT / 'results.json').read_text())
    assert len(rows) == 12
    validate_data(torch.load(ROOT / 'data.pt', weights_only=True))
    tests = (ROOT / 'tests-final.log').read_text()
    assert 'Ran 41 tests' in tests and '\nOK' in tests
    paired = []
    for seed in range(3):
        paths = [ROOT / 'runs' / f'{mode}-seed{seed}' for mode in 'ABCD']
        schedules = [torch.load(p / 'schedule.pt', weights_only=True) for p in paths]
        assert all(torch.equal(schedules[0], x) for x in schedules[1:])
        initials = [torch.load(p / 'checkpoint-00000.pt', weights_only=True)['model'] for p in paths]
        for other in initials[1:]:
            assert all(torch.equal(v, other[k]) for k,v in initials[0].items())
        for path in paths:
            summary = json.loads((path / 'summary.json').read_text())
            assert summary['updates'] == STEPS and summary['parameter_update_norm'] > 0 and summary['complete']
            assert all(r['gradient_norm'] > 0 and math.isfinite(r['ce']) for r in json.loads((path / 'training.json').read_text()))
        paired.append({'seed': seed, 'identical_schedules': True, 'identical_core_initialization': True})
    destination = ROOT / 'source-run'
    for folder in ['wsr', 'tests']:
        for p in Path(folder).glob('*.py'):
            dest = destination / p; dest.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(p,dest)
    for name in ['WSR_V03_PRE.md', 'STATUS.md']:
        if Path(name).exists():shutil.copy2(name, destination / name)
    write_json(ROOT / 'verification.json', {'old_artifacts_unchanged': len(original['artifacts']), 'altered': altered,
               'paired': paired, 'Mamba_loaded_or_updated': False, 'Direct_Dream_updated': False,
               'main_runs': 12, 'main_updates': 12 * STEPS, 'main_answer_draws': 12 * STEPS * 64,
               'random_depth_or_auxiliary_supervision': False,
               'calibration_files_unchanged': len(calibrations), 'unit_tests_passed': 41,
               'task_supervision': 'final weighted count only; all partial-count probes are posthoc',
               'formal_budget': 512, 'schedule_allocated_ceiling': 1024,
               'raw_data_sha256': digest(ROOT/'data.pt'), 'test_model_evaluation_after_all_main_runs': True})
    print('Workspace audit passed', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'pilot', 'train', 'analyze', 'report', 'audit'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    args = parser.parse_args()
    data = initialize()
    if args.stage == 'pilot':pilot(data)
    elif args.stage == 'train':
        gate = json.loads((ROOT / 'pilot-gate.json').read_text())
        assert gate['passed'], 'Diagnose pilot before seed expansion'
        for seed in args.seeds:
            for mode in 'ABCD':train(mode,seed,data)
    elif args.stage == 'analyze':analyze(data); report()
    elif args.stage == 'report':report()
    elif args.stage == 'audit':audit()


if __name__ == '__main__':main()

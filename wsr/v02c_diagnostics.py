"""v0.2c stage 1: layer interfaces and offline state gradients; zero optimizer updates."""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import shutil
import time

import torch

from .layer_edit import cache_inventory, edit_layers_cache, checkpointed_continuation_loss
from .pretrained_evaluate import deserialize_cache, serialize_cache, score
from .pretrained_experiment import continuation, model_load
from .state_dream import edit_cache, history_input
from .v02_data import digest
from .v02_train import load_run, write_json

ROOT = Path('artifacts/wsr-v0.2c')
OLD = Path('artifacts/wsr-v0.2a')
LAYERS = (5, 12, 19)
HISTORIES = (512, 2048, 8192, 32768)
HORIZONS = (64, 512)
SCALES = (1e-4, 1e-3, 1e-2)


def initialize():
    ROOT.mkdir(exist_ok=True)
    for name in ['interface', 'gradients', 'states', 'execution-logs']:
        (ROOT / name).mkdir(exist_ok=True)
    protocol = {'stage': 'interface and offline gradient diagnosis only',
                'edit_layers': list(LAYERS), 'human_layers': [6, 13, 20],
                'histories': list(HISTORIES), 'horizons': list(HORIZONS),
                'negative_gradient_relative_norms': list(SCALES),
                'positive_and_random_control_relative_norm': 1e-3,
                'B0_checkpoints': [512, 1024, 2048], 'B0_seed': 0,
                'dtype': 'FP32', 'future_used_for_gradient': 'offline diagnostic supervision only',
                'panel': 'same two PG validation books at all four histories; first two eligible Wiki validation documents at 512/2048',
                'gradient_horizon': 'full indicated 64 or 512 tokens, no truncation',
                'gradient_memory': '64-token activation checkpoint chunks for horizon512; cache graph remains connected across all chunks',
                'Mamba_optimizer_updates': 0, 'Dream_optimizer_updates': 0,
                'test_split_used': False, 'new_auxiliary_losses': False}
    path = ROOT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Refuse to alter an existing diagnostic protocol')
    write_json(path, protocol)


def panel():
    rows = json.loads((OLD / 'bank/index.json').read_text())['validation']
    wiki_ids = sorted({r['document']['id'] for r in rows if r['document']['dataset'] == 'wikitext103'})[:2]
    chosen = [r for r in rows if r['document']['dataset'] == 'pg19' or r['document']['id'] in wiki_ids]
    # PG2048 was absent from the old bank. Derive it once from the same book's 512 cache.
    for row in list(chosen):
        if row['document']['dataset'] == 'pg19' and row['boundary'] == 512:
            derived = dict(row)
            derived.update({'id': row['id'].rsplit('-', 1)[0] + '-2048', 'boundary': 2048,
                            'path': str(ROOT / 'states' / (row['id'].rsplit('-', 1)[0] + '-2048.pt')),
                            'derived_from': row})
            chosen.append(derived)
    return sorted(chosen, key=lambda r: (r['document']['dataset'], r['document']['id'], r['boundary']))


def load_case(model, row):
    tokens = torch.load(row['token_path'], weights_only=True)['tokens'][None].to(next(model.parameters()).device)
    path = Path(row['path'])
    if not path.exists():
        source = row['derived_from']
        data = torch.load(source['path'], weights_only=True, mmap=True)
        if data['model_revision'] != model.asset_manifest['revision'] or data['document']['sha256'] != row['document']['sha256']:
            raise ValueError('Derived cache provenance mismatch')
        state = deserialize_cache(data, tokens.device)
        state = model.wake(tokens[:, state.processed:row['boundary'] - 1], state)
        data = serialize_cache(state)
        data.update({'model_revision': model.asset_manifest['revision'], 'document': row['document'],
                     'boundary': row['boundary'], 'token_path': row['token_path']})
        torch.save(data, path)
    data = torch.load(path, weights_only=True, mmap=True)
    if data['model_revision'] != model.asset_manifest['revision'] or data['document']['sha256'] != row['document']['sha256']:
        raise ValueError('Cache provenance mismatch')
    state = deserialize_cache(data, tokens.device)
    if state.processed != row['boundary'] - 1 or tokens.shape[1] < row['boundary'] + max(HORIZONS):
        raise ValueError('Invalid anchor or future window')
    return tokens, state


def check_preserved():
    manifest = json.loads((ROOT / 'preservation-before.json').read_text())
    files = manifest['protected_artifacts']
    sources = manifest['original_sources']
    # Older manifests also list working-tree sources. Those may be extended now,
    # but their exact pre-change content must remain in this run's source-before.
    def preserved_path(p):
        return ROOT / 'source-before' / p if p in sources else Path(p)
    altered = [p for p, h in files.items() if not preserved_path(p).exists() or digest(preserved_path(p)) != h]
    if altered:
        raise RuntimeError(f'Old artifacts changed: {altered}')
    for p, h in sources.items():
        archived = ROOT / 'source-before' / p
        if not archived.is_file() or digest(archived) != h:
            raise RuntimeError(f'Original source snapshot mismatch: {p}')
    modified = [p for p, h in sources.items() if digest(p) != h]
    result = {'protected_old_artifact_files': sum(p not in sources for p in files),
              'protected_old_source_entries': sum(p in sources for p in files), 'altered_old_artifacts': altered,
              'modified_sources_archived_in': str(ROOT / 'source-before'), 'modified_source_files': modified}
    write_json(ROOT / 'preservation-check.json', result)
    return result


def verify(model):
    row = next(r for r in panel() if r['document']['dataset'] == 'pg19' and r['boundary'] == 512)
    tokens, cache = load_case(model, row)
    inventory = cache_inventory(model, cache)
    weights = torch.load('artifacts/pretrained-v1/model/pytorch_model.bin', mmap=True, weights_only=True)
    for item in inventory:
        prefix = f"backbone.layers.{item['index']}.mixer."
        item['official_A_log_shape'] = list(weights[prefix + 'A_log'].shape)
        item['official_conv_weight_shape'] = list(weights[prefix + 'conv1d.weight'].shape)
        item['official_out_proj_shape'] = list(weights[prefix + 'out_proj.weight'].shape)
        assert weights[prefix + 'A_log'].numel() == item['nheads']
        assert weights[prefix + 'conv1d.weight'].shape[0] == item['conv_shape'][1]
        assert weights[prefix + 'out_proj.weight'].shape[1] == item['nheads'] * item['headdim']
    del weights
    dream, _, _ = load_run(OLD / 'runs/B0-seed-0/checkpoint-2048.pt', tokens.device)
    dream.requires_grad_(False)
    with torch.no_grad():
        old, old_trace, _ = edit_cache(model, dream, cache, tokens, 512, 128, 12)
        new, traces, _ = edit_layers_cache(model, dream, cache, tokens, 512)
        assert torch.equal(old.layers[12].ssm, new.layers[12].ssm)
        assert all(torch.equal(a, b) for a, b in zip(old_trace, traces[12]))
        old_logits, _ = model.run(tokens[:, 511:575], old)
        new_logits, _ = model.run(tokens[:, 511:575], new)
        assert torch.equal(old_logits, new_logits)
        zero, _, _ = edit_layers_cache(model, None, cache, tokens, 512)
        reference, _ = model.run(tokens[:, 511:575], cache)
        zero_logits, end = model.run(tokens[:, 511:575], zero)
        assert torch.equal(reference, zero_logits) and end.processed == 575
        continuous, _ = model.run(tokens[:, :575])
        split_error = float((continuous[:, 511:] - reference).abs().max())
        torch.testing.assert_close(continuous[:, 511:], reference, atol=5e-4, rtol=5e-4)
        target = tokens[:, 512:576].flatten()
        from torch.nn import functional as F
        split_nll_difference = float((F.cross_entropy(continuous[:, 511:].flatten(0, 1), target)
                                      - F.cross_entropy(reference.flatten(0, 1), target)).abs())
        changed, _, _ = edit_layers_cache(model, dream, cache, tokens, 512, edit_layers=[11, 12, 13])
        changed_future = tokens.clone()
        changed_future[:, 512:] = (changed_future[:, 512:] + 1) % model.lm_head.out_features
        other, _, _ = edit_layers_cache(model, dream, cache, changed_future, 512, edit_layers=[11, 12, 13])
        for i in range(len(cache.layers)):
            assert changed.layers[i].conv is cache.layers[i].conv
            if i in [11, 12, 13]:
                assert torch.equal(changed.layers[i].ssm, other.layers[i].ssm)
            else:
                assert changed.layers[i] is cache.layers[i]
        assert changed.processed == cache.processed
    # All three independent injected states must be reachable through one joint loss.
    states = {i: cache.layers[i].ssm.detach().clone().requires_grad_() for i in LAYERS}
    loss, _ = continuation(model, tokens, cache.edit_many(states), 512, 64)
    grads = torch.autograd.grad(loss, tuple(states.values()))
    norms = {str(i): float(g.norm()) for i, g in zip(LAYERS, grads)}
    assert all(math.isfinite(n) and n > 0 for n in norms.values())
    # The real-weight gate compares a full 128-token graph against 2 connected checkpoints.
    direct, _ = continuation(model, tokens, cache.edit_many(states), 512, 128)
    direct_grads = torch.autograd.grad(direct, tuple(states.values()))
    checked = checkpointed_continuation_loss(model, tokens, cache.edit_many(states), 512, 128)
    checked_grads = torch.autograd.grad(checked, tuple(states.values()))
    checkpoint_errors = {}
    for i, a, b in zip(LAYERS, direct_grads, checked_grads):
        relative = norm(a-b) / norm(a)
        assert relative < 1e-3
        checkpoint_errors[str(i)] = {'relative_gradient_error': relative, 'max_absolute_error': float((a-b).abs().max())}
    assert abs(float(direct.detach()) - float(checked.detach())) < 2e-5
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    result = {'passed': True, 'cache_inventory': inventory, 'legacy_12_edit_and_trace_bitwise_equal': True,
              'legacy_12_logits_bitwise_equal': True, 'no_edit_logits_bitwise_equal': True,
              'continuous_vs_segmented_max_logit_difference': split_error,
              'continuous_vs_segmented_absolute_nll_difference': split_nll_difference,
              'continuous_vs_segmented_tolerance': {'atol': 5e-4, 'rtol': 5e-4},
              'checkpoint128_vs_direct128_gradient_errors': checkpoint_errors,
              'checkpoint128_vs_direct128_absolute_nll_difference': abs(float(direct.detach()) - float(checked.detach())),
              'all_conv_and_unselected_cache_objects_preserved': True, 'future_mutation_edit_invariance': True,
              'injected_state_gradient_norms': norms, 'anchor_processed_after_64': end.processed,
              'official_cache_reference': 'artifacts/pretrained-v1/vendor/mamba2.py:345',
              'official_reference_sha256': digest('artifacts/pretrained-v1/vendor/mamba2.py'),
              'model_revision': model.asset_manifest['revision'], 'optimizer_updates': 0}
    write_json(ROOT / 'interface/verification.json', result)
    print('Official model interface checks passed; segmented max logit error', split_error, flush=True)


def norm(x):
    return float(x.detach().double().norm())


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return float((a.double() * b.double()).sum()) / denominator if denominator else None


def state_gradient(model, tokens, cache, boundary, horizon, layer):
    h = cache.layers[layer].ssm.detach().clone().requires_grad_()
    if horizon == 512:
        loss = checkpointed_continuation_loss(model, tokens, cache.edit_many({layer: h}), boundary, horizon)
    else:
        loss, _ = continuation(model, tokens, cache.edit_many({layer: h}), boundary, horizon)
    grad, = torch.autograd.grad(loss, h)
    return float(loss.detach()), grad.detach()


@torch.no_grad()
def evaluate(model, tokens, cache, boundary, horizon, layer, value):
    losses, _ = score(model, tokens, cache.edit_many({layer: value}), boundary, horizon, chunk=64)
    return sum(losses) / len(losses)


def diagnose(model, only_first=False):
    gate = ROOT / 'interface/verification.json'
    if not gate.exists() or not json.loads(gate.read_text())['passed']:
        raise ValueError('Run official interface verification before state gradients')
    checkpoints = {}
    for step in [512, 1024, 2048]:
        dream, _, _ = load_run(OLD / f'runs/B0-seed-0/checkpoint-{step:04d}.pt', next(model.parameters()).device)
        checkpoints[step] = dream.requires_grad_(False).eval()
    versions = [p._version for p in model.parameters()]
    selected = panel()[:1] if only_first else panel()
    for row in selected:
        tokens, cache = load_case(model, row)
        boundary = row['boundary']
        with torch.no_grad():
            history = history_input(model, tokens, boundary, 128)
            dream_edits = {step: d(cache.layers[12].ssm, history)[0] for step, d in checkpoints.items()}
        for horizon in HORIZONS:
            for layer in LAYERS:
                dest = ROOT / 'gradients' / f"{row['id']}-future{horizon}-layer{layer}.json"
                if dest.exists():
                    continue
                torch.cuda.reset_peak_memory_stats() if tokens.is_cuda else None
                start = time.monotonic()
                baseline, grad = state_gradient(model, tokens, cache, boundary, horizon, layer)
                h = cache.layers[layer].ssm
                hn, gn = norm(h), norm(grad)
                assert math.isfinite(gn) and gn > 0 and torch.isfinite(grad).all()
                probes = []
                with torch.no_grad():
                    # Fixed norm radius in relative state coordinates for every layer/history.
                    direction = -grad / gn
                    for scale in SCALES:
                        delta = direction * (hn * scale)
                        value = evaluate(model, tokens, cache, boundary, horizon, layer, h + delta)
                        probes.append({'kind': 'negative_gradient', 'requested_relative_norm': scale,
                                       'actual_relative_norm': norm((h + delta) - h) / hn,
                                       'nll': value, 'delta_nll': value - baseline,
                                       'first_order_delta_nll': float((grad.double() * delta.double()).sum())})
                    generator = torch.Generator(device=tokens.device).manual_seed(20261117 + layer)
                    random = torch.randn(h.shape, device=tokens.device, dtype=h.dtype, generator=generator)
                    controls = {'positive_gradient': grad / gn, 'random': random / norm(random)}
                    for kind, vector in controls.items():
                        delta = vector * (hn * 1e-3)
                        value = evaluate(model, tokens, cache, boundary, horizon, layer, h + delta)
                        probes.append({'kind': kind, 'requested_relative_norm': 1e-3,
                                       'actual_relative_norm': norm((h + delta) - h) / hn,
                                       'nll': value, 'delta_nll': value - baseline})
                    alignment = []
                    if layer == 12:
                        for step, edited in dream_edits.items():
                            delta = edited - h
                            cos = cosine(delta, -grad)
                            nll = evaluate(model, tokens, cache, boundary, horizon, layer, edited)
                            alignment.append({'checkpoint': step, 'seed': 0, 'relative_edit_norm': norm(delta) / hn,
                                              'cosine_to_negative_gradient': cos,
                                              'angle_to_negative_gradient_degrees': math.degrees(math.acos(max(-1, min(1, cos)))) if cos is not None else None,
                                              'first_order_delta_nll': float((grad.double() * delta.double()).sum()),
                                              'actual_delta_nll': nll - baseline, 'nll': nll})
                result = {'id': row['id'], 'dataset': row['document']['dataset'], 'document': row['document']['id'],
                          'history': boundary, 'horizon': horizon, 'layer_index': layer, 'human_layer': layer + 1,
                          'baseline_nll': baseline, 'state_norm': hn, 'gradient_norm': gn,
                          'gradient_rms': gn / math.sqrt(grad.numel()), 'gradient_elements': grad.numel(),
                          'relative_L2_sensitivity': hn * gn,
                          'elementwise_relative_gradient_norm': norm(grad * h),
                          'probes': probes, 'B0_alignment': alignment,
                          'gradient_horizon_tokens': horizon, 'temporal_gradient_detach': False,
                          'activation_checkpoint_chunk': 64 if horizon == 512 else None,
                          'seconds': time.monotonic() - start,
                          'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if tokens.is_cuda else 0,
                          'cache_sha256': digest(row['path']), 'token_file_sha256': digest(row['token_path'])}
                assert math.isfinite(baseline) and all(math.isfinite(p['nll']) for p in probes)
                torch.save({'gradient': grad.cpu(), 'layer_index': layer, 'history': boundary, 'horizon': horizon,
                            'id': row['id'], 'cache_path': row['path']}, dest.with_suffix('.pt'))
                write_json(dest, result)
                print(row['id'], 'future', horizon, 'layer', layer + 1, 'NLL', round(baseline, 5),
                      'g', round(gn, 6), 'best delta', round(min(p['delta_nll'] for p in probes), 7),
                      'seconds', round(result['seconds'], 2), flush=True)
                del grad
                gc.collect()
        del cache, tokens, dream_edits
        gc.collect()
    assert [p._version for p in model.parameters()] == versions
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    write_json(ROOT / 'gradient-execution-check.json', {
        'all_backbone_parameter_versions_unchanged': True, 'all_backbone_gradients_none': True,
        'Mamba_optimizer_updates': 0, 'Dream_optimizer_updates': 0,
        'panel_mode': 'first_case_smoke' if only_first else 'full_panel',
        'model_revision': model.asset_manifest['revision']})


def report():
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / 'gradients').glob('*.json'))]
    expected = len(panel()) * len(HORIZONS) * len(LAYERS)
    flat = []
    for r in rows:
        for p in r['probes']:
            flat.append({k: r[k] for k in ['id', 'dataset', 'document', 'history', 'horizon', 'human_layer',
                                          'baseline_nll', 'state_norm', 'gradient_norm', 'relative_L2_sensitivity']} | p)
    if flat:
        with (ROOT / 'state-gradients.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
            writer.writeheader();writer.writerows(flat)
    alignment = [{k: r[k] for k in ['id', 'history', 'horizon', 'baseline_nll']} | a for r in rows for a in r['B0_alignment']]
    if alignment:
        with (ROOT / 'B0-alignment.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(alignment[0]));writer.writeheader();writer.writerows(alignment)
    summary = []
    for history in HISTORIES:
        for horizon in HORIZONS:
            for layer in LAYERS:
                group = [r for r in rows if (r['history'], r['horizon'], r['layer_index']) == (history, horizon, layer)]
                if not group:
                    continue
                def mean(values):
                    return sum(values) / len(values)
                at_scale = [p for r in group for p in r['probes'] if p['kind'] == 'negative_gradient' and p['requested_relative_norm'] == 1e-3]
                small = [p for r in group for p in r['probes'] if p['kind'] == 'negative_gradient' and p['requested_relative_norm'] == 1e-4]
                summary.append({'history': history, 'horizon': horizon, 'layer': layer + 1, 'cases': len(group),
                                'mean_gradient_norm': mean([r['gradient_norm'] for r in group]),
                                'mean_relative_L2_sensitivity': mean([r['relative_L2_sensitivity'] for r in group]),
                                'mean_delta_nll_at_0.0001': mean([p['delta_nll'] for p in small]),
                                'negative_gradient_small_improved_cases': sum(p['delta_nll'] < 0 for p in small),
                                'mean_delta_nll_at_0.001': mean([p['delta_nll'] for p in at_scale]),
                                'negative_gradient_improved_cases': sum(p['delta_nll'] < 0 for p in at_scale)})
    result = {'complete': len(rows) == expected, 'gradient_cases': len(rows), 'expected': expected,
              'summary': summary, 'B0_alignment': alignment, 'optimizer_updates': 0,
              'full_finite_gradients': all(math.isfinite(r['gradient_norm']) and r['gradient_norm'] > 0 for r in rows),
              'total_diagnostic_seconds': sum(r['seconds'] for r in rows),
              'peak_gpu_allocated_bytes': max([r['peak_gpu_allocated_bytes'] for r in rows], default=0)}
    write_json(ROOT / 'results.json', result)
    lines = ['# WSR-v0.2c 第一阶段：状态层位诊断', '',
             f"完成 {len(rows)}/{expected} 个层位×历史×窗口诊断；主模型和 Dream 优化器更新均为 0。",
             '所有梯度使用真实未来，仅作离线分析，不是可部署的 Dream 输入或性能收益。', '',
             '| 历史 | 预测窗口 | 层 | 样本数 | 梯度范数均值 | 相对 L2 敏感性 | 0.01% 负梯度 ΔNLL | 0.1% 负梯度 ΔNLL | 0.1% 改善样本 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['history']} | {r['horizon']} | {r['layer']} | {r['cases']} | {r['mean_gradient_norm']:.6g} | {r['mean_relative_L2_sensitivity']:.6g} | {r['mean_delta_nll_at_0.0001']:+.7f} | {r['mean_delta_nll_at_0.001']:+.7f} | {r['negative_gradient_improved_cases']}/{r['cases']} |")
    lines += ['', '## 旧 B0 编辑方向与实际预测', '',
              '同一旧 seed0 的三个检查点，按历史长度汇总两种预测窗口；角度相对负梯度，ΔNLL>0 为退化。', '',
              '| 历史 | 旧检查点更新数 | 案例数 | 相对修改范数均值 | 负梯度余弦均值 | 实际 ΔNLL 均值 |',
              '|---:|---:|---:|---:|---:|---:|']
    for history in HISTORIES:
        for step in [512, 1024, 2048]:
            group = [a for a in alignment if a['history'] == history and a['checkpoint'] == step]
            if not group:
                continue
            average = lambda key: sum(a[key] for a in group) / len(group)
            lines.append(f"| {history} | {step} | {len(group)} | {average('relative_edit_norm'):.5f} | {average('cosine_to_negative_gradient'):+.6f} | {average('actual_delta_nll'):+.7f} |")
    lines += ['', '## 解释范围', '',
              '- 第 6/13/20 层使用同一基底缓存、同一未来与相同相对 L2 修改半径；三层从原始缓存独立干预，不叠加。',
              '- 两本 PG 验证书籍覆盖四个历史长度；两篇 Wiki 验证文章覆盖 512/2048。同长度内可配对比较层位；跨长度优先使用同一 PG 书籍，不能忽视样本和语料差异。',
              '- 梯度范数不是可训练性排名；相对敏感性为 ||H||₂ ||∇H L||₂，它是相对 L2 半径下的一阶最陡方向尺度。',
              '- 小幅负梯度收益是已知答案的局部可改进空间，不证明无未来输入的编辑器能学会该方向。',
              '- 512-token 梯度完整回传；使用 64-token 分块重计算，块间缓存仍在计算图内，没有 detach 或截短为 64 token。',
              '- B0 方向只使用旧 seed0 的 512/1024/2048 检查点；不等于三个 Dream 种子的结论。',
              '- 尚未训练第 6/20 层或三层编辑器；不能回答多层协调是否已有收益。', '',
              '详表见 state-gradients.csv、B0-alignment.csv；每个案例的原始梯度保存在 gradients/*.pt。', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['interface', 'gradients', 'smoke', 'report', 'audit'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    initialize()
    if args.stage in ['interface', 'gradients', 'smoke']:
        model = model_load(args.device)
        if args.stage == 'interface':
            verify(model)
        else:
            diagnose(model, only_first=args.stage == 'smoke')
            report()
    elif args.stage == 'report':
        report()
    else:
        result = report()
        preservation = check_preserved()
        assert result['complete'] and result['full_finite_gradients']
        interface = json.loads((ROOT / 'interface/verification.json').read_text())
        execution = json.loads((ROOT / 'gradient-execution-check.json').read_text())
        assert interface['passed'] and execution['panel_mode'] == 'full_panel'
        unique = set()
        baseline_groups = {}
        for path in (ROOT / 'gradients').glob('*.json'):
            row = json.loads(path.read_text())
            key = (row['id'], row['horizon'], row['layer_index'])
            assert key not in unique
            unique.add(key)
            baseline_groups.setdefault(key[:2], []).append(row['baseline_nll'])
            saved = torch.load(path.with_suffix('.pt'), weights_only=True)
            gradient = saved['gradient']
            assert saved['id'] == row['id'] and saved['layer_index'] == row['layer_index']
            assert list(gradient.shape) == interface['cache_inventory'][row['layer_index']]['ssm_shape']
            assert torch.isfinite(gradient).all() and norm(gradient) > 0
            assert row['temporal_gradient_detach'] is False
            assert row['gradient_horizon_tokens'] == row['horizon']
            for probe in row['probes']:
                assert math.isfinite(probe['nll'])
                assert abs(probe['actual_relative_norm'] / probe['requested_relative_norm'] - 1) < 1e-3
        assert all(max(values) - min(values) < 1e-6 for values in baseline_groups.values())
        tests = (ROOT / 'tests-final.log').read_text()
        assert 'Ran 36 tests' in tests and '\nOK' in tests
        destination = ROOT / 'source-run'
        for directory in ['wsr', 'tests']:
            for p in Path(directory).glob('*.py'):
                dest = destination / p; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
        for name in ['WSR_V02C.md', 'STATUS.md']:
            if Path(name).exists():
                shutil.copy2(name, destination / name)
        write_json(ROOT / 'verification.json', {'complete': True, 'gradient_cases': result['gradient_cases'],
                   'preservation': preservation, 'Mamba_optimizer_updates': 0, 'Dream_optimizer_updates': 0,
                   'matched_unedited_baselines': len(baseline_groups), 'gradient_files_finite': True,
                   'relative_norm_radius_checks_passed': True, 'unit_tests_passed': 36,
                   'interface_verified': True, 'execution': execution, 'test_split_used': False})
        print('v0.2c audit passed', flush=True)


if __name__ == '__main__':
    main()

"""Paired, base-only timescale experiment; original model equations are unchanged."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .model import Backbone
from .diagnose import Profile, batch, objective, read, state_ablation

VARIANTS = ['original', 'slow', 'multiscale', 'slow_curriculum']
TARGETS = [2., 8., 32., 128.]
CALIBRATION_SEEDS = range(700000, 700016)
PROBE_SEEDS = range(800000, 800016)
EVAL_SEED_START = 1000000


def profiles():
    return {
        'copy8': Profile('copy', length=8),
        'copy24': Profile('copy', length=24),
        'copy64': Profile('copy', length=64),
        'copy64_interference': Profile('copy', length=64, distractors=True),
        'copy128_ood': Profile('copy', length=128, distractors=True),
        'count2': Profile('memory', length=2),
        'count4': Profile('memory', length=4),
        'count_original': Profile('memory', episodes=3),
        'lookup2': Profile('lookup', pairs=2),
        'redundancy_original': Profile('redundancy', episodes=3),
        'rules_heldout': Profile('rules', episodes=3),
    }


def train_profile(step, steps, variant):
    """All variants share task order and world seeds; only curriculum changes difficulty."""
    task = step % 3
    stage = min(2, step * 4 // steps) if variant == 'slow_curriculum' else 2
    if task == 0:
        choices = [[2, 8], [8, 24], [2, 8, 24, 64]][stage]
        # Interleave neutral and alias distractors at every training length.
        visit = step // 3
        length = choices[(visit // 2) % len(choices)]
        return Profile('copy', length=length, distractors=bool(visit % 2)), stage
    if task == 1:
        return [Profile('memory', length=2), Profile('memory', length=4, episodes=3),
                Profile('memory', episodes=3)][stage], stage
    return [Profile('lookup', pairs=1), Profile('lookup', pairs=2),
            Profile('redundancy', episodes=3)][stage], stage


def quantiles(values):
    levels = torch.tensor([0., .1, .25, .5, .75, .9, 1.], dtype=torch.float64)
    return dict(zip(['min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'],
                    torch.quantile(values.double().flatten(), levels).tolist()))


@torch.no_grad()
def probe(model, profile, seeds=PROBE_SEEDS):
    rates, rms, maxima, innovations = [], [], [], []
    state = None
    previous = model.initial(len(seeds), next(model.parameters()).device).ssm.double()

    def observe(dt, decay, current):
        nonlocal previous
        h = current.ssm.detach().double()
        rate = dt.detach().double() * model.log_a.detach().exp().double()
        rates.append(rate)
        rms.append(h.square().mean((2, 3)).sqrt())
        maxima.append(h.abs().amax((2, 3)))
        write = h - decay.detach().double()[..., None, None] * previous
        innovations.append(write.square().mean((2, 3)).sqrt())
        previous = h

    correct, total, losses = 0, 0, []
    for tokens, answer in batch(seeds, profile):
        _, state = model.run(tokens, state, observer=observe)
        loss, logits = read(model, state, answer)
        losses.append(float(loss))
        correct += int((logits.argmax(-1) == answer).sum())
        total += len(answer)
    rate = torch.stack(rates, 1)
    h_rms = torch.stack(rms, 1)
    h_max = torch.stack(maxima, 1)
    write_rms = torch.stack(innovations, 1)
    if bool((rate <= 0).any()) or not all(bool(torch.isfinite(x).all()) for x in [rate, h_rms, h_max, write_rms]):
        raise RuntimeError('Nonfinite probe dynamics')
    half = math.log(2) / rate
    summary = {'profile': asdict(profile), 'worlds': len(seeds),
               'tokens_per_world': rate.shape[1], 'log_a': model.log_a.tolist(),
               'accuracy': correct / total, 'loss': sum(losses) / len(losses), 'heads': []}
    bins = torch.tensor([0., .5, 2., 8., 32., 128., 512., float('inf')], dtype=torch.float64)
    for head in range(model.heads):
        effective = math.log(2) / rate[:, :, head].mean(1)
        head_half = half[:, :, head]
        hist = torch.histogram(head_half.flatten(), bins=bins).hist.tolist()
        survival_loss = rate[:, 1:, head].cumsum(1)
        crossed = survival_loss >= math.log(2)
        observed = crossed.any(1)
        hitting = crossed.to(torch.int64).argmax(1) + 1
        summary['heads'].append({
            'head': head, 'channels_sharing_decay': model.width // model.heads,
            'ssm_coordinates_sharing_decay': model.width // model.heads * model.memory,
            'effective_half_life': float(math.log(2) / rate[:, :, head].mean()),
            'effective_half_life_per_world': effective.tolist(),
            'first_state_half_survival_tokens': [int(t) if bool(ok) else None for t,ok in zip(hitting,observed)],
            'survival_observation_horizon': rate.shape[1]-1,
            'survival_right_censored_fraction': float((~observed).double().mean()),
            'instant_half_life_quantiles': quantiles(head_half),
            'half_life_histogram': {'edges': [0., .5, 2., 8., 32., 128., 512., 'inf'], 'counts': hist},
            'state_rms_quantiles': quantiles(h_rms[:, :, head]),
            'state_abs_max': float(h_max[:, :, head].max()),
            'write_rms_quantiles': quantiles(write_rms[:, :, head]),
            'last_over_first_state_rms': float(h_rms[:, -1, head].mean() / h_rms[:, 0, head].mean().clamp_min(1e-12)),
            'near_frozen_fraction_decay_gt_0999': float((rate[:, :, head] < -math.log(.999)).double().mean()),
            'overwrite_fraction_decay_lt_01': float((rate[:, :, head] > -math.log(.1)).double().mean()),
        })
    raw = {'decay_rate': rate, 'state_rms': h_rms, 'state_abs_max': h_max, 'write_rms': write_rms}
    return summary, raw


@torch.no_grad()
def initialize(model, variant, targets=TARGETS):
    if variant == 'original':
        return {'log_a': model.log_a.tolist(), 'method': 'unchanged: log_a=0'}
    if variant in ['slow', 'slow_curriculum']:
        model.log_a.fill_(-3.)
        return {'log_a': model.log_a.tolist(), 'method': 'unchanged slow baseline: log_a=-3'}
    if variant != 'multiscale' or len(targets) != model.heads or min(targets) <= 0:
        raise ValueError('Multiscale needs one positive target per head')
    # Selective dt depends on local inputs. Calibrate on training-side visible inputs only.
    dt_chunks = []
    def observe(dt, decay, state):
        dt_chunks.append(dt.double())
    calibration = [Profile('memory', episodes=3), Profile('redundancy', episodes=3),
                   Profile('copy', length=24)]
    for profile in calibration:
        state = None
        for tokens, _ in batch(CALIBRATION_SEEDS, profile):
            _, state = model.run(tokens, state, observer=observe)
    mean_dt = torch.cat(dt_chunks, 0).mean(0)
    desired = torch.as_tensor(targets, dtype=torch.float64)
    log_a = torch.log(math.log(2) / (desired * mean_dt))
    model.log_a.copy_(log_a.to(model.log_a))
    actual = math.log(2) / (model.log_a.exp().double() * mean_dt)
    return {'log_a': model.log_a.tolist(), 'method': 'calibrated using mean dt, token weighted',
            'targets': list(targets), 'calibration_mean_dt': mean_dt.tolist(),
            'calibration_effective_half_life': actual.tolist(),
            'calibration_profiles': [asdict(p) for p in calibration],
            'calibration_world_seeds': [CALIBRATION_SEEDS.start, CALIBRATION_SEEDS.stop]}


@torch.no_grad()
def evaluate_all(model, worlds):
    result = {}
    seeds = range(EVAL_SEED_START, EVAL_SEED_START + worlds)
    for name, profile in profiles().items():
        # Batched in chunks to bound peak graph-free memory.
        all_logits, all_targets, all_losses, episode_rows = [], [], [], []
        for start in range(0, len(seeds), 16):
            chunk = range(seeds.start + start, seeds.start + min(len(seeds), start + 16))
            state = None
            for episode, (tokens, target) in enumerate(batch(chunk, profile)):
                _, state = model.run(tokens, state)
                loss, logits = read(model, state, target)
                all_logits.append(logits)
                all_targets.append(target)
                all_losses.append((float(loss), len(target)))
                episode_rows.extend({'world': int(seed), 'episode': episode,
                                     'label': int(y), 'prediction': int(pred), 'correct': bool(pred == y)}
                                    for seed, y, pred in zip(chunk, target, logits.argmax(-1)))
        logits, target = torch.cat(all_logits), torch.cat(all_targets)
        counts = torch.bincount(target, minlength=8)
        per_episode = {}
        for ep in range(profile.episodes):
            rows = [r for r in episode_rows if r['episode'] == ep]
            per_episode[str(ep)] = sum(r['correct'] for r in rows) / len(rows)
        result[name] = {'profile': asdict(profile), 'n': len(target),
                        'accuracy': float((logits.argmax(-1) == target).double().mean()),
                        'loss': sum(loss*n for loss,n in all_losses) / len(target),
                        'answer_uniform_baseline': .125, 'majority_baseline': float(counts.max() / len(target)),
                        'label_counts': counts.tolist(), 'per_episode_accuracy': per_episode,
                        'rows': episode_rows}
    return result


@torch.no_grad()
def monitor(model, step, directory):
    summaries, raw = {}, {}
    for name in ['copy24', 'copy64_interference', 'count_original', 'redundancy_original', 'rules_heldout']:
        summaries[name], raw[name] = probe(model, profiles()[name])
    torch.save(raw, directory / f'dynamics-step-{step:04d}.pt')
    (directory / f'dynamics-step-{step:04d}.json').write_text(json.dumps(summaries, indent=2))
    return summaries


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name != 'log_a':
            digest.update(name.encode())
            digest.update(bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def fit(config, variant, seed, directory):
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = Backbone()
    nondecay_fingerprint = fingerprint(model)
    init = initialize(model, variant, config['targets'])
    assert fingerprint(model) == nondecay_fingerprint
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    start = time.monotonic()
    monitors = {'0': monitor(model, 0, directory)}
    before = evaluate_all(model, config['worlds'])
    trace, stage_budgets = [], {}
    milestone_steps = sorted(set([config['steps']//3, 2*config['steps']//3, config['steps']]))
    for step in range(config['steps']):
        profile, stage = train_profile(step, config['steps'], variant)
        seeds = range(400000 + step*config['batch'], 400000 + (step+1)*config['batch'])
        data = batch(seeds, profile)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective(model, data)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss {variant} seed={seed} step={step}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        key = f'stage{stage}-{profile.family}'
        budget = stage_budgets.setdefault(key, {'updates': 0, 'worlds': 0, 'input_tokens': 0, 'supervised_answers': 0})
        budget['updates'] += 1
        budget['worlds'] += len(seeds)
        budget['input_tokens'] += sum(tokens.numel()+len(y) for tokens,y in data)
        budget['supervised_answers'] += sum(len(y) for _,y in data)
        if (step+1)%50 == 0 or step == 0:
            trace.append({'step': step+1, 'profile': asdict(profile), 'curriculum_stage': stage,
                          'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                          'log_a': model.log_a.detach().tolist()})
        if (step+1)%100 == 0:
            print(f'{variant} seed={seed} step={step+1}/{config["steps"]} loss={float(loss.detach()):.4f}', flush=True)
        if step+1 in milestone_steps:
            monitors[str(step+1)] = monitor(model, step+1, directory)
    after = evaluate_all(model, config['worlds'])
    ablations = {name: state_ablation(model, batch(PROBE_SEEDS, profiles()[name]))
                 for name in ['copy24', 'copy64_interference']}
    result = {'variant': variant, 'seed': seed, 'init': init, 'config': config,
              'initial_nondecay_fingerprint': nondecay_fingerprint,
              'before': before, 'after': after, 'dynamics': monitors,
              'state_ablation': ablations, 'trace': trace, 'budgets': stage_budgets,
              'seconds': time.monotonic()-start}
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': config, 'variant': variant, 'seed': seed, 'initialization': init}, directory/'base.pt')
    (directory/'result.json').write_text(json.dumps(result, indent=2))
    print('completed', variant, seed, {k: round(v['accuracy'],3) for k,v in after.items()}, flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, default=Path('artifacts/timescale-v1'))
    p.add_argument('--steps', type=int, default=900)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--worlds', type=int, default=128)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    p.add_argument('--targets', type=float, nargs=4, default=TARGETS)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.steps < 12 or args.steps % 12 or min(args.batch,args.worlds,min(args.targets)) <= 0:
        p.error('positive sizes and targets required; --steps must be a multiple of 12')
    torch.set_num_threads(1)
    args.directory.mkdir(parents=True, exist_ok=True)
    config = {'steps': args.steps, 'batch': args.batch, 'worlds': args.worlds, 'lr': .002,
              'targets': args.targets, 'seeds': args.seeds, 'variants': args.variants,
              'torch_version': str(torch.__version__), 'device': 'cpu', 'dtype': 'float32',
              'train_seed_start': 400000, 'probe_seed_start': PROBE_SEEDS.start,
              'eval_seed_start': EVAL_SEED_START,
              'note': 'Matched optimizer steps; curriculum does not match tokens or answer counts.'}
    path = args.directory/'config.json'
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != config:
            raise FileExistsError('Existing experiment: use matching --resume or a new directory')
    else:
        path.write_text(json.dumps(config, indent=2))
    for seed in args.seeds:
        for variant in args.variants:
            directory = args.directory/f'{variant}-seed-{seed}'
            if args.resume and (directory/'result.json').exists():
                continue
            if directory.exists():
                raise FileExistsError(f'Incomplete run preserved at {directory}; choose a new directory')
            fit(config, variant, seed, directory)

if __name__ == '__main__':
    main()

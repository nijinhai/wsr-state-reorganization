"""Portable orchestration of the unchanged v0.3 sandbox protocol."""
import argparse
import json

from .precompute_experiment import ROOT, CONFIGS, prepare, train, analyze, preservation, write


def smoke(data):
    scores = {}
    for name in ('B', 'C8', 'D8'):
        _, final = train(name, 0, data, until=1024, small=True)
        scores[name] = final['train']['seen_query']['all']['accuracy']
    if min(scores.values()) < .99:
        raise RuntimeError(f'Small-sample gate failed: {scores}. Diagnose before expanding.')
    return scores


def pilot(data, scores):
    for name in CONFIGS:
        train(name, 0, data, until=512)
    decision = ROOT / 'pilot-decision.json'
    if not decision.exists():
        write(decision, {'decision': 'continue fixed task and all configurations to declared budget',
                         'small32_training_queries': scores,
                         'criterion': 'interface/finite gradient and small-sample gate, never C superiority',
                         'task_adjustments': [], 'test_model_evaluations_so_far': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'pilot', 'full'], default='smoke')
    args = parser.parse_args()
    data = prepare()
    if args.stage == 'prepare':
        return
    scores = smoke(data)
    if args.stage == 'smoke':
        return
    pilot(data, scores)
    if args.stage == 'pilot':
        return
    for seed in range(3):
        for name in CONFIGS:
            train(name, seed, data)
    result = analyze(data)
    # New runs get their own evidence summary; committed results/ is never modified.
    lines = ['# Reproduced run summary', '',
             'Final 2048-step checkpoints, three paired seeds. No test-based model selection.', '',
             '| Model | Seen query accuracy | Held query accuracy |', '|---|---:|---:|']
    for name, item in result['aggregate'].items():
        seen, held = item['seen_query'], item['held_query']
        lines.append(f'| {name} | {100*seen["mean"]:.2f}% | {100*held["mean"]:.2f}% |')
    lines += ['', 'Numerical agreement may vary with hardware and PyTorch version. '
              'These are fresh results; the historical interpretation is in the repository results/ directory.']
    (ROOT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    from .precompute_report import main as report
    report()
    preservation()
    print('Fresh comparison complete:', ROOT)


if __name__ == '__main__':
    main()

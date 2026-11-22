"""Static scientific figures; run with the local analysis environment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('artifacts/wsr-v0.3-pre')
    results = json.loads((root / 'results.json').read_text())
    colors = {'A':'#377eb8', 'B':'#984ea3', 'C':'#4daf4a', 'D':'#e41a1c'}
    labels = {'A':'A: ordinary', 'B':'B: round encoding', 'C':'C: soft coverage', 'D':'D: hard mask'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for mode in 'ABCD':
        curves = [json.loads((root / f'runs/{mode}-seed{s}/learning-curves.json').read_text()) for s in range(3)]
        steps = [r['step'] for r in curves[0] if r['step'] <= 512]
        values = np.array([[next(r for r in curve if r['step']==n)['evaluation']['validation_new']['all']['accuracy'] for n in steps] for curve in curves])*100
        mean, sd = values.mean(0), values.std(0,ddof=1)
        axes[0,0].plot(steps, mean, color=colors[mode], label=labels[mode])
        axes[0,0].fill_between(steps, mean-sd, mean+sd, color=colors[mode], alpha=.12)
        group = [r for r in results if r['mode']==mode]
        rounds = [1,2,4,6,8,10,12]
        accuracy = np.array([[r['rounds'][str(n)]['all']['accuracy'] for n in rounds] for r in group])*100
        axes[1,0].plot(rounds, accuracy.mean(0), marker='o', color=colors[mode], label=labels[mode])
        probes = np.array([[p['heldout_rounded_accuracy'] for p in r['intermediate']['posthoc_count_probes']] for r in group])*100
        axes[1,1].plot(range(1,9), probes.mean(0), marker='o', color=colors[mode], label=labels[mode])
    x = np.arange(4)
    for j, (family, color) in enumerate([('distinct','#6baed6'),('reread','#fdae6b')]):
        scores = np.array([[r['metrics']['test_new'][family]['accuracy'] for r in results if r['mode']==mode] for mode in 'ABCD'])*100
        axes[0,1].bar(x+(j-.5)*.35, scores.mean(1), .35, yerr=scores.std(1,ddof=1), color=color, capsize=3, label=family)
    axes[0,0].set(title='Paired-seed validation curves', xlabel='Optimizer updates', ylabel='Accuracy (%)', ylim=(0,100))
    axes[0,1].set(title='Held-out query combinations', xticks=x, xticklabels=list('ABCD'), ylabel='Accuracy (%)', ylim=(0,100))
    axes[1,0].set(title='Inference depth: fixed original target', xlabel='Read/update rounds', ylabel='Accuracy (%)', ylim=(0,100))
    axes[1,0].axvline(8,color='gray',linestyle=':',alpha=.6)
    axes[1,1].set(title='Posthoc partial-count linear readout', xlabel='Workspace round', ylabel='Rounded probe accuracy (%)', ylim=(0,105))
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=9)
    fig.suptitle('WSR-v0.3 precursor: read-only bank, 16-D workspace, final-answer supervision',fontsize=13)
    fig.savefig(root/'overview.png',dpi=180)
    fig.savefig(root/'overview.pdf')
    plt.close(fig)


if __name__=='__main__':main()

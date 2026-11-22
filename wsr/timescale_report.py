"""Summarize completed timescale runs and render standalone scientific figures."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import torch

from .timescales import VARIANTS, TARGETS

NAMES = {'original': 'Original', 'slow': 'Slow', 'multiscale': 'Multiscale', 'slow_curriculum': 'Slow + curriculum'}
CN_NAMES = {'original': '原始衰减', 'slow': '慢衰减', 'multiscale': '多时间尺度', 'slow_curriculum': '慢衰减＋课程'}


def mean_sd(values, percent=False):
    factor = 100 if percent else 1
    return f'{factor*statistics.mean(values):.1f} ± {factor*statistics.stdev(values):.1f}' if len(values)>1 else f'{factor*values[0]:.1f}'


def collect(directory):
    return [json.loads(path.read_text()) for path in sorted(directory.glob('*-seed-*/result.json'))]


def write_csv(path, rows):
    with path.open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(directory,runs,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    groups={v:[r for r in runs if r['variant']==v] for v in config['variants']}
    tasks=['copy24','copy64','copy64_interference','copy128_ood','count_original','redundancy_original','rules_heldout']
    labels=['Copy 24','Copy 64','Copy 64\ninterference','Copy 128\nOOD','Count 12\n3 episodes','Redundancy\n3 episodes','Rules\nuntrained']
    fig,ax=plt.subplots(figsize=(13,5.2),layout='constrained')
    colors=['#999999','#286fbb','#d26a20','#479b67']
    x=np.arange(len(tasks))
    width=.19
    for index,(variant,color) in enumerate(zip(config['variants'],colors)):
        values=[[r['after'][task]['accuracy']*100 for r in groups[variant]] for task in tasks]
        means=[statistics.mean(v) for v in values]
        std=[statistics.stdev(v) if len(v)>1 else 0 for v in values]
        offset=(index-(len(groups)-1)/2)*width
        ax.bar(x+offset,means,width,color=color,label=NAMES[variant],yerr=std,capsize=2,alpha=.85)
        for center,observations in zip(x+offset,values):
            ax.scatter([center]*len(observations),observations,color='#222222',s=10,zorder=3)
    ax.axhline(12.5,color='#555555',ls='--',lw=1,label='Uniform 8-class chance')
    ax.set(xticks=x,xticklabels=labels,ylabel='Accuracy (%)',ylim=(0,120),
           title=f'Joint base training: {config["steps"]} updates, {len(config["seeds"])} paired seeds')
    ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.24))
    ax.grid(axis='y',alpha=.15)
    for suffix in ['png','svg','pdf']:
        fig.savefig(directory/f'task-performance.{suffix}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,len(groups),figsize=(14,4.8),sharey=True)
    fig.subplots_adjust(left=.065,right=.995,bottom=.17,top=.79,wspace=.06)
    axes=np.atleast_1d(axes)
    for ax,variant in zip(axes,config['variants']):
        for phase,shift,color in [('0',-.16,'#286fbb'),(str(config['steps']),.16,'#d26a20')]:
            arrays=[]
            for head in range(4):
                pieces=[]
                for r in groups[variant]:
                    data=torch.load(directory/f'{variant}-seed-{r["seed"]}'/f'dynamics-step-{int(phase):04d}.pt',weights_only=True)
                    pieces.append((math.log(2)/data['count_original']['decay_rate'][:,:,head]).flatten().numpy())
                arrays.append(np.concatenate(pieces))
            bp=ax.boxplot(arrays,positions=np.arange(1,5)+shift,widths=.27,whis=(10,90),showfliers=False,patch_artist=True,
                          manage_ticks=False,medianprops={'color':'#222222'})
            for patch in bp['boxes']:
                patch.set(facecolor=color,alpha=.7)
            for part in ['whiskers','caps']:
                for line in bp[part]: line.set_color(color)
        for target in TARGETS:
            ax.axhline(target,lw=.7,color='#aaaaaa',ls=':')
        ax.set(yscale='log',xticks=[1,2,3,4],xticklabels=['H0','H1','H2','H3'],title=NAMES[variant],xlabel='Shared-decay head')
        ax.grid(axis='y',alpha=.12)
    axes[0].set_ylabel('Instantaneous half-life (tokens, log scale)')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#286fbb',label='Before training'),Patch(color='#d26a20',label='After training')],
               loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.suptitle('Count-12 fixed probe: boxes = quartiles, whiskers = p10/p90; pooled tokens/worlds/seeds',fontsize=11,y=.995)
    for suffix in ['png','svg','pdf']:
        fig.savefig(directory/f'half-life-distributions.{suffix}',dpi=180)
    plt.close(fig)


def report(directory,config,runs):
    by_variant={v:[r for r in runs if r['variant']==v] for v in config['variants']}
    metric_rows=[]
    half_rows=[]
    for r in runs:
        for phase in ['before','after']:
            for task,values in r[phase].items():
                metric_rows.append({'variant':r['variant'],'seed':r['seed'],'phase':phase,'task':task,
                                    'accuracy':values['accuracy'],'loss':values['loss'],'answers':values['n']})
        for step,measurements in r['dynamics'].items():
            for task,values in measurements.items():
                for h in values['heads']:
                    q=h['instant_half_life_quantiles']
                    observed=[t for t in h['first_state_half_survival_tokens'] if t is not None]
                    half_rows.append({'variant':r['variant'],'seed':r['seed'],'step':step,'task':task,'head':h['head'],
                        'log_a':values['log_a'][h['head']], 'effective_half_life':h['effective_half_life'],
                        **{f'instant_{k}':v for k,v in q.items()},
                        'first_state_hitting_median_observed':statistics.median(observed) if observed else '',
                        'first_state_hitting_right_censored_fraction':h['survival_right_censored_fraction'],
                        'state_rms_median':h['state_rms_quantiles']['median'],'state_rms_p90':h['state_rms_quantiles']['p90'],
                        'state_abs_max':h['state_abs_max'],'state_growth':h['last_over_first_state_rms']})
    write_csv(directory/'metrics.csv',metric_rows)
    write_csv(directory/'half-lives.csv',half_rows)
    labels={'copy8':'复制 8','copy24':'复制 24','copy64':'复制 64','copy64_interference':'复制 64＋干扰',
            'copy128_ood':'复制 128＋干扰（未训练长度）','count2':'两步计数','count4':'四步计数',
            'count_original':'原十二步计数，连续三轮','lookup2':'两对映射检索',
            'redundancy_original':'原冗余表示，连续三轮','rules_heldout':'未训练规则链，连续三轮'}
    lines=['时间尺度与基础能力：受控实验','',
           '本轮只训练基础 SSM；没有训练 Dream。递推方程、参数数量和任务 token 接口保持一致，原实验结果已保留。', '',
           f'四组、{len(config["seeds"])} 个配对种子，共 {len(runs)} 次运行；每次 {config["steps"]} 个 Adam 更新，lr={config["lr"]}，每批 {config["batch"]} 个独立世界，梯度裁剪 1。复制、计数、冗余表示按固定顺序轮换，使用相同训练世界 seed。测试每项 {config["worlds"]} 个世界，原任务每世界三轮连续状态；不同世界重置。训练和测试 seed 不重叠。测试不参与初始化校准。', '',
           '原始组 log_a=0；慢衰减组及课程组 log_a=-3；多尺度组以训练侧固定输入的 mean(dt) 校准四个 head，目标有效半衰期约 2、8、32、128 token。参数 log_a 仍可训练。每个 head 内的 8 个通道 × 8 个状态坐标共享衰减，因此这是四组时间尺度，并非 256 个独立时间常数。其他初始参数的哈希在同一 seed 下完全一致。', '',
           '课程组前 1/4 更新使用短复制、两步计数、单对映射；第二个 1/4 使用中等复制、四步三轮计数、两对映射；后 1/2 使用与其他组相同的挑战任务。每阶段仍轮换三个任务，减轻整段切换任务造成的遗忘。非课程组始终使用挑战训练，复制长度为 2、8、24、64，均包含中性填充和随机别名干扰。课程只匹配总更新数，没有匹配 token 数、答案数和困难任务曝光次数，实际预算另存 JSON。', '',
           '以下为三个种子的测试正确率均值 ± 样本标准差（百分点）；同一行来自同一批测试世界，表内所有任务由同一个共享模型完成。多数类基线、逐世界预测和分轮正确率均保存在每次运行的 result.json。复制 128 是长度外推，规则链未训练，二者作为挑战测试。', '',
           '| 测试 | '+' | '.join(CN_NAMES[v] for v in config['variants'])+' |',
           '| --- | '+' | '.join('---' for _ in config['variants'])+' |']
    for task,label in labels.items():
        lines.append('| '+label+' | '+' | '.join(mean_sd([r['after'][task]['accuracy'] for r in by_variant[v]],percent=True) for v in config['variants'])+' |')
    reference=runs[0]['after']
    lines+=['',f'同一测试集上，原计数的经验多数类比例为 {reference["count_original"]["majority_baseline"]*100:.1f}%，原冗余表示为 {reference["redundancy_original"]["majority_baseline"]*100:.1f}%；它们高于八类均匀猜测的 12.5%。两对映射检索若只从当前可见的两个标签随机选择，也可达 50%，本轮结果没有达到这一捷径基线。']
    lines+=['','图中黑点为各个种子，误差线为种子间样本标准差，不是置信区间。','','![任务表现](task-performance.png)','','通道时间尺度的测量使用固定 16 个独立探针世界，训练前以及 1/3、2/3、结束时重复测量；逐项记录四个 head，不只汇总均值。瞬时半衰期为 ln(2)/(dt_t × exp(log_a_h))，有效半衰期为 ln(2)/mean(dt_t × exp(log_a_h))。还记录首个状态贡献实际衰减到 1/2 所需 token 数；轨迹结束前没有达到者标为右删失，不冒充已测得的长半衰期。轨迹只包含实际传递的历史状态，不包含丢弃的虚拟 READ 状态。', '',
            '下表展示 seed=0 原计数探针的每个 head，格式为有效半衰期，以及逐 token 的 p10–p90。所有种子、任务、检查点、完整分位数及直方图见 half-lives.csv 与 dynamics-step-*.json；原始逐 token 衰减率和状态幅度见对应 .pt。', '',
            '| 配置 | Head | 训练前：有效值 [p10–p90] | 训练后：有效值 [p10–p90] |',
            '| --- | --- | --- | --- |']
    for variant in config['variants']:
        r=next(r for r in by_variant[variant] if r['seed']==config['seeds'][0])
        for h in range(4):
            values=[]
            for step in ['0',str(config['steps'])]:
                head=r['dynamics'][step]['count_original']['heads'][h]
                q=head['instant_half_life_quantiles']
                values.append(f'{head["effective_half_life"]:.2f} [{q["p10"]:.2f}–{q["p90"]:.2f}]')
            lines.append(f'| {CN_NAMES[variant]} | {h} | '+ ' | '.join(values)+' |')
    lines+=['','![逐 head 半衰期分布](half-life-distributions.png)','','干扰与状态幅度同样保留为观察对象。下表的干扰差值是同长度 64 的中性复制正确率减去随机别名干扰复制正确率，避免把长度差异混进干扰效应。状态 RMS 和最大绝对值是动态范围指标；此 SSM 状态没有硬饱和边界，仅凭数值变大不能断言发生饱和或信息混淆。', '',
            '| 配置 | 64-token 复制干扰损失（百分点） | 原冗余任务训练后各 head 的最大绝对状态值范围（跨种子） |',
            '| --- | --- | --- |']
    for variant in config['variants']:
        rows=by_variant[variant]
        delta=[r['after']['copy64']['accuracy']-r['after']['copy64_interference']['accuracy'] for r in rows]
        maxima=[h['state_abs_max'] for r in rows for h in r['dynamics'][str(config['steps'])]['redundancy_original']['heads']]
        lines.append(f'| {CN_NAMES[variant]} | {mean_sd(delta,percent=True)} | {min(maxima):.2f}–{max(maxima):.2f} |')
    lines+=['','实际训练预算（每个种子相同，输入 token 数包含虚拟 READ 评分计算）：','',
            '| 配置 | 更新步 | 世界数 | 输入 token 数 | 监督答案数 |',
            '| --- | --- | --- | --- | --- |']
    for variant in config['variants']:
        budgets=by_variant[variant][0]['budgets'].values()
        totals={key:sum(b[key] for b in budgets) for key in ['updates','worlds','input_tokens','supervised_answers']}
        lines.append(f'| {CN_NAMES[variant]} | {totals["updates"]} | {totals["worlds"]} | {totals["input_tokens"]:,} | {totals["supervised_answers"]:,} |')
    means={variant:{task:statistics.mean(r['after'][task]['accuracy'] for r in by_variant[variant])
                    for task in labels} for variant in config['variants']}
    lines+=['','对本轮候选的解读：','',
            '多尺度的四组长短差异在训练后仍然存在；具体变化不能只看参数 log_a，因为输入选择性 dt 同时会学习。下表保留原计数探针训练后有效半衰期在各 seed 间的范围：','',
            '| 配置 | H0 | H1 | H2 | H3 |', '| --- | --- | --- | --- | --- |']
    for variant in config['variants']:
        columns=[]
        for head in range(4):
            values=[r['dynamics'][str(config['steps'])]['count_original']['heads'][head]['effective_half_life'] for r in by_variant[variant]]
            columns.append(f'{min(values):.1f}–{max(values):.1f}')
        lines.append('| '+CN_NAMES[variant]+' | '+' | '.join(columns)+' |')
    lines+=['',
            f'24-token 复制均值从原始组 {means["original"]["copy24"]*100:.1f}% 变为慢衰减 {means["slow"]["copy24"]*100:.1f}%、多尺度 {means["multiscale"]["copy24"]*100:.1f}%、慢衰减加课程 {means["slow_curriculum"]["copy24"]*100:.1f}%。原始计数和原冗余表示没有呈现相应幅度的提升；因此，在这个共享训练协议下，较长状态保留和有效任务运算可以分离。多尺度候选并未展现稳定超越统一慢衰减的证据，不代表其他尺度分配或预算也无效。', '',
            '课程组的复制改善不能直接归因于课程本身的某一阶段：任务难度、监督答案数和 token 数都随课程变化。规则链未直接训练，接近随机的结果也不能用来证明模型完全无法处理规则。', '',
            '状态幅度与干扰指标提示仍需区分“历史存在”与“能按需读取和组合”。本轮没有发现非有限数值；没有独立证据可将性能下降具体归因为状态饱和。当前主要任务尚未通过基础能力验收，不据此启动 Dream。']
    lines+=['','这里只对预先列出的候选做探索性比较，没有将最长半衰期视作最优，也没有据此调整测试集。三个种子不足以支持普遍性结论。规则链缺少直接训练，结果应解释为当前协议下的迁移表现，而非模型能否学习规则处理的绝对上限。', '',
            '所有检查点保存在 `<配置>-seed-<种子>/base.pt`，包含模型、优化器、初始化与完整运行配置；观察数据与性能逐世界保留。原基础权重及顶层评估/冒烟文件的哈希保存在 prior-artifacts.sha256。本轮结果不会自动替换 WSR-v0.1 的共享基底，也不会触发 Dream 训练。']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,default=Path('artifacts/timescale-v1'))
    args=p.parse_args()
    config=json.loads((args.directory/'config.json').read_text())
    runs=collect(args.directory)
    expected={(v,s) for v in config['variants'] for s in config['seeds']}
    if {(r['variant'],r['seed']) for r in runs}!=expected:
        raise RuntimeError(f'Incomplete run set: {len(runs)}/{len(expected)}')
    for seed in config['seeds']:
        assert len({r['initial_nondecay_fingerprint'] for r in runs if r['seed']==seed})==1
    render(args.directory,runs,config)
    report(args.directory,config,runs)
    print(args.directory/'REPORT.md')

if __name__=='__main__': main()

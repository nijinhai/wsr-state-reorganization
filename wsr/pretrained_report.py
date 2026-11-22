"""Paired NLL/trajectory analysis for the frozen-pretrained WSR matrix."""
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path('artifacts/pretrained-v1')
LABELS={'baseline':'无 Dream / 无破坏','corrupt':'无 Dream / 10% 遮蔽','05-state-only':'D / T1 / 无破坏 / 仅状态'}
LABELS.update({f'{i+1:02d}':f'{m} / {t} / '+('10% 遮蔽' if c else '无破坏') for i,(m,t,c) in enumerate((m,t,c) for m in ['A','D'] for t in ['T1','T2'] for c in [False,True])})
NAMES=['baseline','corrupt']+[f'{i:02d}' for i in range(1,9)]+['05-state-only']


def load(name,seed,kind):
    path=ROOT/'evaluation'/f'{name}-seed-{seed}'/(kind+'.json')
    return json.loads(path.read_text()) if path.exists() else None


def weighted(rows):
    values=[x for row in rows for x in row['losses']]
    return statistics.mean(values)


def pair_delta(rows,reference):
    lookup={(r['document']['id'],r['history_visible']):r for r in reference}
    total=0.;count=0
    for row in rows:
        other=lookup[(row['document']['id'],row['history_visible'])]
        assert len(row['losses'])==len(other['losses'])
        total+=sum(a-b for a,b in zip(row['losses'],other['losses']))
        count+=len(row['losses'])
    return total/count


def csv_file(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def mean_sd(values,digits=6):
    mean=statistics.mean(values)
    return f'{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}' if len(values)>1 else f'{mean:.{digits}f}'


def collect():
    summary=[];distances=[];events=[];rounds=[]
    baseline=load('baseline',0,'short')['result']
    corrupted=load('corrupt',0,'short')['result']
    for name in NAMES:
        seed_values=[];delta_values=[];condition_values=[]
        seeds=[0,1,2] if name.isdigit() else [0]
        for seed in seeds:
            data=load(name,seed,'short')
            if data is None:raise RuntimeError(f'Missing short {name}/{seed}')
            rows=data['result']
            seed_values.append(weighted(rows));delta_values.append(pair_delta(rows,baseline))
            ref=corrupted if name in ['corrupt','02','04','06','08'] else baseline
            condition_values.append(pair_delta(rows,ref))
            for row in rows:
                for distance,nll in row['distance_bins'].items():
                    distances.append({'name':name,'seed':seed,'kind':'short','document':row['document']['id'],
                                      'history_visible':row['history_visible'],'distance':distance,'nll':nll})
        summary.append({'name':name,'seeds':len(seed_values),'mean_nll':statistics.mean(seed_values),
                        'nll_sd':statistics.stdev(seed_values) if len(seed_values)>1 else 0,
                        'delta_to_clean':statistics.mean(delta_values),'delta_clean_sd':statistics.stdev(delta_values) if len(delta_values)>1 else 0,
                        'delta_to_same_state_condition':statistics.mean(condition_values),
                        'nll_values':seed_values,'delta_values':delta_values})
        data=load(name,0,'long-single')
        if data is None:raise RuntimeError(f'Missing long single {name}')
        for row in data['result']:
            for distance,nll in row['distance_bins'].items():
                distances.append({'name':name,'seed':0,'kind':'long-single','document':row['document']['id'],
                                  'history_visible':row['history_visible'],'distance':distance,'nll':nll})
        periodic=load(name,0,'periodic')
        if periodic is None:raise RuntimeError(f'Missing periodic {name}')
        reference=load('baseline',0,'periodic')['result']['events']
        for row,ref in zip(periodic['result']['events'],reference):
            assert row['history_visible']==ref['history_visible']
            events.append({'name':name,'event':row['event'],'history_visible':row['history_visible'],
                           'nll':row['nll'],'delta_to_clean':row['nll']-ref['nll'],
                           'corruption_relative_norm':row['corruption_relative_norm'],
                           'total_relative_norm':row['total_relative_norm'],
                           'dream_relative_norm':row['edit_trace'][-1]['relative_total_edit'] if row['edit_trace'] else 0})
        if name not in ['baseline','corrupt']:
            data=load(name,0,'rounds')
            if data is None:raise RuntimeError(f'Missing round sweep {name}')
            for row in data['result']['rows']:
                rounds.append({'name':name,'round':row['round'],'nll':row['nll'],
                               'relative_edit':0 if row['round']==0 else data['result']['trace'][row['round']-1]['relative_total_edit']})
    csv_file(ROOT/'short-summary.csv',[{k:v for k,v in r.items() if k not in ['nll_values','delta_values']} for r in summary])
    csv_file(ROOT/'future-distance.csv',distances);csv_file(ROOT/'periodic-events.csv',events);csv_file(ROOT/'rounds.csv',rounds)
    return summary,distances,events,rounds


def figures(summary,distances,events,rounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['png','svg','pdf']:fig.savefig(ROOT/(name+'.'+ext),dpi=180)
        plt.close(fig)
    lookup={r['name']:r for r in summary}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,names,reference,title in [(axes[0],['01','03','05','07','05-state-only'],'baseline','Clean state'),
                                     (axes[1],['02','04','06','08'],'corrupt','Masked state')]:
        ref=lookup[reference]['mean_nll']
        y=[(lookup[n]['mean_nll']-ref)*1000 for n in names]
        err=[lookup[n]['nll_sd']*1000 for n in names]
        ax.bar(range(len(names)),y,yerr=err,capsize=3,color=['#286fbb' if n.startswith('0') and int(n[:2])<=4 else '#d26a20' for n in names])
        ax.axhline(0,color='black',lw=1)
        ax.set(xticks=range(len(names)),xticklabels=names,ylabel=f'Delta NLL vs {reference} (milli-nats/token)',title=title)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('Short validation: paired loss differences, mean +/- SD across 3 training seeds\nState-only control uses seed 0; lower is better',fontsize=12)
    save(fig,'short-effects')

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors={'05':'#286fbb','07':'#d26a20','06':'#479b67','08':'#9264b7','corrupt':'#666666'}
    bins=['1-32','33-128','129-512','513-2048']
    base=load('baseline',0,'long-single')['result']
    for ax,history in zip(axes,[8192,32768]):
        reference={(r['document']['id'],r['history_visible']):r for r in base}
        for name in ['05','07','06','08','corrupt']:
            rows=load(name,0,'long-single')['result']
            delta=[]
            for bin_name in bins:
                values=[r['distance_bins'][bin_name]-reference[(r['document']['id'],history)]['distance_bins'][bin_name]
                        for r in rows if r['history_visible']==history]
                delta.append(statistics.mean(values)*1000)
            ax.plot(range(4),delta,marker='o',label=name,color=colors[name])
        ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.15)
        ax.set(xticks=range(4),xticklabels=bins,xlabel='Future token distance',ylabel='Delta NLL vs clean baseline (milli-nats)',title=f'Visible history: {history} tokens')
        ax.legend(frameon=False,ncol=3)
    fig.suptitle('Single Dream on PG-19: seed 0, two validation books',fontsize=12)
    save(fig,'distance-effects')

    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for column,names,ref_name in [(0,['01','03','05','07'],'baseline'),(1,['02','04','06','08'],'corrupt')]:
        reference=load(ref_name,0,'periodic')['result']['events']
        for name in names:
            rows=load(name,0,'periodic')['result']['events']
            cumulative=[];s=0.;n=0
            per_event=[];x=[]
            for row,ref in zip(rows,reference):
                ds=sum(a-b for a,b in zip(row['losses'],ref['losses']))
                s+=ds;n+=len(row['losses']);cumulative.append(s/n*1000)
                per_event.append(ds/len(row['losses'])*1000);x.append(row['history_visible']/1024)
            axes[0,column].plot(x,cumulative,label=name,marker='.',lw=1.4)
            axes[1,column].plot(x,per_event,label=name,marker='.',lw=1.1)
        for ax in axes[:,column]:
            ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.15);ax.legend(frameon=False,ncol=4)
            ax.set_xlabel('Visible history before intervention (Ki tokens)')
        axes[0,column].set(title=f'Cumulative delta vs {ref_name}',ylabel='milli-nats/token')
        axes[1,column].set(title=f'Per-event delta vs {ref_name}',ylabel='milli-nats/token')
    fig.suptitle('Periodic Wake-Dream on one PG-19 book: seed 0, 2048-token interval, no resets',fontsize=12)
    save(fig,'periodic-effects')

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in ['01','03','05','07','05-state-only']:
        rows=[r for r in rounds if r['name']==name]
        zero=rows[0]['nll']
        axes[0].plot([r['round'] for r in rows],[r['relative_edit']*100 for r in rows],label=name,marker='.')
        axes[1].plot([r['round'] for r in rows],[(r['nll']-zero)*1000 for r in rows],label=name,marker='.')
    axes[0].set(xlabel='Dream round',ylabel='Total edit / initial state norm (%)')
    axes[1].set(xlabel='Dream round',ylabel='Delta future NLL vs round 0 (milli-nats)')
    for ax in axes:ax.grid(alpha=.15);ax.legend(frameon=False,ncol=2)
    fig.suptitle('Iteration ablation: fixed validation prefix and 512 future tokens; seed 0',fontsize=12)
    save(fig,'round-effects')


def write_report(summary):
    lookup={r['name']:r for r in summary}
    interface=json.loads((ROOT/'interface-verification.json').read_text())
    plan=json.loads((ROOT/'protocol-plan.json').read_text())
    data=json.loads((ROOT/'data/manifest.json').read_text())
    lines=['WSR-v0.1：官方预训练 Mamba2-130M 状态重组实验','',
           '本轮已改为冻结官方预训练权重的真实文本实验。结果为低预算探索性验证，不预设改写必须提升计数、检索或规则能力。旧合成与时间尺度实验及其归档没有被覆盖。', '',
           '基底：state-spaces/mamba2-130m，revision `3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49`，严格匹配全部权重键与维度，128,989,632 个参数冻结。FP32 可微 PyTorch 后端保留完整预训练架构，原生 Mamba CUDA kernels 未参与实验；与固定官方 v2.2.5 Python step/SSD 定义独立比较。', '',
           f'接口验收：连续/分段 logit 最大绝对差 {interface["continuous_vs_split_logits"]["max_absolute"]:.6g}；与官方完整逐 token 路径最大差 {interface["official_step_logits"]["max_absolute"]:.6g}；注入状态的真实语言预测梯度范数 {interface["injected_state_gradient_norm"]:.6g}。清零或遮蔽状态会改变后续预测，所有原参数无梯度。另有 14 项单元测试及独立双精度 SSD gradcheck。8K 前缀用 128/256-token 两种 wake 分段时，后续 512-token 平均绝对 NLL 差约 7.77e-6；同一缓存的零改写逐 token NLL 差为零，配对评价各组使用相同分段与干净历史缓存。', '',
           '改写层为索引 12（第 13 层）的完整 SSM `[1,24,64,128]`。所有卷积缓存与其他层状态在干预瞬间保留。边界 b 的历史可见长度为 b，缓存累计处理 b−1；最后一个可见 token 留作 anchor，在编辑后处理，重新产生第一个未来 token 的 logit。所有比较使用同一约定，后续连续运行不重复 anchor，也不使用旧 logits。', '',
           '8 组各三个种子，共 24 个主训练；另有 seed=0 的仅输入状态 D/T1/无破坏对照。共享编辑器沿状态行工作，Dream 共 137,664 个参数，hidden=64，八轮；D 的 GRU 记忆在一次 Dream 的八轮间传递、下次 Dream 重新初始化。A 不传递独立循环隐状态。比较的是这两个实现，未声称严格匹配有效参数/计算量或完全隔离所有额外状态聚合操作。', '',
           '每组 32 次未来预测更新，T2 额外八次去噪并沿用同一 Adam 状态，不属于等计算预算对照；未来预测阶段的文档、边界与随机破坏种子与 T1 对齐。batch=1，每更新两个连续周期，干预间隔 256，lr=3e-4、梯度裁剪 1。只对未来 64 token 反传，周期间保留缓存值并截断梯度。长历史 8K/32K 是评价，不是长距离梯度优化。没有按短程收益筛掉任何一组。', '',
           '训练文档池为原划分的 12 个 WikiText-103 文档和 4 本 PG-19 书，实际采样文档与片段逐项写入 training.json。验证文章按长度筛选两个完整且不少于 4096 token 的文档，未按损失选取；PG-19 长程单次评价用两本 40K 自然前缀，周期性评价用其中一本至 32K。文章/书之间重置，同一书内部周期性传递缓存。原测试集保持封存，以下均是验证结果；不能称作最终测试成绩。', '',
           '短中程结果使用两个 WikiText 验证文档，在 512、2048 个可见历史 token 后预测 2048 个 token。损失为 tokenizer 级自然对数 NLL，未换算成 PG-19 官方 word-level PPL。均值 ± 标准差只描述三个训练种子，不是置信区间；无 Dream 与仅状态对照只使用 seed=0。评价破坏掩码按文档和边界固定，各训练种子共享；尚未进行多个评价掩码的重复检验。Δ 为配对损失，负数更好；“同状态对照”对无破坏组用原基线、对破坏组用破坏后不修复。', '',
           '| 配置 | 架构 / 训练 / 破坏 | NLL | 相对无 Dream Δ（milli-nats/token） | 相对同状态对照 Δ（milli-nats/token） |',
           '| --- | --- | --- | --- | --- |']
    for r in summary:
        lines.append(f'| {r["name"]} | {LABELS[r["name"]]} | {mean_sd(r["nll_values"])} | {mean_sd([1000*x for x in r["delta_values"]],digits=3)} | {r["delta_to_same_state_condition"]*1000:.3f} |')
    lines+=['','![短程配对结果](short-effects.png)','','单次 Dream 的长历史结果（seed=0，两本 PG-19 书，均预测后续 2048 token）。周期性一列来自其中一本书的整个 32K 流，文本与计分区间不同，不应与单次列直接相减；应在各自列内与同位置基线比较：','',
            '| 配置 | 8K 历史 NLL | 32K 历史 NLL | 周期性至 32K NLL |',
            '| --- | --- | --- | --- |']
    for name in NAMES:
        rows=load(name,0,'long-single')['result']
        short=weighted([r for r in rows if r['history_visible']==8192])
        long=weighted([r for r in rows if r['history_visible']==32768])
        periodic=load(name,0,'periodic')['result']['mean_nll']
        lines.append(f'| {name} | {short:.6f} | {long:.6f} | {periodic:.6f} |')
    lines+=['','![不同未来距离](distance-effects.png)','','周期性结果保持同一书籍状态，首次在 512 可见 token 干预，其后每 2048 token 干预，直到 32K，没有在周期间重置。以下均相对无 Dream 的同位置预测，正数表示退化。本表单位为 milli-nats/token（千分之一 nat）。首/末四个周期采用 token 加权；8K/32K 是从首次干预起至该全局长度的累计配对损失。只测一个验证书、一个训练种子，不能据此概括所有书籍上的积累规律。','',
            '| 配置 | 前四周期 Δ | 后四周期 Δ | 累计至 8K Δ | 累计至 32K Δ |',
            '| --- | --- | --- | --- | --- |']
    ref=load('baseline',0,'periodic')['result']
    def delta_block(a,b):
        return sum(sum(x-y for x,y in zip(r['losses'],s['losses'])) for r,s in zip(a,b))/sum(len(r['losses']) for r in a)*1000
    for name in NAMES:
        row=load(name,0,'periodic')['result'];events=row['events']
        values=[delta_block(events[:4],ref['events'][:4]),delta_block(events[-4:],ref['events'][-4:])]
        values += [(row['stream_prefix_nll'][str(n)]-ref['stream_prefix_nll'][str(n)])*1000 for n in [8192,32768]]
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','![周期性重组](periodic-effects.png)','','固定验证前缀的轮次扫描（同一个训练八轮的模型，seed=0，后续 512 tokens）：','',
            '| 配置 | 八轮后相对状态修改幅度 | Round 8 − Round 0 的 NLL（milli-nats） |',
            '| --- | --- | --- |']
    for name in NAMES[2:]:
        row=load(name,0,'rounds')['result'];rs=row['rows'];amount=row['trace'][-1]['relative_total_edit']
        lines.append(f'| {name} | {amount*100:.4f}% | {(rs[-1]["nll"]-rs[0]["nll"])*1000:.3f} |')
    lines+=['','![编辑轮次轨迹](round-effects.png)','','曲线与逐 token CSV 用于检查短期退化/长期改善，以及多轮修改是否趋稳；不以单个最短 horizon 选择配置。轮次扫描并非独立训练的单次编辑器，不能单凭扫描排除普通状态适配的解释。历史仅状态对照在 Dream 时也不读取 pending anchor，该信息差异在 history_window 与 state_history 中明确记录。', '',
            '所有原始逐 token 损失、训练数据身份、每轮/每 head 修改范数、第一/最后训练样本及代表性评价的完整状态轨迹均保留。轨迹中相对编辑量以 Dream 输入状态为分母；破坏后的改写量与破坏本身的改变量分开记录。长程仅 seed=0，验证文本较少，当前优化预算很小，效果大小与种子差异均需结合原始数据解释。', '',
            '产物：`short-summary.csv`、`future-distance.csv`、`periodic-events.csv`、`rounds.csv`；主模型/分词器校验见 `model/manifest.json`，原文档划分与哈希见 `data/manifest.json`，权重与优化器检查点见 `formal/`，接口检查见 `interface-verification.json`。旧实验归档为 `artifacts/legacy-2026-11-13/`。']
    findings=ROOT/'FINDINGS.md'
    if findings.exists():lines+=['',findings.read_text()]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    summary,distances,events,rounds=collect()
    figures(summary,distances,events,rounds)
    write_report(summary)
    print(ROOT/'REPORT.md')

if __name__=='__main__':main()

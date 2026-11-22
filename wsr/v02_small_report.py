"""Standalone plots and a report for the three bounded diagnostics."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('artifacts/wsr-v0.2a-diagnostics')
SCALES=[-.25,0,.125,.25,.5,1,1.5]


def read(name):return json.loads((ROOT/name).read_text())
def mean(values):return statistics.mean(values)
def csv_write(name,rows):
    with (ROOT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def figure(name):
    plt.tight_layout()
    for ext in ['png','svg','pdf']:plt.savefig(ROOT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close()


def cluster_interval(values,documents):
    groups={d:[] for d in set(documents)}
    for v,d in zip(values,documents):groups[d].append(v)
    arrays=list(groups.values());rng=np.random.default_rng(70)
    boot=[mean([v for i in rng.integers(0,len(arrays),len(arrays)) for v in arrays[i]]) for _ in range(2000)]
    return {'mean':mean(values),'cluster_95_percentile_interval':np.quantile(boot,[.025,.975]).tolist(),'documents':len(groups)}


def main():
    amplitudes=[r for path in sorted(ROOT.glob('amplitude-*.json')) for r in json.loads(path.read_text())]
    assert len(amplitudes)==336
    base={(r['group'],r['seed'],r['id']):r for r in amplitudes if r['scale']==0}
    detail=[];summary=[]
    for row in amplitudes:
        reference=base[(row['group'],row['seed'],row['id'])]
        for horizon in [64,512,2048]:
            n=min(horizon,len(row['losses']))
            delta=mean([a-b for a,b in zip(row['losses'][:n],reference['losses'][:n])])
            detail.append({k:row[k] for k in ['group','seed','id','dataset','document','history','scale','relative_edit']}|
                          {'horizon':horizon,'nll':mean(row['losses'][:n]),'delta_no_edit':delta})
    for group in ['B0','P']:
        for history in [512,2048,8192,32768]:
            for horizon in [64,512,2048]:
                for scale in SCALES:
                    a=[r for r in detail if r['group']==group and r['history']==history and r['horizon']==horizon and r['scale']==scale]
                    points=[mean([r['delta_no_edit'] for r in a if r['seed']==seed]) for seed in range(3)]
                    summary.append({'group':group,'history':history,'horizon':horizon,'scale':scale,
                                    'delta_no_edit':mean(points),'seed_sd':statistics.stdev(points),
                                    'seeds_degraded':sum(v>0 for v in points),'mean_relative_edit':mean([r['relative_edit'] for r in a])})
    csv_write('amplitude-detail.csv',detail);csv_write('amplitude-summary.csv',summary)
    fig,axes=plt.subplots(1,4,figsize=(14,3.8))
    for ax,h in zip(axes,[512,2048,8192,32768]):
        for group,color in [('B0','#4c78a8'),('P','#54a24b')]:
            rows=[r for r in summary if r['group']==group and r['history']==h and r['horizon']==2048]
            y=np.array([r['delta_no_edit'] for r in rows]);sd=np.array([r['seed_sd'] for r in rows])
            ax.plot(SCALES,y,'o-',label=group,color=color);ax.fill_between(SCALES,y-sd,y+sd,color=color,alpha=.15)
        ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='gray',lw=.5);ax.set_title(f'History={h}');ax.set_xlabel('Final edit multiplier');ax.legend()
    axes[0].set_ylabel('Future 2048 NLL change vs zero edit');figure('amplitude-scan')
    s=read('S-results.json');sq=read('S-query-control.json');p=read('P-results.json');pn=read('P-nonlinear.json');pv=read('P-visible-control.json')
    fact_rows=[{'readout':name,**row['validation']} for name,row in s['levels'].items()]
    csv_write('S-facts.csv',fact_rows)
    fig,ax=plt.subplots(figsize=(9,4));names=[r['readout'] for r in fact_rows];x=np.arange(len(names))
    ax.bar(x-.2,[r['link_accuracy'] for r in fact_rows],.4,label='Link facts');ax.bar(x+.2,[r['color_accuracy'] for r in fact_rows],.4,label='Color facts')
    ax.axhline(.25,color='black',ls=':',lw=1);ax.set_xticks(x,names,rotation=20);ax.set_ylim(0,1.05);ax.set_ylabel('Unseen-rule fact accuracy');ax.legend();figure('S-readability')
    p_rows=[]
    for name,values in p['ridge'].items():
        for condition in ['validation','same_document_shuffled_training_pairing','same_document_swapped_validation_history']:
            for d,metric in values[condition]['per_distance'].items():p_rows.append({'head':'ridge','input':name,'condition':condition,'distance':d,**metric})
    for name,values in pn.items():
        for condition in ['validation','validation_swapped_history']:
            for d,metric in values[condition]['per_distance'].items():p_rows.append({'head':'InfoNCE_MLP','input':name,'condition':condition,'distance':d,**metric})
    csv_write('P-by-distance.csv',p_rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,color in [('raw_axes','#4c78a8'),('visible64','#54a24b')]:
        for pairing,style in [('aligned','-'),('same_document_shuffled',':')]:
            row=pn[name+'-'+pairing]['validation']['per_distance']
            axes[0].plot([128,512,2048],[row[str(d)]['ce'] for d in [128,512,2048]],'o'+style,label=name+'/'+pairing,color=color)
            axes[1].plot([128,512,2048],[row[str(d)]['accuracy'] for d in [128,512,2048]],'o'+style,label=name+'/'+pairing,color=color)
    axes[0].axhline(np.log(8),color='black',ls='--',lw=1);axes[1].axhline(.125,color='black',ls='--',lw=1)
    for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks([128,512,2048],[128,512,2048]);ax.set_xlabel('Future distance');ax.legend(fontsize=7)
    axes[0].set_ylabel('Validation InfoNCE (lower is better)');axes[1].set_ylabel('8-way retrieval accuracy');figure('P-predictability')
    docs=[r['document'] for r in p['validation_rows']];intervals={}
    uniform=p['controls']['uniform_logits']['row_ce']
    for name,values in pn.items():
        intervals[name+'_minus_uniform']=cluster_interval([a-b for a,b in zip(values['validation']['row_ce'],uniform)],docs)
    for name in ['raw_axes','visible64']:
        intervals[name+'_aligned_minus_shuffled_train']=cluster_interval([a-b for a,b in zip(pn[name+'-aligned']['validation']['row_ce'],pn[name+'-same_document_shuffled']['validation']['row_ce'])],docs)
    result={'amplitude':summary,'S':s,'S_query_head_control':sq,'P':p,'P_InfoNCE_heads':pn,'P_visible_control':pv,
            'P_document_cluster_intervals':intervals,
            'interpretation':'Validation diagnostics, not a new production setting or a final test. No Mamba/Dream updates. S head controls use privileged fact labels and extra query draws, not a formal S rerun.'}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# WSR-v0.2a 三项小诊断','','使用原检查点与缓存；冻结 Mamba 和 Dream，不启动新一轮正式训练。S 的新计算仅为本次授权的读出诊断，不恢复 S 主训练。S+P、W1/W2 均未启动。所有结论均来自 validation，最终 test 未读取。','','## 1. 状态修改幅度扫描','','固定八轮输出方向，续算状态为 `H(s)=H0+s*(H8-H0)`。卷积缓存和其他层不变；s=0/1 使用原 FP32 状态端点，避免减加舍入。其余点重新运行后续 LM。B0/P 各三个种子，2048 步检查点。两篇 Wiki（一篇原曲线、一篇曲线外）各两个历史边界，以及两本 PG 各 8K/32K 边界，共八个前缀。没有周期性幅度扫描。','','下表为未来 2048 tokens 的 ΔNLL，先对两个文档平均，再对三个种子平均；负值表示改善：','','| 组/历史 | −0.25 | 0.125 | 0.25 | 0.5 | 1.0 | 1.5 |','|---|---:|---:|---:|---:|---:|---:|']
    for group in ['B0','P']:
        for history in [512,2048,8192,32768]:
            vals=[next(r['delta_no_edit'] for r in summary if r['group']==group and r['history']==history and r['horizon']==2048 and r['scale']==scale) for scale in [-.25,.125,.25,.5,1.,1.5]]
            lines.append('| '+f'{group}/{history}'+' | '+' | '.join(f'{v:+.7f}' for v in vals)+' |')
    lines+=['','![修改幅度扫描](amplitude-scan.png)','','阴影是三个种子的标准差，不是文档总体置信区间。CSV 同时提供未来 64/512/2048 和实际相对编辑范数；所有扫描点均展示，不以 validation 的最优点替换正式配置。两个 Wiki 文档的结果不代表全部 Wiki 验证集。','','**判断：**缩小幅度确实减少 B0 的长历史损害，但 8K/32K 上正向 0.125 仍有退化，反向 −0.25 反而改善。过度改写不是充分解释；编辑方向在长历史分布下也不合适。短历史允许小幅正向收益，存在历史/文本依赖。P 缓解损害，但长历史均值也没有稳定的正向有益区间。未据此修改任何训练模型或部署缩放系数。','','## 2. S 的三级信息读取','','一级：直接解析实际可见文本，并核对原规则标签；1280 个训练/验证规则均精确对齐，确定性查询执行 100%。另对按 token 位置保留的冻结 embedding 投影训练线性事实读出，检验输入信息的可学习性。','','二级：原始 SSM；三级：原 S-seed0 的八轮改写状态。两层分别读取既有 QueryDecoder 的冻结 128 维 StateReader 输出，以及保持头/P 通道位置的固定随机 N 维投影（128→8 后展平，含逐头尺度，共 12312 维）。没有改变正式 Dream 架构。另读出冻结历史 GRU 的最终 64 维上下文。','','探针只预测四条链接和四个基础颜色，1024 训练规则、256 个未见规则；正则从训练规则内部 768/256 划分选择，再拟合全训练集。标签只参与探针拟合。查询答案由预测事实执行确定性规则得到，属于信息可读性上界，不是训练出的组合能力。','','| 读出 | 链接事实 | 颜色事实 | 八事实全对 | 未见查询，符号执行 |','|---|---:|---:|---:|---:|']
    for row in fact_rows:lines.append('| '+row['readout']+' | '+' | '.join(f'{100*row[k]:.2f}%' for k in ['link_accuracy','color_accuracy','all8_facts_accuracy','heldout_queries_symbolic_accuracy'])+' |')
    lines+=['','![S 信息可读性](S-readability.png)','','打乱训练规则标签后，验证事实读出回到随机附近；32 样本拟合对大多数读出为 100%，历史 GRU 上未完全拟合。固定文本格式、原规则划分与单一模型种子的限制保留；失败的线性读出不能证明完全没有信息。','','读出头对照：对原始 reader、改写 reader，以及显式基础事实 one-hot（特权正对照）分别重新训练同结构查询 MLP；只训练新头，原 reader 和 Dream 不更新。每头 512 步、batch16、八查询，65536 查询标签 draws，是原 S 单种子查询监督 draws 的四倍；三种输入用同初始化、样本日程与预算。','','| 查询头输入 | 训练规则/已见查询 | 新规则/已见查询 | 新规则/未见查询 |','|---|---:|---:|---:|']
    for name,values in sq.items():lines.append('| '+name+' | '+' | '.join(f"{100*values[k]['accuracy']:.2f}%" for k in ['train_seen_queries','validation_seen_queries','validation_unseen_queries'])+' |')
    lines+=['','新规则/未见查询的可见颜色频率捷径为 53.91%；这些新查询头约 52%，仍未超过该捷径。原正式 S 头同集合为 25.88%。','','**判断：**信息已经存在于原始 SSM，甚至现有 StateReader 的低维输出也足够线性恢复事实。Dream 没有增加事实读出价值。主要瓶颈至少包含下游查询学习/关系组合，而不是单纯缺少事实。显式历史 GRU 较弱仍是局部问题，但不能用它解释 S 的全部失败；也不能据此归因于缺少工作区。特权事实和符号执行的成功不代表 Dream 学会了中间计算。','','## 3. P 的未来特征可预测性','','使用原三个距离 128/512/2048、原冻结未来编码器/特征标准化/InfoNCE 温度 0.1，每个正确片段配七个同文档、相互不重叠的困难负样本。每个前缀固定一套候选，各读出共享；均匀 CE=ln(8)=2.07944，随机化 top1 期望 12.5%。训练 860 个前缀、144 文档；验证 32 个前缀、18 个独立文档，共 96 个相关距离查询。','','输入：原 P-seed0 的冻结 reader 输出、原始 SSM 的位置保留投影，以及最近 64 个可见 tokens 的冻结编码特征；后者使用与目标相同的独立片段编码器，不含未来。Ridge 从训练文档的 116/28 文档隔离划分选正则，再拟合全训练集。额外 InfoNCE MLP 为 hidden128、512 步、batch32、固定 lr=1e−3/AdamW；只训练新头，不训练 Dream。四个头共 2048 次轻量更新，196608 个距离查询 draws，不能与正式 P 的计算预算等同。','','| 读出/训练配对 | 验证 CE | 验证 top1 |','|---|---:|---:|']
    for name,values in p['controls'].items():
        if name=='uniform_logits':lines.append('| 均匀（top1 为随机化期望） | 2.07944 | 12.50% |')
        else:lines.append(f"| {name} | {values['ce']:.5f} | {100*values['accuracy']:.2f}% |")
    for name,values in p['ridge'].items():lines.append(f"| ridge/{name} | {values['validation']['ce']:.5f} | {100*values['validation']['accuracy']:.2f}% |")
    for name,values in pn.items():lines.append(f"| InfoNCE/{name} | {values['validation']['ce']:.5f} | {100*values['validation']['accuracy']:.2f}% |")
    lines+=['','![P 可预测性](P-predictability.png)','','虚线配对打乱保持文档身份和样本量，只打乱训练文档内部历史—未来关联；验证时另对同文档输入轮换，81.25% 的前缀真正改变，其余单前缀文档保持不动。高容量探针训练准确率可接近 100%，而文档外验证接近随机，说明能记住训练候选不等于学会未来对应关系。','','相同文档的聚类 bootstrap（2000 次，18 文档；仍仅单个探针种子）对 InfoNCE CE 的配对变化：','','| 对比 | 均值 | 95% 聚类区间 |','|---|---:|---|']
    for name,values in intervals.items():
        lo,hi=values['cluster_95_percentile_interval'];lines.append(f"| {name} | {values['mean']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines+=['','已可见片段的正对照：目标换成最近 64 个可见 tokens 的特征；负样本仍为七个同文档非重叠片段。这是管线正对照，不是未来预测结果。','','| 读出 | 已可见目标 CE | 已可见目标准确率 |','|---|---:|---:|']
    for name,values in pv['readouts'].items():lines.append(f"| {name} | {values['ce']:.5f} | {100*values['accuracy']:.2f}% |")
    lines+=['','**判断：**在本轮数据量、两类新读出头与三个距离下，没有证实可泛化的未来特征可预测性。流程能学到可见目标，未来候选自匹配也为 100%，故不像候选索引或损失链路坏掉；但高容量头明显过拟合，仍无法区分目标本身的条件不确定性、读出器局限和训练文档不足。结果不证明所有历史都不含未来信息。P 对 LM 退化的缓解应继续单独解释，不能归功于已学会未来特征预测。','','## 执行边界与复现','','原检查点、旧报告和已完成结果的 SHA256 对照见 verification.json；旧文件不覆盖。新增事实/查询/特征头与固定缓存均在本诊断目录。Mamba/Dream 优化器更新均为零，没有 W1/W2、新记忆槽或 S+P。所有扫描/读出均属验证诊断，没有用测试集选参数。','','源码和运行方式见项目根目录 WSR_V02A_DIAGNOSTICS.md；完整数值见 results.json、amplitude-detail.csv、S-facts.csv、P-by-distance.csv。图片同时提供 PNG/SVG/PDF。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    findings=['# 三项诊断结论','','1. **幅度：损害不只是过度改写。** B0 在 8K/32K 历史下缩到 0.125 倍仍退化，反向 −0.25 倍改善；短历史可有小幅正向收益。方向的历史依赖是下一步要解释的问题，不能只调小幅度就宣称修复。',
              '2. **S：事实可读，查询组合未学好。** 原始/改写 SSM 的基础事实都可完全读出，既有 128 维 reader 也约 99.9%；新查询 MLP 即使直接获得真实基础事实，也只有约 52% 的未见组合准确率，未超过颜色频率捷径。显式历史 GRU 的弱敏感度是局部问题，主要瓶颈不能直接归结为信息缺失或需要工作区。',
              '3. **P：尚未证实未来目标的泛化可预测性。** Ridge 和独立 InfoNCE 头在文档外验证的 CE 都未优于均匀基线；训练可接近完全拟合，打乱配对对照优势不一致。相同管线能预测可见片段，未来自匹配也正常，因此应优先审查目标/读出/数据规模的适配，而不是扩大 Dream 训练。',
              '', '本次没有训练 Mamba 或 Dream；只拟合诊断头。S 正式训练保持暂停，S+P、W1/W2 未启动。单一探针种子、固定格式 S 和有限验证文档限制了结论，失败读出不等于理论上不可预测。',
              '', '[完整方法、表格与图](REPORT.md) · [核验](verification.json)']
    (ROOT/'FINDINGS.md').write_text('\n'.join(findings)+'\n')
    print('Three-diagnostic report and standalone figures saved',ROOT,flush=True)


if __name__=='__main__':main()

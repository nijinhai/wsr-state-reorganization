"""Reports separate accuracy, serial compute, and amortized query cost."""
import csv
import json
import statistics as st

from .precompute_experiment import ROOT, CONFIGS, write


def msd(values, percent=True):
    factor = 100 if percent else 1
    return f'{st.mean(values)*factor:.2f} ± {st.stdev(values)*factor:.2f}'


def table_csv(name, rows):
    with (ROOT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    result = json.loads((ROOT / 'results.json').read_text())
    runs = result['runs']
    def rows(name):
        return [runs[f'{name}-seed{s}'] for s in range(3)]
    def accuracies(name, query='held_query', hops='all'):
        return [x['test'][query][hops]['accuracy'] for x in rows(name)]
    curves = {name: [json.loads((ROOT/'runs'/f'{name}-seed{s}'/'learning-curves.json').read_text())
                     for s in range(3)] for name in CONFIGS}
    paired = []
    for other in ('A', 'B', 'B-budget', 'C2', 'C4', 'D8'):
        for query in ('seen_query', 'held_query'):
            values = [100*(a-b) for a, b in zip(accuracies('C8', query), accuracies(other, query))]
            paired.append({'contrast': 'C8 - '+other, 'queries': query,
                           'seed0_pp': values[0], 'seed1_pp': values[1], 'seed2_pp': values[2],
                           'mean_pp': st.mean(values), 'sd_pp': st.stdev(values)})
    table_csv('paired-differences.csv', paired)
    costs = []
    for name in CONFIGS:
        for seed, r in enumerate(rows(name)):
            for count, c in r['cost'].items():
                costs.append({'model': name, 'seed': seed, 'queries': int(count),
                              'parameters': r['parameters'], 'training_seconds': r['training_seconds'], **c})
    table_csv('costs.csv', costs)
    depth_rows = []
    for name, k in [('B', 1), ('C2', 2), ('C4', 4), ('C8', 8)]:
        for seed, r in enumerate(rows(name)):
            depth_rows.append({'kind': 'independently trained fixed depth', 'model': name, 'seed': seed,
                               'rounds': k, 'seen_accuracy': r['test']['seen_query']['all']['accuracy'],
                               'held_accuracy': r['test']['held_query']['all']['accuracy']})
    for name in ('C8', 'D8'):
        for seed, r in enumerate(rows(name)):
            for k, val in r['inference_depth'].items():
                depth_rows.append({'kind': 'inference truncation of depth8 model', 'model': name, 'seed': seed,
                                   'rounds': int(k), 'seen_accuracy': val['seen_query']['all']['accuracy'],
                                   'held_accuracy': val['held_query']['all']['accuracy']})
    table_csv('rounds.csv', depth_rows)

    lines = ['# WSR-v0.3：无查询预计算实验报告', '',
             '本报告使用统一2048更新终点、3个配对初始化种子；没有按测试集选择模型。'
             '所有训练只用最终颜色CE，未加载Mamba或旧Dream。', '',
             '## 任务、目标函数与接口', '',
             '随机6节点有向函数图，每节点一条非自环出边、一个颜色；三色各两节点。'
             '查询为起点s及1–4跳，答案为终点颜色。输入仅基础出边/颜色one-hot；'
             '标签由生成器计算，不把路径、传递关系或答案表输入模型。', '',
             '`W = Precompute(raw_facts)`；随后才提供多个查询，'
             '`L = mean CE(W[s] @ head_weight[h] + head_bias[h], answer)`。'
             'D在初始化及每轮更新中额外输入查询，为每个查询分别计算。', '',
             'W为6×32自由特征；按起点选择输出位置和按步数选择线性分类器是预设接口，'
             '不含查询阶段隐藏层或循环。B/C每轮的全局MLP都同时看到全部规则及全部W，'
             '任何一轮都可完成任意关系组合，不限制一轮一跳。所有可训练参数由最终CE更新；'
             '无事实对齐、中间状态标签或辅助损失。读取头的表达能力和查询隔离已单元验证。', '',
             '训练/验证/测试分别有1024/128/128个互不重复的出边图；三种颜色轮换不跨划分，'
             '共3072/384/384实例。每个具体查询的三类答案等频，随机及多数类准确率33.33%。'
             '24个起点×步数组合中16个训练、8个留出，所有起点/步数分别都出现于训练。'
             '三个种子共享固定数据，SD是初始化/训练差异，不是新数据集总体置信区间。', '',
             '## 模型效果：测试规则均未见', '',
             '| 模型 | 参数 | 训练规则/训练查询 | 训练规则/留出查询 | 新规则/训练查询 | 新规则/留出查询 | 优化训练耗时，秒 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in CONFIGS:
        train = [c[-1]['train']['seen_query']['all']['accuracy'] for c in curves[name]]
        train_held = [c[-1]['train']['held_query']['all']['accuracy'] for c in curves[name]]
        lines.append(f'| {name} | {rows(name)[0]["parameters"]:,} | {msd(train)}% | '
                     f'{msd(train_held)}% | '
                     f'{msd(accuracies(name,"seen_query"))}% | {msd(accuracies(name))}% | '
                     f'{msd([x["training_seconds"] for x in rows(name)],False)} |')
    lines += ['', 'B就是C1的精确等价实现与结果，未重复训练。'
              'B/C具有相同参数量和工作区；B-budget扩大单次MLP宽度以匹配C8预计算线性MACs，'
              '其参数量更大。D8在每个查询上执行8轮，训练计算量约为共享预计算的8倍，'
              '因此不把其准确率差作为相同总成本的胜负。', '',
              '| 模型 | 留出组合1跳 | 2跳 | 3跳 | 4跳 |', '|---|---:|---:|---:|---:|']
    for name in CONFIGS:
        lines.append('| '+name+' | '+' | '.join(msd(accuracies(name, hops=f'h{h}'))+'%' for h in range(1,5))+' |')
    lines += ['', '### C8的配对差异（百分点）', '',
              '| 对照 | 查询组合 | seed0 | seed1 | seed2 | 均值 |', '|---|---|---:|---:|---:|---:|']
    for p in paired:
        lines.append(f'| {p["contrast"]} | {p["queries"]} | {p["seed0_pp"]:+.2f} | '
                     f'{p["seed1_pp"]:+.2f} | {p["seed2_pp"]:+.2f} | {p["mean_pp"]:+.2f} |')
    lines += ['', '## 轮数：独立训练与推理截断分别报告', '',
              '| 独立训练的固定轮数 | 新规则/训练查询 | 新规则/留出查询 |', '|---|---:|---:|']
    for name, k in [('B',1),('C2',2),('C4',4),('C8',8)]:
        lines.append(f'| {k} | {msd(accuracies(name,"seen_query"))}% | {msd(accuracies(name))}% |')
    lines += ['', '| 固定训练8轮的模型 | 推理轮数 | 新规则/训练查询 | 新规则/留出查询 |', '|---|---:|---:|---:|']
    for name in ('C8','D8'):
        for k in ('1','2','4','8'):
            vals = [[r['inference_depth'][k][g]['all']['accuracy'] for r in rows(name)]
                    for g in ('seen_query','held_query')]
            lines.append(f'| {name} | {k} | {msd(vals[0])}% | {msd(vals[1])}% |')
    lines += ['', '截断推理是分布变化，不能替代独立训练轮数对照，也不证明自主停止。'
              '各轮更新幅度/方向仅作动力学描述，不能由状态变化推出有用分工。', '',
              '## 成本与预计算复用', '',
              '以下为稠密线性MACs（1乘加计1MAC，不等于精确硬件FLOPs），'
              '排除one-hot、索引、SiLU、LayerNorm与偏置。实测为CPU FP32单线程、'
              'batch16向量化、预热3次、9次中位数，再跨种子取均值；包括整个前向，'
              '不含反传。训练耗时是优化更新耗时之和，不含评价/检查点/I/O。', '',
              '| 模型 | 查询数 | 总MACs/规则 | MACs/查询 | 总毫秒/规则 | 微秒/查询 |',
              '|---|---:|---:|---:|---:|---:|']
    for name in CONFIGS:
        for count in ('1','4','16','64'):
            vals = [r['cost'][count] for r in rows(name)]
            lines.append(f'| {name} | {count} | {vals[0]["total"]:,} | {vals[0]["MACs_per_query"]:,.1f} | '
                         f'{st.mean(v["seconds_per_world"] for v in vals)*1000:.3f} | '
                         f'{st.mean(v["seconds_per_query"] for v in vals)*1e6:.2f} |')
    lines += ['', 'C成本为一次预计算加Q次轻量读取；D为Q次查询定向计算。'
              '在1/4/16/64个查询的计时中没有重复调用C预计算。'
              '查询从24个组合中有放回独立抽取；数量增加只是摊销，不改变单个查询的预测函数。'
              '本表未启用D的答案缓存；由于只有24个不同查询，若允许按查询键缓存D输出，'
              '重复查询无需重算，因此64查询的线性成本不是D可实现的最优成本。'
              '实测速度还受向量化/批量大小影响，不能用MAC比直接冒充时延比。'
              '所有组训练固定每规则8查询，本轮没有测试训练查询数量的独立作用。', '',
              '## 数据、捷径与置换诊断', '',
              '| 简单基线 | 全查询 | 1跳 | 2跳 | 3跳 | 4跳 |', '|---|---:|---:|---:|---:|---:|']
    for name, val in result['shortcuts']['test'].items():
        lines.append('| '+name+' | '+' | '.join(f'{100*val[h]["accuracy"]:.2f}%' for h in ('all','h1','h2','h3','h4'))+' |')
    lines += ['', '固定一步基线使用真实基础出边进行一次符号读取，是任务捷径诊断，'
              '不是与轻量读取头同预算的已训练组。短周期会使不同跳数答案相关，'
              '必须结合逐跳准确率，不能把超随机准确率直接解释为完整多跳计算。', '',
              '| 模型 | 节点一致置换后：训练组合 | 节点一致置换后：留出组合 | 全局颜色轮换后：全查询 |',
              '|---|---:|---:|---:|']
    for name in CONFIGS:
        vals = rows(name)
        ps = [v['diagnostics']['node_permutation_mapped_split_novel_rules']['seen_query']['all']['accuracy'] for v in vals]
        ph = [v['diagnostics']['node_permutation_mapped_split_novel_rules']['held_query']['all']['accuracy'] for v in vals]
        color = [v['diagnostics']['color_rotation']['all']['accuracy'] for v in vals]
        lines.append(f'| {name} | {msd(ps)}% | {msd(ph)}% | {msd(color)}% |')
    meta = rows('C8')[0]['diagnostics']['node_permutation_metadata']
    lines += ['', '置换同时修改节点槽、出边目标与查询起点；答案语义保持。'
              f'上表按映射后的查询组合归属评价，并剔除碰巧匹配训练图的{meta["excluded_worlds_matching_training_graph"]}个实例，'
              f'保留{meta["novel_worlds"]}个。原8个留出组合中有{meta["original_held_queries_becoming_seen"]}个映射成训练组合；'
              '原组别汇总仍保留在results.json，但不能当作留出组合不变的比较。'
              '全局稠密网络并非结构上置换等变，因此该项检验学习到的泛化而非保证。', '',
              '面板本身包含全部三种颜色轮换，颜色轮换汇总并不是额外独立的泛化挑战。', '',
              '## 实验记录、检查与局限', '',
              '- 只有一个任务配置，没有为C筛选难度或修改任务。32实例B/C8/D8均能100%拟合训练组合；'
              '留出组合不是该过拟合检查的验收目标。',
              '- 先512步seed0预实验，再不改任务续训至预先声明的2048步，扩展3种子；'
              '预实验计入seed0预算，未重复计数。21个主运行共43,008更新、5,505,024个最终答案监督；'
              '小样本检查另外记录。',
              '- 所有模型相同Adam、学习率、梯度裁剪、规则/查询日程与答案数量；'
              '这不保证各结构均达到最佳优化状态。没有为B或C单独调参，因此结果只描述当前预算。',
              '- 最终读取位置按起点预设，分类器按步数选择；循环轮数和全局MLP架构也由实现提供。'
              '不能声称自主寻址、自主停止或已发现特定逐轮算法。',
              '- 本轮按用户收窄范围，先完成准确率、轮数与预算对照；'
              '未加入随机轮数、Anytime、Coverage、轮次编码或新模块。',
              '- 原始产物保护见`preservation-after.json`；代码、协议、曲线、梯度、日程、'
              '检查点及全部测试日志保留。', '',
              '![准确率、轮数与计算成本](overview.png)', '',
              '方法：[仓库协议](../../docs/WSR_V03.md)。原始表：'
              '[main-results.csv](main-results.csv)、[rounds.csv](rounds.csv)、'
              '[costs.csv](costs.csv)、[paired-differences.csv](paired-differences.csv)、'
              '[results.json](results.json)。研究决策另见[FINDINGS.md](FINDINGS.md)。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    palette = dict(zip(CONFIGS, ['#777777','#cc8030','#9055aa','#81aa66','#369858','#13603b','#367ac8']))
    for name in CONFIGS:
        steps = [x['step'] for x in curves[name][0]]
        scores = [[c[i]['validation']['held_query']['all']['accuracy']*100 for c in curves[name]]
                  for i in range(len(steps))]
        axes[0,0].plot(steps, [st.mean(v) for v in scores], label=name, color=palette[name])
        ys = [st.mean(accuracies(name, hops=f'h{h}'))*100 for h in range(1,5)]
        axes[0,1].plot(range(1,5), ys, marker='o', label=name, color=palette[name])
    axes[0,0].set(xlabel='Optimizer updates', ylabel='Held-query validation accuracy (%)', title='Learning curves (mean of 3 seeds)')
    axes[0,1].set(xlabel='Query hops', ylabel='Held-query test accuracy (%)', title='Generalization by requested hops', xticks=range(1,5))
    ks = [1,2,4,8]
    axes[1,0].errorbar(ks,[st.mean(accuracies(n))*100 for n in ('B','C2','C4','C8')],
                        yerr=[st.stdev(accuracies(n))*100 for n in ('B','C2','C4','C8')],
                        marker='o', label='Independently trained fixed K')
    axes[1,0].plot(ks,[st.mean(r['inference_depth'][str(k)]['held_query']['all']['accuracy'] for r in rows('C8'))*100
                       for k in ks], '--o', label='C8 truncated at inference')
    axes[1,0].set(xlabel='Precompute rounds', ylabel='Held-query test accuracy (%)', title='Training depth vs inference truncation', xticks=ks)
    for name in ('B','B-budget','C8','D8'):
        axes[1,1].plot([1,4,16,64],[rows(name)[0]['cost'][str(n)]['MACs_per_query']/1e3 for n in (1,4,16,64)],
                       marker='o', label=name, color=palette[name])
    axes[1,1].set(xlabel='Queries sharing one rule world', ylabel='Linear kMACs per query', title='Amortized cost: B-budget = C8', xscale='log', yscale='log')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(ROOT/'overview.png', dpi=170)
    fig.savefig(ROOT/'overview.pdf')
    plt.close(fig)
    write(ROOT/'accounting.json', {'main_runs':21,'optimizer_updates':43008,'final_answer_labels':5505024,
          'optimization_seconds':sum(x['training_seconds'] for x in runs.values()),
          'C1_alias':'B','test_gate':'test-gate.json','new_mechanisms':[]})
    print('report and figures saved', ROOT, flush=True)


if __name__ == '__main__':
    main()

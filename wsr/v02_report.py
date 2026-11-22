"""Standalone scientific figures and a numerical report for the completed v0.2a matrix."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('artifacts/wsr-v0.2a')
GROUPS=('B0','S','P')
LM_GROUPS=('B0','P')
COLORS={'B0':'#4c78a8','S':'#f58518','P':'#54a24b'}


def read(path):return json.loads(path.read_text())
def mean(values):return sum(values)/len(values)
def sd(values):return statistics.stdev(values) if len(values)>1 else 0.


def savefig(name):
    plt.tight_layout()
    for ext in ['png','svg','pdf']:plt.savefig(ROOT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close()


def write_csv(name,rows):
    with (ROOT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def coverage(training):
    intervals={}
    for row in training:
        key=(row['dataset'],row['document'])
        for cycle in range(2):
            b=row['boundary']+256*cycle;intervals.setdefault(key,[]).append((b,b+64))
    unique=0
    for spans in intervals.values():
        end=-1
        for lo,hi in sorted(spans):
            unique+=max(0,hi-max(lo,end));end=max(end,hi)
    return unique


def main():
    base={r['id']:r for r in read(ROOT/'evaluation/baseline-seed-0/lm.json')}
    baseline_curve=read(ROOT/'no-dream-curve.json')['mean_nll']
    curve=[];resources=[];lm=[];panels=[];future=[];curve_cases=[];repeat_rows=[]
    curve_reference={r['id']:r for r in read(ROOT/'no-dream-curve.json')['rows']}
    actual={}
    for group in GROUPS:
        for seed in range(3):
            run=ROOT/'runs'/f'{group}-seed-{seed}';evaluation=ROOT/'evaluation'/f'{group}-seed-{seed}'
            summary=read(run/'summary.json');actual[(group,seed)]=summary
            training=read(run/'training.json');unique_targets=coverage(training)
            by_prefix={}
            for item in training:by_prefix.setdefault(item['id'],[]).append(item)
            for dataset in ['wikitext103','pg19']:
                pairs=[rows for rows in by_prefix.values() if len(rows)>=2 and rows[0]['dataset']==dataset]
                repeat_rows.append({'group':group,'seed':seed,'dataset':dataset,'repeated_prefixes':len(pairs),
                    'mean_last_minus_first_train_NLL':mean([rows[-1]['lm_nll']-rows[0]['lm_nll'] for rows in pairs]),
                    'mean_first_update':mean([rows[0]['step'] for rows in pairs]),'mean_last_update':mean([rows[-1]['step'] for rows in pairs])})
            resources.append({'group':group,'seed':seed,'updates':summary['updates'],
                 'independent_control_head_updates':0 if group=='B0' else summary['updates'],
                 'LM_target_tokens':summary['LM_supervised_tokens'],'LM_actual_forward_tokens':summary['actual_LM_forward_tokens'],
                 'train_documents':summary['unique_train_documents'],'prefixes':summary['unique_prefixes'],
                 'unique_LM_target_positions':unique_targets,'LM_target_draw_repeat_factor':summary['LM_supervised_tokens']/unique_targets,
                 'train_seconds':summary['training_seconds_sum'],'timed_wall_seconds':summary['wall_seconds_this_invocation'],
                 'LM_target_tokens_per_second':summary['LM_target_tokens_per_second'],
                 'peak_allocated_MiB':summary['peak_allocated_bytes']/2**20})
            for r in read(run/'validation.json'):
                curve.append({'group':group,'seed':seed,'step':r['step'],'nll':r['mean_nll'],
                              'delta_no_Dream':r['mean_nll']-baseline_curve,'edit_relative':mean([x['edit_relative'] for x in r['rows']])})
                for case in r['rows']:
                    curve_cases.append({'group':group,'seed':seed,'step':r['step'],'id':case['id'],'dataset':case['dataset'],
                        'nll':case['nll'],'delta_no_Dream':case['nll']-curve_reference[case['id']]['nll'],
                        'edit_relative':case['edit_relative']})
            if group not in LM_GROUPS:continue
            for r in read(evaluation/'lm.json'):
                reference=base[r['id']]
                for distance,value in r['distance_bins'].items():
                    lo,hi=map(int,distance.split('-'));canonical={1:32,33:128,129:512,513:2048}[lo]
                    future.append({'group':group,'seed':seed,'id':r['id'],'dataset':r['document']['dataset'],
                                   'history':r['boundary'],'future_bin':f'{lo}-{canonical}','available_through':hi,'tokens':hi-lo+1,'nll':value,
                                   'delta_no_Dream':value-reference['distance_bins'][distance]})
                lm.append({'group':group,'seed':seed,'id':r['id'],'dataset':r['document']['dataset'],
                           'document':r['document']['id'],'boundary':r['boundary'],'nll':r['nll'],
                           'delta_no_Dream':r['nll']-reference['nll'],'first64_delta':r['first_64_nll']-reference['first_64_nll'],
                           'tokens':r['tokens'],'final_relative_edit':r['trajectory'][-1]['relative_total']})
            for r in read(evaluation/'panel.json'):
                intact512=r['rounds']['8']['cumulative_horizons']['512']
                for n,row in r['rounds'].items():
                    panels.append({'group':group,'seed':seed,'id':r['id'],'condition':'round_'+n,
                                   'nll512':row['cumulative_horizons']['512'],
                                   'nll2048':row['nll'],'delta512_vs_intact8':row['cumulative_horizons']['512']-intact512,
                                   'post_intervention_delta_cosine':None,'final_state_delta_relative':None})
                for name,row in r['interventions'].items():
                    panels.append({'group':group,'seed':seed,'id':r['id'],'condition':name,
                                   'nll512':row['nll'],'nll2048':None,'delta512_vs_intact8':row['nll']-intact512,
                                   'post_intervention_delta_cosine':row['first_post_intervention_delta_cosine'],
                                   'final_state_delta_relative':row['final_change_vs_intact_relative']})
                matched=r['norm_matched_first_edit']
                panels.append({'group':group,'seed':seed,'id':r['id'],'condition':'norm_matched_first_edit',
                    'nll512':matched['cumulative_horizons']['512'],'nll2048':matched['nll'],
                    'delta512_vs_intact8':matched['cumulative_horizons']['512']-intact512,
                    'post_intervention_delta_cosine':None,'final_state_delta_relative':None})
    write_csv('validation-curves.csv',curve);write_csv('resources.csv',resources)
    write_csv('validation-by-dataset.csv',curve_cases)
    write_csv('LM-results.csv',lm);write_csv('rounds-interventions.csv',panels)
    write_csv('future-distance.csv',future)
    write_csv('training-repeated-prefixes.csv',repeat_rows)
    # Budget curve: paired seeds, fixed validation panel, no best-checkpoint selection.
    plt.figure(figsize=(8,4))
    for group in GROUPS:
        steps=sorted(set(r['step'] for r in curve if r['group']==group))
        y=[mean([r['delta_no_Dream'] for r in curve if r['group']==group and r['step']==s]) for s in steps]
        err=[sd([r['delta_no_Dream'] for r in curve if r['group']==group and r['step']==s]) for s in steps]
        plt.plot(steps,y,'o-',label=group,color=COLORS[group]);plt.fill_between(steps,np.array(y)-err,np.array(y)+err,color=COLORS[group],alpha=.15)
    plt.axhline(0,color='black',lw=.7);plt.xlabel('Optimizer updates');plt.ylabel('Validation NLL change vs no Dream');plt.legend();savefig('budget-curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,dataset in zip(axes,['wikitext103','pg19']):
        for group in GROUPS:
            steps=sorted(set(r['step'] for r in curve_cases if r['group']==group))
            values=[];errors=[]
            for step in steps:
                points=[mean([r['delta_no_Dream'] for r in curve_cases if r['group']==group and r['dataset']==dataset and r['step']==step and r['seed']==seed]) for seed in range(3)]
                values.append(mean(points));errors.append(sd(points))
            ax.plot(steps,values,'o-',color=COLORS[group],label=group)
            ax.fill_between(steps,np.array(values)-errors,np.array(values)+errors,color=COLORS[group],alpha=.15)
        ax.axhline(0,color='black',lw=.7);ax.set_title(dataset+(' / history=512' if dataset=='wikitext103' else ' / history=8192'))
        ax.set_xlabel('Optimizer updates');ax.legend()
    axes[0].set_ylabel('Validation NLL change vs no Dream');savefig('budget-by-dataset')
    main_rows=[]
    for group in LM_GROUPS:
        for dataset,boundary in [('wikitext103',512),('wikitext103',2048),('pg19',8192),('pg19',32768)]:
            values=[mean([r['delta_no_Dream'] for r in lm if r['group']==group and r['seed']==seed and r['dataset']==dataset and r['boundary']==boundary]) for seed in range(3)]
            b0=[mean([r['delta_no_Dream'] for r in lm if r['group']=='B0' and r['seed']==seed and r['dataset']==dataset and r['boundary']==boundary]) for seed in range(3)]
            main_rows.append({'group':group,'dataset':dataset,'history':boundary,'delta_no_Dream':mean(values),'seed_sd':sd(values),
                              'paired_delta_B0':mean([a-b for a,b in zip(values,b0)]),'per_seed_delta_no_Dream':values})
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,dataset in enumerate(['wikitext103','pg19']):
        histories=[512,2048] if i==0 else [8192,32768]
        for j,group in enumerate(LM_GROUPS):
            rows=[r for r in main_rows if r['group']==group and r['dataset']==dataset]
            axes[i].errorbar(np.arange(2)+(j-1)*.1,[r['delta_no_Dream'] for r in rows],yerr=[r['seed_sd'] for r in rows],fmt='o',label=group,color=COLORS[group])
        axes[i].axhline(0,color='black',lw=.7);axes[i].set_xticks(range(2),histories);axes[i].set_title(dataset);axes[i].set_xlabel('Visible history tokens');axes[i].set_ylabel('NLL change vs no Dream');axes[i].legend()
    savefig('LM-effects')
    fig,axes=plt.subplots(1,4,figsize=(14,3.5))
    for ax,(dataset,history) in zip(axes,[('wikitext103',512),('wikitext103',2048),('pg19',8192),('pg19',32768)]):
        for group in LM_GROUPS:
            subset=[r for r in future if r['group']==group and r['dataset']==dataset and r['history']==history]
            bins=sorted(set(r['future_bin'] for r in subset),key=lambda x:int(x.split('-')[0]))
            ax.plot(range(len(bins)),[mean([r['delta_no_Dream'] for r in subset if r['future_bin']==b]) for b in bins],'o-',label=group,color=COLORS[group])
            ax.set_xticks(range(len(bins)),bins,rotation=35)
        ax.axhline(0,color='black',lw=.7);ax.set_title(f'{dataset} / H={history}');ax.set_xlabel('Post-Dream token distance')
    axes[0].set_ylabel('NLL change vs no Dream');axes[-1].legend();savefig('future-distance-effects')
    # Exact S/P generalization and head controls.
    s_rows=[];p_rows=[];prediction_collapse=[]
    for seed in range(3):
        s=read(ROOT/'evaluation'/f'S-seed-{seed}'/'auxiliary-2048.json')
        for split in ['new_rules_seen_queries','new_rules_new_queries','seen_rules_new_queries']:
            for condition,metric in s[split].items():s_rows.append({'seed':seed,'split':split,'condition':condition,**metric})
        p=read(ROOT/'evaluation'/f'P-seed-{seed}'/'auxiliary-2048.json')
        for condition,distances in p['prediction_collapse_diagnostics'].items():
            for distance,metrics in distances.items():prediction_collapse.append({'seed':seed,'condition':condition,'distance':int(distance),**metrics})
        for row in p['rows']:
            for mode,readings in row['metrics'].items():
                for condition,value in readings.items():
                    for r in value['per_distance']:p_rows.append({'seed':seed,'id':row['id'],'mode':mode,'condition':condition,**r})
    write_csv('S-generalization.csv',s_rows);write_csv('P-generalization.csv',p_rows)
    write_csv('prediction-collapse.csv',prediction_collapse)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    s_conditions=['0','1','2','4','8']
    for split,color in zip(['new_rules_seen_queries','new_rules_new_queries','seen_rules_new_queries'],['#4c78a8','#f58518','#54a24b']):
        values=[mean([r['accuracy'] for r in s_rows if r['split']==split and r['condition']==c]) for c in s_conditions]
        axes[0].plot([0,1,2,4,8],values,'o-',label=split,color=color)
        control=mean([r['accuracy'] for r in s_rows if r['split']==split and r['condition']=='no_Dream_trained_head'])
        axes[0].axhline(control,color=color,ls=':',lw=1)
    axes[0].axhline(.25,color='black',lw=.7);axes[0].set_title('S: shared-query accuracy');axes[0].set_xlabel('Dream rounds (same trained model)');axes[0].legend(fontsize=7)
    for mode,color in [('within_document','#4c78a8'),('cross_document','#f58518')]:
        values=[mean([r['correct'] for r in p_rows if r['mode']==mode and r['condition']==c]) for c in s_conditions]
        axes[1].plot([0,1,2,4,8],values,'o-',label=mode,color=color)
        control=mean([r['correct'] for r in p_rows if r['mode']==mode and r['condition']=='no_Dream_trained_head'])
        axes[1].axhline(control,color=color,ls=':',lw=1)
    axes[1].axhline(.125,color='black',lw=.7);axes[1].set_title('P: 8-candidate retrieval accuracy');axes[1].set_xlabel('Dream rounds (same trained model)');axes[1].legend(fontsize=8)
    savefig('auxiliary-effects')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    trajectory_rows=[]
    for group in LM_GROUPS:
        traces=[]
        for seed in range(3):
            for row in read(ROOT/'evaluation'/f'{group}-seed-{seed}'/'panel.json'):
                for point in row['trajectory']:trajectory_rows.append({'group':group,'seed':seed,'id':row['id'],**point})
                traces.append(row['trajectory'])
        axes[0].plot(range(1,9),[mean([t[i]['relative_total'] for t in traces]) for i in range(8)],'o-',label=group,color=COLORS[group])
        axes[1].plot(range(1,9),[mean([t[i]['cosine_with_first_delta'] for t in traces]) for i in range(8)],'o-',label=group,color=COLORS[group])
        interventions=['reset_state','reset_memory','reverse_state']
        axes[2].plot(range(3),[mean([r['delta512_vs_intact8'] for r in panels if r['group']==group and r['condition']==c]) for c in interventions],'o-',label=group,color=COLORS[group])
    axes[0].set_ylabel('Relative cumulative state edit');axes[1].set_ylabel('Cosine with first edit');axes[2].set_ylabel('NLL change vs intact 8 rounds')
    for ax in axes[:2]:ax.set_xlabel('Dream round');ax.legend()
    axes[2].axhline(0,color='black',lw=.7);axes[2].set_xticks(range(3),['Reset H','Reset GRU','Reverse H']);axes[2].legend();savefig('state-trajectories')
    write_csv('state-trajectories.csv',trajectory_rows)
    # Periodic curves, matching the exact books and positions, no reset within book.
    periodic_base=read(ROOT/'evaluation/baseline-seed-0/periodic.json');periodic_rows=[]
    plt.figure(figsize=(8,4))
    for group in LM_GROUPS:
        books=read(ROOT/'evaluation'/f'{group}-seed-0'/'periodic.json')
        cumulative=[]
        for book,reference in zip(books,periodic_base):
            if book['document']['sha256']!=reference['document']['sha256']:raise ValueError('Periodic book mismatch')
            sums=0.;count=0;values=[]
            for event,base_event in zip(book['events'],reference['events']):
                n=len(event['losses']);delta=sum(a-b for a,b in zip(event['losses'],base_event['losses']))
                sums+=delta;count+=n;values.append(sums/count)
                periodic_rows.append({'group':group,'book':book['document']['id'],'history':event['history'],
                     'event_delta_nll':delta/n,'cumulative_delta_nll':sums/count,'relative_edit':event['relative_edit']})
            cumulative.append(values)
        x=[e['processed_after']+1 for e in books[0]['events']]
        plt.plot(x,np.mean(cumulative,axis=0),label=group,color=COLORS[group])
    plt.axhline(0,color='black',lw=.7);plt.xlabel('Continuous visible tokens');plt.ylabel('Cumulative NLL change vs no Dream');plt.legend();savefig('periodic-effects')
    write_csv('periodic-effects.csv',periodic_rows)
    calibration=read(ROOT/'calibration.json');features=read(ROOT/'feature-diagnostics.json')
    matched_targets=read(ROOT/'matched-P-target-diagnostics.json')
    sensitivity=read(ROOT/'S-history-sensitivity.json')
    s_quality=read(ROOT/'S-data-quality.json')
    assert all(str(seed) in sensitivity for seed in range(3))
    summary={'scope':read(ROOT/'scope-update.json'),'LM':main_rows,'resources':resources,'S':{},'P':{},'rounds':{},'interventions':{},'calibration_weights':calibration['weights'],
             'repeated_prefix_training_NLL':repeat_rows,
             'prediction_collapse':prediction_collapse,'matched_P_target_collapse':matched_targets,
             'S_history_sensitivity':{seed:{step:{k:v for k,v in values.items() if k!='rows'} for step,values in row['checkpoints'].items()} for seed,row in sensitivity.items()},
             'S_data_shortcuts':s_quality['validation'],
             'feature_diagnostics':features,'periodic':{group:mean([r['cumulative_delta_nll'] for r in periodic_rows if r['group']==group and r['history']==periodic_rows[-1]['history']]) for group in LM_GROUPS}}
    for split in ['new_rules_seen_queries','new_rules_new_queries','seen_rules_new_queries']:
        summary['S'][split]={c:{'accuracy':mean([r['accuracy'] for r in s_rows if r['split']==split and r['condition']==c]),
                               'ce':mean([r['ce'] for r in s_rows if r['split']==split and r['condition']==c])} for c in s_conditions+['no_Dream_trained_head','norm_matched_first_edit','shuffled_queries_8','visible_color_frequency_baseline']}
    for mode in ['within_document','cross_document']:
        summary['P'][mode]={}
        for condition in s_conditions+['no_Dream_trained_head','permuted_targets','norm_matched_first_edit','distance_only_train_prior','candidate_centrality']:
            subset=[r for r in p_rows if r['mode']==mode and r['condition']==condition]
            summary['P'][mode][condition]={'accuracy':mean([r['correct'] for r in subset]),'ce':mean([r['ce'] for r in subset]),'queries':len(subset)}
    for group in LM_GROUPS:
        summary['rounds'][group]={c:mean([r['nll2048'] for r in panels if r['group']==group and r['condition']=='round_'+c]) for c in s_conditions}
        summary['rounds'][group]['norm_matched_first_edit']=mean([r['nll2048'] for r in panels if r['group']==group and r['condition']=='norm_matched_first_edit'])
        summary['interventions'][group]={c:{'delta512_vs_intact8':mean([r['delta512_vs_intact8'] for r in panels if r['group']==group and r['condition']==c]),
            'final_state_change_relative':mean([r['final_state_delta_relative'] for r in panels if r['group']==group and r['condition']==c]),
            'post_intervention_delta_cosine':mean([r['post_intervention_delta_cosine'] for r in panels if r['group']==group and r['condition']==c])} for c in ['reset_state','reset_memory','reverse_state']}
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2))
    # Report keeps interpretation separate from the reproducible numeric tables.
    lines=['# WSR-v0.2a 结果报告','',
        'B0/P 各完成三个配对种子、每种子 2048 步训练。S 的三个种子训练与辅助评价在暂停指令前已完成并保留。按最新指令，仅继续 B0/P 扩展语言评价与连续运行，S 不再新增训练或扩展评价。保留 512/1024/2048 检查点、验证曲线和原始逐样本结果。官方 Mamba 参数冻结；仅训练 Direct Dream 和对应辅助头。没有实现 W1/W2 或新记忆槽。','',
        '本报告所有语言损失单位为 nats/token；负的配对差值表示改善。结果来自原 validation 划分，测试集未读取。三种子标准差衡量初始化/训练顺序变化，不是文档总体的置信区间。','',
        '## 1. 预算与真实语言预测','',
        '| 组 | 历史/数据 | ΔNLL 相对无 Dream，均值 ± 种子 SD | 配对 ΔNLL 相对 B0 |','|---|---|---:|---:|']
    for row in main_rows:lines.append(f"| {row['group']} | {row['dataset']} / {row['history']} | {row['delta_no_Dream']:+.7f} ± {row['seed_sd']:.7f} | {row['paired_delta_B0']:+.7f} |")
    curve_docs={r['id'].rsplit('-',1)[0] for r in read(ROOT/'no-dream-curve.json')['rows']}
    novel=[]
    lines+=['','未出现在训练曲线面板中的 Wiki 验证文档，两个历史边界合并：','',
            '| 组 | ΔNLL 相对无 Dream | 配对 ΔNLL 相对 B0 | 文档数 |','|---|---:|---:|---:|']
    for group in LM_GROUPS:
        subset=[r for r in lm if r['group']==group and r['dataset']=='wikitext103' and r['id'].rsplit('-',1)[0] not in curve_docs]
        per_seed=[mean([r['delta_no_Dream'] for r in subset if r['seed']==seed]) for seed in range(3)]
        matched=[mean([r['delta_no_Dream'] for r in lm if r['group']=='B0' and r['seed']==seed and r['dataset']=='wikitext103' and r['id'].rsplit('-',1)[0] not in curve_docs]) for seed in range(3)]
        point={'group':group,'delta_no_Dream':mean(per_seed),'seed_sd':sd(per_seed),'paired_delta_B0':mean([a-b for a,b in zip(per_seed,matched)]),
               'documents':len(set(r['document'] for r in subset)),'per_seed_delta_no_Dream':per_seed}
        novel.append(point)
        lines.append(f"| {group} | {point['delta_no_Dream']:+.7f} ± {point['seed_sd']:.7f} | {point['paired_delta_B0']:+.7f} | {point['documents']} |")
    summary['Wiki_outside_curve_panel']=novel
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2))
    lines+=['','上述短程表包含 16 篇 Wiki validation 文章，长程表包含两本 PG validation 书。每个边界后最多评分 2048 tokens；表内先按文档边界求平均，再跨三个种子求均值。不能把不同历史长度下的差异都归因于历史长度，因为对应的后续文本也不同。','',
             '![训练预算曲线](budget-curves.png)','','固定曲线面板为四篇 Wiki 文章与两本 PG 书；阴影为三个种子的标准差。所有预算检查点都展示，不以最好点替代 2048 步终点。','',
             '![分数据集预算曲线](budget-by-dataset.png)','','短历史 Wiki 与 8K 历史 PG 分开呈现，避免合并平均掩盖方向相反的变化。','',
             '![真实文本效果](LM-effects.png)','','![不同未来距离的损失变化](future-distance-effects.png)','','各距离点可能包括不同数量的文档：短文档后续不足 2048 tokens 时使用自然可用长度，逐位置与同文档基线配对；未拼接填充。','',
             '## 2. 目标函数与权重','',
             '`L_LM = mean(-log p(future_64 | modified_state))`；每个更新内两个连续周期再平均。',
             '`L_S = mean_i CE(QueryDecoder(Dream(H,X), q_i), a_i)`，同一组查询共享一次 Dream。',
             '`L_P = CE(normalize(Predictor(H_new,d)) · stop_gradient(normalize(FrozenEncoder(F_candidates))) / 0.1, positive_index)`，三种距离和两个周期再平均。','',
             f"B0 仅用 L_LM；S 使用 alpha={calibration['weights']['S']:.8g}；P 使用 beta={calibration['weights']['P']:.8g}。两项权重分别根据 16 个训练样本的初始 Dream 梯度中位数校准，并经过两个候选各八步的训练稳定性试验；不由验证收益选权重。见 [calibration.json](calibration.json)。正式训练每 256 步记录 LM/辅助梯度范数和夹角。",'',
             '续算中冻结主模型权重但保留对输入状态的反向路径；编辑后重新计算 anchor 的 logits。未来文本/特征绝不进入 Dream；编码器前向为 no_grad，预测目标不带梯度。辅助任务只监督第八轮，1/2/4 轮为事后读出。','',
             '工程修复：有限差分发现辅助 StateReader 对已编辑状态的归一化尺度存在不必要的 detach。已移除并加入沿缩放方向的梯度检查；21 项测试通过。修复前 S/P 各 448 步及其校准被完整归档到 engineering-smoke-reader-stopgrad，不纳入正式对照；S/P 重新校准并从共同初始化训练。B0 与冻结 LM 续算不受该读出头错误影响。见 reader-gradient-diagnostic.json。','',
             '## 3. S：共享查询泛化与预测头贡献','',
             '四地点链接置换＋颜色事实，查询沿链接走 1–3 步后的终点颜色；四选一，随机准确率 25%，均匀预测 CE=ln(4)≈1.38629。1024 个训练规则实例与 256 个新实例独立；八种训练查询组合与四种留出组合独立。查询解码器只读取状态和分开编码的起点/步数，不读取历史。','',
             '| 验证集合 | 独立无 Dream 头 | 联合头读原始状态 | 联合头读 1 轮 | 联合头读 8 轮 | 查询错配 | 颜色频率基线 |','|---|---:|---:|---:|---:|---:|---:|']
    for split,metrics in summary['S'].items():lines.append('| '+split+' | '+' | '.join(f"{100*metrics[c]['accuracy']:.2f}%" for c in ['no_Dream_trained_head','0','1','8','shuffled_queries_8','visible_color_frequency_baseline'])+' |')
    lines+=['','独立无 Dream 头与联合头同初始化、同样本查询、同更新数和学习率；它只在未编辑状态上训练。联合头读原始状态是另一个同头消融。新查询验证不会给 Dream 提供查询或答案，已见规则集合只从实际采样过的训练规则中选取。','',
             '颜色频率基线直接根据可见事实选最常见颜色，忽略链接和查询；它可能明显高于 25%。查询错配则保持编辑状态不变，把查询循环置换后仍对原标签评分。只有超过这些捷径并表现出查询依赖，才更支持关系组合能力，不能仅凭高于随机准确率判断。','',
             f"进一步的数据级捷径检查：在新规则/新查询上，直接取起点颜色为 {100*s_quality['validation']['heldout_combinations']['zero_hop_lookup_accuracy']:.2f}%，始终只走一步、忽略所请求步数为 {100*s_quality['validation']['heldout_combinations']['one_hop_lookup_ignoring_requested_hops_accuracy']:.2f}%。这些基线直接获得可见结构事实，用来揭示任务捷径，不是同容量的神经状态读出器。它们再次说明高于随机准确率不足以证明多步组合。",'',
             'S 的历史通路诊断（每种子 32 个未见规则，不更新参数）：','',
             '| 检查点 | 颜色词上的历史梯度质量 | 最后 16 tokens 上的梯度质量 | 只置换颜色词时最大概率变化，均值 |','|---|---:|---:|---:|']
    for step in ['0','2048']:
        points=[sensitivity[str(seed)]['checkpoints'][step] for seed in range(3)]
        lines.append(f"| {step} | {100*mean([r['mean_color_gradient_mass'] for r in points]):.6f}% | {100*mean([r['mean_last16_gradient_mass'] for r in points]):.4f}% | {mean([r['mean_probability_change_color_permutation'] for r in points]):.3g} |")
    lines+=['','这里保持原始 SSM 不变，只对显式历史 embedding 通路求梯度/置换颜色词。规则全文 94 tokens 完整可见；末尾 16 tokens 是通用任务说明。这个诊断衡量该通路对事实的敏感度，不检验原始 SSM 究竟保留了多少信息；弱敏感度是当前读取链路的瓶颈线索，不能据此否定 S 目标或证明必须增加工作区。详见 S-history-sensitivity.json。','',
             '## 4. P：未来预测、困难负样本与捷径','',
             'FrozenEncoder 为同一冻结 Mamba 的最终归一化隐藏表示，独立编码每个 64-token 未来片段后均值池化。特征使用训练集统计做逐坐标标准化再 L2 归一化。每个正确未来配七个同文档且不与正确片段重叠的负样本，候选乱序；随机准确率 12.5%，均匀预测 CE=ln(8)≈2.07944。距离为 128、512、2048。','',
             '| 候选方式 | 独立无 Dream 头 | 联合头读原始状态 | 联合头读 1 轮 | 联合头读 8 轮 | 目标错配 |','|---|---:|---:|---:|---:|---:|']
    for mode,metrics in summary['P'].items():lines.append('| '+mode+' | '+' | '.join(f"{100*metrics[c]['accuracy']:.2f}%" for c in ['no_Dream_trained_head','0','1','8','permuted_targets'])+' |')
    lines+=['','同文档困难候选按距离分解：','',
             '| 未来距离 | 独立无 Dream 头 | 联合头读原始状态 | 联合头读 8 轮 | 仅距离训练先验 | 候选中心性 | CE：8轮−原始 |','|---|---:|---:|---:|---:|---:|---:|']
    for distance in [128,512,2048]:
        def values(condition,key):return mean([r[key] for r in p_rows if r['mode']=='within_document' and r['distance']==distance and r['condition']==condition])
        lines.append(f"| {distance} | {100*values('no_Dream_trained_head','correct'):.2f}% | {100*values('0','correct'):.2f}% | {100*values('8','correct'):.2f}% | {100*values('distance_only_train_prior','correct'):.2f}% | {100*values('candidate_centrality','correct'):.2f}% | {values('8','ce')-values('0','ce'):+.5f} |")
    lines+=['','对同一联合头，读取第八轮状态相对原始状态的差异，才直接衡量该头是否从编辑中获益。跨文档负样本更容易通过主题识别完成；它只作为捷径诊断，训练始终用同文档困难负样本。目标错配破坏输入与正确标签的关联。候选重复读出与三个距离不是完全独立样本，见逐距离 CSV。','',
             '“仅距离训练先验”从训练目标特征求每个距离的固定均值原型，Wiki/PG 各占一半，不读取当前历史或状态；“候选中心性”仅根据当前候选集合选中心向量，不读取状态，也不作为正式预测器。两者用于检查分布/采样捷径，后者不是合法的状态预测头。','',
             '特征塌缩检查：','',
             '| 划分 | 特征数 | 范数范围 | 有效秩 | 第一主成分能量 | 同文档/跨文档平均 cosine |','|---|---:|---|---:|---:|---|']
    for split,r in features.items():lines.append(f"| {split} | {r['features']} | {r['norm_min']:.6f}–{r['norm_max']:.6f} | {r['effective_rank_entropy']:.2f} | {r['top_principal_energy_fraction']:.3f} | {r['mean_same_document_cosine']:.3f} / {r['mean_cross_document_cosine']:.3f} |")
    lines+=['',f"目标向量自匹配：训练 {features['train']['candidate_sets_checked']} 组、验证 {features['validation']['candidate_sets_checked']} 组均为 100%；正负特征 cosine>0.999 的比例为 0。这个检查验证候选标签对齐和特征可区分性，不说明真实未来能由当前历史唯一确定。",'']
    lines+=['','P 预测输出也按距离单独检查，避免距离 embedding 本身掩盖跨状态塌缩：','',
             '| 状态/头 | 距离 | 输出有效秩，三种子均值 | 跨状态平均 cosine |','|---|---:|---:|---:|']
    for condition in ['0','8','no_Dream_trained_head']:
        for distance in [128,512,2048]:
            subset=[r for r in prediction_collapse if r['condition']==condition and r['distance']==distance]
            lines.append(f"| {condition} | {distance} | {mean([r['effective_rank'] for r in subset]):.2f} | {mean([r['mean_pairwise_cosine'] for r in subset]):.4f} |")
    lines+=['','相同 32 个验证前缀、逐距离匹配的真实目标特征对照：','',
             '| 距离 | 目标有效秩 | 目标跨样本 cosine |','|---|---:|---:|']
    for distance,values in matched_targets.items():
        lines.append(f"| {distance} | {values['effective_rank']:.2f} | {values['mean_pairwise_cosine']:.4f} |")
    lines+=['','这组对照与预测输出使用相同样本数和边界。预测向量较目标明显集中，提示部分输出塌缩/原型化；并非所有输出完全相同，也不能仅凭它断言瓶颈来自 Dream 而不是预测头。见 matched-P-target-diagnostics.json。','']
    lines+=['','![辅助任务与头控制](auxiliary-effects.png)','','图中虚线为独立训练的无 Dream 预测头，实线为联合头在同一已训练 Dream 不同轮次状态上的读出。辅助任务改善不等于真实 LM 改善。','',
             '## 5. 多轮计算与中间干预','',
             '| 组 | 1 轮 NLL | 2 轮 | 4 轮 | 8 轮 | 单轮方向、匹配八轮幅度 | 8−1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,values in summary['rounds'].items():lines.append(f"| {group} | {values['1']:.7f} | {values['2']:.7f} | {values['4']:.7f} | {values['8']:.7f} | {values['norm_matched_first_edit']:.7f} | {values['8']-values['1']:+.7f} |")
    lines+=['','这些是同一个训练为八轮的模型的提前读出，不是分别训练的单轮与多轮模型。轮数增加也增加累计编辑幅度，收益不能单独归因于循环记忆。','',
             '| 组 | 第四轮后干预 | ΔNLL，未来 512，相对完整八轮 | 最终状态差/原状态范数 | 下一轮编辑与原编辑 cosine |','|---|---|---:|---:|---:|']
    for group,interventions in summary['interventions'].items():
        for name,r in interventions.items():lines.append(f"| {group} | {name} | {r['delta512_vs_intact8']:+.7f} | {r['final_state_change_relative']:.5f} | {r['post_intervention_delta_cosine']:.5f} |")
    lines+=['','reset_state 只把 H 还原为最初状态，保留 GRU 记忆；reset_memory 只清空 GRU，保留 H；reverse_state 反转前四轮累计修改。随后继续原第 5–8 轮，不训练新模块。最终状态、下一轮编辑和 LM 损失的变化可证明计算依赖；只有同时带来有益预测，才支持有用中间计算的解释。','',
             '![状态轨迹与干预](state-trajectories.png)','','每组每种子还保存一个 Wiki 和一个 PG 历史的完整八轮 SSM、GRU 记忆与干预后最终状态，见 evaluation/*/state-trajectories.pt；其余文档保留逐轮范数与方向诊断，可由检查点及原始前缀缓存重建。','',
             '## 6. 连续 Wake–Dream','',
             'seed=0 在两本 PG validation 书上，从 512-token 历史起每 2048 tokens 重组，连续到 32K，书内不重置缓存。表内按整段相同位置比较无 Dream。','',
             '| 组 | 两本书平均 ΔNLL |','|---|---:|']
    for group,value in summary['periodic'].items():lines.append(f'| {group} | {value:+.7f} |')
    lines+=['','![连续运行](periodic-effects.png)','','周期曲线同时受具体文本变化影响；不把累计收益曲线变平称为状态动力学收敛。长程仅评价，没有长程梯度优化。','',
             '## 7. 实际预算与资源','',
             '| 组 | 每种子目标 tokens | 主优化器更新 | 独立头对照更新 | 实际 LM 续算 tokens | 训练步骤秒数均值 | 目标 tokens/秒 | 峰值分配显存 MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        rows=[r for r in resources if r['group']==group]
        lines.append(f"| {group} | {rows[0]['LM_target_tokens']} | {rows[0]['updates']} | {rows[0]['independent_control_head_updates']} | {rows[0]['LM_actual_forward_tokens']} | {mean([r['train_seconds'] for r in rows]):.1f} | {mean([r['LM_target_tokens_per_second'] for r in rows]):.1f} | {max(r['peak_allocated_MiB'] for r in rows):.0f} |")
    lines+=['','有效 batch 为一个文档、两个连续周期，128 个 LM 目标；每个周期的梯度长度仍为 64。S 每步另有八个共享查询，P 每步另有六个距离查询。独立无 Dream 头是额外对照，每种子各 2048 次独立更新，不更新 Dream；其计算包含在表内步骤耗时中，验证和保存耗时另在 resources.csv 的 wall 字段记录。训练与 LM 评价使用 GPU FP32，全部辅助读出统一使用 CPU FP32。相同主更新数和 LM tokens 不等于相同总 FLOPs。缓存准备、校准和弃稿成本在 preparation-accounting.json 中分开列出。','',
             f"每个种子累计监督 262144 个 LM targets，但唯一监督位置为 {resources[0]['unique_LM_target_positions']}，存在重复采样。PG/Wiki 各占一半更新，而 PG 训练书只有 16 本，PG 短前缀复用更多。resources.csv 同时报告唯一目标数和重复倍数；不能把累计 tokens 当作同等数量的独立数据。",'',
             '每组实际覆盖的文档数、唯一边界、逐步损失/梯度/耗时见 resources.csv 与 runs/*/training.json；原语料元数据和哈希见 data/manifest.json。旧结果完整性、配对日程、冻结检查和缓存连续性见 verification.json。','',
             '## 8. 解释边界与决策','',
             '最终研究判断见 [FINDINGS.md](FINDINGS.md)。必须分别回答：头是否学会辅助任务、编辑状态是否优于原始状态、多轮是否优于早期读出、干预是否改变后续计算。不能以其中一项代替其他三项，也不能仅以辅助准确率声称 Dream 学会有效中间计算。','',
             '本轮同时扩大了训练文档覆盖，故相对 v0.1 的变化混合了数据覆盖与预算因素；B0 本轮固定语料上的 512→1024→2048 曲线可用于分析预算趋势。只有三个训练种子、16 篇 Wiki 和两本长书；数据按原划分隔离但不是全量 benchmark。未探索 S+P，未实现 W1/W2，未运行 128K。','',
             '完整目标、数据生成器、采样、负样本、权重校准和命令见项目根目录 [WSR_V02A.md](../../WSR_V02A.md)。原始 CSV、PNG/SVG/PDF 和 results.json 与本报告并存。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Report, results, CSV and PNG/SVG/PDF figures written',ROOT,flush=True)


if __name__=='__main__':main()

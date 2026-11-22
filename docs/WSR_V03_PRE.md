> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR-v0.3 前置：不对称读取与多轮工作区

本轮独立研究小型循环工作区，**未加载或修改 Mamba、Direct Dream、StateReader**，不接入语言模型。源码为 `wsr/workspace_model.py`、`workspace_tasks.py`、`workspace_experiment.py`；产物为 `artifacts/wsr-v0.3-pre/`。

## 最终采用的任务

只读信息库有 16 个槽，每槽一个二进制事实 `b_j`。可见查询给出 8 个地址与 8 个权重，权重固定包含四个 1 和四个 2。最终答案：

\[
y=\sum_{k=1}^{8} w_k b_{q_k},\qquad y\in\{0,1,\ldots,12\}.
\]

- **连续新读**：8 个地址均不同。
- **返回重读**：地址顺序为 `a,b,c,d,a,e,f,g`，7 个不同地址，第一项在第五轮再次使用，权重可以不同。

任务要求再次利用旧事实，但不证明任何算法都必须再次物理读取；网络也可能在工作区缓存它。这正是硬屏蔽可否由内部记忆补偿的一部分问题。

所有组均只得到事实库、查询地址与权重；没有部分和、参考步骤值、正确工作区状态或辅助损失。答案头仅访问最终 W，没有完整事实库旁路。

结构提供当前查询地址/权重的固定游标，**不要求模型自己发现下一地址**。事实值被 one-hot 后经共享可学习编码器读入；GRU 自己学习计数更新，没有硬编码乘加到 W。这项实验不能证明自主内容寻址或通用程序推理。

## 四组模型

工作区均为 16 维，原始信息库只读，固定 8 轮。读取分数由当前地址嵌入与 W 的可学习投影共同产生。

\[
s_k=\frac{(e(q_k)+R W_k+t_k)K^T}{\sqrt{16}},\quad
v_k=\sum_j a_{k,j}\,\mathrm{ValueEncoder}(b_j),
\]
\[
W_{k+1}=\mathrm{GRUCell}(v_k+\mathrm{WeightEmbedding}(w_k)+t_k,W_k).
\]

| 组 | 读取机制 | W 之外的机制状态 | 参数量 |
|---|---|---|---:|
| A | `softmax(scores)` | 无 | 2,733 |
| B | A + 可学习轮次向量 `t_k`，影响读取及更新 | 无 | 2,861 |
| C | `softmax(scores − lambda * coverage)` | 16 维软 Coverage | 2,733 |
| D | 已读槽的读取分数置为负无穷 | 16 位已读掩码 | 2,733 |

B 的 8×16 轮次参数在公共参数之后创建，初始全零；所有组公共参数逐位配对。C 的 `lambda=1`、`rho=0.9` 在本任务试验前固定：

\[
c_0=0,\quad a_k=\mathrm{softmax}(s_k-\lambda c_k),\quad c_{k+1}=\rho c_k+a_k.
\]

历史 attention 不 detach；C 可以重新读取，不擦除库中的内容。D 每轮标记 attention 的 argmax，后续硬屏蔽该槽；不是把所有获得非零软 attention 的槽一律屏蔽。构造器支持可调 `penalty/rho`，最终另做冻结 C 的推理期 lambda 扫描，未将其冒充重新训练的模型。

## 数据与预算

- 训练 8,008 个独立事实库，两个任务族各 4,004 个。
- 验证同池、验证留出组合、测试同池、测试留出组合各 520 个实例，两个任务族各 260 个。
- 每个面板的 13 类答案严格等频，均匀与多数类基线均为 `1/13 ≈ 7.69%`。
- 跨所有面板的事实库逐一去重。完整查询（地址序列 + 权重序列）的 SHA-256 bucket 用于划分：训练/同池为 0–5，验证新组合为 6–7，测试新组合为 8–9。
- **同池不意味着每个具体查询已在训练中出现**；它表示相同的允许查询集合。留出组合则由 bucket 保证与训练、验证组合不交叉。最终报告据此使用“同查询池新实例”。
- 每个主实验 Adam lr=1e−3，clip=1，batch=64，**512 次更新**；CPU FP32、单线程，共 4 组×3 配对种子，393,216 个最终答案 draws。每种子完全相同的样本日程。
- 最初分配的日程上限为 1,024；仅依据 A-seed0 在 512 步约 88% 的训练/验证表现，正式预算定为 512，保留学习空间，并在种子扩展及最终模型测试前记录 `budget-decision.json`。旧上限协议保留在 `protocol-initial-ceiling1024.json`，当前 `protocol.json` 为正式预算。每个 run 的原始 `planned_updates=1024` 表示上限，实际 `updates=512` 是正式预算。
- 32 条固定样本检查运行 1,024 步，达到 100%；它不计入主对照更新数。模型测试面板只在 12 个主配置全部完成后评价，没有按测试或最佳验证检查点替换统一终点。

## 功能检查和任务筛选记录

较困难任务未通过普通 A 的跨实例学习检查，较简单任务又达到天花板，因此在扩展种子前做了有限筛选。它们均完整归档，**不作为机制有效性证据，也不与主实验直接比较**：

| 归档目录 | 观察 | 处理 |
|---|---|---|
| `wsr-v0.3-pre-fourstep-pilot` | 双关系四跳：A 训练98.34%，新规则验证26.37% | 保留为内容寻址挑战，未扩种子 |
| `wsr-v0.3-pre-small-pool-pilot` | 链接—选择索引—查表：A 训练80.18%，验证26.56% | 保留，未扩种子 |
| `wsr-v0.3-pre-pointer-large-pilot` | 16,384实例并检查查询嵌入量级后，A 验证25.98% | 保留，仍未发现完整寻址链路 |
| `wsr-v0.3-pre-affine-pilot` | 已知地址的模4变换，A/B/C/D 仍近随机25% | 不将失败归因于读取抑制机制 |
| `wsr-v0.3-pre-count4-control` | 4轮计数 A/B/C 到100% | 保留为功能正对照；主任务扩至8轮加权计数 |

数据/难度筛选没有读取这些预试验的最终模型测试表现。前四项说明本轮不能回答自主寻址的收益；最后一项验证相同最终答案训练链路能学会计算。主任务调整不是给工作区增加中间监督。

## 评价和因果分析

1. 训练、同池新实例、留出查询组合，分连续新读与返回重读报告 CE/准确率。
2. 推理 1/2/4/6/8/10/12 轮，训练始终 8 轮。额外轮复用最后一个地址/权重，B 复用最后一个训练轮次编码；目标仍是原 8 轮答案，因此超额轮数不是等价任务或随机轮数训练。
3. 保存各轮 W、attention、Coverage，以及 top-1 重读率、不同槽数、软 overlap、熵、参考地址 attention。
4. 第4轮的 evidence 清零/跨样本替换，高 attention 与低 attention 槽的内容替换；该轮后 W 清零/替换，保持其他状态一致。
5. 翻转最后一个独有槽或第一个/重复槽的事实位，按其总权重重新计算反事实答案；原始数据另存不覆盖。这些事实仍在原来的二进制值空间。
6. 冻结模型后，用训练 W 拟合逐轮线性 ridge 部分和探针，在留出面板评价。额外部分和标签仅用于事后分析，不回传工作区。
7. 第4轮后，从同族、相同首位事实的其他实例中，选择同/不同部分和的 W 捐赠者。使用接收实例的剩余事实继续，比较原答案与 `原总和 − 接收部分和 + 捐赠部分和`。捐赠者的分析标签只用于干预选择和评分。
8. C 关闭 Coverage、途中清零 Coverage、推理时改变 lambda；D 关闭屏蔽，均与原训练模型分开标注。

干预改变结果不自动代表有用；结合正确率、事实反事实跟随和语义状态替换判断。相同部分和替换若仍退化，也不能声称 W 是完全解耦的纯符号计数器。

## 复现与保护

```bash
.venv-lm/bin/python -m unittest discover -s tests
.venv-lm/bin/python -m wsr.workspace_experiment prepare
.venv-lm/bin/python -m wsr.workspace_experiment pilot
.venv-lm/bin/python -m wsr.workspace_experiment train
.venv-lm/bin/python -m wsr.workspace_experiment analyze
.venv-lm/bin/python -m wsr.workspace_experiment report
.venv-lm/bin/python -m wsr.workspace_experiment audit
```

完成的训练保留，状态/优化器可从 `last.pt` 继续；同名配置不允许静默改写。原 v0.2a/b/c 产物保留原位，旧源码归档于本轮 `source-before`，预试验也保留源码/检查点/日志并单独做哈希保护。没有新的 Mamba 训练、独立工作区接入、随机轮数训练或辅助监督退火。

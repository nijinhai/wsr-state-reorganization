> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR-v0.3：无查询预计算的训练动力

本轮为独立静态规则沙盒。只训练新建的 A/B/C/D 与 B-budget，不加载 Mamba、旧 Dream、StateReader，不覆盖 v0.2b 或 v0.3 前置实验。

## 固定任务与划分

6 个节点，每个节点一条非自环出边、一个颜色；3 种颜色，每个规则实例各出现两次。查询为起点 `s` 和步数 `h=1..4`，最终答案为沿出边走 h 次后的颜色。输入只有出边和颜色的 one-hot，不含路径、中间节点、传递关系或答案表。生成器另存最终答案作监督。

训练/验证/测试分别取 1024/128/128 个不同出边图；每个图的颜色做三种全局循环置换，得到 3072/384/384 个实例。所有颜色变体留在同一划分，出边图跨划分不重合；每个具体查询的三类答案严格等频。均匀、总体多数类基线均为 1/3。起点颜色、固定一步与节点多数颜色另做诊断。

24 个 `(s,h)` 中，`(s+h)%3 != 0` 的 16 个组合用于训练；剩余 8 个用于留出组合评价。每个起点和每种步数都在训练中出现。验证与测试同时报告训练组合和留出组合；“已见组合”不代表见过该规则实例。

## 状态与统一读取头

W 为 `[6,32]`，每个起点有一个可选择的输出位置。查询阶段只取 `W[s]`，使用按 `h` 选择、跨起点共享的线性颜色分类器：

`logits = W[s] @ head_weight[h] + head_bias[h]`

没有查询阶段隐藏层、循环、事实库旁路或重新访问规则。A/B/C/B-budget 的读取头结构和参数量相同，D 也沿用此头。单元测试人为构造能够正确读取全部目标的 W，验证接口表达能力；这些构造不进入训练。

起点可索引位置和步数分类器是显式结构先验；W 的各维无事实槽或各跳答案对齐监督。容量足以表达所有答案，但不规定内部必须形成答案表。

## 模型与公平性

- A：将每个节点的原始出边/颜色 one-hot 固定填零至32维，直接读取；仅训练答案头，无预计算网络。
- B：全局原始规则线性初始化 W，执行一次全局两层 SiLU MLP 残差更新与 LayerNorm；隐藏宽度128。
- C：与 B 完全相同的初始化及更新模块，共享参数执行2/4/8轮；C1与B的前向、初始化、优化与样本日程完全相同，复用B结果，不伪作独立运行。
- B-budget：保持 W 与读取头不变，单次更新隐藏宽度1024；匹配C8的预计算稠密线性MACs，参数量更大，另行记录。
- D8：查询 one-hot 在初始化和每轮更新前提供，其余容量及最终头沿用；每个查询独立运行8轮。

每次 B/C 更新都能同时访问**全部原始规则和全部 W**，没有局部邻接传播、每轮走一步或固定分工限制。B 可以一次计算所有组合，C任意一轮也可以如此。LayerNorm与残差为共有工程结构，无Coverage、轮次编码、Anytime Loss、中间标签或新记忆机制。

## 训练与执行顺序

CPU、FP32、单线程；Adam lr=1e−3、clip=1；每更新16条规则，每规则8个不同训练查询，共128个最终答案，均值CE。A/B/C每规则只执行一次预计算；D为每个查询分别计算。正式2048更新、3配对种子，各模型相同规则与查询日程。公共头及B/C核心初始化配对。

先验证32条固定实例的B/C8/D8过拟合，再运行各组seed0的512更新预实验。预实验仅判断接口、优化和可学习性，**不以C胜出筛任务**。通过后保存决定，再原样续训seed0并扩展其余种子。保留全部曲线、原始梯度范数、检查点及耗时；主测试集模型评价在21个正式运行完成后才执行，统一使用2048终点。

本轮只有这一任务配置；如发现实现错误，修复及失败轨迹另存记录，不静默替换已运行模型。若B/C同时达到天花板，或C没有额外收益，如实报告，不加入新机制修补。

## 评价与成本

分别报告训练/验证/测试、训练/留出查询组合和1–4跳准确率及CE。区分独立训练的C1/2/4/8与仅对C8改变推理轮数；后者不当作公平训练对照。

对同一规则抽取1/4/16/64个独立查询（有放回，允许超过24个不同组合总数）。B/C每次只预计算一次，D分别计算。报告稠密线性MACs及CPU实测时间，预热3次后9次中位数，batch16向量化；时间包含激活/归一化等，MACs不包括它们，也不当作精确硬件FLOPs。拆分预计算与轻量读取的解析成本。查询数量变化的抽样准确率不能解释为能力变化。

上述D成本按每次查询重新计算的指定协议记录，未启用重复查询答案缓存。由于总共只有24种不同查询，缓存可使D的重复查询成本降低；64查询的线性成本不代表D可实现的最优成本。

所有模型训练固定8查询共享监督；本轮不额外比较训练查询数，因此不声称已测定多查询训练的独立收益。推理时复用摊销与准确率分开判断。

执行节点/规则一致置换与颜色置换诊断；C保存逐轮修改范数/方向。若C显示稳定额外收益，再决定中间状态因果实验，当前不预先叠加随机轮数或其他训练机制。

## 复现和产物

源码：`wsr/precompute_model.py`、`precompute_tasks.py`、`precompute_experiment.py`；测试：`tests/test_precompute.py`。新产物：`artifacts/wsr-v0.3-precompute/`。旧artifact全量SHA-256记录在`preservation-before.json`，完成后复核；旧源码保存于`source-before/`，新运行源码另存`source-run/`。

```bash
.venv-lm/bin/python -m unittest tests.test_precompute -v
.venv-lm/bin/python -m wsr.precompute_experiment prepare
.venv-lm/bin/python -m wsr.precompute_experiment smoke
.venv-lm/bin/python -m wsr.precompute_experiment pilot
# 检查预实验并记录 pilot-decision.json 后
.venv-lm/bin/python -m wsr.precompute_experiment train
.venv-lm/bin/python -m wsr.precompute_experiment analyze
.venv/bin/python -m wsr.precompute_report  # 使用已有含 matplotlib 的分析环境
.venv-lm/bin/python -m wsr.precompute_experiment audit
```

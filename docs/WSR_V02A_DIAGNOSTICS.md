> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR-v0.2a 后续三项小诊断（2026-11-16）

新产物独立写入 `artifacts/wsr-v0.2a-diagnostics/`。官方 Mamba、全部原 Dream 和原 S/P 读出器冻结；不恢复 S 主训练，不启动 S+P、W1/W2。保留原检查点、报告与数据缓存，以 SHA256 审计。

1. 幅度：B0/P 2048 步、三个种子，固定 H8−H0 的方向，扫描 −0.25/0/0.125/0.25/0.5/1/1.5。使用两个 Wiki 文档在 512/2048 的边界、两本 PG 在 8192/32768 的边界。未来 2048 tokens 逐位置评分，同时汇总 64/512/2048。零与一端点保留原 FP32 状态并重算首个案例，验证与 v0.2a 原结果一致；其余端点复用原逐 token 结果。卷积/其余层状态与 anchor 时间对齐不变。没有周期性扫描，不从本轮 validation 选生产配置。
2. S 三级：实际可见文本事实→原始 SSM→Dream 后 SSM。S-seed0 原 reader 的冻结128维输出与保留 P 位置的随机投影分别作为信息读出；同时探查原历史 GRU context 和 token 位置保留表示。探针只学习四个颜色与四条链接，后续符号执行是信息可用性上界。1024训练规则/256未见规则；正则只由训练规则内部768/256选择。打乱标签、32样本拟合为控制。追加的新查询头只在冻结特征上训练，三种输入为原 reader/改写 reader/显式真实基础事实；各512步、batch16、8查询、lr1e−3。每头查询标签 draws 为原 S 单种子四倍，不能当作正式 S 的预算匹配重跑。
3. P 可预测性：冻结原未来编码器和特征统计，距离128/512/2048、温度0.1、八候选，七个同文档困难负样本互不重叠。输入为未编辑 SSM 的原 reader/位置保留投影，或最近64可见 token 的冻结编码。训练860前缀144文档，内部116/28文档选择ridge正则；验证32前缀18文档，不读取test。对照含均匀、训练均值、Wiki/PG平衡均值、历史特征持续、目标自匹配、同文档配对打乱和验证历史轮换（81.25%实际改变）。额外四个轻量InfoNCE头（两输入×两配对）固定512步、batch32、hidden128、AdamW lr1e−3/weight_decay1e−3。所有未来数据只用于读出器监督/评价。正对照把目标改成已可见片段以检查管线，不能作为未来预测收益。

Probe 标准化只读取各次拟合的训练输入统计。固定投影种子20261117，保留头与P通道位置，仅将N维128投影到8维；含24个头尺度后共12312维。Ridge使用输入维度归一化的Gram矩阵，候选正则0.001/0.01/0.1/1/10；S按训练内one-hot MSE，P按文档内留出的InfoNCE选择。最终模型种子/轻量头种子0；幅度扫描有三个模型种子。

下游 S 查询头和 P 可见目标控制是执行中根据失败现象追加的定位检查，不是预先验证过的正式新训练信号。全部数值、预算与局限见诊断报告和 `probe-budget.json`。没有通过本轮小探针的失败断言信息在理论上不存在。

```bash
.venv-lm/bin/python -m wsr.v02_small_diagnostics amplitude
.venv-lm/bin/python -m wsr.v02_small_diagnostics prepare
.venv-lm/bin/python -m wsr.v02_small_diagnostics S
.venv-lm/bin/python -m wsr.v02_small_diagnostics S-query
.venv-lm/bin/python -m wsr.v02_small_diagnostics P
.venv-lm/bin/python -m wsr.v02_small_diagnostics P-nonlinear
.venv-lm/bin/python -m wsr.v02_small_diagnostics P-visible
.venv-lm/bin/python -m unittest discover -s tests
.venv/bin/python -m wsr.v02_small_report
.venv-lm/bin/python -m wsr.v02_small_diagnostics audit
```

幅度与特征准备可恢复已有产物；不要并发启动相同组/种子的写入。S/query和ridge探针是确定性的小规模重拟合；InfoNCE头已有结果则保留。所有命令只向新目录写入。

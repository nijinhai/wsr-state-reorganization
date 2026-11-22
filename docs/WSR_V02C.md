> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR-v0.2c：层位与跨层协调

当前执行范围：**接口改造与第一阶段离线诊断**。不训练 Mamba 或 Dream，不添加 S/P、Coverage、W1/W2；先用结果决定单层探索预算，不直接扩大到多层或 24 层训练。

旧源码存于 `artifacts/wsr-v0.2c/source-before/`；旧实验代码快照、检查点和报告保留原位，并使用 `preservation-before.json` 校验。新产物仅写入 `artifacts/wsr-v0.2c/`。

## 状态接口

```python
from wsr.layer_edit import edit_layers_cache

config = {"edit_layers": [12]}  # 也支持 [5]、[19]、[11, 12, 13]
new_cache, layer_traces, original_states = edit_layers_cache(
    model, dream, cache, tokens,
    boundary=boundary, history_window=128,
    edit_layers=config["edit_layers"], rounds=8,
)
```

旧 `edit_cache(..., layer=12)` 和 StateDream 权重格式均保留。新接口默认 `[12]`，返回轨迹与原始状态均按层索引组织。层号从零开始；界面和报告另给出从一开始的自然层号。

- `cache_inventory` 逐层对照实际 mixer 参数检查 SSM/卷积形状。官方权重加载仍要求严格匹配；另外逐层核对官方 A_log、卷积及 out_proj 权重形状，并引用本地归档的官方 `mamba2.py` 缓存分配函数。
- 当前官方 130M 的 24 层实际均为 SSM `[B,24,64,128]`、卷积 `[B,1792,4]`。这是检查结果，不是新接口的先验假设。
- `Cache.edit_many` 在注入前验证全部层号、形状、精度和设备；保留未编辑层及所有卷积缓存的对象，保留 processed 计数，操作不消耗 token。
- 多层编辑从同一份原始缓存分别生成修改，再一次性注入；不会把前一层编辑误当成已经完成了一轮 Mamba 续算。
- 共享编辑器只接受与各层实际头数、state_dim 兼容的形状；不兼容时明确拒绝。接口也接受 `{layer: editor}` 映射以兼容异构层，当前仅测试该能力，没有训练独立多层编辑器。
- anchor 不变：缓存已处理 `boundary−1` 个 token；编辑后从最后一个可见 token 开始续算，不重放整个历史或复用编辑前 logits。

## 第一阶段面板与指标

使用同一官方冻结主模型、旧验证语料和未编辑完整缓存。两本 PG-19 验证书籍各取 512/2048/8192/32768 历史；两篇 WikiText 验证文章各取 512/2048。共 12 个历史状态 × 2 个预测窗口（64/512）× 3 个层位（6/13/20），共 **72 个梯度案例**。PG-19 的 2048 缓存此前未保存，故从该书 512 缓存自然续算至 2047，保存到本轮目录。

每个案例计算：

\[
g_\ell=\nabla_{H_\ell} L_{\mathrm{LM}},\qquad
s_\ell=\|H_\ell\|_2\|g_\ell\|_2.
\]

报告梯度 L2/RMS、状态 L2、相对 L2 敏感性，以及逐元素 `H*g` 范数。相对 L2 敏感性对应整层按同一个相对范数半径扰动的一阶最大变化量，不是逐坐标相同比例编辑，也不是训练难易排名。

\[
\Delta H=-\rho\|H\|_2\frac{g}{\|g\|_2},
\quad \rho\in\{10^{-4},10^{-3},10^{-2}\}.
\]

在相同原始缓存上独立评价每个半径；同时提供半径 `1e−3` 的正梯度与固定随机单位方向对照。每次均记录实际注入后的相对范数，避免 FP32 加法误差掩盖半径差异。

对第 13 层，加载旧 B0-seed0 的 512/1024/2048 检查点，仅作冻结推理。记录 8 轮总编辑与负梯度的余弦/角度、修改幅度、一阶预测及实际 NLL 变化。这是单个旧 Dream 种子的早中晚检查点，不是三种子证据。高维方向接近 90° 本身不足以判断失败，需要结合符号、投影和实际损失。

所有未来只进入离线损失与梯度分析，不进入 Dream。所有比较从相同的未编辑缓存出发；不叠加不同层的梯度干预。

## 内存工程与数值验证

首次 512-token 反传尝试使用 CPU pinned-memory 卸载，遇到 CUDA out-of-memory；失败日志保留在 `smoke.log`，原工程协议保留在 `protocol-before-memory-fix.json`。没有因此更新参数或缩短目标。

现在使用 `checkpointed_continuation_loss`：以 64-token 块做激活重计算，跨块同时传递全部 SSM 和卷积缓存，计算图不 detach，完整监督并回传 512 个 token。64 只是重计算块大小，不是梯度截断窗口。FP32 分段计算与一次运行允许舍入误差，不声称逐位等价。

测试包括：

- 默认 `[12]` 与旧状态编辑、8 轮轨迹及后续 logits 的逐位一致性；无编辑 logits 逐位一致。
- 官方模型完整前向与分段前向的一致性，报告实际最大 logits 和 NLL 差。
- 单层/三层注入、卷积缓存和未编辑层保留、不修改原缓存、不重复消费 token、未来变更不影响编辑。
- 每个选定 SSM 都能得到有限非零梯度，梯度能到达共享 Dream；主模型没有梯度或参数更新。
- 异构层缓存与匹配编辑器映射，不兼容共享编辑器明确报错。
- 小模型直接反传与重计算反传对照，以及真实官方模型 128-token 直接反传与两个连通重计算块的梯度误差对照。

## 复现

```bash
.venv-lm/bin/python -m unittest discover -s tests
.venv-lm/bin/python -m wsr.v02c_diagnostics interface
.venv-lm/bin/python -m wsr.v02c_diagnostics smoke
.venv-lm/bin/python -m wsr.v02c_diagnostics gradients
.venv-lm/bin/python -m wsr.v02c_diagnostics report
.venv-lm/bin/python -m wsr.v02c_diagnostics audit
```

诊断完成案例按独立文件保留，重复运行跳过已完成案例。`smoke` 只取第一个历史状态，但也测试两种完整预测窗口；没有任何训练步骤。

主报告：`artifacts/wsr-v0.2c/REPORT.md`；逐案例梯度与记录：`gradients/`；统一表：`state-gradients.csv`、`B0-alignment.csv`；接口结果：`interface/verification.json`。

后续单层训练仍需保持旧 B0 的文档采样日程、初始化种子、Adam lr=3e−4、8 轮、无破坏、每周期未来 64 token、每更新两个连续周期与周期间 detach。若完全保持旧条件，第 13 层优先复用旧检查点，不应重复训练一个同样基线；若日程变化则必须重跑配对基线。先完成 seed0 的 512 更新探索，再依据其学习曲线与跨历史评价决定是否扩至 2048×3，不依据离线 oracle 梯度收益直接扩大。

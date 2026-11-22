> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

时间尺度与基础能力实验

本轮比较四种初始化/训练配置，保持 SSM 递推、参数数量、输入输出接口一致，不训练 Dream。旧基础模型、诊断结果和配置保留。新结果位于 `artifacts/timescale-v1/`，完成后由报告工具生成完整表格、通道分布与静态科学图（PNG/SVG/PDF）。

| 配置 | 初始化 | 训练 |
| --- | --- | --- |
| original | 原始 log_a=0 | 直接混合任务训练 |
| slow | log_a=-3 | 直接混合任务训练 |
| multiscale | 四个 head 校准至约 2、8、32、128 token | 直接混合任务训练 |
| slow_curriculum | log_a=-3 | 由短复制/短计数/单映射到原挑战任务 |

每个 seed 的四组其他初始参数完全相同，记录 SHA-256。默认三个种子、每次 900 个更新、batch=16 个独立世界、Adam lr=0.002、梯度裁剪 1。同一个共享模型轮换学习复制、计数和冗余表示；不把独立专项模型的指标混称为一个共享基底的能力。不同世界重置状态，同世界原任务连续三轮传递 SSM 与卷积缓存。

课程的前 1/4 更新训练短复制、两步计数和单对映射；第二个 1/4 训练中等复制、四步计数和两对映射；后 1/2 训练原计数、原冗余表示及与其他组相同的复制分布。各阶段仍轮换三个任务。课程组匹配优化步数，没有匹配 token 数、监督答案数和挑战任务曝光次数；逐组实际预算保存在 result.json。

当前模型每个 head 的 8 个通道 × 8 个 SSM 坐标共享一个衰减率，只支持四组独立时间尺度。多尺度校准只修改 log_a，根据训练侧独立可见输入测得的 mean(dt) 设置：

```text
A_h = ln(2) / (target_half_life_h * mean(dt_h))
log_a_h = log(A_h)
```

因此目标值是校准流的有效半衰期，不是保证每种输入或每个 token 恰好相同。时间尺度仍可训练。`--targets` 可指定其他候选，不把这四个值固化为模型默认值。

训练前、1/3、2/3、结束时使用同一批独立探针世界观察每个 head 的完整分布。记录瞬时半衰期 `ln(2)/(dt_t*A_h)`、按累计对数衰减计算的有效半衰期、p10/p25/中位数/p75/p90 与极值、直方图、逐世界值，以及首个状态贡献真正衰减到 1/2 的时间。观察长度不足时记录右删失，不伪造长半衰期。

同时记录逐 token 状态 RMS、最大绝对值、写入 RMS、近冻结/强覆盖比例及非有限值；这些用于观察状态动态范围，不能单凭数值幅度认定发生“饱和”。同长度 64-token 复制的中性填充/随机别名干扰为配对输入，有助于分开长度与干扰影响。128-token 复制为未训练长度的外推，规则链为未训练的规则迁移挑战，分别报告。

运行与生成报告（新目录避免覆盖）：

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m wsr.timescales --directory artifacts/new-timescales
.venv/bin/python -m wsr.timescale_report --directory artifacts/new-timescales
```

绘图依赖可通过 `pip install -e '.[analysis]'` 安装。完整结果、初始校准、每轮训练轨迹与逐世界预测保存在 `<配置>-seed-<种子>/result.json`，检查点为同目录的 `base.pt`。四次探针的 JSON 记录逐 head 统计，对应 `.pt` 保留逐 token 原始数据。`metrics.csv` 与 `half-lives.csv` 用于导出比较。

同名实验默认拒绝覆盖；参数完全相同的 `--resume` 可跳过已完成运行，保留未完成目录并明确提示，不静默替换原结果。

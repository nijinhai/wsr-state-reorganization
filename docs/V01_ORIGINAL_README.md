> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR-v0.1：冻结官方预训练 Mamba2-130M 的状态重组

WSR-v0.2a 的 Direct Dream 预算与 S/P 训练信号实验见 [执行协议](WSR_V02A.md)，独立入口为 `python -m wsr.v02_pipeline`，产物在 `artifacts/wsr-v0.2a/`。下文记录 v0.1 的设置与复现。

当前基底为 [state-spaces/mamba2-130m](https://huggingface.co/state-spaces/mamba2-130m)，不再从零训练基础 SSM。原合成任务、时间尺度实验与结果保留在原路径，代码与文件哈希另归档于 `artifacts/legacy-2026-11-13/`。旧入口改为 `python -m wsr.legacy_experiment`；当前 `python -m wsr.experiment` 使用真实文本与冻结预训练模型。

本轮训练、对照和长程评价已完成，结果与图表见 [完整报告](../results/pretrained-v1/REPORT.md)，关键观察见 [FINDINGS](../results/pretrained-v1/FINDINGS.md)。

研究设计保持 A/D × T1/T2 × 无破坏/10% 遮蔽，共八组。A 为共享网络的重复残差改写；D 加入八轮内独立 GRU 记忆。两者均读取单层真实 SSM 状态和最近 128 个可见 token 的冻结 embedding，执行八轮。另训练 D/T1/无破坏/仅状态对照，并保留无 Dream、破坏后不修复的对照。

| 编号 | Dream | 训练 | 状态破坏 |
| --- | --- | --- | --- |
| 01 | A | T1 未来预测 | 无 |
| 02 | A | T1 未来预测 | 10% |
| 03 | A | T2 去噪→预测 | 无 |
| 04 | A | T2 去噪→预测 | 10% |
| 05 | D | T1 未来预测 | 无 |
| 06 | D | T1 未来预测 | 10% |
| 07 | D | T2 去噪→预测 | 无 |
| 08 | D | T2 去噪→预测 | 10% |

模型权重固定 revision `3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49`，分词器为固定 revision 的 GPT-NeoX tokenizer。全部 128,989,632 个原参数冻结；以 FP32 执行，保留原 24 层、全部状态维度、卷积、门控 RMSNorm 和残差结构。默认只编辑索引 12 的 SSM，形状 `[1,24,64,128]`，其他 SSM/卷积缓存不在 Dream 时修改。编辑网络沿状态维共享参数，额外循环不更新主模型权重。

当前可微后端使用官方方程的普通 PyTorch SSD 收缩，缓存更新为函数式操作；它不是之前的简化 SSM，也没有改变预训练架构。以固定 [官方 v2.2.5 源码](https://github.com/state-spaces/mamba/tree/v2.2.5) 的逐 token 完整模型和最小 SSD 定义独立验证。官方源文件与 Apache-2.0 许可证保留在 `artifacts/pretrained-v1/vendor/`。本机原生 CUDA 扩展未参与结果，不能把参考后端的验证说成原生 CUDA kernel 验证。

关键时间对齐：边界 `b` 有 `b` 个可见 token，缓存只处理了 `b-1` 个，最后一个可见 token 作为待处理 anchor。Dream 修改缓存后，输入 `tokens[b-1:b+L-1]`，预测目标为 `tokens[b:b+L]`。第一个未来 token 的 logit 因而在修改后重新计算。下轮直接延续得到的状态，不重复输入 anchor、不使用编辑前 logits。JSON 同时记录 `history_visible=b` 和 `state_history=b-1`。

## 环境与执行顺序

旧 `.venv` 保留；本机新环境 `.venv-lm` 使用经过实际验证的 PyTorch 2.6.0+cu124，在 GTX 1660 上以 FP32 运行。按本机环境复现：

```bash
python3 -m venv .venv-lm
.venv-lm/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-lm/bin/pip install -e '.[pretrained]'
.venv-lm/bin/python -m wsr.fetch_pretrained_assets
.venv-lm/bin/python -m wsr.pretrained_data
.venv-lm/bin/python -m wsr.verify_pretrained
.venv-lm/bin/python -m unittest discover -s tests
.venv-lm/bin/python -m wsr.experiment smoke
.venv-lm/bin/python -m wsr.experiment pilot
.venv-lm/bin/python -m wsr.experiment train --case 05 --seed 0
```

接口→冒烟→先导检查通过后，才允许 `train` 进入正式矩阵。先导门槛检查有限且非零的梯度、真实状态改变，不以单个短程损失改善筛选方案。每组重新初始化 Dream，使用同一预训练权重、相同未来预测文档/边界/破坏种子；T2 只在预测前额外增加八步去噪，其未来预测数据序列与 T1 对齐。

首轮工程预算：三个种子，各组 32 次预测更新；T2 额外八步去噪并沿用 Adam 状态；lr=3e-4，Dream 梯度裁剪 1，每次两个连续 Wake–Dream 周期。训练历史长度 512/1024/2048，干预间隔 256。每次只对未来 64 token 反传，周期之间截断梯度但保留缓存值；这属于短程优化，8K/32K 只作连续运行评价。检查点每四步保存，可按同配置继续未完成训练。

## 数据与评价

WikiText 使用 `Salesforce/wikitext` 的 `wikitext-103-raw-v1`，按顶层文章标题恢复完整文档，绝不把小标题误当新文档。准备 12 个训练文档，并从原验证划分按长度选定两篇不少于 4096 token 的完整文章；扫描时保留的额外文档见清单。PG-19 根据原始划分列表从官方 GCS 下载单本书，准备 4 本训练、2 本验证长书，保存最多 40K-token 的自然前缀。文档身份、文本哈希、源位置和截取长度均写入清单，并检查跨划分身份与精确文本重复。未读测试集作配置选择。

短程评价使用两个 WikiText 验证文档、512/2048 的历史边界；长程单次干预使用两个 PG-19 验证书籍、8192/32768 的历史边界。后续预测最多 2048 tokens，分别记录 1–32、33–128、129–512、513–2048 的损失及累计长度结果。周期性评价对同一本 PG-19 书连续运行至 32K，每 2048 tokens 干预一次，内部不重置缓存。长程首轮先报告 seed=0，短程报告三个种子；范围差异在报告中标明。128K 不阻塞首轮。

```bash
.venv-lm/bin/python -m wsr.pretrained_evaluate short --name baseline
.venv-lm/bin/python -m wsr.pretrained_evaluate short --name corrupt
.venv-lm/bin/python -m wsr.pretrained_evaluate short --name 05 --seed 0
.venv-lm/bin/python -m wsr.pretrained_evaluate long-single --name 05
.venv-lm/bin/python -m wsr.pretrained_evaluate periodic --name 05
.venv-lm/bin/python -m wsr.pretrained_evaluate rounds --name 05
.venv-lm/bin/python -m wsr.experiment train --case 05 --history-window 0
.venv-lm/bin/python -m wsr.pretrained_evaluate short --name 05-state-only
```

损失为 tokenizer 级自然对数 NLL，未转换成 PG-19 官方 word-level perplexity。周期性不同方案比较的是相同书籍、位置、分词和边界的配对损失；允许早期退化、后期改善。逐 token 损失、不同历史长度、逐轮修改幅度、逐 head 修改范数与代表性完整编辑轨迹均保留。轮次扫描使用同一个训练八轮的模型，不冒充独立训练的单次编辑器。

正式检查点与训练轨迹：`artifacts/pretrained-v1/formal/`；数据清单：`data/manifest.json`；验证：`interface-verification.json`；完整计划：`protocol-plan.json`；评价：`evaluation/`。旧报告 [DIAGNOSIS.md](DIAGNOSIS.md)、[TIMESCALES.md](TIMESCALES.md) 仅描述此前简化基底的实验，不代表本轮官方模型。

评价可按有界批次继续，已完成的 JSON 不覆盖：

```bash
.venv-lm/bin/python -m wsr.run_pretrained_evaluation --phase all --max-jobs 4
.venv/bin/python -m wsr.pretrained_report
```

报告绘图使用带 `[analysis]` 依赖的环境；本机沿用旧 CPU `.venv` 绘图，不修改新 CUDA 训练环境。最终检查见 `artifacts/pretrained-v1/final-verification.json`，长前缀与零修改数值检查见 `long-interface-verification.json`。

## 本地对话与续写试用

```bash
.venv-lm/bin/python -m wsr.chat_server
```

打开 http://localhost:7860 。默认使用 CUDA，提供对话历史、纯文本续写、温度与生成长度设置。这里加载原始官方预训练权重，不加载 Dream 编辑器，也没有指令微调；对话模式仅通过 `User:` / `Assistant:` 提示词模拟聊天。建议先用英文，生成内容可能不准确或继续编写对话。输入最多 2048 tokens，单次生成最多 256 tokens，同一时间执行一个生成请求。可通过 `--port` 和 `--device cpu` 修改启动配置。

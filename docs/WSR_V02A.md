> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR-v0.2a：Direct Dream 的训练预算、共享查询与未来特征预测

本轮延续官方冻结 `state-spaces/mamba2-130m`、完整 FP32 可微 PyTorch 后端。没有工作区或新增记忆槽。旧产物以原路径和 SHA256 清单保留；新产物独立写入 `artifacts/wsr-v0.2a/`。运行完成后以该目录的 `REPORT.md`、`verification.json` 为准。

固定 D 架构、索引 12 的单层 SSM、8 轮共享参数编辑、最近 128 个可见 token 的历史接口。无随机破坏，无去噪预训练。保持卷积缓存与其他 SSM 缓存；原 LM 参数全程 `requires_grad=False`，续算不包裹 `no_grad`。

## 预算与数据

B0、S、P 从同种子的同一 Dream 初始化开始，每组 2048 次 Adam 更新，学习率 3e-4，在 512、1024、2048 步保存完整检查点与优化器。先运行 seed=0，通过有限梯度、参数实际改变、验证损失有限的检查后才扩展 seed=1、2；不以是否改善 NLL 筛选种子。验证曲线另记录 0、32、128、256、768、1536 步。

每次优化器更新处理一个真实文档、两个连续 Wake–Dream 周期，各监督未来 64 tokens，共 128 个 LM 目标；周期间隔 256 tokens，周期间截断梯度但保持状态值。每步实际续算 320 tokens（128 个监督输入、192 个未监督 wake 输入）。冻结的历史前缀只计算一次并按 FP32 缓存，之后训练复用，缓存创建成本另计；不会复用 Dream 后的上层状态代替真实续算。

请求覆盖 128 篇 WikiText-103 原 train 文章和 16 本 PG-19 原 train 书籍，每文档最多六个 512–2048 范围的历史边界。所有正式组同种子使用完全相同的文档、边界和 LM 目标日程。每两个更新分别采样 WikiText 与 PG-19。数据准备保留完整文章/书籍的自然前缀与原划分，不拼接不同文档。原 validation 中另外选定 16 篇 WikiText 文章和原两本 PG-19 书籍。测试集不读取。实际数量以 manifest 和 bank/index 为准。

四篇固定 WikiText validation 文章的 512 历史边界和两本 PG validation 书的 8K 边界用于训练曲线；其余 WikiText validation 文章用于扩展评价。最终测试结论仍不由这些 validation 结果代替。所有 checkpoint 都按预定预算保留，不事后只选最好步数。

## 精确目标与梯度

`L_LM = mean(-log p(future_64 | modified_state))`，一个更新内两个周期再取平均。与 v0.1 一样，可见历史有 b 个 token，缓存只处理 b−1 个，最后一个可见 token 作为 anchor；编辑后重新输入 anchor，预测第一个未来 token，绝不复用编辑前 logits。主模型参数冻结，但从未来 logits 到注入状态和 Dream 的梯度保留。

三组分别是 `B0: L_LM`，`S: L_LM + alpha L_S`，`P: L_LM + beta L_P`。alpha/beta 分别由 16 个训练日程上的初始 Dream 梯度范数校准，目标为加权辅助梯度约为 LM 梯度的 0.3 倍；对 0.5x 和 1x 两个候选各做八步训练稳定性试验，选择有限梯度且中位加权梯度比不超过 1 的 1x，失败才退到 0.5x。如果半权重试验仍超过该比值，再根据实测中位比缩小到 0.3。校准不读取验证结果，试验权重不进入正式组初始化。具体测量、权重与试验保存在 calibration.json，正式训练中继续测量梯度大小与夹角。

### S：同一共享状态，多查询

生成四个地点的有向链接置换和各地点四选一颜色。历史只列链接和基础颜色事实；查询要求从某个起点沿链接走 1–3 步后回答终点颜色。共有 `24 × 4^4 = 6144` 个具体规则实例，固定随机划分 1024 个训练、256 个未见规则实例。

12 种“起点×步数”组合划为八种训练组合与四种未见组合。解码器对起点、步数分别编码，使留出的组合不依赖一个从未训练的整体查询 ID。对同一历史只调用一次 Dream；所有查询在共享状态形成后才送入同一轻量解码器。

`H_new = Dream(H, X)`，`L_S = mean_i CE(QueryDecoder(H_new, q_i), a_i)`。解码器只能读取状态与查询，不能读历史。真实 LM 更新之外，每步多采样一个规则历史并监督八个查询；不改变 LM 的 64-token 未来窗口。报告新规则/已见查询、已见规则/新查询、新规则/新查询三个评价。

### P：归一化未来特征的 InfoNCE

在距离 128、512、2048 的位置各取 64-token 真实未来片段。距离 1 指第一个未来 token。FrozenEncoder 为同一个冻结 Mamba 的最终归一化隐藏表示，对每个未来片段独立从零状态编码并均值池化。预计算过程在 `no_grad` 下执行；用训练特征的逐坐标均值与标准差做标准化，再做 L2 归一化，验证只沿用训练统计。

`z = stop_gradient(normalize(FrozenEncoder(F)))`。轻量预测头从最终 Dream 状态和距离标识预测归一化向量，使用 `CE(dot(z_pred, candidate_features)/0.1, positive_index)`，即八候选 InfoNCE。每个距离包含一个正确未来和七个来自同文档、互不与正确片段重叠的困难负样本；候选顺序随机。一个周期对三个距离取平均，再对两个周期取平均。真实未来及候选特征只进入损失，不作为 Dream 输入。

只监督第八轮状态；第 1、2、4 轮的特征预测作为分析读出。另评估跨文档负样本、候选特征错配、原始状态和独立训练无 Dream 头。记录特征范数、方差、有效秩、主成分占比，以及同文档/跨文档相似度，检查塌缩与主题捷径。

## 分开衡量预测头与状态编辑

S/P 各自附带独立的无 Dream 预测头：与联合头同初始化、相同辅助样本/候选顺序、相同更新次数和学习率，仅从未改写原始状态学习，独立优化，不向 Dream 传梯度。P 的第二周期控制状态也由未编辑的原始历史单独计算，不能使用第一周期 Dream 已改变的状态。额外控制头计算属于诊断开销，单独披露。

评价同时给出：独立无 Dream 头、联合头读原始状态、联合头读第 1/2/4/8 轮状态。前者衡量头本身学习，后两者检验同一个头是否从编辑状态获益；辅助准确率不能直接当作形成了有用中间计算。

真实文本报告 NLL 与不同未来距离损失。另在第四轮后分别重置 SSM、反转累计编辑、重置 GRU 循环记忆，继续原第 5–8 轮，记录后续编辑方向、幅度、最终状态和 LM 损失的变化。这些是因果干预诊断；造成影响不自动等于这种计算对任务有帮助。两本 PG validation 书籍另作 32K 连续周期评价，书内不重置。

预算按优化器更新和 LM 监督 tokens 匹配，辅助任务及其控制头带来额外计算，耗时与显存必须同时报告。训练仍仅有 64-token 梯度范围，长上下文评价不称作长程优化。

## 复现

```bash
.venv-lm/bin/pip install -e '.[pretrained,v02]'
.venv-lm/bin/python -m wsr.v02_data documents
.venv-lm/bin/python -m wsr.v02_data bank
.venv-lm/bin/python -m wsr.v02_aux S
.venv-lm/bin/python -m wsr.v02_aux P
.venv-lm/bin/python -m unittest discover -s tests
.venv-lm/bin/python -m wsr.v02_pipeline
.venv/bin/python -m wsr.v02_report
```

单组入口为 `python -m wsr.v02_train --groups B0 --seeds 0 --steps 2048`。已有 last.pt 按相同日程和优化器继续；已完成组保持不动。S/P 需要先完成 calibration。新文件均在独立目录，旧结果不覆盖。

## 2026-11-16 执行范围更新

按用户最新指令继续 B0/P 剩余种子及配对评价。S 的三个种子和辅助读出、历史敏感度诊断在暂停指令前已完成，全部保留；暂停新增 S 训练与扩展语言/连续运行评价。现有训练曲线和辅助结果仍如实报告。S+P 不启动，W1/W2 不实现。流水线入口现在只继续 B0/P；S 旧结果不覆盖。

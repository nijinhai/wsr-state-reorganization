> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR-v0.2b 结论

1. 原 MLP 可以拟合小样本，也可通过预算/学习率/容量改善完整训练集拟合；先前55%不是表达能力上限。但未见起点×步数组合仍差，不能只归结为优化不足。
2. 3092参数共享GRU，在oracle事实上仅凭最终答案，三个种子都学到1–3步组合和未见查询；3081参数匹配MLP仍未学会相同泛化。
3. 直接接回原始/编辑后reader，未见组合分别约54.6%/58.4%，能力只部分保留；额外事实监督的线性桥接可恢复约99.9–100%，这不是Dream收益。
4. 事后节点读出和同节点/异节点的跨规则状态替换支持有用的中间内容。但地点槽、起点地址和执行步数由实现提供；固定三次更新时泛化变差，不能称自主发现程序或独立工作区。

Mamba、Direct Dream与原StateReader全程冻结。所有旧结果保留；S/P和W1/W2未扩展。

[完整方法、曲线和逐步结果](REPORT.md) · 审计 (`verification.json`; local archival artifact, not distributed)

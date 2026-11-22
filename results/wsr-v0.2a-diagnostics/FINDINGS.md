> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# 三项诊断结论

1. **幅度：损害不只是过度改写。** B0 在 8K/32K 历史下缩到 0.125 倍仍退化，反向 −0.25 倍改善；短历史可有小幅正向收益。方向的历史依赖是下一步要解释的问题，不能只调小幅度就宣称修复。
2. **S：事实可读，查询组合未学好。** 原始/改写 SSM 的基础事实都可完全读出，既有 128 维 reader 也约 99.9%；新查询 MLP 即使直接获得真实基础事实，也只有约 52% 的未见组合准确率，未超过颜色频率捷径。显式历史 GRU 的弱敏感度是局部问题，主要瓶颈不能直接归结为信息缺失或需要工作区。
3. **P：尚未证实未来目标的泛化可预测性。** Ridge 和独立 InfoNCE 头在文档外验证的 CE 都未优于均匀基线；训练可接近完全拟合，打乱配对对照优势不一致。相同管线能预测可见片段，未来自匹配也正常，因此应优先审查目标/读出/数据规模的适配，而不是扩大 Dream 训练。

本次没有训练 Mamba 或 Dream；只拟合诊断头。S 正式训练保持暂停，S+P、W1/W2 未启动。单一探针种子、固定格式 S 和有限验证文档限制了结论，失败读出不等于理论上不可预测。

[完整方法、表格与图](REPORT.md) · 核验 (`verification.json`; local archival artifact, not distributed)

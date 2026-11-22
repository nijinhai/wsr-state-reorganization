# Project summary for applications

## One sentence

Investigated whether shared recurrent computation can reorganize frozen model state or prepare reusable representations, using differentiable Mamba-2 state interfaces, final-answer supervision, causal diagnostics and compute-matched baselines.

## 中文简历表述

**冻结状态空间模型的状态重组与可复用预计算研究（WSR）**

- 设计并推进冻结Mamba2-130M的状态重组实验，组织实现可微SSM截取、注入与续算，验证缓存一致性、下一token对齐及梯度路径，支持32K连续评价。
- 通过事实探针、特权输入、查询网络替换及72个分层梯度案例，区分信息保留、读取接口与组合计算瓶颈。
- 完成7配置×3配对种子、各2048更新的无查询预计算对照，控制查询读取预算和答案监督，分别分析准确率、参数效率及1/4/16/64查询复用成本。
- 发现多轮模型优于同参数量单次模型，但留出组合不及计算量匹配的宽前馈基线；保留负结果与实现先验，未将沙盒收益解释为大模型能力提升。

## Interview framing

The initial state-editing hypothesis did not produce stable language-model gains. Rather than increasing complexity immediately, the research decomposed the failure into information availability, readout learning and computation. Small oracle tasks demonstrated useful recurrent intermediate states; later compute-matched controls showed why those results do not imply an inherent advantage for recurrence. The resulting work demonstrates experimental engineering, failure diagnosis and careful interpretation.

Implementation and documentation were AI-assisted. Use verbs such as “designed,” “led,” “organized implementation,” or “implemented” according to the applicant's actual contribution. The repository does not claim autonomous reasoning, improved production LLM performance or a peer-reviewed publication.

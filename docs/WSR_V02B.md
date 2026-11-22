> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR-v0.2b：中间计算链路诊断（2026-11-17）

产物独立保存在 `artifacts/wsr-v0.2b/`。所有旧代码、检查点、报告与结果保留。Mamba、原Direct Dream及原StateReader均不更新；只训练新的小型查询网络。没有新的LM训练、S/P重设计、W1/W2或S+P。

## 数据与预算

保留原1024训练/256验证规则，八种训练查询与四种留出起点×步数组合。额外112条规则来自同一个原始6144规则空间：链接为四环、四种颜色互异，且与原训练和验证实例均不重合；只作验证诊断，不进入优化器。既有frozen reader特征复用v0.2a诊断缓存；新112规则用冻结官方Mamba前13层、原S-seed0 Dream/reader提取，先处理history[:-1]取得SSM再让Dream读取全部可见history，查询尚未提供。

第一阶段：原oracle MLP在32规则上训练2048更新，完整训练8192更新；lr为1e−3，额外3e−4、3e−3控制，另有紧凑hidden128容量控制。仅新查询头优化，逐步记录0/32/128/256/512/1024/2048/4096/8192。

主对照A原MLP、A'匹配MLP、B oracle循环、C raw reader循环、D edited reader循环，各3个配对种子、8192次Adam更新、lr1e−3、clip1、每步16规则×8查询。每种子使用相同规则日程与1048576个最终答案draws。普通MLP约9596名义参数（oracle填充对应6144个死权重）；匹配紧凑MLP3081参数；循环oracle3092参数；C/D均7220参数，额外128→32适配器也只由最终答案训练。C/D从相同初始化开始，没有先加载训练完的oracle查询核心。没有中间节点损失、teacher forcing或事实标签辅助损失。

循环小样本检查4096更新，固定一轮/三轮控制各8192更新、seed0。所有阶段总共22个有界小查询配置；完整预算以runs及results.json为准。

## 结构预设与学到的内容

oracle事实按地点分四槽，每槽链接one-hot4+颜色one-hot4。训练的LayerNorm/Linear/SiLU编码到16维；训练的地点embedding同时给出起点向量和attention key。soft attention读事实值，GRUCell更新工作向量，参数跨1–3轮共享，最后再读一次事实、MLP预测颜色。步数embedding可供网络读取；主循环由查询h指定h次有效更新。三次潜在更新均展开，超出h的更新被mask。没有硬编码沿链接执行的矩阵乘法或正确中间节点。

起点地址、事实分槽、地点间共享与h→h次更新均是人为提供的结构；主对照不隔离各自贡献，也不匹配FLOPs。固定1/3轮的同结构网络仍提供步数embedding，用来检查直接给定循环次数的影响。不能把结构先验称为模型自己发现程序控制流。

## 分析与额外特权对照

按1/2/3步、已见/未见查询、原验证/四环集分别报告准确率、CE和macro accuracy。改变起点/步数时，仅在正确答案应变化的样本上统计变化与新答案正确率。

第一轮后清零、反转更新、跨规则替换工作状态。相邻四环规则常共享映射，因此补充同节点/不同节点的跨规则捐赠状态，继续使用接收规则的事实进行后续计算，并对照相应反事实路径。节点标签只用于事后选择捐赠者与评分；不传入网络或用于训练。另在冻结网络上拟合事后线性节点探针。验证和四环的1/2/3轮状态、attention和logits保存在analysis，所有工作状态干预可从检查点复现。

C/D只部分成功后，用线性事实探针定位：额外监督训练规则的八项基础事实，再将预测事实输入冻结的已训练oracle查询核心。这是单列的特权事实桥接，不属于仅最终答案监督的C/D，也不计作Dream能力。正则沿用先前S事实探针的训练内选择；没有新Dream训练。

## 复现

```bash
.venv-lm/bin/python -m wsr.v02b_query stage1
.venv-lm/bin/python -m wsr.v02b_query stage2-seed0
# oracle-gate.json通过后；两个后续阶段可独立运行
.venv-lm/bin/python -m wsr.v02b_query stage2-more
.venv-lm/bin/python -m wsr.v02b_query stage3
.venv-lm/bin/python -m wsr.v02b_analyze
.venv/bin/python -m wsr.v02b_report
.venv-lm/bin/python -m unittest discover -s tests
.venv-lm/bin/python -m wsr.v02b_audit
```

已完成训练自动保留，不应并发启动同一run。未完成训练没有通用中途恢复接口；先保留其目录，再决定如何恢复，不用同名任务覆盖。门槛是oracle训练>=99%，原验证/四环的已见/未见查询均>=90%；若未通过，不运行reader扩展。

所有结果均为有限四节点、1–3步和指定规则分布上的验证诊断，不是长度外推或最终benchmark测试。报告明确区分初始化种子变化、固定旧Dream checkpoint、额外事实监督与新查询网络计算。

> Historical research record. This public export includes selected reports, numerical tables and figures; original checkpoints/caches are not distributed. Paths below describe the original experiment unless linked to a public file.

# WSR 当前完整实验架构

截至 2026-11-18，覆盖 v0.2a、v0.2b、v0.2c、v0.3 前置实验与 v0.3 无查询预计算沙盒。以下均为 Mermaid 文本图；各版本是不同阶段的实验视图，不是同时联合训练的一个大模型。

**最新边界：v0.2c 仅完成接口与离线梯度诊断；v0.3 前置及无查询预计算沙盒均完成三种子对照，均未接入 Mamba。** 软 Coverage 无稳定收益；新的 C8 优于同参数量单次 B，但未在留出组合上超过计算量匹配的 B-budget，不据此启动 Mamba 工作区训练。

## 1. 实验总览

```mermaid
flowchart TB
    X["可见历史 X<br/>真实文档 / 规则文本，分别采样"] --> M["官方预训练 Mamba-2 130M<br/>权重始终冻结"]
    M --> CACHE["完整运行缓存<br/>24 层 SSM + 卷积缓存"]
    CACHE --> H0["H0：第 13 层 SSM<br/>编辑层索引 12"]
    X --> E["最近 128 个可见 token 的嵌入<br/>不含未来与具体查询"]
    H0 --> DREAM["Direct Dream D<br/>共享参数，8 轮残差改写"]
    E --> DREAM
    DREAM --> H8["H8：编辑后的 SSM"]

    subgraph HIST["v0.2a：已完成的独立训练组"]
        LM["真实文本：注入 H8 → 冻结 Mamba 续算<br/>L_LM：未来 64 token 的 NLL"]
        S["S：另采样规则历史，运行同一 Dream<br/>共享状态 + 多查询 → 最终颜色 CE"]
        P["P：真实文本 H8 → 特征预测头<br/>未来距离 128 / 512 / 2048 的 InfoNCE"]
        GROUP["B0：L_LM<br/>S：L_LM + α L_S<br/>P：L_LM + β L_P<br/>三组分别训练，不是 S+P 联训"]
        LM --> GROUP
        S --> GROUP
        P --> GROUP
    end
    H8 --> LM
    DREAM -. "共享 Dream 权重；使用另一份规则历史" .-> S
    H8 --> P

    subgraph NOW["v0.2b：已完成的中间计算诊断"]
        O["正确基础事实 oracle"]
        RR["冻结原 StateReader<br/>读取原始 H0"]
        RD["冻结原 StateReader<br/>读取编辑后 H8"]
        Q["Dream 完成后提供查询<br/>起点 + 步数 h，h = 1 / 2 / 3"]
        A["A / A′：oracle + MLP<br/>原规模 / 匹配规模"]
        B["B：oracle + 循环查询网络"]
        C["C：原始 reader + 新适配器<br/>+ 循环查询网络"]
        D["D：编辑 reader + 新适配器<br/>+ 循环查询网络"]
        ANS["预测终点颜色<br/>仅最终答案 CE 训练新查询网络"]
        O --> A & B
        RR --> C
        RD --> D
        Q --> A & B & C & D
        A & B & C & D --> ANS
    end
    H0 --> RR
    H8 --> RD
    ANS --> TEST["评价与干预<br/>未见规则 / 未见查询组合 / 1–3 步<br/>工作状态替换、清零、强制循环深度"]

    subgraph LAYER["v0.2c：接口与离线诊断完成；无新训练"]
        SELECT["逐层核对完整缓存<br/>单层索引 5 / 12 / 19<br/>相邻三层接口 11 / 12 / 13"]
        GRAD["同一原始缓存上独立诊断各层<br/>未来 64 / 512 token 的状态梯度"]
        LOCAL["等相对范数扰动 + 旧 B0 方向对照<br/>72 个案例；不能当作 Dream 收益"]
        SELECT --> GRAD --> LOCAL
    end
    CACHE --> SELECT

    subgraph PRE["v0.3 前置：独立任务与工作区；不连接上方 Mamba 链路"]
        BANK["只读二进制事实库：16 槽<br/>8 项地址与权重查询"]
        WORK["16 维 W，共享 GRU，固定 8 轮<br/>A 普通 / B 轮次编码<br/>C 软 Coverage / D 硬屏蔽"]
        COUNT["仅最终加权计数 CE<br/>4 组 × 3 配对种子 × 512 更新"]
        CAUSAL["留出组合、读取轨迹与状态干预<br/>普通循环学会有用计算<br/>Coverage 无稳定额外收益"]
        BANK --> WORK --> COUNT --> CAUSAL
    end

    subgraph BLIND["v0.3：无查询预计算；另一项独立沙盒"]
        RULES["随机 6 节点规则<br/>基础出边 + 颜色，查询 1–4 跳"]
        PRECOMP["A 原始状态 / B 单次 / B-budget<br/>C 无查询共享更新 1 / 2 / 4 / 8 轮"]
        LIGHT["预计算后才提供多个查询<br/>按起点取 W + 按步数线性分类"]
        DIRECT["D8：先提供查询<br/>每个查询单独计算 8 轮"]
        COMPARE["仅最终答案 CE<br/>准确率、轮数与复用成本分别比较"]
        RULES --> PRECOMP --> LIGHT --> COMPARE
        RULES --> DIRECT --> COMPARE
    end

    classDef frozen fill:#e8eef7,stroke:#45658c,color:#172638;
    classDef current fill:#e8f4e9,stroke:#508351,color:#173519;
    class M,CACHE,RR,RD frozen;
    class A,B,C,D,ANS current;
```

图例：蓝色为冻结基底或原读取器，绿色为 v0.2b 新训练的查询模块与其监督。Direct Dream 在 v0.2a 可训练，在 v0.2b/v0.2c 完全冻结。v0.2b、v0.3 前置与 v0.3 无查询预计算各有独立的 A/B/C/D 编号，也不同于 v0.1 的 Dream 架构 A/D。

## 2. Direct Dream 与可微状态续算（原始第 13 层路径）

```mermaid
flowchart LR
    X["历史 X，共 b 个可见 token"] --> PREFIX["冻结 Mamba 处理 X 前 b−1 个 token"]
    PREFIX --> CACHE["完整缓存 C0"]
    CACHE --> H["抽出第 13 层 H0<br/>B × 24 × 64 × 128"]
    X --> EMB["最后至多 128 个可见 token<br/>冻结词嵌入，768 维"]
    EMB --> HIST["历史投影 768→64<br/>SiLU + 历史 GRU → 上下文 e"]

    subgraph LOOP["Direct Dream D：同一套参数重复 8 轮"]
        READ["当前 Hk / 固定逐头 RMS<br/>投影 128→64 + SiLU"]
        Z["全局均值读出 + e<br/>GRUCell 更新 zk → zk+1<br/>z0 = 0，64 维"]
        EDIT["逐行读出 + zk+1 + 头嵌入<br/>共享 MLP → ΔHk<br/>按 RMS 缩放"]
        ADD["Hk+1 = Hk + ΔHk"]
        READ --> Z --> EDIT --> ADD
        READ --> EDIT
        ADD -->|"下一轮"| READ
    end
    H --> READ
    HIST --> Z
    ADD --> H8["保存 H1…H8 与每轮修改量"]
    H8 --> INJECT["仅替换第 13 层 SSM<br/>其他层与全部卷积缓存原样保留"]
    CACHE --> INJECT
    INJECT --> ANCHOR["重新处理最后一个可见 token<br/>pending anchor = X 第 b 个 token"]
    ANCHOR --> FUT["冻结 Mamba 可微续算<br/>预测未来 64 token"]
    LABEL["未来真实 token<br/>仅监督 / teacher forcing"] --> FUT
    FUT --> LOSS["L_LM"]
    LOSS -. "v0.2a 梯度经冻结主模型回传到 Dream" .-> EDIT
    FUT --> NEXT["后续 Wake–Dream 周期<br/>传递编辑后的完整缓存"]
```

- 主模型为 `state-spaces/mamba2-130m`：24 层、模型宽度 768，128,989,632 个冻结参数。运行路径为载入官方权重的 FP32 可微 PyTorch 实现，不能等同于已启用官方融合 CUDA kernel。
- 编辑层的状态形状为 `[batch, 24 heads, 64 head_dim, 128 state_dim]`。8 轮共享编辑参数，历史编码 GRU 与编辑循环 GRU 是不同模块。
- 每个 Dream 调用都重置内部 `z`；可跨周期积累的是 Mamba 缓存。此 Direct Dream 分支未接入独立工作区或新增持久记忆槽；v0.3 前置的 W 是另一项独立实验。
- 缓存只处理到可见历史的倒数第二个 token。注入后重跑最后一个可见 token，才得到编辑后的首个未来预测，避免使用编辑前已算好的 logits。
- 未来文本不进入 Dream；续算时的 teacher forcing 不等于向 Dream 泄漏答案。主模型参数冻结，续算对注入状态的梯度仍保留。
- v0.2a 每更新包含两个连续周期，各监督 64 token；周期之间保留缓存数值、截断梯度。因此每更新监督 128 个 LM token，但单次反传范围仍为 64 token，不能称为长程梯度优化。不同文档或书籍之间重置状态。
- 上图保留旧检查点对应的 `[12]` 路径；v0.2c 将编辑层改为可配置，默认路径与旧路径逐位一致，详见第 5 节。

## 3. v0.2a：B0 / S / P 的实际训练信号

```mermaid
flowchart TB
    REAL["真实文档<br/>WikiText-103 / PG-19，按原划分"] --> RAW["冻结 Mamba → 原始状态 H0<br/>限定可见历史 E"]
    RAW --> DR["可训练 Direct Dream D<br/>8 轮，无随机破坏，T1"]
    DR --> HR["真实文档编辑状态 H8"]
    HR --> LM["注入 → anchor → 冻结主模型续算<br/>未来 64 token NLL：L_LM"]

    RULE["另采样一份规则文本历史<br/>4 地点、4 链接事实、4 颜色事实"] --> RM["同一冻结 Mamba<br/>规则状态 H0_s 与历史 E_s"]
    RM --> DS["同一组 Dream 参数<br/>该规则历史只运行一次 Dream"]
    DS --> HS["共享规则状态 H8_s"]
    Q["查询组 q1…qm<br/>仅在 Dream 之后提供"] --> QD["可训练 QueryDecoder<br/>StateReader + 查询 embedding + MLP<br/>不能访问原始历史"]
    HS --> QD
    QD --> LS["L_S = 多查询最终颜色 CE 均值"]
    AS["答案 ai，仅作监督"] --> LS
    RM -. "无 Dream 对照" .-> QS["独立训练相同结构查询头"]
    Q --> QS

    HR --> PRED["可训练 FeaturePredictor<br/>StateReader + 距离 embedding<br/>归一化 768 维预测特征"]
    F["未来 64-token 片段<br/>距离 128 / 512 / 2048"] --> ENC["冻结 Mamba 特征编码器<br/>片段独立编码，均值池化<br/>训练内标准化 + L2 归一化"]
    ENC --> TARGET["stop_gradient 目标特征"]
    TARGET --> LP["L_P：InfoNCE，温度 0.1<br/>1 正样本 + 7 同文档不重叠困难负样本"]
    PRED --> LP
    RAW -. "无 Dream 对照" .-> PP["独立训练同结构特征预测头"]

    LM --> OBJ["独立组目标<br/>B0：L_LM<br/>S：L_LM + α L_S<br/>P：L_LM + β L_P"]
    LS --> OBJ
    LP --> OBJ
    OBJ -. "只更新各组 Dream 与相应辅助头" .-> DR
    OBJ -. "S 分支共享的 Dream 权重" .-> DS
```

S 与真实文本 LM 使用不同历史、不同状态，只共享主模型和 Dream 参数；P 使用真实文本对应的编辑状态。辅助头无 Dream 基线单独训练，用于区分头本身学会任务与状态编辑的贡献。α、β 按训练内初始梯度量级校准，不是默认等权。S+P 未启动。

StateReader 对每个头的 SSM 做 RMS 归一化，逐行投影 `128→16`，沿 head_dim 取均值，拼接逐头 log-RMS，最终 `408→128`。原 S 查询 MLP 再拼接起点及步数 embedding；它没有访问全部历史的旁路。

## 4. v0.2b：最终答案监督下的循环查询计算

```mermaid
flowchart TB
    ORACLE["oracle 基础事实<br/>4 个地点 × 链接 one-hot 4 + 颜色 one-hot 4"] --> SLOT["4 个事实槽，每槽 8 维"]
    RAW["C：冻结 Mamba 原始 H0"] --> RC["冻结原 StateReader → 128 维"]
    EDITED["D：冻结旧 Dream 编辑 H8"] --> RD["冻结原 StateReader → 128 维"]
    RC --> AC["C 的新适配器 128→32"]
    RD --> AD["D 的新适配器 128→32"]
    AC & AD --> SLOT
    SLOT --> VALUE["共享可学习事实值编码<br/>LayerNorm + Linear + SiLU → 每槽 16 维"]

    QUERY["查询：起点 s，步数 h ∈ 1…3"] --> INIT["W0 = 地点 embedding(s)<br/>+ 步数 embedding(h)，16 维"]
    KEY["共享可训练地点 embedding<br/>同时充当起点编码与注意力 key"] --> INIT
    KEY --> ATT
    INIT --> ATT["soft attention<br/>由 Wk 查询四个地点槽"]
    VALUE --> ATT
    ATT --> EV["加权事实 evidence"]
    EV --> GRU["同一 GRUCell<br/>Wk+1 = update(Wk, evidence)"]
    INIT --> GRU
    GRU --> STATE["W1 / W2 / W3<br/>保存中间状态与 attention"]
    STATE -->|"未满 h 次，继续更新"| ATT
    STATE -->|"完成 h 次有效更新"| FINAL["再进行一次注意力事实读取"]
    VALUE --> FINAL
    KEY --> FINAL
    FINAL --> HEAD["concat(Wh, 最终 evidence)<br/>MLP 32→32→4"]
    STATE --> HEAD
    HEAD --> CE["仅最终颜色答案 CE"]
    CE -. "更新适配器与查询网络；不更新旧模块" .-> GRU
    CE -. "仅 C / D" .-> AC & AD

    STATE -. "仅分析，不作中间训练标签" .-> INTER["清零 / 反转更新 / 跨规则替换<br/>节点保持或改变的反事实干预<br/>事后线性节点探针"]
    STATE -. "另设固定轮数对照" .-> DEPTH["固定 1 / 3 轮训练对照<br/>强制 0 / 1 / 2 / 3 轮评价"]
```

图中 oracle、C、D 是互斥输入条件；三条输入不会相加或拼接。C/D 适配器输出虽按四槽排列，但没有基础事实标签监督，不预设其已经学会可解释事实。

| 条件 | 输入 | 本轮训练参数 | 参数量 |
|---|---|---|---:|
| A | oracle 事实填零至 128 维 | 原查询 MLP（含查询嵌入） | 9,596，含 6,144 个零填充对应的无效权重 |
| A′ | oracle 32 维事实 | 匹配规模 MLP（含查询嵌入） | 3,081 |
| B | oracle 四槽事实 | 事实编码、注意力键、共享 GRU、答案头 | 3,092 |
| C | 原始 H0 的冻结 reader 特征 | 新适配器 + 与 B 同结构查询核心 | 7,220 |
| D | 编辑 H8 的冻结 reader 特征 | 新适配器 + 与 B 同结构查询核心 | 7,220 |

每组独立训练；C/D 同种子从匹配初始化开始，不加载已经学好的 oracle 核心。三种子共享训练日程，每组主实验 8,192 次更新，每更新 16 规则 × 8 查询。使用原 1,024 训练规则、256 验证规则，8 种训练查询组合、4 种留出组合；另有 112 条未见四色四环规则只作诊断。

本轮 Mamba、旧 Dream 与原 StateReader 全冻结。C/D 均使用同一份旧 S-seed0、2048 更新的 Dream/reader 检查点；三个种子是新查询网络种子，不是三个 Dream 种子。

另有独立的特权诊断：`冻结 reader → 额外事实标签监督的线性桥 → 冻结已训练 oracle 查询核心`。它不属于 C/D 最终答案监督主链路，不能算作 Dream 的新能力。

结构中人为规定了地点分槽、共享地址 embedding 和 `h 步查询 → h 次有效更新`。网络学习的是软读取、状态转移与答案映射；没有硬编码符号链接执行，也没有中间节点损失。与 MLP 的对照同时改变了槽共享、注意力和循环结构，未匹配 FLOPs，不能将全部差异只归因于循环。

## 5. v0.2c：可配置状态层位与离线梯度诊断

```mermaid
flowchart TB
    C0["冻结 Mamba 的原始完整缓存<br/>逐层核对 SSM / conv 实际形状"] --> PICK["edit_layers_cache / Cache.edit_many<br/>默认 [12]；单层 [5] / [19]<br/>多层接口 [11, 12, 13]"]
    OLD["旧 Direct Dream 检查点<br/>所有参数冻结，仅推理"] --> PICK
    PICK --> ATOMIC["各层从同一原始快照生成修改<br/>一次性注入选定 SSM<br/>其余层、conv、processed 保持不变"]
    ATOMIC --> RUN["沿用 pending anchor<br/>冻结主模型可微续算，不重复消费 token"]
    RUN --> CHECK["默认 [12] 与旧路径逐位一致<br/>检查注入、续算与梯度连通"]

    C0 --> PANEL["离线面板：原始缓存独立分支<br/>12 个历史状态 × 2 窗口 × 3 层<br/>层 6 / 13 / 20，索引 5 / 12 / 19"]
    PANEL --> LOSS["未来 64 / 512 token 的 LM NLL<br/>历史 512 / 2048 / 8192 / 32768"]
    LABEL["真实未来只作离线监督<br/>不得输入 Dream"] --> LOSS
    LOSS --> G["g_l = 对该层 SSM 的损失梯度<br/>范数、归一化敏感性"]
    G --> STEP["沿负梯度按相同相对 L2 半径扰动<br/>1e−4 / 1e−3 / 1e−2<br/>另设正梯度与随机方向对照"]
    STEP --> EVAL["重新续算，测实际 NLL 变化<br/>每层每次均从原始缓存出发"]
    G --> ALIGN["第 13 层：比较旧 B0 的编辑方向<br/>seed0，512 / 1024 / 2048 检查点<br/>余弦、修改幅度、实际 NLL"]
```

实际检查后，官方 130M 的 24 层 SSM 均为 `[B,24,64,128]`，卷积缓存均为 `[B,1792,4]`。新接口仍逐层校验，不把这一结果硬编码为所有模型的假设；支持共享编辑器及匹配形状的逐层编辑器映射。相邻三层注入与梯度接口已经验证，**不代表三层联合训练或协调收益已经验证**。

诊断使用两本 PG-19 验证书的四种历史长度，以及两篇 WikiText 验证文章的 512/2048 历史，共 72 个案例。512-token 反传采用 64-token 块的激活重计算，跨块计算图不 detach，不能与 v0.2a 的 64-token 截断反传混淆。

结果：相对修改半径 `1e−4` 时，三层各 24/24 案例均能沿负梯度降损；更大半径下第 20 层更易退化。没有证据表明第 13 层明显不适合改写；旧 B0 后期的长历史退化仍在。上述梯度使用真实未来，只说明局部可改进空间，不能算作无未来输入的 Dream 推理收益。

**本阶段 Mamba 与 Dream 均为 0 次优化器更新。** L6/L20 的新训练、2048 步三种子扩展，以及相邻三层共享/独立编辑器与范数控制实验均尚未运行。

## 6. v0.3 前置：独立多轮工作区与不对称读取

这一分支未加载 Mamba、Direct Dream 或 StateReader。下图的信息库不是 Mamba SSM；工作区 W 不回写语言模型缓存。

```mermaid
flowchart TB
    BANK["只读信息库：16 槽二进制事实<br/>原始内容不擦除、不改写"] --> VALUE["共享可学习 ValueEncoder<br/>每槽事实值编码为 16 维"]
    QUERY["8 个查询地址及权重<br/>固定游标提供本轮地址 qk、权重 wk"] --> SCORE["scores：当前地址 embedding<br/>+ W 的投影 + 可选轮次编码<br/>与地址 keys 点积"]
    W0["可学习初始 W0：16 维"] --> SCORE

    subgraph POLICY["四种互斥读取条件，共享其余核心结构"]
        A["A 普通循环<br/>softmax(scores)"]
        B["B 轮次编码<br/>读取和更新加入可学习 tk"]
        C["C 软 Coverage<br/>softmax(scores − lambda × ck)<br/>ck+1 = rho × ck + attention"]
        D["D 硬屏蔽<br/>屏蔽历史 argmax 槽<br/>本轮再记录 argmax"]
    end
    SCORE --> A & B & C & D
    A & B & C & D --> ATT["attention 加权读取事实 evidence"]
    VALUE --> ATT
    ATT --> UPDATE["共享 GRUCell<br/>输入 evidence + 权重 embedding<br/>B 另加入本轮编码"]
    W0 --> UPDATE
    UPDATE --> WK["W1…W8<br/>保存逐轮状态、attention 和读取统计"]
    WK -->|"下一轮，固定共 8 轮"| SCORE
    WK --> FINAL["答案头只访问 W8<br/>13 类：加权和 0…12"]
    TARGET["最终答案 y = sum wk × bit[qk]<br/>无中间步骤监督"] --> CE["最终答案交叉熵 CE"]
    FINAL --> CE
    CE -. "更新工作区、读取模块及答案头" .-> UPDATE
    WK -. "冻结后分析，标签不回传工作区" .-> PROBE["部分和线性探针<br/>中间 W 清零 / 跨样本替换<br/>保持或改变部分和的语义干预"]
    ATT -. "因果与行为分析" .-> INTER["evidence / 槽内容干预<br/>事实翻转、权重变化<br/>必要重读与软 overlap"]
```

两类任务各占一半：**连续新读**访问 8 个不同槽；**返回重读**按 `a,b,c,d,a,e,f,g` 访问，第一项在第五轮再次使用。权重为四个 1、四个 2。模型可能把旧事实缓存在 W 中，因此任务要求再次利用旧信息，不证明必须物理重读。查询游标与地址由实现提供，网络学习软读取、状态更新和答案映射；未硬编码加权累加，也未要求模型自主发现下一地址。

| 组别 | 额外机制状态 / 参数 | 留出查询组合准确率，三种子均值 ± SD |
|---|---|---:|
| A：普通循环 | 无；2,733 参数 | 89.81 ± 2.99% |
| B：轮次编码 | 8×16 可学习轮次参数；总计 2,861 参数 | 89.04 ± 1.33% |
| C：软 Coverage | 16 维累计 attention；2,733 参数 | 89.55 ± 2.16% |
| D：硬屏蔽 | 16 位已读掩码；2,733 参数 | 64.17 ± 1.46% |

C 使用 `lambda=1、rho=0.9`，Coverage 不 detach，允许重读；D 只屏蔽每轮 attention 的 argmax 槽。四组均为 16 维 W、共享 GRU、固定 8 轮、仅最终答案 CE，没有 S/P 辅助损失。A/B/C/D 公共参数和样本日程按种子配对。

正式预算为每组 **3 个配对种子 × 512 次 Adam 更新，batch=64**，CPU FP32 单线程。训练集 8,008 个事实库；验证/测试各含同查询池新实例与留出组合两个面板，每面板 520 条。事实库跨面板去重，完整查询按哈希划分；13 类严格等频，随机及多数类基线均为 7.69%。同查询池不等于具体查询已在训练中出现。32 条样本过拟合检查达到 100%，不计入主实验预算；预试验失败任务与四轮计数正对照均独立归档，不作为机制收益证据。

主要结论与边界：

- 普通 A 形成了有用的中间状态：第 4 轮 evidence 清零后准确率降至 42.44%；改变部分和的 W 替换后，按捐赠部分和计算的新答案准确率约 72.24%。结合事实反事实与事后探针，支持中间计数信息参与后续计算，但不能声称 W 是完全解耦的符号计数器。
- C 改变了软 attention 分配，未稳定改善任务效果；C−A 配对平均差为 −0.26 个百分点。D 尤其损害重读任务，不能把读取多样性本身视为成功。B 也无稳定改善。
- 推理轮数扫描为 1/2/4/6/8/10/12，训练始终 8 轮。少轮省略查询、多轮重复最后地址与权重，目标仍是原 8 项答案；这不是等价计算预算对照，不能据此声称学会自主停止。
- 当前暂不接入 Mamba，不启动大规模 v0.3，也未叠加随机轮数训练或辅助监督退火。

## 7. v0.3：无查询预计算与极轻量查询读取

```mermaid
flowchart TB
    RAW["随机规则：6 节点，每节点出边 + 颜色<br/>基础 one-hot，不含路径或答案表"] --> A["A：原始节点编码固定填零<br/>无预计算网络"]
    RAW --> INIT["B / C：全局线性初始化 W<br/>6 个起点位置 × 32 维自由特征"]
    INIT --> B["B = C1：一次全局 MLP 更新<br/>隐藏宽度 128"]
    INIT --> C["C2 / C4 / C8：共享同一更新模块<br/>每轮读取全部规则与全部 W<br/>不限定一轮一跳"]
    RAW --> BB["B-budget：一次更宽 MLP 更新<br/>宽度 1024，匹配 C8 线性 MACs<br/>相同 W 与查询头"]
    A & B & C & BB --> W["同一规则只形成一次 W<br/>预计算不接收具体查询"]
    Q["此后提供多个未知查询 s、h"] --> HEAD["取 W[s]<br/>按 h 选择共享线性颜色分类器<br/>没有查询阶段隐藏层或循环"]
    W --> HEAD
    RAW --> D["D8：查询条件初始化及 8 轮更新<br/>每个查询独立计算"]
    Q --> D
    D --> DH["同结构轻量查询头"]
    HEAD & DH --> CE["仅最终颜色 CE<br/>每规则 8 查询，均值损失"]
    HEAD & DH --> COST["推理查询数 1 / 4 / 16 / 64<br/>准确率与成本摊销分别报告"]
```

规则/颜色均随机生成；三色平衡、跨划分出边图不重复；24 个起点×步数组合中 16 个用于训练、8 个留出。A/B/C/B-budget 的查询读取预算完全相同，接口能表达正确答案。工作区按起点索引是结构先验，但各维不接受事实对齐或中间结果监督。没有 Coverage、轮次编码、Anytime 或随机轮数训练。

7 个配置 × 3 配对种子，各 2048 更新；C1 与 B 精确等价并复用结果。新规则的训练/留出组合准确率均值：B 为 61.57%/43.42%，B-budget 为 63.52%/48.06%，C8 为 64.43%/45.92%，D8 为 63.45%/44.94%。C8 优于同参数量 B，但对计算量匹配 B-budget，训练组合略高、留出组合三个种子均更低；尚不支持多轮预计算的独特优势。

无查询预计算可以复用于多个查询，降低平均成本；B 和 B-budget 也具备这项收益。固定训练轮数由实现指定，不能称自主停止；本轮无自主寻址或逐轮分工的因果结论，不接入 Mamba。

## 8. 对照、干预与版本边界

| 范围 | 已有对照 / 分析 | 主要区分 |
|---|---|---|
| v0.1 归档 | Dream A/D × T1/T2 × 有无破坏，共 8 组；无 Dream、破坏不修复 | 循环记忆、去噪预训练、随机干预 |
| v0.2a | B0/S/P、独立无 Dream 辅助头、1/2/4/8 轮、不同未来距离、周期性 Wake–Dream | 主 LM 收益、辅助头收益、多轮与连续运行 |
| v0.2a 三项诊断 | 修改幅度 `H0+s(H8−H0)`；S 三级信息读取；P 特征可预测性与困难负样本 | 过度改写、信息与读取瓶颈、主题捷径 |
| v0.2b | oracle 与原始/编辑 reader；MLP 与循环；固定轮数；工作状态因果干预 | 查询优化、组合泛化、信息读取和有用中间状态 |
| v0.2c | 可配置单层/多层接口；层 6/13/20 的等范数梯度扰动；旧 B0 方向对齐 | 层位敏感性与局部可改进空间；未训练新层位或多层 Dream |
| v0.3 前置 | 独立任务 A/B/C/D；三配对种子；读取、事实及中间状态干预 | 普通循环的有用计算与 Coverage 的额外贡献；未接入 Mamba |
| v0.3 无查询预计算 | A/B/B-budget/C1–8/D8；统一轻量查询头、答案数量；复用成本 | 预计算收益、参数与计算量、多轮的额外价值；未接入 Mamba |

短中程真实文本训练与 8K/32K 长程评价分开记录；128K 是后续探索项。各版本保留训练/验证边界，最终测试集不用于调参；v0.2c 仅使用验证语料，v0.3 前置的任务测试在主配置全部训练完成后评价。v0.2b 之后没有新的基础模型或 Direct Dream 训练，也没有把独立工作区接入 Mamba。

四种循环必须分开理解：**Direct Dream 的 8 轮改写 Mamba SSM；v0.2b 查询网络在查询到来后组合事实；v0.3 前置按已给地址逐轮操作二进制库；v0.3 的 C 在查询到来前预计算随机规则。** 后三者的 W 都不回写 Mamba，两个 v0.3 沙盒均未从 Mamba 状态读取信息。

## 9. 代码与报告依据

- [Direct Dream、历史窗口与缓存编辑](../wsr/state_dream.py)
- [冻结预训练主模型与状态续算](../wsr/pretrained_model.py)
- [v0.2a 训练目标与分支](../wsr/v02_train.py)
- [v0.2a 实验协议](WSR_V02A.md)
- [v0.2b 查询网络、事实适配器与训练](../wsr/v02b_query.py)
- [v0.2b 中间状态分析](../wsr/v02b_analyze.py)
- [v0.2b 完整报告与结果](../results/wsr-v0.2b/REPORT.md)
- [v0.2b 冻结范围与复现说明](WSR_V02B.md)
- [v0.2c 可配置编辑层与缓存检查](../wsr/layer_edit.py)
- [v0.2c 层位梯度诊断与分块反传](../wsr/v02c_diagnostics.py)
- [v0.2c 协议、接口与阶段边界](WSR_V02C.md)
- [v0.2c 诊断结论](../results/wsr-v0.2c/FINDINGS.md)
- [v0.2c 完整报告](../results/wsr-v0.2c/REPORT.md)
- [v0.3 前置工作区与四种读取机制](../wsr/workspace_model.py)
- [v0.3 前置任务与数据划分](../wsr/workspace_tasks.py)
- [v0.3 前置训练、干预与分析](../wsr/workspace_experiment.py)
- [v0.3 前置协议与预试验归档](WSR_V03_PRE.md)
- [v0.3 前置结论与决策](../results/wsr-v0.3-pre/FINDINGS.md)
- [v0.3 前置完整报告](../results/wsr-v0.3-pre/REPORT.md)
- [v0.3 无查询预计算模型与轻量读取头](../wsr/precompute_model.py)
- [v0.3 随机规则、答案生成与划分](../wsr/precompute_tasks.py)
- [v0.3 训练、成本与评价](../wsr/precompute_experiment.py)
- [v0.3 协议与复现](WSR_V03.md)
- [v0.3 无查询预计算结论](../results/wsr-v0.3-precompute/FINDINGS.md)
- [v0.3 无查询预计算完整报告](../results/wsr-v0.3-precompute/REPORT.md)

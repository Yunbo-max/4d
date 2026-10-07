# 重新研究：修复几何时，怎样保留真实动作？

2026-10-06，基于 `Yunbo-max/4d@078bfa636d53c7e0a7258166400007ee16a7a597` 重新开启数学发现。任务仍是输入视频到拓扑一致的动态 mesh，frozen ActionMesh、原生 ActionBench、一张 2080 Ti、八小时窗口；目标是有实证价值的研究，不以写满方法模块为完成标准。

**这轮已经完成 20 个条件数学构造 → 逐项推导审查 → 全池排序选 15 个 → 条件代码与完整比较设计；没有新增候选实现或 GPU 实验。** 推导使用新版 research-autopilot 的误差拆解、KKT、概率分解、坐标变化、零空间、Schur 消元等操作。每条都有实际推导步骤、成立条件、方法表达、独特预测、反例／否定规则及最近对照，见 [20 张数学卡](evidence/math) 与 [全池排名](evidence/selection.json)。

2026-10-07 的继续工作已接受一份新的本地设置回传：127/127 软件检查通过，原始开发资产、原生环境、三条简单对照臂及评分请求已就绪；仍然没有运行官方评分器或候选实验。下一段可执行源码是[官方 ActionBench CLI 与忠实封装器的独立一致性回放](ACTIONBENCH_SCORER_PARITY.md)，但它被完整的来源支撑原生协议和预先冻结判据有意阻塞。最新机器状态以 [CURRENT.json](CURRENT.json) 和 [STEPWISE_PROGRESS.md](STEPWISE_PROGRESS.md) 为准。

20 条是待检验的构造与研究分支，不能称为 20 个原创贡献。经典数学工具明确归属；C01、C14 当前主要承担诊断／强简单对照，C11、C12 等有明显既有方法或工程工具风险。C01/C03、C06/C07 存在嵌套，共同底座与共享机制不算独立发现。若功能审查确认完全覆盖，应合并／降级并补充真正有根据的候选，不能用参数、seed、消融或更名补足数量。旧十项原型及负证据保留出处，不回填“数学先行”历史。

## 新的核心问题

值得检验的问题是：**当参考形状或对应关系不准确时，几何修复究竟去除了误差，还是一并削弱输入支持的动作？** 当前隔夜分数不能诊断其原因；它只提供开发期原生／静止对照。

新文献边界限制了泛泛的“注意力对应”“局部刚性”“时间平滑”“形状与动作分离”。[STAC](https://arxiv.org/html/2605.19786v1)已覆盖前三类；[Shape Flow Guidance](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/8876.pdf)也有差分锚定与 root-space 解耦。研究必须提出它们没有排除的具体解释，并在同信息／同预算强对照下检验，阅读范围和未完成审查见 [primary notes](evidence/primary-source-notes.json)。

优先的四条竞争解释是相关参考误差、动作保护子空间、多解平均以及相位／幅度不识别。它们并不要求同时成立；实验应允许强普通方法解决问题和新增分支被否定。

### C02：动作保护不能无限制

设几何修复给出的目标步为 d，W 正定，C 是明确选定的粗动作线性观测。由 KKT：

$$\delta^*=d-W^{-1}C^\top(CW^{-1}C^\top)^\dagger Cd,\qquad C\delta^*=0.$$

该表达保留的是 C 定义的特征，不是未知真实动作或全部 CD-M。若 C 满列秩，唯一可行更新是 0；若 d 已在零空间，保护操作没有作用。这两个边界决定了对照和失败规则，而不是仅增加一个 regularizer。

### C03：偏差相减需要足够相关性与可信标定

先把真实后续误差和可观察的 self-map residual 中心化。对 `e_t-B r`，矩阵风险的理想解为

$$B^*=C_{ta}C_{aa}^{-1},\qquad R(B)-R(B^*)=\operatorname{tr}[(B-B^*)C_{aa}(B-B^*)^\top].$$

实际方法还需拟合均值，均值误差会增加平方 bias；低秩、ridge 和数据分布变化也不能继承理想收益。由此得到跨资产标定能否 transfer 的可检验问题。C01 的固定 unit subtraction 是必要强对照，不能把两条嵌套构造算成两次独立机制证明。

### C05：多解均值可能缩小形状

任意归一化有限权重满足 `E||Y||²-||EY||²=tr Cov(Y)`；两个相反旋转分支的均值可收缩。这是合法反例，不证明真实输入普遍失败。候选改为有空间一致性项的 joint discrete label decision，必须与 independent top-1、局部均值和调温度均值使用同一 mode bank、lift 和搜索成本比较。实际 mode 接口尚未资格通过。

### C20：时序错位与动作不足可能不可区分

对小相位 `tau`，`Y(t+tau)=Y(t)+Ydot(t)tau+O(tau²)`。把相位和幅度分别放入 P、B 后，weighted Schur projection 可以消去未约束相位；active monotonicity 需要完整 QP。两类 span 重叠时，输入可能无法识别原因。该方法内部可调整相位，但 evaluator 始终评分原始时间，不修改 reference。

## 20 条及前 15 排名

下表的“入选”是条件研究优先级，所有条目均未测。

| 排名 | ID / 数学卡 | 构造的具体变化 | 角色 |
|---:|---|---|---|
| 1 | [C02](evidence/math/4d-math-20261006-c02.json) | W 度量下投影到动作观测零空间 | 入选：优先候选 |
| 2 | [C03](evidence/math/4d-math-20261006-c03.json) | 带均值、cross-covariance 和估计风险的矩阵修正 | 入选：标定待验证 |
| 3 | [C05](evidence/math/4d-math-20261006-c05.json) | 多解保留的空间联合离散决策 | 入选：接口待验证 |
| 4 | [C20](evidence/math/4d-math-20261006-c20.json) | endpoint-preserving phase / amplitude 分块推断 | 入选：合法目标待验证 |
| 5 | [C01](evidence/math/4d-math-20261006-c01.json) | `X+F_t-F_a` 与完整交叉误差条件 | 入选：诊断／强对照 |
| 6 | [C04](evidence/math/4d-math-20261006-c04.json) | 不确定动作梯度的 ellipsoid 最坏约束 | 入选：uncertainty 待验证 |
| 7 | [C08](evidence/math/4d-math-20261006-c08.json) | 路径分布的 endpoint KL bridge | 入选：支持与端点待验证 |
| 8 | [C10](evidence/math/4d-math-20261006-c10.json) | differential field 的 pinned 可积投影 | 入选：经典底座／待比较 |
| 9 | [C13](evidence/math/4d-math-20261006-c13.json) | 向量 group acceleration 稀疏先验 | 入选：与 Gaussian 比较 |
| 10 | [C14](evidence/math/4d-math-20261006-c14.json) | 在 frozen 共转坐标中修 residual | 入选：强简单对照优先 |
| 11 | [C06](evidence/math/4d-math-20261006-c06.json) | 对应运输双方 surface-area 边际 | 入选：守恒假设待验证 |
| 12 | [C07](evidence/math/4d-math-20261006-c07.json) | unmatched mass + 完整 native fallback | 入选：部分运输待比较 |
| 13 | [C15](evidence/math/4d-math-20261006-c15.json) | protected Q 保留、complement SVT | 入选：低秩假设待验证 |
| 14 | [C11](evidence/math/4d-math-20261006-c11.json) | proper polar 的 stretch-only 谱约束 | 入选：既有方法覆盖风险高 |
| 15 | [C12](evidence/math/4d-math-20261006-c12.json) | face-area quadratic 的首个可行步边界 | 入选：可能仅工程修复 |
| 16 | [C09](evidence/math/4d-math-20261006-c09.json) | SO(3) relative rotation 图同步 | 保留：已有强覆盖 |
| 17 | [C16](evidence/math/4d-math-20261006-c16.json) | anchor covariance / sensitivity 传播选择 | 保留：标定与成本高 |
| 18 | [C17](evidence/math/4d-math-20261006-c17.json) | `n_j ∝ A_j kappa_j^(2/3)` 查询预算 | 保留：旧 m07 近似覆盖 |
| 19 | [C18](evidence/math/4d-math-20261006-c18.json) | 相机 nuisance 的 weighted Schur 消元 | 保留：合法输入与识别未闭合 |
| 20 | [C19](evidence/math/4d-math-20261006-c19.json) | `K=JU` 输出度量双路系数优化 | 保留：旧 controllability/m09 重合 |

## verified 的精确范围

| 环节 | 当前证据 |
|---|---|
| 条件数学卡与逐项审查 | 20/20；同上下文 Codex 自审，不是独立／人工 theorem 审查 |
| 有限代数核查 | 48 项通过，覆盖 20 构造；没有使用 benchmark 样本或 scorer，不是一般证明或效果实验 |
| 全池排名及 top-15 | `selection_verified=true`；全 20 行有五项理由和当前内容绑定 |
| 条件代码／比较规格 | 15 份；完整步骤→拟定函数、臂、端点、边界和依赖 |
| 新候选实现 / code_verified | 0 / 0 |
| 冻结设计 / design_verified | 0 / 0；numeric margins、真实 split、代码和成本待补齐 |
| 完整候选 native 比较 | 0 |
| 确认成功 / 正式失败 | 0 / 0；全部待验证 |

核验器实际执行退出 0，见 [执行记录](evidence/verification-execution.json)和 [派生报告](evidence/verification-report.json)。checker 核查的是内容绑定及已写明的语义审查；它不自动证明数学、审查者身份、创新性、科学准入或效果。有限代数核查脚本只属于证据检查，不能计为新方法 code。

## 接下来怎样推进

先闭合上轮的原始证据和强简单对照。仓库报告 16/16 原生／静止 pairs 完成，但 published `completion-verification.json` 实际仍是早期 1/16 快照，raw mesh、逐臂 scorer/receipt、attempt/env 与完整 telemetry 尚未回传。现有统计不足以确认某个自然几何／动作失败机制，也不足以证明新增方法必要。

当前 [research-autopilot SKILL.md](https://github.com/Yunbo-max/Research_Autopilot/blob/main/SKILL.md) 要求：

> “Natural Gate 0 PASS and IPCG CONCURRENT before implementation or Gate A freezing.”

因此本次不将条件设计升级为已准入候选实现／GPU 队列。该限制来自当前 skill，具体证据缺口是最终原始核验、strongest simple qualification 和功能撞车／重要性审查；它不是另设用户许可。补齐后按各分支必要性编码和验证，未准入分支保留数学推导与待验证状态。

后续可直接审查 [代码设计](CODE_DESIGN.md)、[逐方法实验与八小时窗口](EXPERIMENTS.md) 和 [15 份机器规格](specs)。一张卡按完整多臂单元校准，保留 30 分钟核验与收尾；没有实测成本前，不承诺八小时完成 15 项。只有真的完成原生强对照、独立确认和 E04，才在逐方法表中填写成功或失败。

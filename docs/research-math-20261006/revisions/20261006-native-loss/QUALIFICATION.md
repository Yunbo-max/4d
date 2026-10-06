# 下一窗口：资格实验与新方法实施的边界

状态：条件资格计划，**未冻结、不可直接派发候选实验**。固定任务为原生 ActionMesh → 16 帧固定拓扑 mesh → 原始 ActionBench 全评分；不扩成自造编辑 benchmark 或轨迹 point-cloud 替代任务。

## 先从完整证据定位自然机制

1. 在原 GPU 主机运行证据导出；带回所有归档块、manifest 和终态日志。分别读 geometry、self-map、材料对应、动作方向、时间相位、失败与成功例；不能仅根据 CD-M 高的三个已选例推断普遍机制。
2. 使用原作者 scorer 的完整预算，核对 original/stationary 两臂的原始输入和分数；trusted replay 与哈希检查分开记录。若原始 cache 或完整评分不合格，先修复资格，不承认新方法有效。
3. 对原始输入、GT 和 source/weights 做合法访问与 exposure 审查。此前 8+16 UID 共 24 个已暴露；剩余 104 个不自动是独立 fresh data，需要家族／重复与既往访问核查。
4. 从全开发 cohort 的成功、失败和边界定位竞争解释。开发结果可以否定研究分支；确认 split、数值判据、方法和强对照选择必须在确认前冻结。

## 强简单对照的真实状态

| 对照／代码入口 | 当前可用性与允许用途 | 资格要求 |
|---|---|---|
| 原 ActionMesh + stationary：`research_overnight/generation.py` | 历史实际报告 16 pairs；完整 raw 尚未带回 | 终态闭包与 unchanged native replay |
| 官方 full scorer：`research_census_eval.py` | 完整 16 帧、100k surface、10k ICP、24 rotations、200 iterations、seed44 包装；无需新 scorer | 原来源、I/O、源版本和实际全执行证据 |
| 普通时间 Gaussian | 需要资格验证完整 mesh adapter，尚无当前全比较 | anchor 固定、同拓扑／原时间；开发选择 sigma；全 native 三指标与总成本 |
| 全局刚体／共转坐标修复 | 经典 Procrustes/坐标变换，是 C14 的强基线角色 | 无 inference GT/camera；与普通 Gaussian 同目标、同信息、同预算；合法原 mesh 输出 |
| 普通局部 ARAP／Laplacian 修复 | 需要同一 mesh 目标下的完整 adapter，旧工具存在不等于资格通过 | 明确其自带全局平移不变性；C02 不能靠只保留全局 centroid 声称额外作用 |
| 固定 unit self-residual subtraction（C01 baseline） | 有数学规格，尚未当前资格验证；不当作原创贡献 | 同 Stage-II 完整上下文、时间条件、query、normals；验证 direct coordinate 语义及精确 anchor；全 native 比较 |
| STAC / SFG | 原文强相关，功能重叠限制创新叙事；未在本环境资格验证 author implementation | 若官方接口可用，使用原作者路径及合法输入／完整 scorer；代理实现只能称 proxy，不冒称论文复现 |

当前 GitHub `actionmesh/research_math/run_controls.py` 明确运行 constructed benchmark/demo；不进入 native qualification。`research_census_motion_controls.py` 是两个旧 UID 的 point-trajectory 诊断，复用另存 transform，文件本身也不声称与历史完整 evaluator bitwise 相同；它不代替上述完整 mesh 多臂比较，也不能作为新候选成绩。相关确切源码身份与阅读范围见 [baseline-source-audit.json](baseline-source-audit.json)。

## C02 的普通对照已有什么保证

ARAP 原作者的差分能量见 [Sorkine & Alexa 2007, Eq. (3)/(7)](https://igl.ethz.ch/projects/ARAP/arap_web.pdf)。其 edge energy 对所有顶点同时加同一个平移不变。我们据此推导一个具体对照边界，而不把它声称为作者 handle-constrained solver 的一般结论：

$$J(Z)=\frac12\sum_i m_i\|Z_i-Y_i\|^2+\lambda E_{edge}(Z),\quad m_i>0.$$

在逐帧、无硬空间 pin 或额外 absolute/temporal 项的自由问题中，沿全局平移求一阶条件得 $\sum_i m_i(Z_i-Y_i)=0$。因此任何满足该条件的驻点已保留质量加权全局质心；C02 若只加同一质心约束，不能据此声称增加动作保护。

跨部位连接的边能量一般不对每个 patch 的独立平移不变，所以 patch/part centroid 保护仍是不同的条件问题，但也未证明真实动作更好。硬 pin、相机项或时间耦合会改变上述条件，须审查实际完整目标。由此把“是否需要新增保护观测”变成明确对照，尚不判定 C02 实测成败。

## 三条优先机制怎样被否定

| 分支 | 必须先建立的自然事实 | 能排除“加模块就涨分”的区别性比较 |
|---|---|---|
| C02 动作保护投影 | 普通几何修复在全 cohort 的某个合法粗动作观测上有系统偏移；该观测与输入动作有实际联系 | same d/W；普通修复、保留全局 centroid 的普通修复、明确 patch/part 动作观测零空间投影。若粗观测已由普通 ARAP 保留或 projected step 为零，则新增必要性不成立 |
| C05 多解联合决策 | 真实 native mode 接口有多峰与有意义的空间连续性；不是人为制造两个旋转分支 | 同 mode bank/lift/信息/搜索成本的局部均值、调温度、independent top-1、joint spatial decision。若单峰，或普通 top-1 已解决，则不支持联合决策必要性 |
| C20 相位／幅度分块 | 未改变 scorer 的原时间上，合法输入 evidence 能区分 timing 与 amplitude；相应 spans 不完全重合 | 原输出、普通时序修复、只调幅度、只调内部相位、联合约束解；active monotonicity 用完整 QP。若不可识别，则不以重排 evaluator 时间掩盖 |

C03 排第 7；需先建立 development decoded-query 材料对应与跨家族标定。既有数学反例是损失关系的边界，不能当作“自然长尾存在”或 native 干预结果。所有分支最终以最强可执行简单对照和完整原生三指标检验，失败、OOME、超时和 missing 不从分母消失。

## 八小时怎样安排

证据导出是 CPU 文件核验，不占方法 GPU 预算，但实际 I/O 时长尚未测量。下一 GPU 窗口先做 qualified baseline 的一个完整多臂 pilot，记录 inference/query/solver/native score/packing 全耗时、真实峰值显存、源身份和 receipt。

每窗最多 28,800 秒，保留 1,800 秒核验／收尾。用完整 pilot 的保守耗时准入剩余完整单元，不用旧 original/stationary 平均 899 秒替代 4–9 臂的新成本。若最小可信单元放不下，跨窗口继续，不降低16帧或原生抽样预算。当前不声称八小时能完成15项。

在 Natural Gate 0 PASS、最强简单对照和 IPCG 准入后，才实施选中的必要分支、运行 software/native interface 检查，冻结 Gate A 和 fresh confirmation；进入 E04 后按预定 precision/效应/非劣判据逐方法填写 PASS/KILL/REVISE/INCONCLUSIVE。不可用的相邻 author code、未经验证的 mapping 或预算不足是具体 blocker，不是科学失败。

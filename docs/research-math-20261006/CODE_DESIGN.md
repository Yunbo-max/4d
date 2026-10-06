# 15 个入选分支的条件代码设计

这是完成数学池核验和 20→15 排序之后写的接口与实现设计，**没有新增候选算法实现，也不是已冻结的运行协议**。逐项计划见 [specs](specs)，实际状态由 [核验报告](evidence/verification-report.json)给出。新代码开始前仍需自然问题、强简单基线、功能撞车和重要性准入。

## 共同输入、输出和可用信息

共同输入包包含原始 16 帧 RGBA、原始时间 IDs、锚点帧 ID、锚点顶点 `X[V,3]` 和 faces、输入坐标变换、Stage-I 完整 latent/context、源与目标时间参数、锚点 normals，以及原生解码输出 `Y[T,V,3]`。每份内容记录 UID、seed、源码／权重／数据摘要。现有顶层调用只返回网格，不能假定上轮保存了 latent 或注意力。

读取的 `temporal_autoencoder.py` 默认 `prediction_mode=direct`，输出是绝对坐标。设其查询为 `F_t(X)`，位移只能由 `F_t(X)-X` 得到。所有方法保留原顶点身份、faces 和原始时间；输入的视频证据与生成对象不能错配。`F_a` 必须在同一 latent/context、源时间、query 及 normals 上查询。

候选输出包统一为完整 `Y'[T,V,3]`、参数与终止原因、全部 solver residual、操作日志及成本；方法需同时输出未修正版本以便配对。不可用输入、非有限数、无解或越界都记录，按冻结策略回退原生结果或标为执行失败。回退不从实验分母删除，也不将无法执行解释为科学失败。不得无记录截断坐标来维持数学恒等式。

GT 点云、GT 对应、官方 ICP 变换和评分输出只进入隔离的评价路径；C03 允许明确标注的开发标签标定，确认期推理不能访问它们。合法相机、motion surrogate 和 sparse modes 必须分别资格核查，不能以文件存在作为可用输入证明。

## 共同底座

拟建立 `NativeContext`、`ProtectionOperator`、`GeometryRepair`、`SparseModes`、`TransportSupport`、`LegalMotionSurrogate` 六类接口；它们目前是设计名，不是已存在模块。

- `GeometryRepair` 先用相同锚点的普通 ARAP／弹性修复，定义几何目标、权重、pins 和 solver stopping；所有相关臂共用。强输入视频约束变形另行资格核查。
- `ProtectionOperator` 的初始可检验版本：仅用锚点几何划分固定 patch，并在固定面积权重下保护 patch-centroid 的相邻帧速度；锚点 pins 先通过自由变量消元。所保护的是原生预测的粗动作，不保证真实动作或原生 CD-M。patch 数和定义在开发后冻结；若目标修复本来就在该零空间，C02 应没有作用。
- `SparseModes` 只能使用实际模型接口提供的有限候选。先核查 attention capture 是否改变原生输出、FlashAttention 行为和峰值显存；不能假定可保存完整 `(NT)²` 矩阵。
- `TransportSupport` 同时检查零面积、连通性、可行边和原顶点映射。稀疏化会改变解；列边际、slack 或端点残差不能只报告目标值。
- `LegalMotionSurrogate` 要绑定输入来源、权重、有效域和坐标／时间单位。C04、C20 无法得到合格目标时保持阻塞，不能输入 oracle 代替。
- decoder 维持原 FP16/low-ram 路径；small Gram、covariance、根和 CPU 稀疏解尽量用 float64，再记录输出转换误差。solver 不与两个大模型同时驻留 GPU。

## 拟定模块

| 分支 | 拟定主接口 | 数学到实现的重点 | 有意义的检查 |
|---|---|---|---|
| C02 | `project_protected_step(d,C,W,pins)` | 自由变量上的 KKT Gram 解；矩阵自由乘积；冗余约束 | 与独立零空间解比较、约束残差、满秩时零更新；真实 mesh 端到端 |
| C03 | `fit_transfer(dev_labels)` / `correct_transfer(Y,r0,fit)` | 资产级开发拟合均值及矩阵；冻结后应用；记录低秩／ridge 偏差 | 均值交叉项、奇异协方差、跨资产 transfer、确认标签隔离 |
| C05 | `select_spatial_modes(modes,graph,rotations)` | joint discrete objective、近似搜索上限、共用 lift | 独立标签与 joint objective、mode support、搜索成本、lift 是否引入收益 |
| C20 | `fit_phase_amplitude(Y,target,bases,time_ids)` | 小相位 Taylor；inactive 时 Schur；active 时完整 QP | 与完整 block solve 比较、monotonicity、端点、二阶残差、原时间输出 |
| C01 | `correct_self_map(X,F_a,F_t)` | 同 context 差分、严格锚点、完整 error cross term | direct mode 语义、零残差不变、越界和 clipping 记录 |
| C04 | `robust_protected_step(d,g0,S,r,epsilon,pins)` | ellipsoid support 转 SOCP；局部非线性 surrogate 验证 | 最坏方向、半径零极限、PSD、Taylor remainder、拒绝日志 |
| C08 | `endpoint_bridge(chain,endpoints)` | 稀疏 forward/backward、endpoint IPF、最终 path decision | 小状态枚举一致性、结构零不可行、两端残差、decision 非均值伪修复 |
| C10 | `integrate_field(G,W,h,pins)` | 降维 Poisson／QR；每个连通分量 gauge | 解与梯度残差、不一致场正交投影、失效 pins；不可声称无自交 |
| C13 | `repair_group_acceleration(Y,M,D2,pins)` | affine anchor 项、3D group dual ball、真实 timestep | primal/dual gap、off-diagonal metric 项、单位变换、非二次先验 |
| C14 | `repair_body_residual(Y,X,R,c,repair)` | 冻结合法估计的 rigid factors；只修 residual | 刚体不变性、估计不稳定、world/body 同预算、不得取 GT ICP |
| C06 | `area_balanced_transport(cost,a,b,support)` | log-domain scaling、双方 area marginals | 可行支持、两边残差、零面积映射、真实伸缩边界、稀疏误差 |
| C07 | `partial_transport(cost,a,b,gamma,epsilon)` | 双 slack、entropy 域、每个拒绝点 native fallback | 两边质量闭合、epsilon=0 极限、全分母、拒绝并非减少评分对象 |
| C15 | `protected_residual_svt(U,Q,lambda)` | Q 正交且包括实际 anchor basis；只对 complement SVT | `QU'=QU`、anchor、solver 最优性、真实细节动作被低秩删除的边界 |
| C11 | `limit_stretch(local_map,l,u,lift)` | 合格的 full-rank local map、proper polar、stretch spectrum | degenerate/reflection、frame completion、legitimate stretch、共用 integration |
| C12 | `admit_area_step(Y,d,faces,beta)` | exact quadratic、首次真正越界而非切触、所有面 connected interval | 独立 dense alpha 验证只作几何检查；安全步不变；大旋转与自交局限 |

每个分支在 [specs](specs) 中把全部推导步骤映射到拟定函数。这些是 planned mappings，没有 `code_refs` 或 `implementation-card`，因此 `code_verified=0`。

## 缓存、内存和成本

缓存只有在 source tree、输入／权重、seed、上下文、坐标／时间、生成参数和上游操作完全相同时才复用。decoder query 缓存还绑定 query/normals 和 source/target alphas。推理语义变化必须使受影响的下游缓存失效。确认缓存只能来自被冻结的确认协议。

一张 2080 Ti、一个 GPU worker。历史环境实际报告 22528 MiB；标准 11 GiB 配置未验证。完整轨迹与 sparse operators 可放 CPU，不能申请 dense `3TV × 3TV` metric 或完整 attention。记录 allocations、连续 telemetry 的实测峰值及缓存 I/O；单个显存采样不能作容量保证。

同时报告复用缓存的边际成本和从视频独立运行的总成本。共享上游只在总运行预算中计一次，但每臂独立使用成本需要披露。公平控制获得有用的计算／调参额度，不靠无意义额外调用凑相同时间。

## 准入之后的编码顺序

先补齐证据和强简单基线；再按自然失败与 functional delta 决定是否编码 C02 或 C03 等分支。低成本 C01 和 C14 优先用作资格核查／诊断对照，不据此宣称新方法。每次只改变该方法对象，禁止把前一个成功方法叠到下一项后将组合收益算给它。

先保留实现 source identity、数学步骤映射、软件检查与真实 mesh 接口输出；运行 `verify_methods.py --require code_verified --candidate <id>`。完整 native protocol 冻结及 design review 通过后才 dispatch。当前没有发布候选启动命令。

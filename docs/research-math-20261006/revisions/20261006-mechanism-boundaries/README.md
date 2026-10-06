# 第二轮重新研究：先检验方法是否必要、信息是否足够

分析接续 `036b15e99c5816bd353c56a6491c85de47003d3e`；提交前收到 `e874f6497318e791bec00ce2d9482aa35d93f5cf` 的 W0 回传，已审查并在该提交上继续，保留其所有产物及 macOS fixture 修正。任务仍为 frozen ActionMesh、原始 16 帧固定拓扑 mesh 和完整 ActionBench 三指标；单张 2080 Ti、八小时窗口。保留已有 20 个构造与历史，此次修订 C02/C05/C20 的数学卡、条件自审和全池优先级，再更新 15 份当前规格。没有新增候选或候选实现。

详细公式、证明条件、强对照和不可外推结论见 [THEOREMS.md](THEOREMS.md)。

| 分支 | 本轮真正新增的推导／认识 | 由此改变的实验问题 |
|---|---|---|
| C02 动作保护投影 | 保护后局部二次几何收益比为 $1-\rho$；匹配步长的普通缩步几何收益不小于投影 | 收益究竟来自方向保护，还是仅因为少修了几何？必须比较同 $d,W$、同 $W$ 步长的缩步 |
| C20 相位／幅度 | $\kappa=1-\sigma_{\max}(U^TV)^2$；噪声放大率 $1/\sqrt\kappa$；ridge 可逆不代表数据可识别 | 合法输入证据能否分离 timing 与 amplitude？正则项是否只是任意分配两者？ |
| C05 空间多解 | 非线性解码不与平均交换；原 decoder 不返回候选后验；localized surface blend 只保证在凸包内 | 先资格验证真实 bank 和共同 lift，再比较普通 surface-projected mean，区分均值离面与身份选择错误 |

这些不是原生效果结论。C02 的代价公式针对固定局部二次目标，C20 的角度结论针对解除 gauge 后的不等式非激活线性问题，C05 的非交换例子仅否定一般等价。没有新增 benchmark、GT、scorer 或自造 native 评测。

## 全池重排后的前 15

**C02、C01、C10、C13、C14、C04、C03、C20、C11、C12、C15、C05、C08、C06、C07。**

同一批 15 个身份，改变顺序；reserve 仍为 C09、C16、C17、C18、C19。所有 20 项比较与理由见 [selection.json](selection.json)。优先级增加了真实 native 输出接口、简单对照与明确代数边界的权重；不是效应预测、原创性通过或派发许可。C01/C14 具有明确经典对照角色，不能因排位靠前就包装为新贡献。

C05 从第 2 降至第 12、C20 从第 3 降至第 8、C08 从第 6 降至第 13。其公式仍是条件构造；native mode/target/path 接口未闭合，不能以这些未建立的对象作为先写算法的依据。没有因为缺少实验给它们判定科学失败。

## 当前状态

| 环节 | 计数与范围 |
|---|---|
| math_verified | 20；条件内容绑定及同上下文数学自审，不是独立正式证明或创新资格 |
| selection_verified | 全 20 排序、当前前 15；只是条件 shortlist |
| 当前候选代码／完整设计／结果／正式成败 | 全部 0；未升级为 code/design/results/verdict_verified |
| 本轮有限代数核对 | 40 个随机矩阵恒等式例、4 个角度噪声率例与两个边界；只是数学核对 |
| 上轮证据导出工具 | `036b15e` 版本的34工程测试与独立复审作为历史保留；新提交仅改 macOS fixture 路径，生产 exporter 未改，本轮不重新认证其测试记录 |

[batch.json](batch.json) 绑定当前数学卡与审查，[specs](specs) 中每个数学步骤都有计划映射和原生比较要求。新增 $\rho$、$\kappa$ 等仅作机制诊断；不作为替换 native CD3D/CD4D/CD-M 的评分规则。参数、ridge、proof witness 不计为独立方法。

## 实验如何继续

最新 W0 报告及 49,032-byte checkpoint 已读取、核对两文件 SHA256。归档40个成员中，16个 pair 的字节 hash 与各自 final receipt 一致，16组三指标也与既有 summary 完全一致。但该小包仍未包含 final receipts 引用的 **352 个阶段／raw 成员**，包括逐臂评分、mesh、latents；不能把这份 metadata checkpoint 当成完整原始回放。见 [W0 逐项审查](w0-feedback-review.json)。这是实质性终态记录进展，不是新候选效果。

接下来按[已提交的导出入口](../../../../scripts/research_evidence_20261006/README.md)在原 GPU 主机收齐完整 raw 闭包。现有环境仍没有 GPU 主机连接或这些完整原始产物，本轮没有启动 GPU 实验。原生回放、全开发 cohort 的成功／失败比较和最强简单对照资格，沿用[上轮资格计划](../20261006-native-loss/QUALIFICATION.md)。

本轮对该计划的增补是：C02 比较必须匹配 $W$ 步长并记录 $\rho$；C20 先建立目标、gauge、rank 和 $\kappa$，不得用 ridge 可逆代替信息充分；C05/C08/C06/C07 先建立真实 surface 候选及合法链路、输出 lift 和完整成本。共同 strong-baseline、信息／搜索预算、全原生评分、全失败分母与 fresh family confirmation 仍必需。具体 UID、数值效应／非劣界、预算和 precision 未冻结；这些条件规格不能直接作为运行队列。

八小时仍是 28,800 秒，保留 1,800 秒收尾；用 qualified baseline 的完整多臂 pilot 实测决定容量。不改变16帧或原生评分预算，也不把旧两臂平均15分钟当成新多臂成本。

候选实施继续遵循 [research-autopilot SKILL.md](https://github.com/Yunbo-max/Research_Autopilot/blob/main/SKILL.md)：

> Require current parent, Natural Gate 0 PASS and IPCG CONCURRENT before implementation or Gate A freezing.

目前这三项及强简单对照仍未闭合。继续数学／接口资格调查不需要新增用户许可；科学实现不会通过填写布尔值获得准入。

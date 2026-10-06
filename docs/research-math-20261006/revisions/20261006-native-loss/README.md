# 继续研究：先证明损失与方法的关系，再安排原生实验

2026-10-06，接续 `Yunbo-max/4d@e383e7850f4cf5205b23310e654fc755a2c16888`。方向仍是 frozen ActionMesh 的视频到拓扑一致动态网格，重点是“修复几何时如何保留输入支持的动作”，一张 2080 Ti、八小时窗口。历史运行的卡报告 22 GiB；未验证普通 11 GiB 版本。

这次保留上轮 20 个稳定构造及历史，不重新凑数。新增证据促成 **C03 数学边界修订、全池重排、15 份当前条件规格更新**，并实现上轮最终原始证据的导出入口。工程工具不计为研究方法实现。

## C03 的推导不能直接保证原生分数

原 C03 的结论是中心化、真实二阶矩下的平方点误差：

$$B^*=C_{ta}C_{aa}^{-1},\quad R_2(B)=\mathbb E\|e-Br\|^2.$$

ActionBench 使用非平方欧氏距离；CD-M 还包含首帧 ICP、表面抽样与首帧固定对应。两种目标的改善不能互相替代。下面是数学反例，**没有构造 benchmark、网格或 scorer，也不代表真实数据的出现频率**：

| 概率 | e | r |
|---:|---:|---:|
| 9/20 | 0 | 1 |
| 9/20 | 0 | −1 |
| 1/20 | 10 | 5 |
| 1/20 | −10 | −5 |

两者均值为零，$C_{aa}=17/5$、$C_{ta}=5$，故 $B^*=25/17$。精确求和得到

$$\mathbb E e^2=10\;\longrightarrow\;\mathbb E(e-B^*r)^2=45/17,$$
$$\mathbb E|e|=1\;\longrightarrow\;\mathbb E|e-B^*r|=27/17.$$

平方风险下降，非平方风险上升。令三维误差为 $(e,0,0)$，参考特征为 $(r,u,v)$，其中 $u,v$ 是独立对称 ±1，则 $C_{aa}=\operatorname{diag}(17/5,1,1)$ 正定，最优矩阵是 $\operatorname{diag}(25/17,0,0)$，欧氏风险相同。所有误差和特征再缩小十倍后均在 [−1,1] 内，风险顺序仍相反：平方 $1/10\to9/340$，非平方 $1/10\to27/170$。因此该反例不依赖奇异协方差或大坐标。

这否定的是“平方风险下降足以推出非平方风险下降”的一般推论，**不是 C03 的实测失败**；原平方恒等式仍成立。

## 在同一 C03 中推导更接近非平方距离的标定目标

先资格验证开发集的材料对应、坐标和 alignment；tracked GT 点不自动对应解码器 query vertex。它们未建立时，真实误差 $E_i$ 不可用，方法仍不可实例化。以下只是条件构造。

令 $\phi_i=(1,r_{0,i})$、$\Theta=[b,B]$，冻结非负权重 $w_i$ 且总权重大于零，$\tau>0,\lambda>0$：

$$L_\tau(\Theta)=\sum_i w_i\sqrt{\|E_i-\Theta\phi_i\|^2+\tau^2}+\frac\lambda2\|B\|_F^2.$$

由平方根的凹性，令 $s_k$ 为当前残差平方：

$$\sqrt{s+\tau^2}\le\sqrt{s_k+\tau^2}+\frac{s-s_k}{2\sqrt{s_k+\tau^2}}.$$

因此定义

$$q_i=\frac{w_i}{\sqrt{\|E_i-\Theta_k\phi_i\|^2+\tau^2}},$$
$$H=\sum_iq_i\phi_i\phi_i^\top+\lambda\operatorname{diag}(0,I),\quad K=\sum_iq_iE_i\phi_i^\top,$$
$$\Theta_{k+1}=KH^{-1}.$$

对任意向量 $(a,v)$，$\sum_iq_i(a+v^\top r_{0,i})^2+\lambda\|v\|^2=0$ 迫使 $v=0,a=0$，故 $H$ 正定。三维特征时，这是共享的 4×4 系统；截距直接吸收均值修正，不另拟合一组自由均值而重复计算。该公式针对普通点坐标；若引入各资产不同的 ICP 算子，必须另推导相应系统。

精确求解给出 $L_\tau(\Theta_{k+1})\le Q(\Theta_{k+1}\mid\Theta_k)\le L_\tau(\Theta_k)$，且同一 ridge 项下 $0\le L_\tau-L_0\le\tau\sum_iw_i$。秩截断、系数裁剪或不精确求解需要重新核查，不能继承该表达。当前有限代数检查见 [脚本](algebra_audit.py)及[输出](algebra-audit.json)。

这是经典 robust affine regression / IRLS / MM 思路的条件应用，归属与实际阅读范围见 [primary notes](primary-source-notes.json)。损失变体、ridge、平滑参数不是新的候选。冻结神经网络权重也不使开发标签标定变成 zero-shot。普通 covariance 为零只使原平方理想算子的矩阵部分为零；非平方站点方程使用径向加权交叉矩，不能沿用该条件。

**该目标仍是开发期点误差 surrogate；其下降不保证原生 CD-M、CD4D 或 CD3D 改善。** 正式比较保留 full native scorer，并比较原输出、最强简单修复、平方／非平方截距、C01 固定相减、平方／非平方对角算子、平方完整矩阵和 smoothed-unsquared 完整矩阵，保持标签、特征、调参机会和总成本一致。

## 当前 20→15 选择

C03 因目标不一致与材料标定未闭合，由第 2 位降到第 7 位；其余构造相对顺序在全池复审后保留。前 15 为：

**C02、C05、C20、C01、C04、C08、C03、C10、C13、C14、C06、C07、C15、C11、C12。**

五个 reserve 仍为 C09、C16、C17、C18、C19。见[当前全池排序](selection.json)、[逐项绑定 batch](batch.json)和[15 份当前规格](specs)。19 张未改数学卡／审查仍引用上轮确切内容；C03 使用本次 [13 步卡](c03-math.json)及[条件自审](c03-review.json)。历史规格的公共工程和实验说明继续作为背景；本次 C03 的目标、臂和映射以当前规格为准。

| 当前计数 | 数值 | 精确含义 |
|---|---:|---|
| math_verified | 20 | 条件公式及同上下文自审的内容绑定；不验证经验假设或原创性 |
| selection_verified | true | 全 20 构造排序并选前 15；不是科学准入 |
| 新候选 code/design/results_verified | 0/0/0 | 未实施候选、未冻结完整 native 设计、未完成候选比较 |
| 成功／正式失败 | 0/0 | 无经验成败判决 |
| 证据导出工具 | 34 项工程测试通过 | 文件／回执／并发锁／CLI；不运行 scorer 或测试候选效果 |

当前核验器实际退出 0，见 [报告](verification-report.json)及[执行记录](verification-execution.json)。旧 71 项软件检查不计为当前候选效果；新工具的工程检查也不计为 method code_verified。

## 下一步已经做成可运行入口

旧 `check_completion.py` 只打包顶层 `pair.json` 和 `receipt.json`。最终 receipt 本来还引用逐臂 native score、阶段回执、mesh、latents；它们未被旧包一同带回。新 [export_feedback.py](../../../../scripts/research_evidence_20261006/export_feedback.py) 校验完整引用闭包、冻结 source/input/weight hashes、逐臂 scorer 来源与完整预算、16 帧记录、嵌套缓存图片、pair/summary 一致性、进程身份和既有 runner 锁；把整个原 run 的现存文件及冻结代码加入分块归档，保留失败尝试和旧早期快照。34 项回归检查覆盖独立工程审查提出的问题；现有 16 对结果的汇总重算完全一致，不据此增加科学验证计数。

直接按[运行说明](../../../../scripts/research_evidence_20261006/README.md)在原 GPU 主机导出。它只做 CPU 文件核验，原始权重／数据只验证哈希、不复制进归档；收到完整原始输出后仍须 trusted native replay，不能仅凭本机 receipt 自证科学结论。当前网页环境缺少原始文件，实际调用返回 blocked，未生成伪最终包。

[下一窗口资格计划](QUALIFICATION.md)把先后关系、可用代码和不合格的旧捷径写明；它不是冻结队列。现有 24 个已看过的 UID 只用于开发，不能再算 fresh confirmation。强简单对照优先资格通过后，才决定 C02/C05/C20 等分支是否必要。

当前 [research-autopilot SKILL.md](https://github.com/Yunbo-max/Research_Autopilot/blob/main/SKILL.md) 要求：

> Require current parent, Natural Gate 0 PASS and IPCG CONCURRENT before implementation or Gate A freezing.

Natural Gate 0、最强简单对照与 IPCG 未闭合，故本次推进条件数学规格和证据工程，尚不把入选研究分支升级为方法实现。这个限制来自 skill 的科学准入规则，不是新增用户许可。八小时内能放多少完整多臂单元，要用原生 pilot 的完整耗时测量；不能拿旧约 15 分钟原生／静止 pair 当成新方法的成本，也不承诺一个窗口检完 15 项。

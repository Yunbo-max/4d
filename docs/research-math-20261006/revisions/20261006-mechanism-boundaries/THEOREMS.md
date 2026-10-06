# 方法必要性审查：保护、可识别性与真实接口

接续 `4d@036b15e99c5816bd353c56a6491c85de47003d3e`。这些是已有 C02/C05/C20 的条件推导，不新增候选，不构造 benchmark，不证明真实数据上存在相应失败。

## C02：保护动作会付出多少几何修复代价

固定 $W\succ0$，令 $d$ 是已声明局部二次几何目标的无约束最优步：

$$q(\delta)=\frac12\delta^TW\delta-\delta^TWd.$$

保护约束为 $C\delta=0$，包括原始 anchor 约束。设

$$G=CW^{-1}C^T,\quad P_W=I-W^{-1}C^TG^\dagger C.$$

原 C02 的 KKT 推导给出 $\delta_*=P_Wd$。令 $p=P_Wd$、$r=d-p$。因为 $Cp=0$ 且 $r=W^{-1}C^TG^\dagger Cd$，有 $p^TWr=(Cp)^TG^\dagger Cd=0$。因此

$$\|d\|_W^2=\|p\|_W^2+\|r\|_W^2,$$
$$q(0)-q(p)=\frac12\|p\|_W^2,$$
$$q(p)-q(d)=\frac12\|r\|_W^2.$$

当 $d\ne0$，定义 $\rho=\|r\|_W^2/\|d\|_W^2\in[0,1]$。保护后的局部几何改善是无约束改善的 $1-\rho$；$Cd=0$ 时完全相同，$\ker C=\{0\}$ 时完全停止。这比“保持动作、修好几何”更精确：它承诺一个明确的局部代价，而非同时免费改善。

### 与普通缩步做真正匹配

令普通缩步 $\delta_s=s d$，选择 $s=\sqrt{1-\rho}$ 使 $\|\delta_s\|_W=\|p\|_W$。则

$$q(0)-q(\delta_s)=\frac12(2s-s^2)\|d\|_W^2\ge\frac12s^2\|d\|_W^2=q(0)-q(p).$$

因此在相同步长下，普通缩步的该局部几何收益反而不小于投影；投影的用途必须是其 $C p=0$ 的方向保护，而不能把少动一些引起的变化当作新机制。$C\delta_s=sCd$，只有 $Cd\ne0$ 且 $s>0$ 时两者才有这一区别。

这不是对所有 ARAP 非线性迭代或原生 CD 的结论。这里 $W,d,C$ 在子问题内固定；不合法地使用 GT motion 形成 $C$ 会破坏方法的信息约束。若 $C=D\psi(y)$ 只是某个可用非线性动作观测的导数，且局部满足 $\|D^2\psi\|\le L$，则

$$\|\psi(y+p)-\psi(y)\|\le\frac L2\|p\|_2^2,$$

而不是精确保留 $\psi$，更不保证 CD-M。冗余保护行使用原 Gram 伪逆；改变伪逆截断阈值会改变实际子空间，需要记录。

原生检验必须比较相同 $d,W$ 的无约束修复、投影、匹配 $W$ 步长的缩步，以及开发集选择的最强简单对照。$\rho$ 和 $Cp$ 是方法代数诊断；**不能替代原作者三个评分指标**。真实 cohort 中 $\rho$ 的分布与动作观测的价值尚未知。

## C20：ridge 能让矩阵可逆，但不能创造信息

当前问题为 $r\approx P\xi+B\gamma$。$W\succ0$，设 $L=W^{1/2}$、$\bar P=LP$、$\bar B=LB$、$\bar r=Lr$。令 $U$ 是 $\operatorname{range}\bar P$ 的正交基，$M=I-UU^T$。对 phase 消元后：

$$\min_\gamma\frac12\|M(\bar r-\bar B\gamma)\|_2^2.$$

这里讨论 phase 不等式未激活的局部问题；phase 参数的冗余 gauge 要先固定。假设 $\bar B$ 满列秩，作薄 QR $\bar B=VR$，$V^TV=I$、$R$ 可逆，令 $\eta=R\gamma$。可识别性矩阵为

$$S=V^TMV=I-V^TUU^TV.$$

于是

$$\kappa=\lambda_{\min}(S)=1-\sigma_{\max}(U^TV)^2.$$

当 phase 子空间为空时定义交叉奇异值为 0，$\kappa=1$。否则 $\kappa$ 是两个子空间最小 principal angle 的平方正弦。$\kappa>0$ 等价于 $MV$ 满列秩，幅度可在该线性观测中分离；$\kappa=0$ 表示至少一个非零幅度方向也能由 phase 表达。若 $\bar B$ 本身不满列秩，还需先合并幅度冗余，不能假装 QR 的 $R$ 可逆。

### 可分离也可能很不稳定

无 ridge 时，$\hat\eta=(MV)^+M\bar r$。观测扰动 $e$ 导致

$$\|\Delta\eta\|_2\le\frac{\|e\|_2}{\sqrt\kappa},\qquad
\|\Delta\gamma\|_2\le\frac{\|R^{-1}\|_2\|e\|_2}{\sqrt\kappa}.$$

因为 $(MV)^T(MV)=S$，最小奇异值为 $\sqrt\kappa$，上述第一项是精确的最坏方向放大率，而非经验拟合规律。接近重合时，小的合法输入误差或插值误差也可能主导分块结果。

加原坐标 ridge 后，$(\bar B^TM\bar B+\lambda I)^{-1}$ 对 $\lambda>0$ 存在。但若某方向 $Bv=Pu$，变换 $(\xi,\gamma)\mapsto(\xi-au,\gamma+av)$ 不改变预测；ridge 只是偏好某个分配，并没有从观测恢复真实 timing/amplitude。反过来，激活的单调性／边界约束可能排除某些混淆解，此时需分析完整受约束问题，不能直接用无约束不可识别性判死刑。

### 明确时间与余项

原始时刻 $t_j$ 不变，内部时间为 $t_j+\tau_j$。选定 $m>0$ 后，离散单调性约束为

$$\tau_{j+1}-\tau_j\ge-(1-m)(t_{j+1}-t_j),\quad \tau_0=\tau_{T-1}=0.$$

对 piecewise-linear warp，该式使每段导数至少为 $m$。如使用更高阶插值，需要另核查段内导数，不能只检查节点。对 block-diagonal $W=\operatorname{diag}(w_iI)$ 且各轨迹 $\|Y_i''\|\le K_i$，Taylor 余项满足

$$\|\epsilon_{\mathrm{lin}}\|_W\le\frac12\left(\sum_iw_iK_i^2\tau_i^4\right)^{1/2}.$$

这项误差同样经过分解算子的 $1/\sqrt\kappa$ 放大。一般非对角 $W$ 不能直接套逐点权重式。原生比较需使用原时间、原 GT 和原 scorer；phase-only、amplitude-only、joint 与普通时序修复共享合法目标、同一拟合和调用预算。还没找到可靠合法目标时，这些公式是条件分析，不是可派发算法。

## C05：数学混合分布与原生 decoder 不是同一个对象

已读取的冻结 `temporal_autoencoder.py` 中，`fwd_cross_attn` 输出隐藏特征经 `norm_out/proj_out` 的 logits，`forward` 返回 $2\sigma(\mathrm{logits})-1$，`apply_displacement` 的 direct 分支把它作为坐标。公开返回接口是每个 query 的一个坐标，不是每个 query 的候选坐标／概率对。

一般 $g(\sum_kp_kz_k)\ne\sum_kp_kg(z_k)$。仅考虑非线性输出 $g(z)=2\sigma(z)-1$，取两个等权 logits $z_1=0,z_2=2$：左边是 $\tanh(1/2)$，右边是 $\tanh(1)/2$，二者不同。这是“不应交换非线性解码与求均值”的数学边界；没有修改 benchmark 或声称冻结权重在自然样本上恰好发生这种错误。

因此原 C05 的 $\mathbb E\|Y\|^2-\|\mathbb EY\|^2=\operatorname{tr}\operatorname{Cov}(Y)$ 仍正确，但它只能应用到已构造且解释明确的坐标候选分布。不能从一行 latent attention 的熵直接推出真实位置多峰，更不能把 ActionMesh 输出称作候选表面点的 posterior mean。

[STAC 原文 §4.1 Eq. (2)–(4)](https://arxiv.org/html/2605.19786v1) 明确先构造 token-chain score，再对目标 surface 候选做 top-scoring sharp blend；§4.2 再做 landmark、Gaussian 和 rigid skinning。它提供了可研究的具体替代路径，且构成 C05 的强对照。其代码／与冻结 native runner 的接口尚未资格验证；本轮不假称已经复现。softmax 归一化使分数成为一组合法权重，不自动建立真实材料对应的概率校准。

### localized blend 不自动留在真实表面

原文把 localized surface samples 的 blend 与留在目标 surface 联系起来；一般成立的性质仅是 **均值落在候选凸包**。在半径 $R$ 的球面，取 $x_\pm=(R\cos\theta,\pm R\sin\theta,0)$，等权均值为 $(R\cos\theta,0,0)$，当 $0<\theta<\pi/2$ 时在球内部。两候选可任意接近，因此“top-scoring localized”不会使该性质严格成立。

若弦长 $\ell=2R\sin\theta$，径向距离为

$$R-\sqrt{R^2-\ell^2/4}=\frac{\ell^2}{8R}+O(\ell^4/R^3).$$

这给出局部曲率／候选范围有关的区别性预测，而不要求候选权重是校准后验。不过平均并不一定使最终 lifted mesh 或 native 指标更差；实际 lift、非等权、离散 mesh 和 model-generated surface 都必须检验。该球面仅是数学反例，不是生成的评测样本。

因此除 sharp mean 与 top-1 外，必要简单对照还包括 **同一 bank 上将平均位置投回候选表面**。若该直接投影已经解决平均离面，而联合标签没有额外 native 收益，不能把“避免均值离面”单独当作空间联合决策的贡献。表面投影后仍可能选错对称部位；身份歧义与几何离面是两种待区分的解释。

空间联合标签也可以使用非校准的 score 作为 unary energy，但这时只能声称一个明确的离散优化构造，不继承 Bayes posterior 的含义或保证。必须先建立共同候选集、同样的 lift、实际信息访问和同样搜索成本，才能比较 sharp mean、独立 top-1、联合选择。自然 multi-mode prevalence、GT-to-decoded-vertex correspondence、错误对称部位选择以及 native 效果均未核实。

## 本轮决策

保留 20 个已有构造，不把 $\rho$、$\kappa$、ridge、候选温度或代数检查计成新方法。C02 需验证投影相对缩步的必要性；C20 需验证目标与可识别性；C05 需建立真实候选接口。当前优先做可直接作用于原 mesh 的经典对照资格，而不是先投入 correspondence-bank 和 phase-target 分支。

数学公式与有限代数核对仍是自审；原创性、自然失败、候选代码、完整 native 实验与成败均未因此完成。

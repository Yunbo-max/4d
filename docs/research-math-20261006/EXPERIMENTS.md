# 逐方法原生实验与八小时窗口设计

这是 15 个分支的原始条件比较规格。阶段级的 numeric criteria、98 个确认 contrasts、完整失败分母和确定性 family split 规则现已由 [G01_DESIGN.json](longgoal-20261007/G01_DESIGN.json) 与 [G01_DESIGN.md](longgoal-20261007/G01_DESIGN.md) 取代并冻结为 `generated_unexecuted` 源码设计；下文未改写的“拟”“尚未冻结”是历史状态，不得覆盖新设计。C06 现已有 geometry-only 完整源码特化和五角色原生入口，但尚未 Local 执行；源码链覆盖为 10/15。动态 family 来源审查、B* 的 D1-only freeze、每个候选的 Gate 0/IPCG、Local receipts 和 native results 仍未完成，因此 `design_verified=0`、GPU dispatch 仍禁止。

## 不改变原生任务

固定 `facebook/actionbench` revision `2796071cbe6248422fcbeab3101fa9f9886cb7b9`，128 个自然配对资产、16 个原始帧。保留 frozen ActionMesh 权重、100 Stage-0 / 30 Stage-I steps、CFG 7.5、FP16/low-ram 的主比较配置；权重与全部 source tree 需要实际回执。

原生 `compute_chamfer_3d_4d`，sampling seed 44、surface samples 100000、ICP samples 10000、24 rotations、200 ICP iterations。三个指标都是原生代码的非平方距离：CD-3D 逐帧 ICP，CD-4D 仅第一帧 ICP 后保持全序列坐标，CD-M 用同步表面样本和第一帧固定 nearest-neighbor correspondence。CD-4D 不是四维坐标距离；CD-3D 的逐帧对齐不能单独证明动作保真。不得改变 evaluator、参考时间、采样分母或输出帧数。

逐臂保存完整 mesh sequence、原始 scorer 输出与 native receipt。算法 surrogate、面积／体积／运动幅度只解释机制或执行约束，不能替换 native 主端点。长视频、QA、编辑和推理不在本轮结论内。

## 先资格核查的控制

| 控制 | 用途及资格要求 |
|---|---|
| B0 原生 ActionMesh | 同视频、锚点、context、seed 的主基线；先重放完整原生 receipt |
| B-static 同锚点静止 | 弱负控制和动作诊断；不能替代最强普通方法 |
| B-Gauss world / body | 普通 temporal Gaussian，含合法 rigid-aligned 版本；开发调参、真实固定时间、全 native 评分 |
| B-ARAP | 同锚点、完整几何、相同数据项／pins 的普通 ARAP／弹性修复；不得给予它较差 lift 或更少调参 |
| B-video | 合法输入视频 tracks／mask／相机估计约束的普通变形；禁止 GT 3D、GT camera 或评价 ICP 进入推理；接口、成本尚未资格通过 |
| 最近发表方法 | STAC、SFG、Mesh4D 等按任务和可执行性逐一审查；无合格代码时标记 unavailable，不能把自写近似叫作原作者实现 |

在开发资料上选定每方法的 strongest executable simple comparator B*，其身份和参数在确认前固定；确认期不能按结果重新选较弱基线。已发布方法的 native 数字与其他 dataset / metric 数字不能混表比较。相同 information、上游 context、lift、调参额度和总成本是必需项。

## 15 项的完整比较

以下每行都另含 B0 和冻结的 B*；重复臂仅在内容／协议／source identity 相同才共享。逐项机器规格见 [specs](specs)。主比较检验新增操作，全部臂保留完整失败分母。

| 分支 | 必需区分性臂 | 拟定主要 native 端点 | 条件预测与否定边界 |
|---|---|---|---|
| C02 | geometry-only、strength-matched blend、同 repair 的 protected update | CD-3D 改善；CD-M/4D 非劣 | Cd 非零且有可用零空间才产生不同修复；普通弱化修复等效则无必要性 |
| C03 | mean-only、C01 unit、diagonal、declared matrix estimator | CD-M 改善；CD-3D/4D 非劣 | cross-moments 稳定且非各向同性；跨资产估计误差／均值漂移吞噬收益则否定 |
| C05 | localized mean、temperature-matched mean、independent top-1、joint spatial labels | CD-3D 改善；CD-M/4D 非劣 | 合法 modes 分离时区别于调温度；单峰或 top-1 等效时不继续新方法 |
| C20 | phase-only、amplitude-only、simple lag、joint constrained inference | CD-M 改善；CD-3D/4D 非劣 | 合法证据能区分 phase 与 amplitude；重标评分时间或不识别均不能支持方法 |
| C01 | uncorrected、mean-bias、same-context self-map subtraction | CD-M 改善；CD-3D/4D 非劣 | 保留完整 error cross term；严格 anchor 并不保证 Chamfer 改善；主要作为诊断／强对照 |
| C04 | deterministic protection、strength-matched repair、robust conic set | CD-3D 改善；CD-M/4D 非劣 | uncertainty set 有 held-out 语义／coverage 支持；不成立则只剩任意保守约束 |
| C08 | local transition tracker、coordinate smoother、decoder-cycle control、endpoint bridge | CD-M 改善；CD-3D/4D 非劣 | 未来 endpoint 对中间路径提供信息；不比局部 tracking 好则路径约束无必要性 |
| C10 | direct common lift、independent local repair、pinned integrable solve | CD-3D 改善；CD-M/4D 非劣 | non-integrable residual 真实存在；输入 field 已可积时应无变化 |
| C13 | Gaussian、quadratic acceleration、group acceleration | CD-M 改善；CD-3D/4D 非劣 | 保留真实急变并减少噪声；压制真实平滑高频动作是失败边界 |
| C14 | world smoothing、simple rigid-aligned smoothing、declared body residual repair | CD-M 改善；CD-3D/4D 非劣 | exact rigid invariance；普通 pose-factored 平滑足够时归入强对照 |
| C06 | row-softmax、vertex-density transport、area-marginal transport | CD-3D 改善；CD-M/4D 非劣 | 源顶点密度与面积不同且质量假设合理；真实 stretch 可使假设失败 |
| C07 | full-mass transport、confidence-threshold fallback、partial-mass+native fallback | CD-M 改善；CD-3D/4D 非劣 | 有不支持匹配；不能靠少输出点改善；普通拒绝回退等效则无新必要性 |
| C15 | Gaussian、unprotected SVT、same-rank protected SVT | CD-M 改善；CD-3D/4D 非劣 | fixed Q 与 anchor 保持；真实细节落入 discarded directions 会失败 |
| C11 | ARAP、elastic repair、common-lift stretch projection | CD-3D 改善；CD-M/4D 非劣 | 局部 strain artifacts 而非合法伸缩；普通 ARAP 等效则归入已有修复 |
| C12 | fixed damping、generic backtracking、exact quadratic admission | CD-3D 改善；CD-M/4D 非劣 | 安全方向不变、危险方向由 first violation 限制；现成 line search 等效则仅工程工具 |

补充 bound / invariance / solver 检查不构成新方法臂；lambda、patch 数、seed、rank 和 budget 设置也不计新 idea。上述分支存在嵌套和共享机制，不能把两项相关结果说成独立机制复现。

## 开发、确认和完整分母

[历史暴露记录](evidence/exposure-and-contract.json)包含旧 8 项及隔夜 16 项，共 24 个已知 UID，全部归开发。还需检查其余先前浏览／调参曝光、同源 mesh / animation family 和重复几何，不能直接宣称其余 104 个全部独立且未见。UID 前缀不是经过证明的 family 标签。

确认 split 之前，按完整来源 metadata 和输入／几何哈希建立 family group、检查已有曝光。所有同族留在一个 split；若分组不能确认，科学确认保持阻塞。用固定 salt `4d-math-restart-20261006-v1` 的 group-ID SHA256 顺序，拟定 12 个 fresh-family 开发单元 D1、另 12 个 D2，其余未暴露 groups 保留确认；实际 UID、family、split 清单及 hashes 在打开新结果前冻结。这个分配规则不是已经冻结的资产队列，也不声称一定有 80 个独立确认对象。

数学池排序不使用新的方法分数。D1 做合法接口与适用条件、全部臂开发比较；D2 检查独特预测和标定 transfer。候选调参最多两个数值参数、各最多三个预先选择值，同类控制得到相当的搜索／拟合／时间额度；需要更多模型选择时重设公平协议，不偷偷越界。C03 的统计 fitting 及所有 GT 标签接触都仅限开发并留日志。保留其余方法为待验证，不因接口缺失直接 KILL。

确认 seed 拟固定 `[42,314,2718]`，推理臂配对共享 upstream RNG/context，评价 seed 44 固定。独立单位是审查后的 asset family；同对象 seeds、16 帧、100000 点和多个超参数不增加独立样本数。统计前按照冻结的对象／family 权重聚合 seeds；同时披露各 seed 和每 UID 原始值。

历史草案拟用 family 级 paired bootstrap 20000 replicates；现行 G01 源码设计已固定为受信外部 family-cluster bootstrap 200000 replicates、RNG seed 20261006、98 个确认检验和 Bonferroni familywise alpha 0.05。动态 family 证据与受信 live analysis adapter 未资格通过时不能强判；未做正式确认的方法仍列为待验证。

对越小越好的 native 指标设配对差 `d=treatment-control`。现行 G01 设计把主端点 absolute minimum effects 固定为 CD-3D 0.002、CD-M 0.003，把非劣 margins 固定为 CD-3D 0.002、CD-4D 0.004、CD-M 0.003，并以官方同任务 leaderboard 的可见差值作为尺度依据；旧“5% / 2%”工作线废止。PASS 需要每个 indispensable 主对照的 multiplicity-adjusted one-sided upper bound `< -Delta_m`，以及对 B* 的两个 guardrail upper bounds `<= epsilon_m,k`。有效样本数和 precision 仍须由 D1/D2 family 证据前瞻评估；库存无法分辨固定效应一半时记 INCONCLUSIVE，不降低阈值追显著性。

每个模型调用、搜索、solver、scorer 都有冻结的 step / time / memory 上限。无动作、无解、坏面、超时、native fallback、部分臂完成、数据加载失败均保留；只按预先独立定义的技术资格规则处理，不按改善大小排除。效果与 native execution failure rate 分开报告，全部失败原因可回放。确认协议采用实际 eligibility / raw receipts，单独的结果 JSON 或布尔资格不足。

## 各方法怎样作最终判断

| 结果 | 条件 |
|---|---|
| PASS（有限任务成功） | 全部合格臂、强简单基线、原生 receipts / scorer replay、冻结主效应与非劣界、区分性控制、确认和 E04 完成；仅支持 frozen claims |
| KILL（该构造或必要性失败） | 实现／基线／评价已合格，有足够精度的自然证据否定机制、显示 native 损害，或最强普通方案已经解决问题；说明否定的是哪条构造与条件 |
| REVISE | 混合结果支持事先允许的子假设；重新推导、审查、冻结并使用新的确认，不重用同一数据当新证据 |
| INCONCLUSIVE | 未测、缺接口、缺 raw proof、目标／统计不识别、样本或功效不足；不能称成功或失败 |

已知经典／已有实现的分支可成为有效 baseline 或工程修复；有效也不自动变成原创论文方法。当前 15 项全部未测，成功和正式失败均为 0。

## 单张 2080 Ti 的八小时安排

窗口硬预算默认 28800 秒，预留 1800 秒给预检、收尾核验与打包；剩余 27000 秒用于完整单元。历史同配置双臂单元平均 899.21 秒、最大 910.24 秒；历史先验 `ceil(1.35*910.24)=1229` 秒仅适用于**未改动的同原生双臂类别**，对应最多 21 个保守等价单元，不能直接用于新方法、多臂评分或拟合。

若只有 12 个资产、15 个方法、每个最低四个臂，已有 720 个 arm/asset slots（实际多项超过四臂）；共享上游与相同原生臂可减少实际调用，但没有实测 scorer/solver/capture 成本前不能说八小时能完成 15 项。全部验证需要多个窗口，八小时不是科研有效性的保证。

| 窗口 | 内容 | 何时结束／进入下一步 |
|---|---|---|
| W0 证据与强对照 | 在旧 GPU 工作区重跑原冻结 completion checker，回传最终 16/16 proof；复用已有 mesh 作诊断与可执行强对照资格核查 | raw hashes / arm scorer、环境、完整 telemetry、自然问题与 baseline qualification 可回放；已完成有效输出不重生成 |
| W1 条件主比较 | 门禁通过后，先一项完整方法+全部区分性臂，2–3 个完整开发单元校准新成本类别；其余按完整单元接纳 | 端到端计时包含 load/capture/solver/all scoring/receipt；失败和慢例参与估计与分母 |
| W2… 筛查与确认 | 按同等强度测各 admitted 方法，再冻结独立确认；所有 15 方法在总表保留自己的状态 | 必需 native criteria / controls 先完成；没有有价值且就绪的任务则收尾 |

新成本类别使用兼容实测完整单元最大耗时乘 1.35 的保守接纳估计，另记录未知 I/O／archive 成本。旧 runner 的 2400 秒下限和固定 16 项尚未接入新成本卡，启动前必须修改并验证预算／恢复行为；本次没有发布可执行候选 queue 或改动 runner。

core 和有科学必要性的 reserve IDs、所有 arms 和参数在启动前冻结。reserve 是否激活只根据已冻结准入、余时和完整单元可完成性，不看效果／显著性；不得随机追加“好看”样本。记录窗口累计 elapsed、失败和已用 budget，恢复不重置。只有足够余时完成全部臂且预留核验时才开新单元；超时保留部分状态并停止接纳。

回传最终包包含 source/data/weights/env identity、全 core/reserve/attempt 库存、每臂 mesh/metrics/native receipt、完整 stdout/stderr、阶段和端到端成本、峰值 telemetry、所有失败以及最后完成时间之后生成的完整核验。大输出可用带 SHA256 的归档和可下载地址。现有连接支持 GitHub 文档交付，尚无当前 GPU lease 或原始输出访问，因此本轮没有启动 GPU 实验。

# 最新实现复核与修正版运行说明

2026-10-04 更新：当前优先任务改为 [自然失败诊断轮次](../failure-diagnosis/protocol-20261004.md)。下文H9入口保留用于以后有证据支持的screening；先完成失败/反例、竞争原因和简单基线审查，不能因H9代码可运行就直接推进方法实验。本更新没有改变或追溯重新解释旧结果。

复核基线：`81f4f48330aef6d0c00dce9826335a3b1287850e`，2026-10-03 的 main。本修订属于 screening code repair；旧结果保留原协议和脚本哈希，不追溯更改。

**十个候选都有代码，算法库基本到位；十个 idea 的自然数据独立对照尚未到位。** 现有自然 census 已完成 8 个资产、10 次原生生成，不等于十个方法分别通过实验。原 v1 的71项软件测试及第2、9项GPU接口检查是有效进展，不能据此证明方法优于基线。

## 逐项判断

| # | 已做到 | 缺口与本次处理 |
|---|---|---|
| 1 | 观测权重、ARAP、均匀/均值/幅度/打乱对照 | 缺实测轨迹和相机；新增独立 `geometry --method 1` 输出与失败记录 |
| 2 | 真正的解码回环/组合、图求解，真实GPU32点检查 | 缺自然对应误差及完整网格对照；回环变小不等于对应正确 |
| 3 | 显式多候选上的二阶联合DP | 候选匹配器获取未实现；单条decoder轨迹不能当top-k匹配 |
| 4 | 闭合遮挡段的可见端点/邻点约束修复 | 仍是轨迹原型，缺真实可见性和图像双向再跟踪接口 |
| 5 | 稳定区域Sim(3)、bbox/全局Sim(3)对照 | 缺可靠稳定区域和自然尺度错误资格；新增 `geometry --method 5` |
| 6 | 动作门槛、排序和生成/评分/重建回调 | 缺真实Wan候选、统一冻结评分器；合成评分表不是自然效果 |
| 7 | 计入probes的固定预算分配、分块距离 | 仍是欧氏插值，缺真实质量/墙钟曲线及合格Fast4DMesh对照 |
| 8 | 有界Dykstra法向修复、标准投影和残差 | 缺可靠有符号接触及IPC/ARAP强对照；新增 `geometry --method 8` |
| 9 | 实时冲突投影及标量/范数/随机对照，两步GPU检查 | smoke原先遗漏正式加法flow设置；现修正并新增完整30步、同噪声、全网格入口 |
| 10 | 前向和未来边界反向重采样，硬锚点保持 | 原不足16帧尾窗口会被native拒绝；末窗现前移、记录真实重叠；自然长视频/表征资格仍缺 |

## 先运行第9项完整对照

在修订后的 checkout 的 `actionmesh/` 目录、原来能运行ActionMesh的CUDA环境执行。以下路径来自已发布的 mannequin/seed42/attempt2 报告。其他cohort case也可用，不再限制两个UID或seed42。

```bash
source /root/rivermind-data/actionmesh-repro/inference-env/bin/activate
python -m unittest discover -s research_ten/tests -v
python -m research_ten guidance \
  --root /root/rivermind-data/actionmesh-repro \
  --case-dir /root/rivermind-data/actionmesh-repro/outputs/census-20261002/cases/000-037_1358c424008a43cbaa35eba5e58551ac__seed42__attempt2 \
  --output /root/rivermind-data/actionmesh-repro/outputs/paired-h09-20261003/mannequin \
  --max-seconds 1800 \
  --memory-cap-mib 20480
```

复用Stage0、image context、锚点和输入，不重新生成视频。先用原缓存/原查询做冻结StageII replay。四个live arms分别从同一噪声开始：`scalar`、`projection`、`norm_matched`、`random`；都读取正式配置的加法flow、30步、每步三分支，正常完成每组90次denoiser forward。每组StageII查询全部顶点和15个非锚点帧，保存完整16帧 `sequence.npz`。

`official_cached` 是单独列出的历史参考，新增StageI调用为0；原来的60次两分支forward及已有时间记录另列。四个live arms是同checkpoint、精度和计算设置的因果对照。原census未记录denoiser checkpoint哈希，不能证明历史权重与当前逐字节相同，也不能把scalar/cache差异全归因于精度。当前权重和源码会记录哈希并在结束复核。

生成后用已有正式evaluator单独评估。GT只进入以下命令：

```bash
python research_census_eval.py \
  --case-dir /root/rivermind-data/actionmesh-repro/outputs/paired-h09-20261003/mannequin \
  --manifest /root/rivermind-data/actionmesh-repro/outputs/paired-h09-20261003/mannequin/manifest.json \
  --gt-dir /root/rivermind-data/actionmesh-repro/data/actionbench-census-20261002/data \
  --repo-root /root/rivermind-data/actionmesh-repro/repo \
  --output /root/rivermind-data/actionmesh-repro/outputs/paired-h09-20261003/mannequin/official-evaluation.json \
  --device cuda \
  --seed 44
```

不要用两步latent差异决定保留idea。先比较匹配scalar下的CD-M、CD3D/CD4D、锚点、失败率和成本。APG/CFG++强基线、动作保持判据、独立held-out cohort及novelty/gate检查仍待补齐；这是screening入口，不是已完成的论文验证。

## 第1、5、8项的真实观测

每次只改变一个方法。输入NPZ共有：`rest[V,3]`、`trajectories[T,V,3]`、整数 `faces[F,3]`、严格递增 `times[T]`；rest必须等于frame0。可复制已有sequence的vertices/faces/timesteps，不重跑原生模型。

| 方法 | 额外NPZ字段 | evidence资格字段 |
|---|---|---|
| 1 | `observed_uv[T,V,2]`、`confidence[T,V]`、`projection[T,3,4]` | `observed_tracks_and_camera_verified` |
| 5 | 布尔 `stable_mask[V]`，可选 `confidence[T,V]` | `stable_region_verified` |
| 8 | `contact_frame[K]`、`contact_indices[K,4]`、`contact_coefficients[K,4]`、`contact_normals[K,3]`、`contact_gap[K]`，可选整数 `fixed_vertices` | `signed_contacts_verified` |

点ID必须与网格对应。相机来自推理前已知信息或仅由观测估计，禁止使用GT的ICP配准。稳定区域、安全有符号接触应实际确认。空接触可以检查软件，不能证明存在自然穿插。Invisible UV仅在confidence=0时可为NaN。相机坐标和误差量纲需明确，不能默认已解决单目尺度歧义。

evidence是冻结JSON。先实际验证，再把资格字段改为true：

```json
{
  "schema_version": 1,
  "uid": "实际资产UID",
  "method": 5,
  "input_sha256": "输入NPZ的实际SHA256",
  "input_evidence": "natural_observations",
  "measurement_provenance": "获取方式、验证依据和相机坐标说明",
  "qualification": {"stable_region_verified": false},
  "parameters": {"camera_convention": "camera_compensated_world", "robust": true}
}
```

```bash
python -m research_ten geometry \
  --method 5 \
  --input /path/to/qualified-scale-input.npz \
  --evidence /path/to/frozen-evidence.json \
  --output /path/to/new-paired-h05
```

可改为method1或8，默认对照固定。H1：原输入、观测权重、三个均匀权重、均值匹配、幅度和打乱；H5：原输入、稳定区域、bbox、全局Sim(3)；H8：原输入、有界法向修复、标准投影。求解设置在evidence预先固定，不在测试集挑最好超参，不叠加方法。额外GT/评价数组不属于输入契约，会被拒绝。

每组独立保存sequence/report，根目录保存完整分母manifest，可传入同一正式evaluator。不收敛/不可行时保留debug sequence但标记failed，evaluator计为失败。观测真实性是调用方资格声明；哈希只证明一致性。

## 失败、成本和验证边界

新入口拒绝覆盖。来源输入/源码/checkpoint途中改变、删除或读不到时整组失效。GPU超时/OOM保留部分尝试次数与墙钟，区分Python iteration返回和同步后验证完成。资源收集失败不能盖掉原异常或阻止报告保存。

预算在native branch/decoder callback检查，不能抢占卡死kernel。20GiB是Torch allocator cap，不代表整卡实测峰值；报告保存Torch峰值及每秒nvidia-smi样本。哈希、加载和记录开销计入总墙钟。

本修订环境：Python3.12、NumPy2.3.5。**85项测试，83通过，2项因没有Torch跳过**。覆盖70帧native尾窗、非法seed、三项独立输出、不可行接触、完整分母、覆盖保护、hash/资格与来源改变/删除失效。独立复核没有剩余Critical/Important问题。**本环境没有运行新增30步CUDA推理或自然质量评估**；原v1的Mac/远端71项无跳过记录是历史验证。

下一步先跑已有mannequin的H9完整组和正式评估，再按原frozen cohort扩到已有资产，最后锁定独立held-out。有真实观测前提时分别跑H5/H1/H8。其他项先补各自前提和强简单对照；前提缺失应记录blocked/qualification repair，不能制造可验证前提或将toy结果算自然证据。

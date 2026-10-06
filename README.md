# 4D 研究复现

## Local Codex：从这里开始

当前可执行轮次是[基线／原生评分资格交接](rounds/20261006-baseline-qualification/WEB_HANDOFF.md)。Local Codex 在精确交付 commit 先读[根级执行约束](AGENTS.md)与[项目运行手册](LOCAL_AGENT_RUNBOOK.md)，再按手册通过 `research-autopilot/scripts/run_harness.py` 驱动独立 Linux 2080Ti 主机；远端不需要 Codex/GPT，会话控制、Git、SSH 与文件回传由用户电脑承担。

状态仍为 `generated_unexecuted`：三臂强简单对照、原生评分请求和严格输出绑定已生成，但本状态没有新的测试执行、GPU 运行或官方评分结果；候选完整验证设计和 native 结果均为 0/15。不得把数学规格、静态 receipt 或软件测试当作科学资格。

## 当前：方法必要性与可识别性审查

[第二轮研究修订](docs/research-math-20261006/revisions/20261006-mechanism-boundaries/README.md)补推 C02 的保护代价和匹配步长对照、C20 的 principal-angle 可识别性及噪声放大、C05 的原生接口和 localized blend 离面边界。保留20个构造与原15个入选身份，更新全池顺序和15份条件规格。数学仍是条件自审，新候选实现、完整 native 实验及正式成败均为0。

新回传的 W0 checkpoint 已核对16个 pair／final receipt 字节 hash，分数与旧 summary 一致；小包仍缺其回执引用的352个阶段／raw成员。见[反馈审查](docs/research-math-20261006/revisions/20261006-mechanism-boundaries/w0-feedback-review.json)，完整 raw 导出与 native 回放继续待补。

## 2026-10-06：数学指标审查与最终证据回传

当前研究入口：[native-loss 修订与完整 20→15 排序](docs/research-math-20261006/revisions/20261006-native-loss/README.md)。C03 的平方风险收益不能保证原生非平方距离收益；已补精确反例、条件非平方标定推导和15份当前规格。`math_verified=20`、`selection_verified=true`；新候选 code/design/results_verified 仍为0。

[最终原始证据导出工具](scripts/research_evidence_20261006/README.md) 已通过34项文件/回执工程测试，可在原运行主机校验并打包final receipt的mesh/latents和逐臂scorer，保留旧快照与失败日志；这是证据工程，不是方法效果验证，原主机尚未执行此次导出。[下一窗口资格计划](docs/research-math-20261006/revisions/20261006-native-loss/QUALIFICATION.md) 写明强简单对照、完整原生预算和准入条件。

当前项目：**ActionMesh（CVPR 2026）官方预训练权重推理**，使用 RTX 2080 Ti 22 GiB，没有训练。

## 八小时实验反馈（2026-10-06）

[本轮复盘与下一轮设计](docs/research-overnight/FEEDBACK-20261006.md)：固定 16/16 个资产
在约 4 小时 2 分钟内完成，完整比较平均约 15 分钟。已记录实测成本与更强对照、
主任务/备用任务、最终回传核验要求；当前 runner 尚未接入新成本卡。
发布的 completion-verification 仍是早期 1/16 快照，原始回执的最终独立核验待补。

## 2026-10-05 的 8 小时单卡启动方案（历史）

[准备、启动、恢复和结果说明](docs/research-overnight/README.md)。默认复用已有
ActionMesh 环境，检查实际显存，按冻结顺序运行新的 ActionBench 样本和原生评分。
启动后最多 8 小时，重启沿用原截止时间。当前是基线和真实失败调查，尚未验证新方法。

```bash
git pull --ff-only
bash scripts/run_8h_2080ti.sh prepare
bash scripts/run_8h_2080ti.sh doctor
# 晚上 23:00 在 GPU 服务器运行：
nohup bash scripts/run_8h_2080ti.sh run --hours 8 > overnight-console.log 2>&1 &
```

## 研究结果总览（2026-10-03）

[全部十批实验结果](results/README.md) · [十项方法与71项测试](results/ten-methods-20261003/README.md) · [自然样本基线与负面结果](results/census-20261002/README.md) · [原始大归档下载](https://github.com/Yunbo-max/4d/releases/tag/results-2026-10-03)

本机和远端十项候选的代码测试均为71项通过、零跳过；这不等于十项方法已在自然数据上改善质量。全部实验结论、构造测试与真实GPU检查范围见对应报告。

## 四种应用实测

2026-10-02 已补上外部 Wan 视频生成模型，完成文字、图片 + 文字、3D + 文字三个端到端样例。视频应用沿用此前验证的袋鼠基线。此前“视频加载器拒绝文字/单图”的结果只说明入口用错，不能判定项目不支持这些应用。

| 应用 | 实测样例 | 结果 | 前置视频 / ActionMesh 显存峰值 |
|---|---|---|---:|
| 视频 → 4D | 官方袋鼠视频 | 已验证16帧动态网格 | 不适用 / 10,257 MiB |
| 文字 → 4D | 章鱼演奏沙锤 | 已生成16帧动画；局部断连、漂浮，沙锤不准确 | 10,347 / 10,257 MiB |
| 图片 + 文字 → 4D | 蘑菇图片 + 唱歌剧 | 已生成16帧动画；无纹理，有嘴部及肢体动作 | 13,201 / 10,255 MiB |
| 3D + 文字 → 4D | 熊猫GLB + 悠闲行走 | 已生成16帧动画，保留原拓扑和纹理像素 | 13,201 / 10,255 MiB |

已采样的视频生成和ActionMesh阶段最大值为 **13,201 MiB（12.89 GiB）**；单图静态重建另记录PyTorch显存，详见[静态阶段报告](results/applications-20261002/image-mushroom/static-anchor/report.json)。只对应512×512、33帧、50步的前置视频配置和ActionMesh低显存配置，不能推广到默认720p或更长视频。技术流程已通，画面质量仍有局限，并非完整论文benchmark或官网同配置复刻。

| 视频：袋鼠 | 3D + 文字：熊猫 | 图片 + 文字：蘑菇 | 文字：章鱼 |
|---|---|---|---|
| ![袋鼠视频到4D](results/input-modes/kangaroo-video/preview.gif) | ![熊猫动画](results/applications-20261002/mesh-panda/animated-preview.gif) | ![蘑菇动画](results/applications-20261002/image-mushroom/animated-preview.gif) | ![章鱼动画](results/applications-20261002/text-octopus/animated-preview.gif) |

[完整结果、实际提示词、时间/显存、复现命令与效果限制](results/applications-20261002/README.md)

- [视频→4D：袋鼠动画 GLB](results/input-modes/kangaroo-video/animated_mesh.glb)
- [章鱼动画 GLB](results/applications-20261002/text-octopus/animated_mesh.glb)
- [蘑菇动画 GLB](results/applications-20261002/image-mushroom/animated_mesh.glb)
- [熊猫带纹理动画 GLB](results/applications-20261002/mesh-panda/animated_mesh.glb)
- [接入前的根因诊断](results/input-modes/application-debug-20261002.md)

前置采用`Wan-AI/Wan2.2-TI2V-5B-Diffusers`，不是已确认的作者演示模型版本。文字走论文允许的文字→视频→4D路线；另外两例用模型正面渲染和动作文字生成视频，再将视频与网格输入ActionMesh。

## 已有视频格式基线（2026-10-01）

| 视频输入格式 | 官方入口耗时 | 峰值显存 |
|---|---:|---:|
| PNG连续帧（袋鼠） | 649.33 s | 10,255 MiB |
| MP4（袋鼠） | 666.03 s | 10,257 MiB |
| MP4 + GLB（熊猫） | 615.98 s | 10,255 MiB |

均为16帧、FP16、`--low_ram`、seed42、官方默认100/30步，未启用`--fast`。见[视频输入格式实测](results/input-modes/README.md)及[最初袋鼠结果](results/kangaroo/preview.gif)。

## 目录

- `actionmesh/repo/`：固定官方代码与TripoSG，保留原许可证。
- `actionmesh/weights-manifest.json`、`download_weights.py`：ActionMesh相关权重及国内镜像下载校验。
- `actionmesh/video-weights-manifest.json`、`download_video_weights.py`、`verify_video_weights.py`：前置Wan权重及FP16派生审计。
- `actionmesh/prepare_application_anchor.py`、`render_application_mesh.py`：单图三维重建与正面渲染。
- `actionmesh/run_application_case.py`：固定三例的视频、ActionMesh与导出入口。
- `actionmesh/render_animation_check.py`：实际GLB重新导入、动画和材质检查。
- `results/applications-20261002/`：新结果、原始输入、中间视频、逐帧网格、日志与验收记录。

## Linux GPU 环境

需要 Python ≥3.10、可用的 CUDA PyTorch/torchvision，以及 `aria2c`。
本次使用独立 venv 复用服务器已有的 torch 2.12.1+cu130、torchvision 0.27.1+cu130。

```bash
cd actionmesh
python -m venv --system-site-packages inference-env
inference-env/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-inference.txt
inference-env/bin/pip install --no-build-isolation --no-deps -e repo
python download_weights.py
inference-env/bin/python finish_inference.py --verify-only
inference-env/bin/python finish_inference.py --run
```

`--run` 前必须确认显卡空闲。默认保存 16 帧 GLB 网格、变形数组和 `runtime.json`；单文件动画 GLB 导出另需 Blender。
显存占用以实际测量为准。这里没有训练流程。

## 同步到电脑

```bash
cd /Users/yunbo/Documents/4d
git pull --ff-only
```

模型权重通过脚本单独下载，不随 Git 同步。已生成且上传的结果会随 Git 更新。

## 来源与许可证

- [ActionMesh 官方仓库](https://github.com/facebookresearch/actionmesh)，版本 `d5c01f5045df55819e337369c9617f603c667e00`。
- [TripoSG 官方仓库](https://github.com/VAST-AI-Research/TripoSG)，版本 `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`。
- 两个项目的原始 LICENSE 均保留在对应目录。上游代码归原作者所有；本仓库整理复现流程及实际运行记录。

## 旧袋鼠基线的导出

以下`export_result.py`仅用于旧袋鼠基线；新应用请使用`run_application_case.py --stage export`，以保留各自输入和原始纹理。

完成袋鼠推理后，可以使用 Blender 3.5.1 导出单文件动画和预览：

```bash
cd actionmesh
inference-env/bin/python export_result.py outputs/<运行目录> --blender /path/to/blender
```

导出包括 `animated_mesh.glb`、`preview.gif`、`deformations.npz` 和 `per-frame-meshes.zip`。
预览左侧为官方输入，右侧为生成网格，使用统一蓝色材质和固定灯光。
如果 Blender 缺少本地动态库，可以用 `--library-dir` 指定单独准备的库目录。

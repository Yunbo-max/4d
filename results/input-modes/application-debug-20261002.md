# 文字和图片应用重新调试：2026-10-02

> 此文保存接入前的诊断。后续已补齐前置模型并实际生成三类动态网格，见[端到端重测结果](../applications-20261002/README.md)。下面“未接入”指诊断当时的状态。

结论：此前测试调用了错误的层级。视频加载器拒绝文字或单张图片，不等于 ActionMesh 项目不支持文字或图文应用。当前部署缺少前置生成模型及串联流程，不能宣称官网四种应用已经测完。

## 官方完整流程

依据[官方项目页](https://remysabathier.github.io/actionmesh/)与[论文第 3.4 节](https://arxiv.org/html/2601.16148v2#S3.SS4)：

| 应用 | 论文描述的流程 | 本地完成情况 |
|---|---|---|
| 视频 → 4D | 视频 → ActionMesh → 动态网格 | 已验证 |
| 3D + 文字 → 4D | 正面渲染三维模型 → 图片 + 动作文字生成视频 → 视频与原模型输入 ActionMesh | 仅最后的视频 + GLB 部分已验证 |
| 图片 + 文字 → 4D | 图片生成三维模型 → 使用上面的 3D + 文字流程 | TripoSG 权重已准备；整条流程未接入和验证 |
| 文字 → 4D | 文字生成图片 → 使用图片 + 文字流程；论文也描述文字生成视频再转 4D 的另一条路径 | 前置文字生成流程未接入和验证 |

论文将前置视频模型称为现成的外部模型；此审计未确认其具体模型版本，不能仅根据网页素材文件名宣称严格复刻了作者的视频生成配置。

## 代码和服务器证据

- 2026-10-02 查询官方 `main`：`d5c01f5045df55819e337369c9617f603c667e00`，与当前使用版本相同。
- 官方 `inference/` 下有 `video_to_animated_mesh.py` 与 `video_and_3d_to_animated_mesh.py` 两个脚本；参数为视频 `--input`、可选三维模型 `--mesh_input`，没有文字 prompt 参数。
- 自建 `test_input_modes.py` 的负向测试直接调用 `actionmesh.io.video_input.load_frames`，没有运行文字编码器、图像生成器或视频生成器。
- 远端项目的 `weights/` 包含 `ActionMesh`、`TripoSG`、`dinov2`、`RMBG`；项目 `hf-cache/` 为空。该项目没有准备前置视频生成模型权重。本结论不涉及机器其他无关目录。
- 服务器仍为 RTX 2080 Ti，22528 MiB。CUDA FP16 矩阵乘法实测通过；没有在这次入口拒绝测试中遇到 GPU 显存错误。Diffusers 为 0.39.0。

## 本次最小复现

在服务器已有 `inference-env` 中调用相同加载器：

| 传入 `load_frames` 的内容 | 实际结果 |
|---|---|
| `a kangaroo boxing` | `ValueError: Unsupported input ... Expected video file, image pattern, or directory.` |
| `repo/assets/examples/kangaroo/00.png` | 同类 `ValueError` |
| `publication/input-modes/inputs/kangaroo.mp4` | 成功读取 16 帧 |

报错发生在帧加载阶段。文字在这里被当作文件路径；单个 PNG 文件不属于该入口接收的路径类型。此前仅含一帧的目录还会触发视频序列最少 16 帧的检查。不能通过复制一张图片 16 次就宣称实现了文字驱动或图文驱动动作生成。

## 已纠正与尚缺部分

- 已纠正 README、机器可读状态和测试范围，保留原始负向测试错误消息并注明它仅检查视频加载器。
- 已验证的视频推理产物保持有效；约 10 GiB 显存只适用于这些视频阶段，不代表完整文字或图文流程的显存要求。
- 要实际跑通文字与图片应用，还需选择并准备前置生成模型、实现各阶段衔接、保存文字/图片/视频中间产物，最后端到端测量显存并验证动态网格。
- 本次是根因诊断和报告纠正，没有新生成文字驱动或图文驱动的 4D 结果。

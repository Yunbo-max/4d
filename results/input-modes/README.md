# ActionMesh 输入模式实测

使用 RTX 2080 Ti 22GB 执行官方预训练模型推理。所有生成测试均为 CUDA、FP16、`--low_ram`、seed=42、默认推理步数，未启用 `--fast`。

**2026-10-02 更正：此实验测试视频输入格式，不是官网四种应用的完整测试。当时没有接入文字驱动的前置生成步骤。随后三条完整应用已另行完成，见[2026-10-02端到端结果](../applications-20261002/README.md)；[调试证据](application-debug-20261002.md)保留接入前诊断。**

## 实测结果

| 输入模式 | 结果 | 帧数 | 官方入口耗时 | 峰值显存 |
|---|---|---:|---:|---:|
| PNG 连续帧（袋鼠） | 通过 | 16 | 10 分 49 秒 | 10,255 MiB |
| MP4 视频（袋鼠） | 通过 | 16 | 11 分 6 秒 | 10,257 MiB |
| MP4 + GLB（熊猫） | 通过 | 16 | 10 分 16 秒 | 10,255 MiB |
| 文字 → 4D | 不属于本组格式测试；[后续已验证](../applications-20261002/README.md) | — | — | — |
| 图片 + 文字 → 4D | 不属于本组格式测试；[后续已验证](../applications-20261002/README.md) | — | — | — |

熊猫输出保留 9,606 个顶点、11,118 个三角面，2048×2048 原纹理逐像素一致。所有成功案例都包含 16 帧有效顶点、非零运动与单文件 GLB 动画轨道。

- [袋鼠 MP4 动画预览](kangaroo-video/preview.gif) · [动画 GLB](kangaroo-video/animated_mesh.glb)
- [熊猫 MP4 + GLB 动画预览](panda-video-mesh/preview.gif) · [带纹理动画 GLB](panda-video-mesh/animated_mesh.glb) · [GPU 多视角预览](panda-video-mesh/grid_normal.mp4)
- [完整机器可读汇总](summary.json) · [视频加载器的负向测试记录](unsupported-inputs.json) · [首次缺依赖失败日志](failed-attempts/panda-missing-pytorch3d/inference.log)

## 测试设计

| 模式 | 输入 | 检查内容 |
|---|---|---|
| PNG 序列 → 动态网格 | 官方袋鼠，16 帧 RGBA | 使用此前已完成的 [PNG 推理记录](../kangaroo/report.json)，本轮不重复运行 |
| MP4 → 动态网格 | 同一袋鼠的 16 帧，合成 8 fps MP4 | 视频解码、自动去背景、GPU 推理、16 帧有效动态网格 |
| MP4 + GLB → 动态网格 | 官方熊猫 16 帧合成的 MP4 + 官方 `panda.glb` | GPU 推理、保留输入拓扑、动画 GLB 内嵌纹理 |
| 文字 | `a kangaroo boxing` | 输入加载器应拒绝；此测试没有调用文字生成流程 |
| 单张图片 | 单 PNG 文件、仅含一张 PNG 的文件夹 | 输入加载器应拒绝；序列至少需要 16 帧 |

MP4 通过 OpenCV `mp4v` 编码生成，先将原始 RGBA 图片合成到白色背景；模型实际接收 MP4 文件，没有传入透明遮罩。编码与自动去背景会改变输入，因此 PNG 与 MP4 结果不要求逐顶点一致。

这是一组输入流程验证，不是模型精度 benchmark，也不代表所有视频、所有三维模型都能成功。文字与单图只测试了视频帧加载器的拒绝行为，没有调用文字或图文生成流程；这些记录不能用于判定完整应用不受支持。

## 文件说明

每个成功的生成目录包含：

- `animated_mesh.glb`：含动画的三维模型。
- `preview.gif` / `preview.png`：左侧为实际 MP4 解码画面，右侧为生成几何的蓝色材质预览。
- `deformations.npz`：逐帧顶点和固定面索引。
- `per-frame-meshes.zip`：逐帧 GLB 几何，不含熊猫原始纹理。
- `report.json` / `runtime.json`：验收结果、时间、显存指标。
- `command.json`、`inference.log`、`gpu-memory.csv`、`export.log`：运行证据。
- `SHA256SUMS`：文件完整性校验。

熊猫的纹理保存在单文件 `animated_mesh.glb` 中；蓝色 GIF 用于看形变，不代表原纹理被丢弃。

模型推理使用 GPU；Blender 的几何预览使用 CPU 渲染。报告中的时间是官方推理入口耗时：熊猫模式包含入口自带的 PyTorch3D GPU 渲染；两者均不含后续 Blender 导出、预览渲染和文件传输。

## 复现脚本

从 `actionmesh/` 目录使用已安装依赖和下载好权重的 Linux CUDA 环境执行：

```bash
inference-env/bin/python test_input_modes.py
inference-env/bin/python export_input_modes.py
```

脚本保存本次固定实验目录 `outputs/input-modes-20261001`，若目录已存在会停止，以免覆盖结果。可以通过 `ACTIONMESH_WORKDIR` 指向已有的远端工作目录。导出脚本使用该工作目录下 `tools/blender-3.5.1-linux-x64/blender` 和独立的动态库目录。

原始示例与输入 GLB 来自 `actionmesh/repo/assets/examples/`，保留上游许可证。上游模型代码未修改。

## 视频 + GLB 的额外依赖

此路径的 TripoSGVAE 需要 PyTorch3D CUDA 运算。首次测试在模型编码前因缺少 PyTorch3D 退出，失败记录保留；在项目独立环境中从固定官方版本编译后重试。构建信息见 `actionmesh/pytorch3d-build.json`。没有修改全局 PyTorch/CUDA 或上游模型代码。

```bash
inference-env/bin/pip install ninja iopath fvcore
git clone https://github.com/facebookresearch/pytorch3d.git tools/pytorch3d-source
git -C tools/pytorch3d-source checkout 88e182f989c80836f4bd744e0d9cb1852762ce01
PATH="$PWD/inference-env/bin:$PATH" MAX_JOBS=4 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.5 CUB_HOME=/usr/local/cuda/include \
  inference-env/bin/pip install --no-build-isolation --no-deps ./tools/pytorch3d-source
```

`TORCH_CUDA_ARCH_LIST=7.5` 对应本次 RTX 2080 Ti；其他显卡应按对应计算能力设置。重试失败模式前，应将该模式的旧输出目录移到备份位置，然后运行 `test_input_modes.py --retry panda-video-mesh`。

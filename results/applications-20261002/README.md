# ActionMesh 四种应用：补齐文字驱动流程实测

2026-10-02，在 RTX 2080 Ti 22 GiB 上完成三条新的端到端推理流程。视频输入沿用先前已通过的袋鼠基线，本轮未重跑。没有训练模型。

| 应用 | 本次输入与路线 | 视频前置耗时 | ActionMesh 耗时 | 视频 / ActionMesh 显存峰值 | 结果 |
|---|---|---:|---:|---:|---|
| 文字 → 4D | 章鱼演奏沙锤 → Wan T2V → ActionMesh | 162.54 s | 672.20 s | 10,347 / 10,257 MiB | 16 帧动态网格 |
| 图片 + 文字 → 4D | 官网蘑菇图 → TripoSG → 模型正面渲染 + 唱歌剧 → Wan I2V → ActionMesh + 原网格 | 161.51 s | 624.43 s | 13,201 / 10,255 MiB | 16 帧动态网格，原拓扑保留 |
| 3D + 文字 → 4D | panda.glb → 正面渲染 + 悠闲行走 → Wan I2V → ActionMesh + 原网格 | 165.67 s | 612.15 s | 13,201 / 10,255 MiB | 16 帧动态网格，原拓扑和纹理像素保留 |
| 视频 → 4D | 先前官方袋鼠 MP4 基线 | 不适用 | 666.03 s | 不适用 / 10,257 MiB | [已有结果](../input-modes/kangaroo-video/) |

视频前置耗时包含 CPU 文本编码及视频子进程加载/导出。ActionMesh 耗时为官方入口，含其 GPU 渲染，不含输入校验、视频生成及 Blender 导出。这些列不是整条流程的墙钟计时。

蘑菇的静态重建另用 71.37 s（模型加载、处理及导出，不含哈希和 imports），PyTorch 峰值 allocated 6,303,564,800 bytes、reserved 9,636,413,440 bytes；该阶段未单独采集 nvidia-smi。视频生成与ActionMesh阶段按秒采样的整卡显存最大值为 13,201 MiB（12.89 GiB），测试期间任务串行。该结论只适用于这里的短片配置，不能推广到默认 720p 或更长视频。

## 结果与实际效果

| 输入 | 动画 GLB | 最终动画预览 | 输入视频与网格对照 |
|---|---|---|---|
| 文字 | [章鱼 GLB](text-octopus/animated_mesh.glb) | [正面动画](text-octopus/animated-preview.gif) | [对照 GIF](text-octopus/preview.gif) |
| 图片 + 文字 | [蘑菇 GLB](image-mushroom/animated_mesh.glb) | [正面动画](image-mushroom/animated-preview.gif) | [对照 GIF](image-mushroom/preview.gif) |
| 3D + 文字 | [熊猫 GLB](mesh-panda/animated_mesh.glb) | [带纹理正面动画](mesh-panda/animated-preview.gif) | [对照 GIF](mesh-panda/preview.gif) |

三例均验证了有限几何、固定拓扑、非零顶点运动和 GLB 动画。章鱼 19,997 顶点/39,990 面；蘑菇 19,986 顶点/39,970 面；熊猫 9,606 顶点/11,118 面。

画面检查：章鱼会抬起和放下触手，但沙锤不准确，三维结果存在局部断连和漂浮；蘑菇有嘴部及肢体动作，静态模型没有纹理，不能声称精确还原了“唱歌剧”；熊猫有抬脚、手臂变化，视频仍有形状与大小漂移。技术流程跑通不等于高质量资产或完整论文指标复现。所有视频均无音频。

`preview.gif` 左侧为实际送入 ActionMesh 的16帧，右侧为蓝色几何预览；原相机偏向模型侧后方。`animated-preview.gif` 使用实际导出的 GLB 重新导入 Blender，从正面播放并保留原材质。GLB为8fps、GIF约8fps展示，源视频为24fps，导出播放时长不等于原视频时长。

## 固定模型与参数

- 前置：`Wan-AI/Wan2.2-TI2V-5B-Diffusers`，revision `b8fff7315c768468a5333511427288870b2e9635`，512×512、33帧、50步、CFG5、seed42。CPU FP32 UMT5，Transformer FP16 并保留官方要求的 FP32 模块，VAE FP32、CPU offload及tiling，标准PyTorch SDPA。
- 后置：固定官方 ActionMesh 代码，FP16、`--low_ram`、seed42，保留官方100/30步、不开`--fast`。从33帧中均匀选取16个真实帧；索引与时间保存在各例`input.json`，没有复制静态图充当视频。
- 单图静态重建：官方 TripoSG，100步、CFG7.5、目标40,000面；先卸载RMBG再加载三维模型。
- 两次256×256/17帧/2步冒烟测试仅验证能运行，记录在[common/smoke](common/smoke/)，没有用于最终4D结果。

这里补上的是外部前置生成模型。Wan为本次选择，不代表已确认作者展示所用的具体视频模型版本。纯文字采用论文明确允许的 T2V → ActionMesh 路线；另外两例采用模型正面渲染 → I2V → 视频+网格路线。渲染背景是浅灰，非严格纯白。

完整实际提示词在[application-cases.json](common/application-cases.json)，原图和来源在[蘑菇输入](image-mushroom/original-input/)。每例`frontend/`保留生成视频、全部PNG帧、参数、日志及显存记录；prompt embeddings校验后未打包，原始哈希仍在报告。权重从与官方SHA一致的国内镜像下载，FP16派生权重进行了全量有限性审计，见[权重校验](common/verification/)。权重不上传Git。

## 重跑

使用独立Linux CUDA环境及已有ActionMesh权重。GPU/Torch/PyTorch3D按仓库原说明准备；原生PyTorch3D构建版本见`actionmesh/pytorch3d-build.json`，Blender路径按本机实际配置。

```bash
cd actionmesh
inference-env/bin/pip install -r requirements-applications.txt
inference-env/bin/python download_video_weights.py
inference-env/bin/python verify_video_weights.py --root "$PWD"
```

本次配置固定使用`outputs/applications-20261002/`；脚本拒绝覆盖已有输出，重跑时使用新项目控制目录。输入准备如下，`BLENDER`指向现有Blender3.5.1；缺库时设置独立runtime-libs的`LD_LIBRARY_PATH`，不要更改系统CUDA。

```bash
mkdir -p data
cp ../results/applications-20261002/image-mushroom/original-input/mushroom.png data/mushroom.png
cp ../results/applications-20261002/image-mushroom/original-input/mushroom.json data/mushroom.json
inference-env/bin/python prepare_application_anchor.py --root "$PWD" --image data/mushroom.png --output outputs/applications-20261002/image-mushroom-anchor --seed 42
BLENDER=tools/blender-3.5.1-linux-x64/blender
"$BLENDER" -b -t 4 -P render_application_mesh.py -- --mesh outputs/applications-20261002/image-mushroom-anchor/anchor.glb --output outputs/applications-20261002/mushroom-render
"$BLENDER" -b -t 4 -P render_application_mesh.py -- --mesh repo/assets/examples/panda/panda.glb --output outputs/applications-20261002/panda-render
```

打包器会核对本次的两次smoke记录，因此复现完整证据包时先执行：

```bash
inference-env/bin/python run_application_case.py --root "$PWD" --case text-octopus --stage video --smoke
inference-env/bin/python run_application_case.py --root "$PWD" --case image-mushroom --stage video --smoke
```

分别对`text-octopus`、`image-mushroom`、`mesh-panda`执行三阶段；下面展示文字案例。后端会拒绝失败、smoke或哈希不匹配的前置视频。

```bash
inference-env/bin/python run_application_case.py --root "$PWD" --case text-octopus --stage video
inference-env/bin/python run_application_case.py --root "$PWD" --case text-octopus --stage backend
inference-env/bin/python run_application_case.py --root "$PWD" --case text-octopus --stage export
```

三例都导出后可运行`package_application_frontends.py --root "$PWD"`。实际GLB重新导入及正面检查使用：

```bash
"$BLENDER" -b -t 4 -P render_animation_check.py -- --mesh outputs/applications-20261002/mesh-panda-4d/animated_mesh.glb --output outputs/applications-20261002/mesh-panda-animation-front --view-index 3
```

正面渲染通过后，可用`package_animation_preview.py --check outputs/applications-20261002/mesh-panda-animation-front --result publication/applications-20261002/mesh-panda`生成GIF并保留GLB校验记录。追加预览后需重建SHA256SUMS；本仓库已完成全量重建及验证。

## 来源

[官方项目页](https://remysabathier.github.io/actionmesh/) · [论文§3.4](https://arxiv.org/html/2601.16148v2#S3.SS4) · [Wan官方模型](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)。官网蘑菇图片及熊猫原始模型归原作者所有；本目录保存实际实验与来源，不主张原素材所有权。

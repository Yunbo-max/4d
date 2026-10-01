# 4D 研究复现

当前项目：**ActionMesh（CVPR 2026）官方预训练权重推理**。

## 状态

- 官方代码与依赖已在 RTX 2080 Ti 22GB 上完成导入检查。
- 官方 kangaroo 示例包含 16 帧 512×512 RGBA 图像。
- 权重已下载并校验；**推理验证进行中，尚未发布生成结果**。
- 使用 FP16、`--low_ram`、seed=42，保留官方默认推理步数。

## 目录

- `actionmesh/repo/`：官方代码及示例，保留原许可证。
- `actionmesh/repo/third_party/TripoSG/`：对应固定版本的 TripoSG。
- `actionmesh/weights-manifest.json`：固定模型版本、文件大小、官方 SHA256、哈希一致的国内镜像地址。
- `actionmesh/download_weights.py`：生成断点续传下载清单并下载、校验。
- `actionmesh/finish_inference.py`：校验权重、运行官方示例、检查输出并记录显存。
- `results/`：运行记录与后续生成结果。

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
cd /Users/yunbo/Documents/Codex/4d
git pull --ff-only
```

模型权重通过脚本单独下载，不随 Git 同步。已生成且上传的结果会随 Git 更新。

## 来源与许可证

- [ActionMesh 官方仓库](https://github.com/facebookresearch/actionmesh)，版本 `d5c01f5045df55819e337369c9617f603c667e00`。
- [TripoSG 官方仓库](https://github.com/VAST-AI-Research/TripoSG)，版本 `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`。
- 两个项目的原始 LICENSE 均保留在对应目录。上游代码归原作者所有；本仓库整理复现流程及实际运行记录。

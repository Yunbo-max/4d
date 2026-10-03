# 十项 training-free 4D 方法代码

对应用户提供的 [十项研究计划](../../docs/research-ten/user-ten-hypotheses-20261002.md)。每项是独立干预，模型权重冻结；代码实现与软件测试不等于自然数据上的科研有效性。

| # | 实现文件 | 方法 |
|---|---|---|
| 1 | `m01_elasticity.py` | 观测支持的局部弹性约束 |
| 2 | `m02_cycles.py` | 同一点的解码路径与回环一致性 |
| 3 | `m03_correspondence.py` | 多候选对应的时间联合选择 |
| 4 | `m04_occlusion.py` | 再显露约束下的遮挡轨迹回修 |
| 5 | `m05_scale.py` | 稳定区域的尺度与形变分解 |
| 6 | `m06_selection.py` | 动作门槛下的引导视频选择 |
| 7 | `m07_sampling.py` | 固定查询预算的控制点分配 |
| 8 | `m08_contact.py` | 给定接触法向约束的轨迹修复 |
| 9 | `m09_guidance.py` | 网格与视频两路增量的冲突投影 |
| 10 | `m10_windows.py` | 重叠窗口的前向与反向条件更新 |

本机已在 `/Users/yunbo/Documents/4d/.venv-research` 安装 Python 3.12、PyTorch 2.14.1 和 NumPy 2.5.3。启用这个环境后，从项目 `actionmesh` 目录运行：

```bash
cd /Users/yunbo/Documents/4d
source .venv-research/bin/activate
cd actionmesh
python -m research_ten list
python -m unittest discover -s research_ten/tests -v
python -m research_ten controls --output ../results/ten-controls-new
```

只检查部分项：

```bash
python -m research_ten controls --methods 5 1 8 --output ../results/geometry-controls-new
```

`controls` 使用明确标注的构造样例，保存逐项结果、耗时与模块哈希，禁止覆盖已有目录。它不会自动下载数据、生成视频、训练权重或把构造样例算作 ActionBench 成绩。

本机与远端整套71项测试均已通过，零跳过。完整版本、日志和GPU结果见 [本轮验证报告](../../results/ten-methods-20261003/README.md)。本机当前使用CPU执行这些测试；原生ActionMesh的CUDA检查在远端2080 Ti执行。Mac安装方式依据 [PyTorch官方指南](https://pytorch.org/get-started/locally/)，锁定环境记录位于本轮结果目录。

真实输入与公开 API：

- [1、5、8：几何模块](../../docs/research-ten/geometry-api.md)
- [2、9、10：模型接口](../../docs/research-ten/model-api.md)
- [3、4、7：匹配与轨迹模块](../../docs/research-ten/tracking-api.md)
- [6：视频选择](../../docs/research-ten/selection-api.md)

`native_smoke.py` 是既有远端缓存上的 GPU 接口检查：第2项实际执行冻结解码器的路径组合；可选第9项实际执行同一噪声起点下的两次 **2步** Stage-I rollout。它不代替正式30步推理和完整网格指标。

```bash
PYTHONPATH=/path/to/package-parent python -m research_ten.native_smoke \
  --root /root/rivermind-data/actionmesh-repro \
  --case-dir /path/to/existing/verified/case \
  --output /path/to/new/output --native-guidance
```

现有自然数据尚不能覆盖十项的全部前提：3/4需要真实匹配与可见性，5需要经验证的稳定区域，6需要同任务多视频和评分，8需要可靠接触约束/检测，10需要自然长视频。公开的 APG、IPC、Fast4DMesh 等强基线没有因这些数值控制而自动完成复现。自然失败普查、公平调参、未见样本评价和正式新颖性判断仍须分别完成。

# 2080 Ti：提前准备，23:00 开始跑

默认入口只运行 ActionMesh 原生基线调查。选定 16 个未参加上一轮八样本调查的
ActionBench 资产，依次生成 16 帧网格，再与同一首帧锚点的静止控制一起运行官方
CD-3D、CD-4D、CD-M。样本顺序提前冻结，失败样本不会被替换。

这是建立真实失败证据和核实原生评估器的一轮，尚未执行新候选方法。实际完成
多少对取决于新样本和评分耗时，不保证 16 对全部完成。

## 现在：在已有 GPU 服务器准备

从 GitHub 仓库根目录运行。需要 Linux、Python ≥3.10 和已有 CUDA ActionMesh
推理环境/权重。脚本优先使用 `OVERNIGHT_PYTHON`，其次使用已缓存的
`inference-env/bin/python`，最后使用 `python3`。请先用 `plan` 查看冻结任务。

```bash
git pull --ff-only
bash scripts/run_8h_2080ti.sh plan
bash scripts/run_8h_2080ti.sh prepare
bash scripts/run_8h_2080ti.sh doctor
```

已有资产目录会依次从 `--root`、`ACTIONMESH_WORKDIR`、历史缓存
`/root/rivermind-data/actionmesh-repro`、本仓库 `actionmesh/` 查找。自定义时：

```bash
export ACTIONMESH_WORKDIR=/path/to/actionmesh-repro
export OVERNIGHT_PYTHON=/path/to/inference-env/bin/python
bash scripts/run_8h_2080ti.sh prepare
bash scripts/run_8h_2080ti.sh doctor
```

`prepare` 下载固定版本的官方 ActionBench 新样本，校验 16 帧 PNG、相机文件
和 `(16,100000,6)` 的原生 GT，并锁定输入与实际权重文件哈希。这一步不占执行窗口。
有数据可用 `prepare --data-root /path/to/actionbench --offline`；此时本地数据来源
需由提供者保证，脚本会绑定实际字节，不能单凭目录名证明上游版本。
没有完整模型缓存时会停止并指向已有权重准备流程，不会自动下载另一套模型。

`doctor` 必须返回 `ready_for_baseline_calibration`。它核查 GPU、实际显存、源码、
权重、CUDA 和原生评分器依赖；它没有运行真实模型，不保证新样本全部能装下。
已有记录来自 22 GiB 的 2080 Ti。默认生成队列要求空闲显存至少 10769 MiB、
总显存至少 11000 MiB，并要求没有其他计算进程。标准 11 GiB
卡在足够空闲时可尝试首个完整任务；阈值为历史 10257 MiB 峰值加 512 MiB 余量，
这一历史样本不能保证其他样本都能装下。

## 23:00：启动

用户计划窗口是伦敦时间 2026-10-05 23:00 至 2026-10-06 07:00。
截止时间按**实际启动时间 +8 小时**保存。脚本不会预约开机或自动延长。

```bash
nohup bash scripts/run_8h_2080ti.sh run --hours 8 > overnight-console.log 2>&1 &
```

看进度：`tail -f overnight-console.log`。队列只使用一个指定 GPU、一次一个工作
进程，生成阶段保持 FP16、low_ram、seed42、100/30 步、guidance7.5 和 16 帧。
评分保持原生 100000 个表面点、10000 个 ICP 点、24 个旋转及 200 次 ICP。
完整比较的初始准入估计为 2400 秒，后续参考实际完整单元的最大耗时乘 1.35。
剩余时间不足时停止启动新单元，保留 120 秒用于清理和汇总；正在运行的进程组
超时会被终止。显存/OOM、CUDA 或原生评分器失败会停队列，其他失败保留证据。

断点恢复使用**完全相同的命令和输出目录**。已验证完成的输出不重跑，原截止
时间与失败次数不清零。一个输出目录及同一物理 GPU 都有防重复锁；旧工作进程
仍活着时拒绝重启。源码、输入、设备或协议变更会拒绝恢复。

新窗口需要显式新目录，准备与运行命令都传相同 `--output`。已有窗口的输入
不能用 `prepare` 改写。中断使用终端 Ctrl-C 或向启动进程发送 SIGTERM，保留日志
和已完成阶段；重启会核查阶段回执。

## 明早：取结果

默认目录为仓库根目录下 `results/overnight-2080ti-20261005/`。队列退出时自动写
`REPORT.md` 和 `summary.json`；也可手工刷新：

```bash
bash scripts/run_8h_2080ti.sh report
```

`protocol.json`：源码/数据哈希及冻结参数；`window.json`：原始截止时间；
`queue.json`：全部计划 ID、成功、失败、未跑和尝试历史；`preflight.json`：环境
核查；`gpu-telemetry.jsonl`：资源采样；`logs/`：完整尝试日志；`units/`：生成序列、
静止控制、两个官方评分及带哈希的回执。部分阶段不算完成比较。
成对差值与资产 bootstrap 区间仅描述本轮开发样本/固定推理 seed，不构成新方法
有效性、总体失败率或科学门禁 PASS。实际失败标签还需要结合输入和网格审看。

## 可选：4D-Bench 输入组织基线

今晚默认不需要 Qwen。只在已具备完整原生 QA、视频与未量化 Qwen2-VL-7B-Instruct
本地权重时启用，且与生成窗口使用不同输出目录。示例：

```bash
bash scripts/run_8h_2080ti.sh prepare --suite perception --output results/overnight-qa-20261005 \
  --qa-json /path/to/qa.json --qa-videos /path/to/videos --qwen-model /path/to/Qwen2-VL-7B-Instruct
bash scripts/run_8h_2080ti.sh doctor --suite perception --output results/overnight-qa-20261005
bash scripts/run_8h_2080ti.sh run --suite perception --output results/overnight-qa-20261005 --hours 8
```

按完整对象选取最多 48 个问题；同样的 view1/8/16 各 6 帧，比较官方拼接视频
与三个带视角标签的视频。每臂一次 generate、最多 128 新 token，原生答案解析
及分类宏平均。共同使用 FP16/SDPA，不声称与官方默认 auto dtype 逐位一致。
空闲/总显存至少 18000 MiB，实际适配仍需首次完整原生任务校准。局部时空交互、
反向注意力链及其他候选方法均未启用。记录 raw 输出、无效答案、对象数量、
token 成本与全部计划分母。

## 代码核查范围与来源

软件核查命令：

```bash
PYTHONPATH=actionmesh python -m unittest discover -s actionmesh/research_overnight/tests -v
bash -n scripts/run_8h_2080ti.sh
```

此轮 30 项软件测试全部通过，独立代码复查通过；测试只检查调度、进程清理、
恢复、哈希、原生解析与输入约束。交付端
没有连接用户 GPU，真实模型/原生评分执行必须以服务器新生成的回执为准。
科学范围见 [DESIGN.md](DESIGN.md)，候选准入见 [CANDIDATES.md](CANDIDATES.md)。

- ActionMesh 复用本仓库 `81f4f48330aef6d0c00dce9826335a3b1287850e` 的 44 个
  模型/评估关键源码 blob，运行时逐文件校验；保留已有上游许可证。
- ActionBench：`facebook/actionbench`，数据版本
  `2796071cbe6248422fcbeab3101fa9f9886cb7b9`，128 资产总体元数据随代码保存。
- 4D-Bench：[官方代码](https://github.com/WenxuanZhu1103/4D-Bench)，固定版本
  `f40f49a7539c4c5b1485ad5e80370cf006bbaf53`；Qwen 示例仅作原生解析对照，
  原作者代码归属不变，测试用 AST 提取解析函数，运行不导入完整示例。
- 4D-Bench 原生 QA 数据：`vxuanz/4D-Bench`，已识别数据版本
  `c3b799ddac9c21db0690ad964c1e82a42c892744`；可选入口绑定提供的本地字节，
  不声称已由下载元数据核验该本地包的版本。

# 导出最后 16/16 的原始实验反馈

该工具解决旧回传包缺少 final receipt 所引用原始产物的问题。Linux、Python 3.9+ 标准库即可，不需要 GPU、torch 或模型下载；不启动实验、不重算或更改任何分数。原始 inference/scorer/data/weights 必须仍位于被冻结的项目路径内。请在原运行主机执行，避免把其他主机的进程记录当作已退出。

先从本仓库当前提交的 [export_feedback.py](export_feedback.py) 下载文件到 `/tmp/4d-export-feedback-20261006.py`，运行：

```bash
python3 /tmp/4d-export-feedback-20261006.py \
  --project-root /root/rivermind-data/actionmesh-repro \
  --run-dir /root/rivermind-data/actionmesh-repro/outputs/overnight-2080ti-20261005 \
  --destination /root/rivermind-data/actionmesh-repro/outputs/feedback-20261006-final \
  --expected-fingerprint 3c3fe5a7f9b0ec7b75ce32d2ab08a21bcd36cfc64ae7e9a39b88517d08072d0d
```

`--destination` 必须是尚不存在的新目录，且不与原 run 重叠；已存在时原内容会保留，可另取新名字。不会更新旧 `completion-verification.json`。持有原 runner/supervisor 锁、仍在运行、跨主机进程未核实、源文件改变或任何原始产物丢失时，退出 2 并给出 blocked 原因，不生成半成品包。

成功退出 0 后，新目录包含 `manifest.json` 和 `feedback.tar.gz.part0001` 等所有部分。默认每块最多 24 MiB，需全部带回；manifest 记录各块的大小、SHA256、合并后 gzip tar 的 SHA256，以及每个成员的原始大小和 SHA256。块不是各自独立的 tar，按 manifest 顺序合并才能读取完整归档。

```bash
cat feedback.tar.gz.part* > feedback.tar.gz
sha256sum feedback.tar.gz
# 必须先比对 manifest 的 archive_sha256，再查看内容。
tar -tzf feedback.tar.gz
```

归档内 `run/` 保留整个原运行目录的现存文件，包括逐臂评分、manifest、阶段／最终回执、latent/mesh、失败尝试、环境／资源日志和旧 checkpoint；`source/` 保留冻结的 runner 和 native source。数据和权重不复制，原 protocol 与新 manifest 保留其身份，工具会在导出时校验它们。额外环境和 telemetry 被完整收录不意味着它们的原始记录连续或包含真实显存峰值。

核查包括协议与 window fingerprint、完整冻结 cohort、队列、preflight 及各单元尝试的结束时间／日志／进程身份、已有只读共享锁、四个阶段及 final receipt 的所有文件／哈希、native inference 参数、16 帧记录形状、inputs/preprocessing 引用的全部缓存图片。通过 AST 字面量读取已校验 hash 的原 evaluator 全部 18 个协议字段（不导入执行），核对原生 scorer 来源、完整预算、denominator manifest 和实际消费的 sequence/GT/generation_report；逐项核对 metric→receipt→pair→summary 并重算既有汇总的均值及固定 bootstrap 区间。JSON 使用同一份通过 SHA256 校验的字节解析，首次文件快照不会被后续检查覆盖。归档时再次核对文件内容，之后检查原目录与源文件未改变、读取归档块验证哈希。符号链接、越界路径、重复 JSON key、非有限分数会被拒绝。

**成功仅代表本地文件完整性与记录一致性，始终输出 `scientific_gate_pass=false`、`native_replay_performed=false`。** 它不加载数组验证网格语义、不执行官方 scorer，不代替独立回放、Natural Gate 0、强对照或新方法成败判决。工具测试使用普通文件、记录和 opaque bytes；这些不是自造 native benchmark，也没有模拟 scorer。

```bash
python3 -m unittest discover -s scripts/research_evidence_20261006/tests -v
```

34 项工程测试与红绿记录见 [validation](validation.json)。独立工程代码审查提出的文件快照竞态、预算／时间线、嵌套文件、汇总一致性和 preflight 检查问题均增加了回归用例，终审记录见 [code-review](code-review.json)。在现有 16 对已发布结果上，描述性汇总重算逐项完全相等，见 [元数据兼容检查](published-metadata-compatibility.json)。在当前仅有 GitHub 摘要的快照上实际调用返回 blocked，缺少冻结的原始 runner；这次没有确认 GPU 端 final 16/16 原始闭包已经存在。

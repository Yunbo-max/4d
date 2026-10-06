# 完成本批全部 16 项

用户已授权：即使超过原定八小时，也继续完成本批固定的 16 个资产。

`completion.py` 是外置调度程序。它等待原 `runner.lock` 释放，再核验原协议、输入、代码、GPU 和结果收据，仅执行剩余项。原八小时窗口保留作为历史记录；`completion-authorization.json` 明确记录本次运行时间变更。已完成的生成和评分阶段会复用，模型、数据、随机种子和官方评估参数不变。

续跑没有整批总时长上限，仍使用一个 GPU 任务。遇到损坏的收据、资源故障或用尽已有两次尝试的失败项，会保存真实的未完成状态供调试，不能把它们计为成功。原生成及两组官方评估的有效完整收据达到 16 个，才标记完成。

启动示例（实际路径由已有部署决定）：

```bash
PYTHONPATH=/path/to/frozen-checkout/actionmesh \
  /path/to/inference-env/bin/python completion.py \
  --output /path/to/existing-output \
  --authorization /path/to/existing-output/completion-authorization.json
```

授权记录须绑定冻结协议指纹、全部任务 ID 及脚本 SHA256；记录本身不提供操作权限。已有守护进程时，应核对 `completion-status.json` 的 PID 和启动时间，避免重复启动。

测试（无需模型或 GPU）：

```bash
PYTHONPATH=actionmesh:scripts/overnight_completion \
  python3 -m unittest discover -s scripts/overnight_completion -p 'test_*.py' -v
```

2026-10-05：本地及远端各通过 9 项测试，零跳过。覆盖断点复用、截止后继续、失败记录、完整分母、收据损坏、进程互斥、资源错误和中断清理。研究结论仍须等完整结果解释；这些测试仅验证调度行为。

# 十项方法的代码与验证结果

2026-10-03。对应 [用户原始十项计划](../../docs/research-ten/user-ten-hypotheses-20261002.md)。十项算法代码、输入契约、最小对照及测试已实现；本轮未训练模型权重。

## 全部软件测试已执行

| 环境/检查 | 结果 |
|---|---|
| 本机 Python 3.12.6 / PyTorch 2.14.1 / NumPy 2.5.3 | **71 通过，0 失败，0 跳过** |
| 远端既有 Python 3.12 / PyTorch 2.12.1 CUDA 环境 | **71 通过，0 失败，0 跳过** |
| 十项构造控制，本机已安装Torch的环境 | **10/10 完成** |
| 十项构造控制，远端环境 | **10/10 完成** |
| 原生 ActionMesh GPU 接口检查 | 完成；原始缓存未变；解码器/去噪器权重冻结且无梯度 |

最初系统 Python 缺少 Torch 的两项跳过已补齐。本机安装于 `/Users/yunbo/Documents/4d/.venv-research`，当前测试使用 CPU；CUDA 模型检查在远端 RTX 2080 Ti 进行。

- [本机全部71项日志](local-torch-tests-final.log)
- [远端全部71项日志](remote/research/ten-methods-20261003/remote-tests-final.log)
- [本机安装记录](local-install.json)、[环境信息](local-environment.json)、[Mac版本锁定记录](local-macos-requirements.lock)
- [本机十项构造控制](local-torch-controls/report.json)、[远端十项构造控制](remote/outputs/ten-methods-20261003/remote-controls-final/report.json)
- [机器可读汇总](verification-summary.json)

## 原生 GPU 检查

沿用已核验的 mannequin / seed42 / attempt2 缓存，全部16帧 Stage-I latent 固定，不读 GT。总墙钟170.31秒，物理显存采样峰值10229 MiB（约9.99 GiB）。

**第2项：**在0、5、10、15时刻对32个原始材料点执行真实冻结解码路径。原解码重现的 RMS / 锚网格对角线为 `3.743351e-8`，通过 `1e-4` 接口容差。路径探针额外使用一环支持点重算法向，总计939个“点×目标时刻”查询，4次解码调用；另有96个点×时刻的原生重现检查。原始法向和中间帧法向都使用对应官方约定。

最大回环 RMS 残差0.010112，最大复合路径 RMS 残差0.005165，单位为原模型坐标。这些是预测之间的分歧，未测出其与自然GT误差的关系，也未证明一致性拟合提升重建质量。

**第9项：**同一初始噪声分别运行标量CFG和冲突投影CFG，每组真实2步、每步3个模型分支，总计12次原生分支调用。两组锚帧均逐位保持，第二步读取了各自更新后的采样状态；最终latent RMS差异0.183322。**这是2步接口检查，尚非正式30步生成或最终网格质量比较。**

- [GPU原始报告](remote/outputs/ten-methods-20261003/native-smoke-v1/report.json)
- [路径数组](remote/outputs/ten-methods-20261003/native-smoke-v1/native-cycles.npz)
- [两组2步latent](remote/outputs/ten-methods-20261003/native-smoke-v1/guidance-two-step.npz)
- [采样日志](remote/research/ten-methods-20261003/native-smoke-v1.log)
- [GPU测试代码与交付代码哈希一致性](native-source-check.json)

## 科研结论边界与不利证据

全部已有软件测试已经执行；原计划的自然失败普查、16/16/96划分、强外部基线、公平调参和完整benchmark实验尚未完成。本轮不能把构造样例成绩或接口通过写成十个方法有效或新颖。

- 第1项权重打乱控制在共同20次迭代内未达到收敛容差，结果保留；不会用该点宣称已击败充分调优ARAP。
- 第7项在较小预算下与FPS打平，构造结果全部保留；空间IDW插值不是Fast4DMesh复现。
- 第8项在单接触构造例中与标准接触投影打平。ARAP对照中的碰撞顶点断开，明确标为弱对照。采样顶点—三角形约束不提供IPC或连续碰撞保证。
- 第10项修复了“只复用不变的前向重叠”错误；现在从可靠性掩码允许的末端伪边界重新采样，逐窗传回更新的重叠信息，只固定原始输入锚点。末端预测不是GT；尚缺自然长视频及共享latent表示资格验证。
- 第3/4项需要真实匹配及可见性；第5项需要验证过的稳定区域和相机约定；第6项需要同任务候选视频、统一评分及下游排名实验。没有补造这些输入。

`controls-first`、初始红色测试日志及几何扩展诊断保留为开发历史；当前结果以 `local-torch-controls` 与 `remote-controls-final` 为准。完整代码与逐项实际输入示例见 [research_ten说明](../../actionmesh/research_ten/README.md)。

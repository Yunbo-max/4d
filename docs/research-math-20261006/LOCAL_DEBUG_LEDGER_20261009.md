# Local 测试问题与修复总账（2026-10-09）

本页记录 Local 在 SSH 主机上实际遇到的问题、根因、修改与验证证据。失败记录保留；未解决项明确列出。

## 当前验证结果

- 最新代码：`462b6f5e5d23a98d9e71e66e5bb6025d6f42c577`。
- 最新运行：`family-provenance-green-001`，**852 项检查，846 通过、6 跳过、0 失败、0 错误**，127.390 秒。
- 6 项跳过需要真实 C01/C03/C04/C06/C07/C08 工件，仍未完成，不能算通过。
- 全部 workload 通过 committed plan builder + 已安装的 Research_Autopilot harness；上述检查均为零 GPU。
- 15 个 idea 的源码存在；native 方法结果仍 **0/15**，GPU STOP 保持有效。

[最新计划、回执与日志](../../rounds/20261008-native-context/windows/20261009-native-prerequisites/RESULT.json) · [此前七轮完整记录](../../rounds/20261008-native-context/windows/20261009-local-cpu-acceptance/RESULT.json) · [资产下载与三项 CPU 准入](../../rounds/20261008-native-context/windows/20261009-local-asset-admission/REVIEW_PACKET.md)

## 代码问题：根因与解决办法

| 编号 | 实际问题 | 怎么解决 | 修改提交 |
| --- | --- | --- | --- |
| S01 | 隔离 staging 缺少 research_census_case.py，计划找不到依赖 | 把实际导入依赖加入 committed builder 的源码闭包；harness 暂存完整源码。 | [15b89a3](https://github.com/Yunbo-max/4d/commit/15b89a3) |
| S02 | raw archive 的成员顺序与验证器要求不一致，阻断 C01/context 消费 | 生产端输出排序且唯一的成员清单；仍逐项重算文件哈希，身份错误继续拒绝。 | [15b89a3](https://github.com/Yunbo-max/4d/commit/15b89a3) |
| S03 | C01 未完整校验 B0 的原始上下文来源 | 从正确的项目 root 执行 B0 上下文校验；伪造来源/错误引用继续拒绝。 | [15b89a3](https://github.com/Yunbo-max/4d/commit/15b89a3) |
| S04 | C05 使用未定义的 b0_sequence_ref；GPU计划缺少设备UUID清单 | 修正实际 sequence 引用；为harness计划提供实际环境的GPU UUID、显存安全余量及单卡任务限制，移除错误memory_profile_ref。GPU计划仅做CPU检查，未执行。 | [15b89a3](https://github.com/Yunbo-max/4d/commit/15b89a3) |
| S05 | C05 producer/consumer 命名空间、模型清单 JSON 与生成成员声明不一致 | 保留 producer 的真实模型命名空间；仅对已精确绑定的 model_ref 使用专门清单校验，避免递归把外部权重当成 CPU 工件输入；修正 stale fixture 和角色顺序。 | [5557f3a](https://github.com/Yunbo-max/4d/commit/5557f3a) |
| S06 | C14 corotational role/method_id 与固定构造身份不一致 | 修正生产比较记录中的候选角色常量，并让 fixture 使用真实固定 method_id；候选身份校验保留。 | [15b89a3](https://github.com/Yunbo-max/4d/commit/15b89a3) / [5557f3a](https://github.com/Yunbo-max/4d/commit/5557f3a) |
| S07 | C06/C07 平衡运输 Sinkhorn 在冻结迭代数内未闭合边际，后续 sequence 缺失 | 在原稀疏熵对偶上加入有界 Newton-CG 搜索；目标、epsilon、支持、边际及最终误差门槛保持原值。缺失结果没有填成成功。 | [e079ae8](https://github.com/Yunbo-max/4d/commit/e079ae8) |
| S08 | C07 部分运输 3000 次后 complementarity/fixed-point 仍超限 | 加入保持 u,v≤0 的投影 Newton-CG 和真实投影步长线搜索；保持容量约束和 KKT 判定门槛，失败仍终止保留。 | [8bf2a80](https://github.com/Yunbo-max/4d/commit/8bf2a80) |
| S09 | C08 endpoint scaling 不收敛 | 为相同端点缩放问题加入有界 Newton 更新；保留最终收敛残差与失败报告。 | [e079ae8](https://github.com/Yunbo-max/4d/commit/e079ae8) |
| S10 | C13 float32 导出量化后 primal/dual gap 超出 1e-7 | 将数学正确性检查与 float32 导出检查分开；数学检查使用 float64，成功导出 fixture 采用可精确表示的仿射数据；保留量化失败必须拒绝导出的专项检查，没有放宽阈值。 | [5557f3a](https://github.com/Yunbo-max/4d/commit/5557f3a) |
| S11 | 修复代码后 G01 implementation closure 哈希过期或顺序非 canonical | 逐一重算真实代码引用，按规则排序 C03/C07 清单，经独立审查后冻结新 design digest；不是跳过哈希检查。 | [108a703](https://github.com/Yunbo-max/4d/commit/108a703) |
| S12 | C02/C03/C04/C12/控制计划等fixture配置或预期报错过期；native_context测试读取未赋值command | 修正C03跨root校准定位和flat official metrics fixture、C02收敛fixture；C04 fixture的ARAP迭代数由2改100（不改容差）；控制检查更新C13 full_method_source_complete期望；native_context测试改读job[command]。终端失败证据仍强制保留。 | [15b89a3](https://github.com/Yunbo-max/4d/commit/15b89a3) / [5557f3a](https://github.com/Yunbo-max/4d/commit/5557f3a) |
| S13 | C10 把二维 3×3 矩阵按三维索引；精确重建未恢复导出器的 frame-zero anchor | 修正矩阵索引；验证器执行与导出器相同的 frame-zero 原值恢复，继续要求 exact reconstruction。 | [15b89a3](https://github.com/Yunbo-max/4d/commit/15b89a3) / [5557f3a](https://github.com/Yunbo-max/4d/commit/5557f3a) |
| S14 | G01 的 generation identity 负例先被更早的 admission 检查拦下 | 让负例先具备正确绑定的 admission，再仅改变 generation identity，独立验证所针对的拒绝条件。 | [d4c8478](https://github.com/Yunbo-max/4d/commit/d4c8478) |
| S15 | G01 late-review 负例只更新部分重复 envelope，先在请求/回执哈希一致性处失败 | 同步 fixture 内 comparison/scoring/raw/result 的绑定；保留晚于 score 完成的 review 时间和原始拒绝断言。 | [6241605](https://github.com/Yunbo-max/4d/commit/6241605) |
| S16 | C06 正向 fixture 使用 producer 当前不支持的 generation seed | 正向 fixture 固定为实际支持的 seed42；明确保留其他种子未准入，不声明多 seed native 覆盖。 | [8bf2a80](https://github.com/Yunbo-max/4d/commit/8bf2a80) |
| S17 | 新发现：G01 把每个资产的 provenance 哈希当作统计家族身份；预期55家族得到128 | derivation 1.1.0 仅用已验证的上游家族键计算 family_id；metadata/geometry/extractor 全部哈希继续独立校验和绑定。拒绝旧版本/旧规则；增加不同文件同家族、篡改与旧格式拒绝检查。独立复核后 852 项检查通过（6项跳过）。 | [462b6f5](https://github.com/Yunbo-max/4d/commit/462b6f5) |
| S18 | supervisor 诊断优先级与故意重签名的负例链条不一致 | 保留不可忽略的 predecessor/事件完整性错误，调整精确诊断优先级；负例更新重复 envelope 后才测试目标故障。未安装或启动在线 supervisor。 | [5557f3a](https://github.com/Yunbo-max/4d/commit/5557f3a) |

## 每轮实际结果

| 运行 | 总数 | 失败 | 错误 | 跳过 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| common-cpu-55f465a-001 | 848 | 23 | 47 | 6 | failed |
| debug-r1-cpu-001 | 848 | 19 | 30 | 6 | failed |
| debug-r2-cpu-001 | 未完成 | — | — | — | 120秒超时，保留 partial attempt |
| debug-r3-cpu-001 | 849 | 3 | 14 | 6 | failed |
| debug-r4-cpu-001 | 849 | 1 | 0 | 6 | failed |
| debug-r5-cpu-001 | 849 | 1 | 0 | 6 | failed |
| debug-r6-cpu-001 | 849 | 0 | 0 | 6 | completed |
| family-provenance-red-002 | 850 | 1 | 0 | 6 | failed：128 != 55 |
| family-provenance-green-001 | 852 | 0 | 0 | 6 | completed |

120秒超时的处理：保留真实中断目录，在新源码和新 run ID 上显式设置更充足的 CPU 上限；没有更改算法收敛容差或重写旧结果。最新两轮各一次、零重试，内部上限600秒、外部660秒。`family-provenance-red-001` 只生成了120秒计划，审查时发现预算不足，未执行；实际 red 运行是 `-002`。

## 安装、下载和控制器问题

| 问题 | 处理及当前状态 |
| --- | --- |
| 当前设备与历史记录不符 | 实测为 Tesla T4，UUID `GPU-305023ac-8457-0ac4-0432-4ff19b30de46`，所有后续授权须绑定实际设备；不沿用历史2080Ti身份。 |
| Research_Autopilot 版本/安装内容不完整 | 更新并安装完整 skill family 到 `bd368630ef623efa16c6c28fab894d692f82ee4d`，保留原插件。 |
| HF 主站无法访问、镜像分页回到主站 | 使用固定 revision 和 `hf-mirror.com`，仅将相应分页请求路由回镜像；三项资产准入已完成。 |
| 大权重 HTTP 下载过慢 | 切换 hf_xet，保留已完成资产及原始 partial；五个 snapshot 共2356文件已准入。 |
| DINOv2/TripoSG Xet HTTP401 | 查到镜像缓存的公开下载 token 已过期；对 token metadata 请求加 cache-busting nonce；未改资产revision，随后下载成功。 |
| GitHub/submodule/SSH传输中断 | 使用固定提交的 Git bundle 和 legacy SCP，核对完整SHA-256；原始 partial 保留。 |
| 全量/稀疏 Git checkout 触发历史大对象拉取、HTTP2/连接超时 | 停止自己的未完成控制器进程并保留目录；从干净的已接收源码副本创建新隔离目录，增量 bundle 导入准确提交，核对 HEAD 和 clean status 后执行。 |
| 源码复制遇到 dangling pretrained_weights 链接、递归忽略规则漏复制历史 tracked 文件 | 保留符号链接本身，并从原源树恢复所有已跟踪缺失文件；最终 clean status 已核对。用户原工作区未改写。 |
| Conda solve 缺 iopath，官方 iopath channel 只提供旧Python构建 | 添加清华 conda-forge，锁定兼容 Python3.11 的 iopath0.1.10；Torch2.4/CUDA12.1/PyTorch3D0.7.8 core 已安装。 |
| 清华不提供 pytorch3d/nvidia channel | 主包用清华；这两个渠道使用官方Anaconda源，保存精确包URL/MD5/SHA-256清单。 |
| explicit安装意外读取 defaults，触发ToS拒绝 | 显式 override-channels，移除无关默认渠道；未替用户接受新条款，随后安装exit0。 |
| Conda explicit --json 返回空stdout | 安装exit0、Python路径及实际Conda记录可见；保留空日志，不伪造JSON成功结果，后续使用规定harness采集实际元数据。 |
| DISO pip隔离构建报 No module named torch | 已定位到 setup.py 需要导入Torch，临时build环境不包含它；32个其他依赖已按版本/哈希锁安装。DISO须在已安装Torch环境中构建，并准备与Torch匹配的CUDA12.1编译器；当前仍在处理，未宣称完整native环境就绪。 |
| Objaverse mirror API403 | 用官方API固定revision `556637099bf4fa79ea7b239d0d9c328b8a2e9ac8`；固定revision的镜像文件端点可读。全部128对象已匹配索引，89元数据分片已在Local保留。 |
| 原始几何HTTP下载超时/SSL EOF | 记录原失败和partial，改用具有公开token刷新钩子的Xet下载；下载仍进行，未生成家族准入。 |

## 尚未解决 / 不能宣称完成

- 家族分组程序缺陷已修复，但**真实128样本家族映射尚未准入**。共作者、相同字符串或独立文件哈希均不足以证明统计独立；仍需上游谱系、几何证据和独立复核。
- DISO构建、完整native runtime capture/兼容性和baseline/scorer qualification仍在继续。
- 六项真实工件检查及15个候选的Gate0/IPCG/科学准入和官方GPU评分未完成。
- 下载/安装成功、CPU测试通过均不改变GPU STOP，也不替代具体GPU执行授权。

## 所有失败条目索引

从保留的日志逐项抽取了 **71 个不同 FAIL/ERROR 标题、139 次失败/错误记录**。每项包含原始运行、报错、日志行号、SHA-256、关联修复组及最新验证。超时另见上表。

[完整机器可读索引](LOCAL_DEBUG_FAILURE_INDEX_20261009.json)

| 测试模块 | 不同失败条目 | 修复组 | 最新检查 |
| --- | ---: | --- | --- |
| `test_area_admission_candidate` | 1 | S12 | 全部对应检查通过 |
| `test_area_transport_candidate` | 4 | S07, S16 | 全部对应检查通过 |
| `test_c01_native_acceptance_plan` | 1 | S02, S03 | 全部对应检查通过 |
| `test_c01_native_scoring_plan` | 1 | S03, S12 | 全部对应检查通过 |
| `test_c02_native_comparison` | 1 | S01, S12 | 全部对应检查通过 |
| `test_c03_calibration_artifacts` | 1 | S12 | 全部对应检查通过 |
| `test_c03_native_scoring_profile` | 2 | S12 | 全部对应检查通过 |
| `test_c04_native_comparison` | 1 | S12 | 全部对应检查通过 |
| `test_c05_candidate_artifacts` | 9 | S04, S05 | 全部对应检查通过 |
| `test_c05_native_comparison` | 4 | S04, S05 | 全部对应检查通过 |
| `test_c05_plan_staging` | 2 | S01, S04, S05 | 全部对应检查通过 |
| `test_c10_native_comparison` | 1 | S13 | 全部对应检查通过 |
| `test_c14_native_comparison` | 4 | S06, S12 | 全部对应检查通过 |
| `test_c14_native_scoring_plan` | 1 | S06, S12 | 全部对应检查通过 |
| `test_control_plan` | 1 | S12 | 全部对应检查通过 |
| `test_g01_design` | 13 | S11, S14, S15, S17 | 全部对应检查通过 |
| `test_group_acceleration_candidate` | 4 | S10 | 全部对应检查通过 |
| `test_integrable_gradient_candidate` | 2 | S13 | 全部对应检查通过 |
| `test_native_b0_contract` | 3 | S03, S11 | 全部对应检查通过 |
| `test_native_context_consumption_plan` | 2 | S02 | 全部对应检查通过 |
| `test_native_context_delivery` | 2 | S02 | 全部对应检查通过 |
| `test_native_context_plan` | 1 | S02, S12 | 全部对应检查通过 |
| `test_partial_transport_candidate` | 4 | S07, S08 | 全部对应检查通过 |
| `test_research_supervisor` | 3 | S18 | 全部对应检查通过 |
| `test_self_map_plan` | 1 | S02 | 全部对应检查通过 |
| `test_trajectory_bridge_candidate` | 2 | S09 | 全部对应检查通过 |

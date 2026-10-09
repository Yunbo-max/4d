# Local native prerequisites — 20261009-native-prerequisites

## 这次完成了什么

修复 G01 将每个资产的文件哈希误用为统计家族身份的问题。实际的 red run 在源码 `fe82cd1d3574b805d3c68cccad5685f3dde7910c` 上得到 **128 != 55**，850项检查中1失败、6跳过。独立源码复核后，green run 在 `462b6f5e5d23a98d9e71e66e5bb6025d6f42c577` 上完成 **852项检查，846通过、6跳过、0失败/错误**，耗时127.390秒。

新的 derivation 1.1.0 仅使用已验证的 upstream_family_keys 计算 family_id。所有原始metadata、geometry和extractor哈希继续逐项验证并绑定到receipts、derivation、evidence和split。旧版本、旧assignment rule不会自动迁移。新增检查覆盖不同资产文件仍归同一家族、provenance篡改、旧版本/规则拒绝。

家族键必须具有一致的谱系含义，仍需独立复核。此修复没有把真实128样本映射标为已准入，没有改变数值效应门槛、候选实现、12/12/至少30家族切分要求或GPU授权边界。

## 可核对的证据

- [RESULT.json](RESULT.json)：两次真实运行的提交、harness digest、receipt、日志和原始归档SHA-256。
- [evidence/](evidence/)：red/green计划、harness报告、attempt、process guard、execution context及stdout/stderr。
- [setup/](setup/)：固定版本依赖解析、安装记录、控制器下载/传输脚本与失败快照。原始记录仍保留于SSH；下载URL中的签名信息已脱敏。
- [问题与解决办法总账](../../../../docs/research-math-20261006/LOCAL_DEBUG_LEDGER_20261009.md)：当前及前七轮问题、根因、修复提交和所有失败条目索引。

两次检查均通过已安装的Research_Autopilot `bd368630ef623efa16c6c28fab894d692f82ee4d`执行：1 CPU、2048MiB、0 GPU、每次1 attempt、0 retry；内部600秒、外部660秒。先生成的 `family-provenance-red-001` 120秒计划没有执行，仍留在原目录。

Local已校验delivery archive以及两份完整raw archive的SHA-256。完整原始归档位于SSH `/root/rivermind-data/4d-native-prerequisites-20261009/`，匹配副本位于Local `results/native-prerequisites-20261009/`。

## 安装和数据准备状态

隔离Conda core已安装：Python3.11、Torch2.4.0、CUDA12.1、PyTorch3D0.7.8、NumPy1.26.4、SciPy1.13.1。32个额外依赖已按解析得到的固定版本和包哈希从清华镜像安装。DISO因隔离build环境看不到Torch而失败；正准备匹配的CUDA12.1编译工具链，再在已安装Torch的环境中构建。尚未把完整native环境标为合格。

固定Objaverse revision `556637099bf4fa79ea7b239d0d9c328b8a2e9ac8` 的索引已匹配128/128个样本，89个原始元数据分片已在Local保存。SSH原始几何下载的timeout/SSL EOF及partial保留，Xet下载继续。未生成或伪造真实家族review。

## 科学状态

GPU STOP仍有效。6个真实工件检查仍跳过；native方法结果与Local方法验证仍0/15。软件修复通过、下载或安装完成均不构成Gate0/IPCG、baseline/scorer qualification或精确GPU-resume授权。

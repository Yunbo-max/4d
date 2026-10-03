# 全部实验结果索引

更新日期：2026-10-03。这里汇总本地已整理的全部十批实验结果，保留成功、失败、负面结果及开发历史。测试通过、接口可用与真实数据质量提升是不同结论；具体证据范围以每批 README 为准。

| 批次 | 内容与结论范围 |
|---|---|
| [ten-methods-20261003](ten-methods-20261003/README.md) | 十项 training-free 候选；本机与远端各71项测试通过、零跳过；十项构造控制及第2/9项原生GPU接口检查；未完成自然benchmark验证 |
| [math-prototypes-20261002](math-prototypes-20261002/README.md) | 三个数学原型、41项测试、两例冻结解码器局部GPU探针 |
| [census-20261002](census-20261002/README.md) | 8个资产、10次推理与官方评估；阶段、时间方向、表面修复及运动诊断，保留不利结果 |
| [three-ideas-20261002](three-ideas-20261002/README.md) | 第一轮三类假设的运行、编辑与独立审计 |
| [round2-ideas-20261002](round2-ideas-20261002/README.md) | 第二轮几何、时间与效用实验，以及失败/重试记录 |
| [new-directions-20261002](new-directions-20261002/README.md) | 新方向的开发性记录与清单 |
| [preflight-20261003](preflight-20261003/README.md) | 远端环境、输入和源码检查；本批没有启动训练 |
| [applications-20261002](applications-20261002/README.md) | 文字、图片+文字、3D+文字的端到端样例、动画与限制 |
| [input-modes](input-modes/README.md) | 视频输入格式实测及应用入口诊断 |
| [kangaroo](kangaroo/) | 最初的视频→4D袋鼠基线；[预览](kangaroo/preview.gif)，详情见[项目首页](../README.md) |

## 大体积原始归档

九个原始 census 压缩包合计 **1,395,634,341 字节**，作为同一仓库的 [GitHub Release 附件](https://github.com/Yunbo-max/4d/releases/tag/results-2026-10-03) 保存。下载地址、精确字节数和SHA256见 [归档清单](raw-archives-20261003.json)。归档包含中间网格、数组及阶段证据，部分历史包内容重叠；原样保留以便追溯。

普通 `git pull` 获取仓库内的报告、指标、动画、数组与脚本；大归档需另行下载。已配置 GitHub CLI 时可以执行：

```bash
gh release download results-2026-10-03 --repo Yunbo-max/4d --pattern '*.tar.gz' --dir raw-results-20261003
```

[发布文件清单](publication-files-20261003.json) 记录本次结果文件的路径、字节数和SHA256。该清单不包含自身，以避免自引用哈希。模型权重、Python环境及原始下载数据集不属于实验输出，继续通过原有下载流程取得。

## 代码

- [十项方法](../actionmesh/research_ten/README.md)
- [三个数学原型](../actionmesh/research_math/README.md)
- 历史运行、评估、导出脚本位于 `actionmesh/research_*.py`，对应协议和测试一并保留。

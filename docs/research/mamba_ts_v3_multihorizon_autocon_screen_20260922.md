# MambaTS v3 长预测 AutoCon screen（2026-09-22）

## 结论

72/72 个 validation-only 作业成功，覆盖 ETTh1、ETTm1、Electricity、Traffic，H336/H720，seed `2021/2022/2023`，且测试工件为零。严格分析器检查了 phase validation、source hash、manifest、配置、checkpoint 轨迹和逐作业测试隔离。

结果不支持把现有 AutoCon 目标作为精度贡献：同构 `MambaTSv3AutoCon lambda=0.001` 相对 `lambda=0` 的平均验证 MSE 变化全部接近零，最大退化为 ETTm1/H720 的 `+0.110%`，最小改善为 Traffic/H336 的 `-0.005%`。8 个数据集-horizon 单元中没有稳定一致的胜出模式。因此当前实现应保留为可运行的自监督表示学习接口和消融对象，不能写成预测增益。

## 配对结果

| 数据集 | H | `lambda=.001` vs `lambda=0` | 胜出 seed | `lambda=0` vs plain v3 | 胜出 seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 336 | `+0.002%` | 0/3 | `-0.054%` | 1/3 |
| ETTh1 | 720 | `+0.001%` | 1/3 | `-0.358%` | 2/3 |
| ETTm1 | 336 | `+0.005%` | 2/3 | `+0.819%` | 1/3 |
| ETTm1 | 720 | `+0.110%` | 1/3 | `+0.354%` | 1/3 |
| Electricity | 336 | `+0.011%` | 1/3 | `+0.892%` | 0/3 |
| Electricity | 720 | `+0.013%` | 1/3 | `+0.216%` | 1/3 |
| Traffic | 336 | `-0.005%` | 2/3 | `-0.056%` | 2/3 |
| Traffic | 720 | `-0.001%` | 1/3 | `+0.285%` | 0/3 |

第一列是唯一能够归因于增量 AutoCon objective 的对比；`lambda=0` vs plain 只反映 AutoCon head/架构开销，不能作为 objective 的因果结论。

## 复核边界

- 冻结计划：[`mamba_ts_v3_multihorizon_autocon_screen_20260922.json`](mamba_ts_v3_multihorizon_autocon_screen_20260922.json)
- 严格分析器：[`analyze_mamba_ts_v3_multihorizon_autocon_screen.py`](../../scripts/analyze_mamba_ts_v3_multihorizon_autocon_screen.py)
- 原始机器读数：[`readout/`](../../artifacts/exploratory/mamba_ts_v3_multihorizon_autocon_screen_20260922/readout/)
- 计划虽在执行前已冻结，但仍标识为 `authorized=false` exploratory snapshot；不能追溯改称确认性或预注册研究。
- 预算仅 `8` epochs，主指标是最小 pure-forecast validation loss；没有测试指标、外部基线长预测复现、统计显著性或端到端效率证据。

下一轮只有在设计出与 `lambda=0` 有可测差异的目标、并保持三臂与强基线同预算的前提下才值得继续。单纯扩大本 AutoCon 权重或把零增益结果包装成创新，不具备论文价值。

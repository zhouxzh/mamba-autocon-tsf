# MeanStd 公平基线 pilot 复核稿

日期：2026-09-07  
状态：探索性、validation-only、短预算；不是论文确认性结果。

## 复核范围

本轮在同一当前源码快照下完成 84 个作业：ETTh1、ETTm1、Electricity、Traffic；M 特征；H96；seed 2021/2022/2023；DLinear、NLinear、PatchTST、TimesNet、iTransformer、Mamba-1、Mamba-2；FP32、batch 32、4 epoch、patience 2、`AutoCon_wnorm=MeanStd`、`skip_test=true`。

此前 DLinear/NLinear 原实现没有窗口归一化。本轮为它们增加了默认关闭的 opt-in `MeanStd` 路径，默认 `LastVal` 行为不变；因此这批结果只在本轮合同内解释。

完整工件：[`mamba_meanstd_baseline_pilot_20260907`](../../artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907/)  
汇总脚本：[`summarize_meanstd_baseline_pilot.py`](../../scripts/summarize_meanstd_baseline_pilot.py)  
phase 校验：[`phase_validation.json`](../../artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907/phase_validation.json)

## Best validation MSE

下表为三 seed 的 mean +/- sample SD，数值是 validation scaled MSE；不同数据集之间不能横向比较。

| 数据集 | DLinear | NLinear | PatchTST | TimesNet | iTransformer | Mamba-1 | Mamba-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.696868 +/- 0.001945 | 0.698166 +/- 0.002689 | **0.690529 +/- 0.001980** | 0.765462 +/- 0.007307 | 0.693969 +/- 0.000593 | 0.723166 +/- 0.006122 | 0.734823 +/- 0.012380 |
| ETTm1 | 0.408890 +/- 0.001194 | 0.408598 +/- 0.003519 | 0.403197 +/- 0.006634 | 0.440143 +/- 0.009213 | **0.395980 +/- 0.001281** | 0.419661 +/- 0.005393 | 0.411823 +/- 0.007226 |
| Electricity | 0.202337 +/- 0.000040 | 0.202566 +/- 0.000040 | 0.198308 +/- 0.000404 | **0.175228 +/- 0.004626** | 0.198961 +/- 0.000692 | 0.182028 +/- 0.001523 | 0.179172 +/- 0.002852 |
| Traffic | 0.507145 +/- 0.000056 | 0.508260 +/- 0.000127 | 0.441408 +/- 0.005265 | **0.438170 +/- 0.011180** | 0.445266 +/- 0.003601 | 0.493479 +/- 0.001356 | 0.491952 +/- 0.006470 |

相对每个数据集最强非 Mamba 均值，Mamba-1/Mamba-2 的变化分别为：ETTh1 `+4.73%/+6.41%`、ETTm1 `+5.98%/+4.00%`、Electricity `+3.88%/+2.25%`、Traffic `+12.62%/+12.27%`。Traffic 中 Mamba-1/2 对 PatchTST、TimesNet 和 iTransformer 均为 `0/3` seed 胜出；Electricity 中两者对 PatchTST 和 iTransformer 均为 `3/3` 胜出，但对 TimesNet 不占优。

## Prediction-only 资源

RTX 5090 D、device-resident synthetic input、batch 8、10 warm-up、50 timed forwards。median latency（ms）如下；不含数据加载、传输、反向、优化器或 relation loss。

| 数据集 | DLinear | NLinear | PatchTST | TimesNet | iTransformer | Mamba-1 | Mamba-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.095776 | 0.056048 | 0.275344 | 2.694528 | 0.157648 | 0.423728 | 0.706816 |
| ETTm1 | 0.094048 | 0.055664 | 0.270176 | 2.719744 | 0.155344 | 0.420320 | 0.697824 |
| Electricity | 0.091696 | 0.053776 | 0.412272 | 2.863312 | 0.193456 | 0.435056 | 0.719792 |
| Traffic | 0.103056 | 0.063392 | 0.889584 | 2.821360 | 0.343424 | 0.462464 | 0.738288 |

Mamba-1/2 比 TimesNet 快，但比 iTransformer 和线性基线慢；这不支持“普遍更快”，也不是端到端训练速度结论。完整 p95、吞吐、参数量和峰值显存见 artifact 的 [`model_resources.csv`](../../artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907/resources/model_resources.csv)。

## 结论边界

- MeanStd pilot 消除了 DLinear/NLinear 与其他模型之间的一个归一化混杂，但 4 epoch 仍是筛选预算，不代表收敛性能。
- 当前 Mamba-1/2 没有在四个数据集上追平最强非 Mamba 基线；结果明显依赖数据集。
- Mamba 行是普通 Mamba 编码器加仓库预测 head，不使用 ACF memory 或 AutoCon relation objective；不能把它写成 AutoCon 增益。
- 测试集没有读取，84 个作业的 `phase_validation.json` 为 `valid=true`；本轮只适合决定后续完整矩阵和候选，不进入论文主表。
- 论文级结论仍需要冻结 H96/H336/H720、长训练预算、至少 3--5 个 seed、明确 test release，以及 forward/backward/optimizer 的训练资源测量。

## 可复核命令

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_meanstd_baseline_pilot_20260907.json --validate_only
conda run --no-capture-output -n mamba python scripts/validate_training_phase.py \
  --plan docs/research/mamba_meanstd_baseline_pilot_20260907.json \
  --phase meanstd_core_baselines_h96 \
  --manifest artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907/logs/job_manifest.jsonl \
  --provenance artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907/provenance.json \
  --output /tmp/mamba_meanstd_phase_validation.json
conda run --no-capture-output -n mamba python scripts/summarize_meanstd_baseline_pilot.py \
  --phase_root artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907 \
  --plan docs/research/mamba_meanstd_baseline_pilot_20260907.json
```

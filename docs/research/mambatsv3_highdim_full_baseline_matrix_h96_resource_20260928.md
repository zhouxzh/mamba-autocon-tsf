# Electricity / Traffic H96 Resource Readout (2026-09-28)

这是一组与同日八臂 validation matrix 对齐的 prediction-only GPU 资源测量，不读取数据集、不运行训练、不包含 AutoCon relation loss。配置来自该矩阵的实际 `config.json`，模型源码根固定到对应的 immutable source snapshot。资源工具以 explicit-config 模式记录为 `unverified_config`，因此它不是 phase-manifest attestation；结论只限此处明确的固定 shape 和设备。

协议：RTX 5090 D、FP32、device-resident synthetic input、`B=32,L=96,H=96`、25 warm-up、100 timed forwards、输入 seed `3291`。候选的预测接口已经执行 prediction-connected representation fusion；它不暴露独立 `return_repr` 张量，故报告的是完整预测路径，而非额外返回表示的接口成本。资源原始记录见 [`model_resources.csv`](../../artifacts/exploratory/mambatsv3_highdim_full_baseline_matrix_h96_validation_20260928/resources/model_resources.csv)。

| Dataset | Model | Median ms | Trainable params | Candidate interpretation |
| --- | --- | ---: | ---: | --- |
| Electricity | MambaTSv3AutoCon | `.9231` | `106,834` | reference |
| Electricity | DLinear / NLinear / RLinear | `.1139` / `.0664` / `.0766` | `18,624` / `9,312` / `9,954` | candidate is slower |
| Electricity | PatchTST / TimesNet | `1.2594` / `2.7163` | `46,080` / `1,232,257` | candidate is faster by `26.7%` / `66.0%` |
| Electricity | iTransformer / Mamba2 | `.2955` / `.7188` | `14,944` / `80,461` | candidate is slower |
| Traffic | MambaTSv3AutoCon | `2.3551` | `168,508` | reference |
| Traffic | DLinear / NLinear / RLinear | `.2336` / `.1165` / `.1359` | `18,624` / `9,312` / `11,036` | candidate is slower |
| Traffic | PatchTST / TimesNet | `3.9465` / `2.8633` | `46,080` / `1,302,046` | candidate is faster by `40.3%` / `17.8%` |
| Traffic | iTransformer / Mamba2 | `.9167` / `.9384` | `14,944` / `167,562` | candidate is slower |

速度差按 `(baseline median / candidate median - 1)` 计算。该表不构成训练吞吐、端到端延迟、硬件无关性能或所有基线速度优势的主张。

# Electricity / Traffic H96 Relation-Auxiliary-Only Resource Readout (2026-09-28)

这是一组与 relation-auxiliary-only validation confirmation 对齐的 prediction-only GPU 资源测量：
MambaTSv3 v3 保留 `AutoCon_lambda=.001`，关闭 prediction-connected representation fusion。测量不读
数据集、不运行训练、不含 relation loss；配置来自实际 validation `config.json`，模型源码根固定至该 phase
的 immutable snapshot。资源工具以 explicit-config 模式记录为 `unverified_config`，故结果仅限明确的固定
shape 和设备，不是 phase-manifest attestation。

协议：RTX 5090 D、FP32、device-resident synthetic input、`B=32,L=96,H=96`、25 warm-up、100 timed
forwards、输入 seed `3291`。原始记录见
[`model_resources.csv`](../../artifacts/exploratory/mambatsv3_highdim_relation_auxiliary_only_h96_validation_20260928/resources/model_resources.csv)。

| Dataset | Relation-auxiliary-only median ms | Trainable params | Prior fusion candidate median ms | Latency change |
| --- | ---: | ---: | ---: | ---: |
| Electricity | `.8011` | `76,753` | `.9231` | `-13.2%` |
| Traffic | `2.2052` | `103,262` | `2.3551` | `-6.4%` |

关闭预测融合同时降低参数与这两个固定 shape 的前向延迟；相对于已测 PatchTST/TimesNet，该候选仍更快，
但并不快于 DLinear、NLinear、RLinear、iTransformer 或 Mamba2。该表不构成训练吞吐、端到端延迟、
硬件无关性能或所有基线速度优势的主张。

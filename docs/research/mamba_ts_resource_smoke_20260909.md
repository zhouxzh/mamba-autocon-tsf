# MambaTS 资源 smoke（2026-09-09）

这是模型接入后的结构级 GPU 检查，不是训练速度或预测精度结果。环境为 RTX 5090 D、FP32、10 次 warm-up、30 次 timed forward，输入常驻 GPU；AutoCon 记录关闭了 auxiliary representation 返回，因此没有把训练期 relation loss 当成部署前向成本。

| shape / arm | batch | trainable parameters | median ms | p95 ms | sequences/s | peak increment MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 C=7 `MambaTS` | 8 | 44,448 | 0.2176 | 0.2367 | 35,900 | 1.30 |
| ETTh1 C=7 `MambaTSAutoCon` | 8 | 50,848 | 0.2210 | 0.2448 | 35,477 | 1.30 |
| PEMS03 C=358 `MambaTS` | 2 | 44,448 | 0.2250 | 0.2417 | 8,758 | 15.75 |
| PEMS03 C=358 `MambaTSAutoCon` | 2 | 50,848 | 0.2226 | 0.2349 | 8,880 | 15.75 |
| PEMS03 C=358 `DLinear` | 2 | 18,624 | 0.0945 | 0.1061 | 20,322 | 1.84 |
| PEMS03 C=358 `PatchTST` | 2 | 46,080 | 0.2872 | 0.3123 | 6,877 | 9.71 |
| PEMS03 C=358 `Mamba2` | 2 | 86,450 | 0.7341 | 0.7574 | 2,716 | 2.46 |

在 PEMS03/C=358 的同一协议下，MambaTS 比 Mamba-2 快约 `69.4%`、比 PatchTST 快约 `21.7%`；DLinear 的 median latency 低 `58.0%`（即 MambaTS 约为 DLinear 的 2.38 倍）。AutoCon projection 相对 MambaTS 的 median latency 在 PEMS03 下降 `1.05%`、ETTh1 增加 `1.54%`，方向并不稳定。这只是固定 shape 的结构读数，不能外推到 train-step、数据加载或全矩阵速度。原始 JSON/CSV 和源码 hash 保存在 [`artifacts/exploratory/mamba_ts_resource_smoke_20260909`](../../artifacts/exploratory/mamba_ts_resource_smoke_20260909)。

真实预测结论仍需运行已登记的 PEMS validation-only 计划，并保持 `execution.authorized=false` 直到人工复核完成。

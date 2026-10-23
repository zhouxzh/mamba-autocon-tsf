# Mamba patch-token contrastive screen 资源预检

日期：2026-09-07  
状态：已完成 prediction-only 资源预检；24-job 训练 screen 仍未授权、未启动。

## 目的和边界

这是 [`mamba_patch_token_contrastive_screen_plan_20260907.json`](mamba_patch_token_contrastive_screen_plan_20260907.json) 的启动前结构成本检查。它覆盖同一 `MambaACFAutoCon`、H96、`hard_selected + mamba_acf_relation_lag_mask=true` 配置，在 ETTh1、ETTm1、Electricity 和 Traffic 上分别测量 patch-token off/on。

资源测量使用 RTX 5090 D、`B=8`、10 次 warm-up、30 次同步计时、FP32、device-resident synthetic input。测量路径是 prediction-only，`relation_loss_included=false`；不包含数据读取、CPU/GPU 传输、反向、优化器、checkpoint I/O 或完整训练 wall time。因此它只能回答结构预算问题，不能证明端到端速度。

## 原始读数

| 数据集 | patch | 总参数 | 可训练参数 | median forward (ms) | p95 (ms) | peak increment (MB) | sequences/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | off | 58,947 | 55,746 | 1.243984 | 1.273560 | 3.284 | 6,412.8 |
| ETTh1 | on | 105,649 | 102,448 | 1.746752 | 1.791642 | 3.284 | 4,570.7 |
| ETTm1 | off | 58,979 | 55,778 | 1.232832 | 1.273021 | 3.284 | 6,460.0 |
| ETTm1 | on | 105,681 | 102,480 | 1.727808 | 1.772163 | 3.284 | 4,613.2 |
| Electricity | off | 99,453 | 96,252 | 1.241888 | 1.266910 | 12.840 | 6,414.7 |
| Electricity | on | 146,155 | 142,954 | 1.778416 | 1.804291 | 77.238 | 4,486.6 |
| Traffic | off | 169,242 | 166,041 | 1.254912 | 1.312349 | 32.167 | 6,334.8 |
| Traffic | on | 215,944 | 212,743 | 2.104144 | 2.124915 | 208.625 | 3,796.1 |

相对变化按 `(patch-on / patch-off - 1) * 100%` 计算：

| 数据集 | 总参数 | 可训练参数 | median | p95 | peak increment | sequences/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | +79.23% | +83.78% | +40.42% | +40.68% | +0.00% | -28.73% |
| ETTm1 | +79.18% | +83.73% | +40.15% | +39.21% | +0.00% | -28.59% |
| Electricity | +46.96% | +48.52% | +43.20% | +42.42% | +501.55% | -30.06% |
| Traffic | +27.59% | +28.13% | +67.67% | +61.92% | +548.56% | -40.08% |

ETTh1/ETTm1 的报告峰值增量都为 3.284 MB，说明该字段在小通道形状上没有区分出 patch 分支的额外工作集；不能据此推断显存完全不变。Electricity 和 Traffic 的绝对峰值增量分别从 `12.840 -> 77.238 MB`、`32.167 -> 208.625 MB`，是更有信息量的高维读数。

## 判断

- 四个数据集的 patch-on 都降低 synthetic throughput；median latency 增加约 `40--68%`，没有“更快”的结构证据。
- patch-token 增加 `28--84%` 的可训练参数；高维 Electricity/Traffic 的绝对峰值工作集明显上升。
- 这组结果不否定 patch-token 的精度候选价值。修正 relation 合同后的 ETTh1/Traffic 单 seed pilot 仍显示 `-2.79%/-3.24%` validation loss，但该精度信号尚未经过四数据集、多 seed 复验。
- 如果研究目标包含速度优势，当前候选应在训练前降级或改造；如果接受计算代价，只能把它作为“可能改善精度的结构消融”进入人工批准流程。

## 复核入口和下一步

原始 JSON/CSV 位于 [`artifacts/exploratory/mamba_patch_token_contrastive_screen_20260907/resources`](../../artifacts/exploratory/mamba_patch_token_contrastive_screen_20260907/resources)，目录说明见该目录的 `README.md`。精度 pilot 的有效 relation 读数见 [`mamba_patch_token_contrastive_pilot_readout_20260907.md`](mamba_patch_token_contrastive_pilot_readout_20260907.md)。

24-job screen 仍保持 `execution.authorized=false`，没有写入训练 checkpoint 或 test 指标。若人工批准，启动前还需要在 ETTh1/Traffic 形状补做相同有效 batch 和 AMP/FP32 合同下的 forward+backward+optimizer train-step smoke；只有该测量与训练 wall time 才能支持效率讨论。若不批准，保留这些预检作为资源负结果，并将 patch 分支限制为后续低成本结构诊断。

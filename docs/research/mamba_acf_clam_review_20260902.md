# CLAM 候选分支人工复核

更新日期：2026-09-02。CLAM（local ACF agreement modulation）是 MambaACF 的一个未训练候选，不是已验证的精度改进。

## 研究问题

训练集 ACF 先验给出每个通道/lag 的稳定方向；一个新的 encoder window 可能暂时偏离该先验。CLAM 检验：能否只用当前已观测 window 的相关性，调节 static ACF memory 的 lag 权重，同时保留先验 sign 和 AutoCon 表示学习。

## 代码语义

1. 只接受 `AutoCon_acf_source=raw`、static `mamba_use_acf_memory=true`，并使用 train-only prior。
2. 只计算 `length - lag >= mamba_acf_local_min_pairs` 的 lag；默认阈值为 16。当前 window 不含 decoder 或未来 target。
3. 每个合格 lag 的相关性乘以 prior sign 和 pair-count confidence，作为 agreement；`exp(tanh(scale) * agreement)` 只重分配 lag mass。
4. memory value 的方向仍使用 prior sign；没有合格 lag 时走既有 static-memory 路径。
5. ReZero scale 从零开始，额外可训练参数为 1；AutoCon relation loss、projection head 和多尺度输入路径不变。

## 已完成证据

- CPU/CUDA shape、finite-output、input-gradient、CLAM-gradient、prior-sign、zero-init 和 no-eligible fallback 合同已通过。
- 当前 prediction-only synthetic GPU 资源：ETTh1 median `1.401 -> 1.734 ms`（`+23.80%`），Traffic `1.419 -> 1.718 ms`（`+21.08%`）；Traffic 峰值显存增量 `+3.128 MB`（本协议观测值）。详见 [`resource_repr_control_current_20260902_v3`](../../artifacts/exploratory/resource_repr_control_current_20260902_v3/)。
- 这些读数不是预测精度、端到端训练速度或统计重复；目前没有 CLAM 训练结果。

## 建议的最小下一步

在人工批准后，运行 ETTh1/Traffic、H96、3 seeds 的 local off/on 配对 screen。唯一改变 `mamba_acf_local_agreement`，其余 optimizer、split、prior sidecar、AutoCon lambda、训练预算和 Mamba 版本固定。选择规则应提前写明：以 dataset/horizon/seed 配对的 scaled-MSE 为主，MAE 为辅，报告每个 seed、均值差、置信区间、median/p95 latency、峰值显存和完整 source/config/sidecar hash；不要用 pooled windows 当独立重复。

计划未授权前不得启动训练。若 paired accuracy 没有稳定改善，或改善小于当前计算代价的实际影响，应停止扩展，而不是增加更多机制。

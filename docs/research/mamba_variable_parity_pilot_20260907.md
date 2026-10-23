# Variable-token parity pilot

日期：2026-09-07  
状态：探索性 paired pilot；不是确认性论文结果。

本轮只测试 `MambaACFAutoCon` 在公平 `MeanStd` 合同下增加已有
`mamba_variable_mixer=true` 的影响。对照是本轮
[`mamba_normalization_parity_pilot_20260907.md`](mamba_normalization_parity_pilot_20260907.md)
中同数据集、同 seed 的 variable-mixer-off AutoCon arm。固定条件为 ETTh1/Traffic、H96、
seed 2021/2022、4 epoch、batch 32、`d_model=32`、单尺度、causal embedding、
train-only `train_peaks_available` ACF、`MeanStd`、`raw + hard_selected + lag_mask=true`
relation、global encoder representation、lambda `0.03`、skip-test。

唯一结构差异是 variable-token channel-preserving branch。结果只允许报告 paired
validation delta、训练 wall time、relation telemetry 和 prediction-only resource；若
要进入主线，必须至少扩展到 3--5 seed、多数据集和完整预算。

输出根目录：`artifacts/exploratory/mamba_variable_parity_pilot_20260907/`。

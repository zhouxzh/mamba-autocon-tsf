# Variable-token parity third-seed replication

日期：2026-09-07  
状态：探索性第三 seed replication；不是确认性论文结果。

本轮补充 `seed=2023`，分别运行 shared `MambaACFAutoCon`（variable mixer off）与
`mamba_variable_mixer=true`，数据为 ETTh1/Traffic、H96、4 epoch。两臂沿用
[`mamba_normalization_parity_pilot_20260907.md`](mamba_normalization_parity_pilot_20260907.md)
的 `MeanStd`、causal embedding、train-only ACF、`raw + hard_selected + lag_mask=true`
global encoder relation 和 `AutoCon_lambda=0.03` 合同；唯一结构差异是 variable-token branch。

输出根目录：`artifacts/exploratory/mamba_variable_parity_replication_20260907/`。
所有作业 `skip_test=true`，只用于把前两 seed 的 paired direction 扩展到第三个 seed；
即使方向一致，也不能替代 3--5 seed 完整预算确认实验。

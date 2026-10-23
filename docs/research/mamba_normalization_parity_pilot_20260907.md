# Mamba / AutoCon normalization parity pilot

日期：2026-09-07  
状态：启动前固定的探索性 pilot；不是正式预注册，也不是确认性论文结果。

## 目的

在同一数据切分、训练预算和 `MeanStd` 窗口归一化下，比较 PatchTST、普通
Mamba-1/2、plain MambaACF 和带 AutoCon relation representation 的
MambaACF，确认后续结构优化是否有公平的精度起点。`MeanStd` 使用窗口 population
variance，并不是一个新的精度主张。

## 固定合同

- 数据集：ETTh1、Traffic；H96；多变量 (`features=M`)；输入文件使用当前仓库哈希；
- 模型：PatchTST、Mamba1、Mamba2、MambaACF、MambaACFAutoCon；
- seed：2021、2022；每个数据集/模型/seed 一个作业，共 20 个作业；
- 训练：4 epoch、patience=2、batch size=32、Adam、learning rate `1e-3`、cosine schedule、`d_model=32`；
- normalization：所有 arm 显式 `AutoCon_wnorm=MeanStd`；PatchTST 使用其内置同合同；
- Mamba-1/2：`d_state=16`、`d_conv=4`、`expand=2`、`headdim=16`、单尺度、单位 token stride、causal embedding；
- MambaACF：单尺度、causal embedding、train-only ACF peak availability、关闭 forecast memory；
- AutoCon arm：`AutoCon_lambda=0.03`、`AutoCon_acf_source=raw`、`hard_selected`、
  `lag_mask=true`、`relation_scope=global`、`relation_representation=encoder`、
  `relation_batching=lag_pairs`、`relation_lag_balance=balanced`；
- 所有作业 `skip_test=true`，不读取 test split；日志、配置、validation trajectory
  和独立 GPU resource JSON/CSV 保留在同一 fresh artifact root。

## 解释边界

本 pilot 只允许报告 dataset-local validation direction、paired seed delta、训练
wall time 和 prediction-only GPU resource。它不允许写成 SOTA、泛化、显著性、端到端
速度或“AutoCon 必然有效”结论。模型间差异仍可能受到实现容量、预测头和原生模型路径
影响；若候选值得继续，必须新建至少 3--5 seed、完整预算、冻结 test gate 的确认计划。

## 输出

训练输出根目录：`artifacts/exploratory/mamba_normalization_parity_pilot_20260907/`。
资源协议：RTX 5090 D、FP32、device-resident synthetic input、B=8、10 warm-up、50
timed forwards；不包含数据加载、传输、反向、优化器或 relation loss。

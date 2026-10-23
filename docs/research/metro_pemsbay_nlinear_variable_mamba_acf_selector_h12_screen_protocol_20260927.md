# METR-LA / PEMS-BAY H12 selector screen

这是一个 validation-only 的跨数据集确认协议，不能产生或读取 held-out test 指标。

## 固定内容

- 数据：`METR-LA`（207 通道）和 `PEMS-BAY`（325 通道），`seq_len=96`、`pred_len=12`、多变量输入。
- 候选：`NLinearVariableMambaAutoCon`，共享 NLinear base、单次 variable-axis Mamba-1 scan、prediction-connected AutoCon representation、train-only ACF memory，以及零初始化 rank-2 horizon + rank-1 channel selector。
- 对照：selector off；唯一允许的结构差异是 `mamba_variable_acf_memory_selector`。
- 外部基线：DLinear、NLinear、PatchTST、TimeMixer、iTransformer、MambaTSv3。
- 训练：fresh seeds `2711, 2712, 2713`，8 epochs、patience 3、batch 32、相同 chronological split，候选使用 `AutoCon_lambda=0.0001` 和 relation lag-pairs，外部基线关闭候选专属 relation 设置。

## 通过条件

selector-on 必须在两个数据集都降低平均 validation MSE、每个数据集至少赢得 2/3 个配对 seed，且没有超过 2% 的单 seed 退化；同时相对每个数据集的最强外部基线不超过 2%。否则停止后续调参和 held-out test 注册。

## 当前执行状态

计划文件已通过 `run_training_plan.py --validate_only` 的 48-job 校验，dry-run 也通过；数据和图文件哈希已核对。GPU 资源预检已确认候选在 RTX 5090 D 上可以前向，但本协议尚未获得本轮训练授权，因此不能把资源预检写成精度结果。

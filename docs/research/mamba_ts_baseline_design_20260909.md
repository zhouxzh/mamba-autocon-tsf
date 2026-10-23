# MambaTS 基线设计与外部实现核验

## 为什么加入这个基线

现有高维 Traffic/PEMS 结果显示，共享时间 hidden state 的 Mamba-2 不能自动解决跨变量异质性。MambaTS 是一个合适的结构对照：它先把每个变量切成 patch，再按变量组织 patch token 做 Mamba 扫描，最后使用逐变量预测头，保留变量维度的状态。

官方仓库将入口放在 [`models/MambaTS.py`](https://github.com/XiudingCai/MambaTS-pytorch/blob/main/models/MambaTS.py)，并在 README 中说明 VPT/VAST 相关模块位于 `mixer2_seq_simple.py`。源码可辨识的核心路径是 patch embedding、channel-major token flatten、Mamba encoder 和 per-variable `PredictionHead`；官方实现还包含变量顺序学习与其自定义扫描组件。

## 本仓库实现的边界

[`models/MambaTS.py`](../../models/MambaTS.py) 是 **MambaTS-inspired** baseline，不声称逐行复现官方 VAST/VPT：

- 使用当前环境已锁定的 `mamba_ssm==2.3.1` Mamba-1 kernel；不引入外部 vendored kernel、ATSP 求解器或额外 CUDA 依赖。
- 保留 patch embedding、按变量的 patch token 扫描和逐变量 forecast head，便于与现有 Mamba-1/2 及 PatchTST 做结构归因。
- plain 入口是 `MambaTS`；`MambaTSAutoCon` 只增加独立 projection/representation head，复用仓库已有 AutoCon 训练循环。patch token 的通道池化结果会插值回 `seq_len` 的观测时间网格，保证 legacy AutoCon 局部关系使用原始时间 lag 单位。它不启用 MambaACF 专用 signed-lag relation loss，因此可以单独回答“通用 AutoCon 表征学习是否有帮助”。
- `features=MS` 会在反归一化前切到最后一个 target channel；`features=M/S` 要求输出通道与输入通道一致，避免隐式广播。

CMamba 的官方实现可作为低秩跨变量调制的后续参考，TimeMachine 的官方实现可作为多尺度残差路径参考；本次不把这些结构与 MambaTS 同时开启，否则无法区分 patch、变量状态和辅助目标的贡献。

## 复核重点

1. 先比较 `MambaTS` 与 `Mamba2` 的 plain validation MSE/MAE、参数量、显存和训练时间。
2. 再在完全相同的 PEMS 窗口、seed、MeanStd、optimizer 和预算下比较 `MambaTS` 与 `MambaTSAutoCon`。
3. PEMS04/08 固定 `data_feature_index=0`；实验不使用图邻接矩阵，因此不能把结果写成 spatial-GNN 结论。
4. 外部仓库的精度/速度数字不填入本仓库表格；只有当前树生成的、带配置和 hash 的工件才能进入结果汇总。

旧的 192-job 计划仍登记在 [`mamba_pems_baseline_autocon_pilot_20260909.json`](mamba_pems_baseline_autocon_pilot_20260909.json)，保持 `execution.authorized=false`，不能与新结果拼接。当前 v3 的独立 exploratory screen 已完成：[`mamba_pems_ts_v3_fresh_seeds_20260909.md`](mamba_pems_ts_v3_fresh_seeds_20260909.md) 在两个 fresh seeds 上相对 Mamba-2 为 `16/16` 配对改善；[`mamba_pems_ts_v3_strong_baseline_screen_20260909.md`](mamba_pems_ts_v3_strong_baseline_screen_20260909.md) 进一步显示 v3 相对 DLinear/PatchTST 为 `8/8` 改善，相对 TimesNet 为 `5/8`，但 TimesNet 在三个 PEMS 长 horizon 坐标更低。v3 因此是短期预测候选和长期竞争者，不是普适最优模型。

# MambaACF Data-Dependent Channel Mixer Screen

人工复核稿 · 2026-09-03 · 计划已登记，尚未授权或启动训练

## 当前结论

[`mamba_data_dependent_channel_mixer_screen_plan_20260903.json`](mamba_data_dependent_channel_mixer_screen_plan_20260903.json) 是一个 60-job 的 exploratory、validation-only 计划。它使用仓库中已有且已哈希的 ETTh1 与 Traffic CSV，在 H96、`2021/2022/2023` 三个 seed 下比较六个普通预测基线、一个 MambaACF 参考、一个新通道状态混合器，以及同一 AutoCon 架构的 relation null/active 两臂。

当前没有该计划的训练指标、checkpoint、端到端耗时或 GPU 计时。`execution.authorized=false` 且 `training_started=false` 是硬边界；不得把计划、历史结果或结构资源 smoke 当作新候选的精度证据。

## 候选定义

候选是一个 CMamba-inspired 的低秩、窗口条件通道状态适配器：它复用 MambaACF 已计算的 `[B,D]` 窗口摘要，经过 `D -> r -> 2D` 低秩投影生成共享 hidden-space scale/shift，并以零初始化的 ReZero 残差注入 `[B,T,D]` 编码状态。实际秩由 `min(C-1,D-1)` 封顶；该全局状态路径把新增矩阵计算控制为 `O(Dr+rD)`，不使用 dense `C x C` 投影，也不等同于外部 CMamba 实现；跨变量信息通过原有共享 embedding 与 scale-gate 摘要进入适配器。

该分支位于 MambaACF 的共享编码状态路径，因此候选的 forecast 分支和 AutoCon 表示分支共享同一状态。状态来自当前观测窗口，故本计划关闭严格 representation-causality 声明；不能把 `mamba_causal_embedding=true` 误写成整个表示路径严格因果。候选默认零残差，只有训练后才能评估其是否学到有用调制。

## 资源 Smoke（无训练）

在 RTX 5090 D、Torch `2.8.0+cu128`、`mamba_ssm=2.3.1` 上，完成了最终源码的四配置 B=8 isolation spot check（ETTh1/Traffic 均为 `B=8`，warmup 25、timed 100、input seed 2718）。四条记录均是 plain `MambaACF`、`AutoCon=false`、`mamba_use_acf_memory=false`，只隔离测量 channel mixer 开关；每个配置是一次模型构造、一次 100-forward 计时。它不包含数据读取、传输、反向传播、优化器或 checkpoint I/O。逐配置原始记录、环境和源码哈希见 [`mamba_data_dependent_channel_mixer_resource_current_b8_20260903`](../../artifacts/exploratory/mamba_data_dependent_channel_mixer_resource_current_b8_20260903/README.md)。

| 数据集/通道数 | mixer | median ms | p95 ms | peak increment MB | total/trainable params |
| --- | --- | ---: | ---: | ---: | ---: |
| ETTh1 / 7 | off | 0.6626 | 0.6752 | 3.2827 | 40,405 / 35,092 |
| ETTh1 / 7 | on, rank 4 | 0.7070 | 0.7241 | 3.2827 | 40,858 / 35,545 |
| Traffic / 862 | off | 0.6867 | 0.7207 | 15.7598 | 150,700 / 145,387 |
| Traffic / 862 | on, rank 4 | 0.7424 | 0.7600 | 15.7598 | 151,153 / 145,840 |

当前源码 isolation spot check 的 on/off 中位数比为 ETTh1 `1.0671x`、Traffic `1.0811x`；p95 比为 `1.0724x`、`1.0545x`。参数均增加 453，表中的显存数值是每次 run 的 steady-state peak increment，不是 mixer 的 on-minus-off 显存增量。虽然本组统一使用 `B=8`，其余显式 CLI defaults 仍不等于预注册候选（尤其没有 ACF memory/AutoCon 分支及其架构开关），因此只说明固定 synthetic shape 下的隔离结构成本，不能外推到候选、其它 batch/horizon 或端到端训练。

此前的四次重复 smoke（ETTh1 `1.0716x`、Traffic `1.0696x` 的中位数比）以及早期 current-source spot check 均保留为历史测量，绑定不同源码/协议，不能与本 B=8 记录混合汇总。预注册候选（ACF memory、AutoCon relation 和完整 `B=8` 配置）的 resource gate 仍未评估。

## 配对与归因

- `mamba_acf_reference_h96` 与 `mamba_acf_channel_state_h96` 只应在候选开关/秩上不同；主估计量是同一 `dataset × horizon × seed` 的 validation scaled-MSE 比值与差值。
- `mamba_acf_channel_state_relation_null_h96` 使用 `lambda=0`，`mamba_acf_channel_state_relation_h96` 使用固定 `lambda=0.03`。两臂保持同一 MambaACF AutoCon、train-only raw ACF prior、lag sampler、数据切分和训练预算；该对照只估计 relation objective 的增量。
- DLinear、NLinear、PatchTST、TimesNet、Mamba1、Mamba2 是固定预算的描述性基线，不得从本 screen 中选择超参数或追溯性拼接历史结果。

## 预注册门槛

精度门：候选相对 MambaACF reference 在两个数据集都不恶化（均值 validation scaled-MSE ratio `<=1.000`、至少 `2/3` seed 胜、无 seed ratio `>1.05`）；另与每个数据集上均值最好的普通基线报告 `<=1.010` 的 screening gate。relation active 相对 null 也须在两个数据集达到 `<=1.000` 且至少 `2/3` seed 胜。所有 seed 的 paired delta、区间和异质性都必须保留。

资源门：在同一源码快照、同一 GPU 和 device-resident synthetic 输入上运行 `scripts/benchmark_model_resources.py`（batch 8、warmup 25、timed 100、input seed 2718）。候选相对 MambaACF reference 的中位延迟、p95 延迟、steady-state peak allocated memory 和 active trainable parameters 均不得超过 `1.10x`；本页的重复 smoke 只完成了这些结构/推理项的预筛查，训练 manifest 的 `elapsed_sec` 仍须另行汇总。缺失、混设备或 source/config hash 不一致的记录使资源门失效，不能复制历史读数。

只有在 60 个坐标完整、配对工件与 hash 通过、精度和资源门在两个数据集均通过，并重新冻结一份人工审核的 confirmation plan 后，才可讨论晋级。当前只有资源预筛查，没有任何候选精度坐标；单个数据集、单个 seed 或单次 GPU 前向都不足以支持论文结论。

## 复核命令

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_data_dependent_channel_mixer_screen_plan_20260903.json \
  --validate_only
```

预期输出：`Validated 60 jobs (order_seed=20260903)`。该命令只校验计划，不创建训练结果，也不会授权训练。

资源测量应在人工批准、源码/环境/数据 hash 冻结并完成 phase validation 后执行；本页已有的 smoke 仅用于结构成本审计，不能替代批准后的 manifest 资源记录。

## 证据边界

本页可以支持“候选结构、配对合同和预注册门槛已明确”。本页不能支持“候选提高了预测精度”“候选比 Mamba2 或 PatchTST 更快”“AutoCon relation 已带来增益”或任何发表结论。

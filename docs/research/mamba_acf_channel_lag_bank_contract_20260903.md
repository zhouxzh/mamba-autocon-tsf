# MambaACF 逐通道稀疏 Lag Bank 契约

_人工复核稿 · 2026-09-03 · 仅资源/契约检查，未启动精度训练；当前资源工件为 v4 结构记录。strict draft 现被全窗口归一化阻断，未通过 validate-only。_

## 一句话结论

`mamba_acf_channel_lag_bank` 是一个默认关闭的、无可训练参数的 forecast-memory 候选：它在同一个 train-only 全局 ACF lag bank 上，先排除没有合法 forecast step 的 lag，再为每个通道保留 `abs(ACF) * legal forecast-step fraction` 得分最高的若干个槽位。它不是任意逐通道重新发现周期，也不改变 AutoCon relation loss。当前没有该候选的精度结果，不能写成 ACF 增益、Mamba 改进或论文证据。

## 精确定义

设模型最终解析出的共享 lag bank 为

```text
L = (l_1, ..., l_K),   A in R^(C x K)
```

其中 `A[c,k]` 是第 `c` 个通道在 `l_k` 上的 signed、train-only ACF。对固定输入长度 `L` 和预测长度 `H`，实现先计算每个 lag 的合法步比例：

```text
q_k = (1/H) * sum_h 1[0 <= L - 1 + h - l_k < L]
```

先排除 `q_k = 0` 的槽位，再按每个通道的
`score[c,k] = abs(A[c,k]) * q_k` 排序，保留 `top_k` 个槽位，得到 detached dense mask `S in {0,1}^(C x K)`；相同得分时按现有升序 lag bank 做确定性 tie-break。若合法槽位总数少于 `top_k`，当前实现直接抛出错误，不静默降低实际 `top_k`。forecast memory 在每个预测步仍应用逐位置合法性约束：

```text
S[c,k] = 1
and 0 <= seq_len - 1 + h - l_k < seq_len
```

因此实际权重仍是带 causal validity mask 的 signed ACF softmax。候选只改变 forecast-memory 的 support 和由此派生的 reliability；它不读取当前验证/测试窗口的 ACF，也不引入新的 learned 参数或 dense `C x C` projection。固定的 `[C,K]` mask 仍是一个 train-only、channel-specific 的函数规则，不能被描述为完全没有函数容量变化。

### “逐通道”边界

计划的修订方向使用 `mamba_acf_lag_mode=train_peaks_available`：先从完整 train-only prior 的候选峰值中，按共享的合法 forecast-step 支持解析一个 horizon-specific bank，再执行上述逐通道 `abs(ACF) * q_k` top-k 选择。bank 的具体整数由数据和 horizon 决定；配置中的 lag 列表只表达候选槽位数量/约束，不能被写成固定周期清单。

逐通道 mask 只能在这个**已解析的共享 bank**内选择 top-k，不能选择 bank 外的任意 lag。每个 job 必须在 `config.json`、`acf_prior.npz` 和 phase validator 中记录实际 `mamba_acf_effective_lags`、每个 lag 的 `q_k`、support mask 的形状/hash，以及 selected-slot count。修订后的实现保证：只要 resolved bank 至少有 `top_k` 个 `q_k>0` lag，每个通道就得到恰好 `top_k` 个统计 support slots；运行时仍会按预测步应用 causal validity mask。这个保证不等于每个 slot 对所有预测步都可用，也不等于每个 ACF 峰都经过独立显著性检验。

Relation loss 仍使用完整 prior 和其自身的 candidate/selected-lag 语义。因而 `channel_lag_bank_on - channel_lag_bank_off` 的归因只属于 forecast-memory support，不是 relation target、representation fusion 或 channel-mixing 的消融。

## 为什么值得筛选

现有后验诊断显示，共享 lag 选择在 Traffic 上与通道峰值的覆盖不佳，而 ETTh1 的覆盖更高；这只是结构动机，不是 treatment effect。该候选的可检验假设是：在保持共享 Mamba 编码器、relation 设置、训练预算和参数容量不变时，通道特异且按合法性加权的 ACF support 能减少不相关 lag 的混合，至少不会损害 H96 预测。

该假设也可能失败：通道 ACF 峰值可能不稳定，强相关不代表可预测，top-2 可能丢失互补周期，且 bank 外的真实峰值无法被恢复。因此 screen 必须报告每个 seed 的 paired delta、resolved bank、support density 和 seed heterogeneity，而不能只看 pooled mean。

## 已登记但未授权的 screen

计划文件：[`mamba_acf_channel_lag_bank_screen_plan_20260903.json`](mamba_acf_channel_lag_bank_screen_plan_20260903.json)。该草案保留了严格表示因果性要求，但当前 `LastVal` 会读取完整窗口末端，因此现状态为 blocked，不可启动。

为避免把“关闭 strict flag”伪装成同一个研究问题，另行登记了 forecast-level 修订版 [`mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json`](mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json)。修订版明确只检验 `L-1+h-lag` 的 forecast-memory indexing 因果性，保留 `LastVal` 的窗口级 AutoCon representation，并已通过 `validate_only` 展开 12 个 job；它同样是 `execution.authorized=false`，不能在人工批准前启动。

- ETTh1/Traffic，H96；off/on 两臂；seed `2021/2022/2023`，共 12 jobs。
- 两臂均为 `MambaACFAutoCon`，同一 `train_peaks_available` resolved bank、raw static prior、global relation、causal embedding、单位 token stride 和训练预算；唯一结构差异为 support mask 开关。
- 计划方向为 validation-only：修订后的配置应固定 `skip_test=true`、`stream_metrics=false`，并记录 `selection_metric=validation_scaled_mse`。主指标读取 checkpoint `loss_logs.json` 的最佳 `val_loss`（纯 forecasting validation MSE）；`stream_metrics.json` 只由独立 test 阶段生成，不能在跳过 test 时作为必需工件。test 指标不得用于 arm 选择或 stopping/promotion；若未来需要最终 test 描述性评估，应在 validation 选择和 provenance 冻结后另行执行并隔离记录。
- strict 草案的 exploratory 输出根应使用 `artifacts/exploratory/mamba_acf_channel_lag_bank_screen_20260903/...`；forecast-level 修订版则使用 `artifacts/exploratory/mamba_acf_channel_lag_bank_forecast_screen_20260903/...`。两者都不应放入 `artifacts/paper_final/`。此前 strict 草案的 `validate_only` 记录已失效：严格因果 contract 现会返回 `window_full_normalization_forbidden`。若后续实现 prefix-causal normalization，必须重新冻结 JSON、运行 `validate_only` 并更新 hash；未经人工改写研究问题，不得仅关闭 strict flag 来启动这份草案。
- 计划状态仍应为 `execution.authorized=false`、`training_started=false`；在修订后的 JSON、launch sidecar 和 hash 重新冻结前，不得把任何目录解释为已执行结果。
- `top_k=2` 需要成为显式 gate：每个 job 的 resolved bank 至少有 2 个 `q_k>0` lag；实现会在不满足时拒绝，而不是记录降级后的 top-1 结果。
- 评估/恢复时，`mamba_acf_channel_lag_bank` 和 `mamba_acf_channel_lag_top_k` 必须与训练时 `config.json` 完全一致，并在模型构造前校验；CLI 默认值不会再静默覆盖历史 checkpoint 的 mask 拓扑。compact/full ACF 直接 API 也会先验证 lag/value 对齐。

计划中的停止规则（池化退化超过 1% 或某数据集三个 seed 全部退化）只能作为探索性扩展门槛，不能当作统计显著性、非劣性证明或测试集选择结果。

## 当前资源契约（不是速度证据）

资源契约工件：[`artifacts/exploratory/channel_lag_bank_resource_contract_20260903_v4/model_resources.json`](../../artifacts/exploratory/channel_lag_bank_resource_contract_20260903_v4/model_resources.json)。这是按当前源码重新生成的 synthetic/device-resident 结构实例化，没有训练和端到端计时；v3、v2 和更早目录仍保留作历史记录。v3 的 `strict=true`、空 causality limits 和旧源码哈希已过期，不能作为当前严格因果证据：

| 项目 | 读数 |
|---|---:|
| 模型 | `MambaACFAutoCon` |
| 解析示例 bank | 7 slots |
| 通道数 | 7 |
| top-k | 2 |
| selected/total support slots | `14/49` |
| support density | `0.285714` |
| dense `[C,K]` bool mask storage（当前契约，7×7 示例） | **49 bytes**（fp32 等价 196 bytes） |
| selected-entry count（不是物理存储） | 14 |
| total/trainable parameters | `40,405 / 40,372` |

资源读数只能证明当前实现能构造该 mask，并显示 mask 本身没有 learned parameter；它不支持预测精度、显存优势、吞吐优势或端到端训练速度结论。旧 JSON 中的 `14` bytes 是 selected-entry count，不是 dense bool buffer 的物理字节；当前 v4 工件按同形状 `[C=7,K=7]` 报告 `49` bytes、fp32 等价 `196` bytes。sidecar 当前保存 prior 与 lag metadata，mask 按代码和 `top_k` 确定性重建；正式运行应同时保存 source/config hash、mask 形状/hash、每个 lag 的 `q_k` 和 support 统计，避免只凭一个 density 数字复核。

## Claim boundary

当前可以写：

- “实现了一个在共享 train-only ACF bank 上、按 `abs(ACF) * legal forecast-step fraction` 选择的逐通道稀疏 forecast-memory support mask。”
- “该候选不增加 learned 参数或 dense cross-channel projection；relation loss 路径保持独立。”
- “候选代码已具备结构契约；当前 v4 资源记录完成。严格 representation-causality draft 因全窗口归一化而被阻断，尚不能启动 paired exploratory retraining。”

当前不能写：

- “逐通道 ACF 已提升预测精度”或“top-2 是最优 lag 数”；
- “MambaACF 普遍优于 Mamba2、PatchTST 或所有基线”；
- “mask 带来速度/显存/参数优势”（未做统一 device-resident 与 E2E 对照）；
- “ACF 峰值选择具有因果增益”；
- “12-job screen 是确认性或 publication-ready 证据”。

## 人工复核清单

1. 冻结修订后的计划副本、launch sidecar、源码/数据/计划 hash；确认执行授权仍由人工给出。
2. 确认 `train_peaks_available` 的实际解析结果、每个 `q_k` 和 dense `[C,K]` mask 形状/hash；每个 on/off 配对必须完全一致，top-2 不得静默降级。
3. 确认 raw prior 只由 train split 拟合，sidecar 在训练前保存，测试指标不参与选择。
4. 确认 screen 使用 `skip_test=true`、`stream_metrics=false`，并从 `loss_logs.json:val_loss` 做 validation-only stopping；若后续生成 test 输出，必须与 validation 选择及 exploratory 根隔离。
5. 训练后按 `dataset × horizon × seed` 配对，报告 validation selection 与最终 test 描述性结果分离、每通道只作异质性描述。
6. 只有在独立 paired retraining、容量公平、ACF placebo/availability 对照和完整基线门通过后，才考虑提升到更高等级证据。

## 相关入口

- [`mamba_current_status.md`](mamba_current_status.md)：总状态与论文边界。
- [`autocon_mamba_acf_core_review.md`](autocon_mamba_acf_core_review.md)：Mamba/AutoCon/ACF 数据流与归因审计。
- [`mamba_acf_evidence_gap_audit_20260903.md`](mamba_acf_evidence_gap_audit_20260903.md)：基线、E2E、placebo 和 paired-retraining 缺口。
- [`mamba_acf_counterfactual_audit_20260903.md`](mamba_acf_counterfactual_audit_20260903.md)：已训练函数的 memory 分支反事实依赖；不含本候选重训结果。

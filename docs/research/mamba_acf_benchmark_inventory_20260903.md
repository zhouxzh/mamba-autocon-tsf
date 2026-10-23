# MambaACF 基准库存与人工复核清单（历史快照）

**审计日期：** 2026-09-03（pre-glob 历史扫描口径）  
**审计输入：** [`artifacts/benchmark_audit_20260903/`](../../artifacts/benchmark_audit_20260903/)  
**性质：** 只读库存。它确认计划坐标与可用指标文件是否对应，不选择模型、不会合并不同训练协议，也不构成统计检验或论文结论。

> 本页保留旧的 `24` 个计划 / `74` 个 phase 扫描结果，供历史追溯。当前审计器已纳入日期后缀计划；请以 [`benchmark_inventory_current_20260903.md`](benchmark_inventory_current_20260903.md) 及 [`artifacts/benchmark_audit_20260903_final2/`](../../artifacts/benchmark_audit_20260903_final2/) 的 `35` 个计划 / `102` 个 phase 结果为准。

## 可直接确认的状态

本次审计加载了 `24` 个文件名匹配 `*_plan.json` 的注册计划、`74` 个 phase。已发现 `5,875` 个结果配置和 `2,970` 个可用指标工件；其中 `2,751` 个指标位于注册 phase 根目录内，`219` 个位于注册根目录之外，按审计规则均只能视为历史/探索性工件。

| 项目 | 数量 | 解释 |
|---|---:|---|
| 计划坐标 | 6,715 | 所有已纳入审计的 phase 展开后的预期作业数 |
| 已匹配指标坐标 | 2,751 | 有可用 `metrics.npy` 或有效 streamed MSE 的注册坐标 |
| 缺失指标坐标 | 3,964 | 未匹配到可用测试指标 |
| 仅配置、无测试指标 | 2,898 | 全部同时被标记为 `trained_without_test`；它们是 validation-only 选择工件，不是已完成 test 结果 |
| phase 完整覆盖 | 57 / 74 | 仅表示该 phase 的已登记坐标都有指标 |
| phase 零覆盖 | 17 / 74 | 没有可用测试指标 |
| 部分覆盖 phase | 0 | 当前没有介于 0% 与 100% 之间的 phase |
| 审计键上的重复匹配 | 0 | 仅按 dataset/model/features/horizon/seed/lambda 的坐标键计算 |
| 注册根内额外指标行 | 0 | 不代表注册根外的 219 个工件不存在 |

9 个注册数据集 CSV 都存在且表头校验通过；没有未知 CSV，也没有计划解析错误。这说明本次库存的基础路径和数据注册表没有明显断裂。

## 已完成与未完成

“完成”仅指该计划自身登记坐标有可读指标，不等于可与其它计划并入同一比较表。

| 计划/类别 | 已匹配 / 预期 | 当前可支持的用途 | 关键限制 |
|---|---:|---|---|
| `final_training_plan` 的 strict 测试 phase | 1,260 / 2,520 | 历史 S/M、transfer 与 ILI 结果的只读复核 | 两个 selection phase 仅有 validation-only 配置；所有 4 个已完成 baseline phase 都缺 NLinear |
| `corrected_training_plan` | 900 / 2,160 | corrected S/M 历史结果复核 | 两个 selection phase 仅 validation-only；两个已完成 baseline phase 都缺 NLinear |
| `highdim_training_plan` | 270 / 648 | Electricity/Traffic 的高维后验 replay | selection 仅 validation-only；completed matrix 缺 NLinear，且文档已标注为 posthoc-attested，而非原始 launch-time confirmatory evidence |
| 已注册 MambaACF 探索计划（不含当前 confirmation 草案） | 321 / 323 | 机制筛选、memory/relation/variable-token 等方向性证据 | 均为 screening/exploratory；H96、少量 seed、预算和 source snapshot 不同，不能拼成排行榜 |
| `mamba_acf_confirmation_plan` | 0 / 884 | 当前唯一登记的完整 MambaACF 确认路线 | `draft_requires_prelaunch_review`，未启动；不得引用为已有实验结果 |
| `extension_baseline_plan` | 0 / 180 | 计划中的 Electricity/Traffic plain baseline 扩展 | 注册但没有任何可用指标，不能借用其它高维结果填充 |

`mamba_acf_v2_benchmark_plan` 是已完成探索路线中的唯一不完整计划：baseline/high-dim/local-memory 三个 phase 为 `31/31`，原始 `acf_local_relation_screen_M` 为 `0/2`。它随后有独立的 `mamba_acf_v2_relation_retry_plan`，其 retry phase 为 `2/2`；两者的计划 hash、source-tree hash 和输出根不同。因此 retry 是一个新的单-seed 探索记录，不能与原 phase 相加，更不能视为重复作业的第二次样本。

## NLinear 缺口

仓库中 `NLinear` 实现存在，但当前可发现的 NLinear 指标只有 `4` 个：

1. `mamba_acf_v2_baseline_screen_M` 中 ETTh1、Weather、Exchange-Rate 各 `H96/seed=2021` 一个，共 3 个；它们来自 3-epoch、单 seed 的注册 exploratory screen。
2. `artifacts/extension_pilot_20260831/traffic_h96/nlinear/` 中 Traffic `H96/seed=2021` 一个；它位于注册 phase 根之外，且是 batch size 16 的 3-epoch pilot。

这四个点均不能补入其它矩阵。特别是：

- `corrected_confirmatory_{S,M}`、`strict` 的 `confirmatory_core`、`confirmatory_M`、`secondary_transfer`、`exploratory_ili`，以及 `highdim_confirmatory_M` 共 `7` 个已完成、baseline-check 适用的 phase，模型列表本身没有 NLinear；所以即使覆盖率是 100%，也只是**缩减基线集**的完整覆盖。
- `mamba_acf_v3_relation_screen_plan` 的 5 个 phase 也被审计器标为缺少全部六个标准 baseline。这并非某个丢失文件，而是它们本来就是 MambaACF 消融 phase；它们不能被称为独立 baseline comparison。
- `extension_baseline_plan` 与当前 `mamba_acf_confirmation_plan` 都列入了 NLinear，但前者 `0/180`，后者 `0/884`，均未启动。
- [`mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json`](mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json) 已登记一个 ETTh1/Traffic、H96、3-seed 的 6-job 草案；它 `authorized=false`、`training_started=false`，目标输出根不存在。即使将来完成，它也只能是同一新 provenance 下的独立描述性 NLinear 记录，不能反向修补 historical strict/corrected/highdim 或早期 four-model fair screen。

## 不可合并的 provenance 组

以下结果即使数据集、H96 或 seed 名称相同，也不能直接平均、做配对检验或放入同一个主表。

| 组 | 主要差异 | 正确用法 |
|---|---|---|
| strict vs corrected | strict 为历史快照；corrected 使用独立输出根和修正后协议 | 各自只读复核；不能用 live validator 覆盖历史 strict 工件 |
| corrected/highdim vs 当前 MambaACF | 50 epochs、不同宽度/选择绑定；highdim 还是 posthoc-attested replay | 可作为历史基线背景，不能证明当前 MambaACF 的 non-inferiority |
| v2 baseline screen | ETTh1/Weather/Exchange-Rate、单 seed、3 epochs；高维 phase 更短 | 只作早期实现与 baseline trace，不能用于五-seed或高维主表 |
| fair baseline screen | ETTh1/Traffic、H96、3 seeds、8 epochs、raw ACF、单尺度与 causal embedding | 同一 plan 内可做其预先登记的探索性配对；NLinear 未包含 |
| v3 relation screen | 同为 ETTh1/Traffic H96 3 seeds，但 residual ACF source、不同 source snapshot，且比较 hard/soft relation 语义 | 只用于当前 relation 机制方向；不能与 fair baseline 数字混表 |
| variable/live/sidecar screen families | 存在 raw/residual source、causal embedding、memory/relation 开关及 source snapshot 差异；多个 phase 名称相同 | 用各自 plan/provenance ID 识别，不能因 phase 名称相同而把结果视为重复或独立 replication |
| 219 个未注册根工件 | 包含 pilot、standalone replay、资源测量及历史输出 | 不计入任何训练样本量；必要时只能逐一核对 checkpoint、配置和 replay 目的 |

审计器的重复判定也有范围：它只使用 dataset、canonical model、features、horizon、seed 和 AutoCon lambda。它不会比较 epoch、batch size、ACF source、Mamba width、训练数据比例、source snapshot、sidecar 或参数量。因此“重复匹配为 0”只能说明注册 phase 内没有同坐标的多条可用指标，不能证明语义上没有近似重复实验。

## 新草案未被本次自动审计覆盖

历史扫描时的 `scripts/audit_benchmark_matrix.py` 只扫描文件名以 `*_plan.json` 结尾的文件。下表 8 个可执行草案使用 `*_plan_20260902.json` 命名，均在审计生成前已存在，却没有进入当时的 `24` 个计划、`74` 个 phase 和 `6,715` 个坐标统计。这是**历史库存工具的命名匹配盲区**，不是训练完成状态；当前工具已修正为纳入日期后缀计划。

| 草案 | 已登记作业 | 授权/训练 | 输出根状态 |
|---|---:|---|---|
| anchor-copy screen | 12 | `authorized=false`, `training_started=false` | 不存在 |
| relation-lambda calibration | 18 | `authorized=false`, `training_started=false` | 不存在 |
| memory-gate-mode screen | 12 | `authorized=false`, `training_started=false` | 不存在 |
| NLinear baseline-gap screen | 6 | `authorized=false`, `training_started=false` | 不存在 |
| reliability-interpolation screen | 12 | `authorized=false`, `training_started=false` | 不存在 |
| scale-conditioning screen | 162 | `authorized=false`, `training_started=false` | 不存在 |
| single-vs-multiscale screen | 12 | `authorized=false`, `training_started=false` | 不存在 |
| variable-token replication | 60 | `authorized=false`, `training_started=false` | 不存在 |

合计 `294` 个草案作业未在本次自动库存中计数。另有两个同日期 JSON 是 launch sidecar，不是训练计划；`mamba_acf_extension_plan.json` 是 candidate-only 的 12 个 arm，也没有可展开的 `phases`，不应误标为零覆盖训练计划。

## 人工复核入口

在授权任何新的 accuracy 训练前，建议逐项确认：

1. 将 [`mamba_acf_confirmation_plan.json`](mamba_acf_confirmation_plan.json) 作为当前论文级确认路线的唯一候选；它的 `884` 个作业目前均未运行，不能由历史 artifact 替代。
2. 决定 NLinear 的角色：若它是论文级 baseline，必须在同一冻结 provenance/预算下与其它比较对象重跑；若只补 6-job gap screen，报告必须保持 standalone descriptive。
3. 在下次库存前，显式传入日期后缀计划，或调整审计工具的计划发现规则；否则 `294` 个未授权草案将持续从总量、缺失坐标和覆盖率中消失。
4. 对任何要进入同一统计表的坐标同时核对 plan hash、source-tree/source snapshot、dataset hash、split/train ratio、epochs/early-stop、features、模型预算、ACF source、relation objective 与 launch sidecar。只核对 MSE 文件或 phase 名称不足够。
5. 保留原始 v2 local-relation 缺失 phase 与 retry 的分离记录；不要把 retry 计作同一计划的第二个 seed。
6. 将资源-only、same-checkpoint counterfactual 与 standalone replay 明确标为机制/实现证据，不计入训练独立样本数或 speed/accuracy 主结论。

## 当前可写与不可写

当前可以写：仓库已有完整的历史/探索性工件库存、部分 MambaACF 机制筛选和已训练 checkpoint 的分支依赖诊断；MambaACF 及 AutoCon/ACF 的有效性仍是数据集和协议相关的探索性问题。

当前不能写：完整六基线比较、NLinear 已补齐、当前 MambaACF 对 PatchTST/Mamba2 的确认性非劣、跨协议 pooled 平均、或端到端速度优势。上述主张必须等待同一冻结确认计划、完整 baseline、配对 seed 和资源协议完成后再判断。

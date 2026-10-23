# Anchor x Soft-ACF-Memory Interaction Screen

这是一个当前代码树上的最小 2x2 机制筛选计划。原始 JSON 仍是不可变的 draft
（`execution.authorized=false`、`training_started=false`）；但该 draft 已在冻结源码快照
`2837199515e0ee7f...` 下事后执行为 validation-only exploratory screen。四个 phase
各有 `6/6` 个成功 job 和 `valid=true` 的 `phase_validation.json`，合计 `24/24`。这不把
原计划追溯性地变成预注册或确认性实验，详见
[`erratum_20260907`](mamba_acf_anchor_memory_interaction_screen_erratum_20260907.md)。注册的四个 cell 是：

| cell | direct anchor | soft static ACF memory |
|---|---:|---:|
| `plain_h96` | off | off |
| `anchor_only_h96` | on | off |
| `memory_only_h96` | off | on |
| `anchor_memory_h96` | on | on |

范围固定为 ETTh1、Traffic，H96，seed `2021/2022/2023`，共 24 jobs。四个 cell 共享 `MambaACFVariable` alias（实际 executable model 为 `MambaACF`）、MeanStd、raw train-only ACF sidecar、同一优化器/学习率/数据 split/epoch budget；唯一注册因素是 `mamba_acf_anchor_copy` 与 `mamba_use_acf_memory`。

## 运行前检查

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_acf_anchor_memory_interaction_screen_plan_20260904.json \
  --validate_only
```

原计划的验证输出为 24 jobs。该条保留为历史合同；不要覆盖已有 artifact 根目录或改写
原计划，以免破坏其 SHA-256 绑定。

## 推荐执行顺序

实际执行时，`run_frozen_exploratory_plan.py` 已支持 `skip_test=true`、
`stream_metrics=false` 的 validation-only phase，并为每个 phase 写入 lightweight
`phase_validation.json`。这是事后 exploratory 执行方式，不是对原计划授权状态的改写。
以下常规 runner 命令仅保留为合同参考：

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_acf_anchor_memory_interaction_screen_plan_20260904.json \
  --phase plain_h96

conda run --no-capture-output -n mamba python scripts/validate_training_phase.py \
  --plan docs/research/mamba_acf_anchor_memory_interaction_screen_plan_20260904.json \
  --phase plain_h96 \
  --manifest artifacts/exploratory/mamba_acf_anchor_memory_interaction_screen_20260904/plain_h96/logs/job_manifest.jsonl \
  --provenance artifacts/exploratory/mamba_acf_anchor_memory_interaction_screen_20260904/plain_h96/provenance.json \
  --output artifacts/exploratory/mamba_acf_anchor_memory_interaction_screen_20260904/plain_h96/phase_validation.json
```

对其余三个 phase 重复上述流程。分析器把 `phase_validation.json` 视为完整性门，不读取未验证的 manifest。

## 只读分析

```bash
conda run --no-capture-output -n mamba python scripts/analyze_mamba_acf_anchor_memory_interaction_screen.py \
  --plan docs/research/mamba_acf_anchor_memory_interaction_screen_plan_20260904.json
```

分析器使用每个 checkpoint 的 `loss_logs.json:val_loss` 最小值作为 primary metric，不读取 test metrics。它核验 phase validation、manifest/artifact hashes、provenance/source snapshot、raw sidecar hash、paired config，以及 anchor checkpoint key；缺少任何 cell 时返回 `NOT_RUN` 并不写 readout。

主要统计量为四个 simple effects 和 difference-in-differences：

```text
(anchor_memory - memory_only) - (anchor_only - plain)
```

负值表示两个机制的误差下降具有超加性；正值表示干扰或次加性。三个 seed 只能作机制筛选，不能作为论文确认性证据，也不能据此继续堆叠 AutoCon、温度或其他 ACF 因子。

## 已执行读数与边界

合并臂相对 plain 的 validation scaled MSE 为 ETTh1 `-0.656%`（`3/3`）和
Traffic `-0.475%`（`2/3`），但交互量在 ETTh1 为 `+0.000486`（三个 seed 均非负），
Traffic 为 `-0.001027`（`2/3` 为负），pooled 为 `-0.000270`，95% CI
`[-0.001449, +0.000908]`。因此当前结论是“合并臂没有明显损害”，不是“两个 ACF
机制存在稳定协同”；不得据此把组合升级到 test selection 或继续加机制。

该 interaction screen 使用 `mamba_causal_embedding=false`、`skip_test=true`、
`stream_metrics=false`、seeds `2021--2023`；它与 fresh direct-anchor held-out
replication 的 `mamba_causal_embedding=true`、test-enabled、seeds `2024--2028`
不是同一协议，二者不得合并统计。完整 readout 见
[`mamba_acf_anchor_memory_interaction_screen_readout_20260904.md`](mamba_acf_anchor_memory_interaction_screen_readout_20260904.md)。

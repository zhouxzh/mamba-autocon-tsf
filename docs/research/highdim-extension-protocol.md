# 高维真实数据扩展协议

本文定义 ECL electricity 和 Traffic 两个真实高维数据集的预注册扩展。它是独立于 ETT 确认性结果的后续阶段；在所有 gate 通过前，扩展结果不进入主论文结论。需要区分设计与实际证据：当前 `_v3` 工件虽已完整生成并通过 validator，但 selection sidecar 为 `posthoc_verified`、`publication_ready=false`，所以不能把该次执行追溯性地称为 launch-time 预注册确认。

## 数据合同

| 数据集 | 文件 | 值列数 | 频率 | SHA256 |
| --- | --- | ---: | --- | --- |
| electricity | `dataset/electricity/electricity.csv` | 321 | hourly | `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad` |
| traffic | `dataset/traffic/traffic.csv` | 862 | hourly | `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16` |

两份 CSV 都必须以 `date` 开头，目标列为 `OT`，使用 multivariate `features=M`。文件来源、revision、许可证和逐列质量摘要写在计划 JSON 的 `dataset_contract` 及 `artifacts/paper_final/corrected/highdim_data_quality/*.json` 中。运行前重新执行质量审计；哈希、列数、小时 cadence、重复时间戳、有限值和常量列任一不符都停止阶段。Weather 当前存在重复/不规则时间记录，不纳入本扩展。

计划同时固定了两份质量报告的 canonical content SHA256（排除生成时间和本机路径）。正式启动前必须运行
`scripts/validate_highdim_contract.py --strict`；`run_training_plan.py`
对含 `dataset_contract` 的计划也会自动执行同一检查。该 gate 会把当前 CSV
哈希、CSV 结构和质量报告哈希/内容绑定到启动边界。

## 模型和优化

注册的九个 arm 与核心阶段保持一致：DLinear、PatchTST、TimesNet、AutoConNet、AutoConNetAutoCon、Mamba1、Mamba1AutoCon、Mamba2、Mamba2AutoCon。Mamba1/2 使用当前 `mamba_ssm` CUDA 实现；运行环境必须记录 GPU、PyTorch、`mamba_ssm`、`causal_conv1d` 和 Triton 版本。高维配置固定 `seq_len=96`、`label_len=48`、`d_model=d_ff=64`、一层 Mamba block、`d_state=16`、`d_conv=4`、`expand=2`、`headdim=16`、batch size 16、最多 50 epochs、patience 10、cosine learning rate、`weight_decay=1e-5`。这些配置是容量/资源折衷，不代表长上下文能力；论文必须报告参数量和 GPU 资源。

Mamba 实现先把全部通道投影到共享 `d_model`，再用预测头映射回通道。因此它是 channel-mixing 的多变量模型，不是逐通道 independent (CI) Mamba。`headdim=16` 由 Mamba-2 使用；Mamba-1 的实现不暴露该参数，资源记录中应为 `null`。ECL/Traffic 的高维结果应按这一模型定义解释，并补充逐通道宏平均和整体微平均误差。

## 阶段和规模

历史 `_v3` 九臂阶段的唯一注册入口为 [`highdim_training_plan.json`](highdim_training_plan.json)。其正式执行输出到 `artifacts/paper_final/corrected/highdim_selection_M_v3/` 和 `artifacts/paper_final/corrected/highdim_confirmatory_M_v3/`；旧的无后缀目录保存 4-job 预检，`_v2` 目录保存 1-job 的旧 validator 快照预检。另一个 [`extension_baseline_plan.json`](extension_baseline_plan.json) 是独立、未训练的 180-job 高维补基线计划；它不属于 `_v3`，也不得与 `_v3` 或未来 P3/P3b 的结果拼表。

- `highdim_selection_M`：2 数据集 × 3 horizon（96/336/720）× 3 AutoCon arm × 7 个 lambda × 3 个 selection seed = **378 个作业**。只训练/验证，使用 `--skip_test`，不读取测试指标。lambda 候选为 `{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}`；在每个 dataset/model/horizon/seed cell 内归一化最佳 validation MSE，再按模型选择平均相对损失最低者。
- `highdim_confirmatory_M`：2 数据集 × 3 horizon × 9 arm × 5 个确认性 seed = **270 个作业**。lambda 必须来自已验证且独立 attested 的 `highdim_selection_M` 输出，禁止查看测试集后重选。

两阶段合计 648 个作业。高维确认性阶段不保存完整 `pred.npy/true.npy`，而是启用 `stream_metrics=true`：每个测试窗口在线累计 scaled/original MAE、MSE、RMSE、MAPE、MSPE，逐通道 MAE/MSE/RMSE，并按默认 `origin_stride=pred_len` 保留非重叠 origin 汇总和最多四个抽样窗口。`stream_metrics.json` 与 `stream_samples.npz` 是验证所需的紧凑证据；窗口不是独立重复，置信区间仍以 seed 为单位。

## 启动前 gate

按下列顺序执行，任何一步失败都不得继续：

1. 数据审计：
   ```bash
   conda run --no-capture-output -n mamba python scripts/audit_dataset_quality.py \
     --csv dataset/electricity/electricity.csv --dataset electricity \
     --expected-channels 321 --expected-freq h \
     --expected-sha256 7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad \
     --source-repository thuml/Time-Series-Library \
     --source-revision 2b66e59ee19dac8f6f19fb5d4997f289fdfea357 --license CC-BY-4.0 \
     --strict --output artifacts/paper_final/corrected/highdim_data_quality/electricity.json
   conda run --no-capture-output -n mamba python scripts/audit_dataset_quality.py \
     --csv dataset/traffic/traffic.csv --dataset traffic \
     --expected-channels 862 --expected-freq h \
     --expected-sha256 cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16 \
     --source-repository thuml/Time-Series-Library \
     --source-revision 2b66e59ee19dac8f6f19fb5d4997f289fdfea357 --license CC-BY-4.0 \
     --strict --output artifacts/paper_final/corrected/highdim_data_quality/traffic.json
   ```
2. 先做真实 GPU resource benchmark 和单 job GPU smoke；synthetic high-dimensional benchmark 只能证明容量/延迟，不能证明预测效果。当前 smoke 已完成并记录在 [`artifacts/paper_final/corrected/highdim_smoke/README.md`](../../artifacts/paper_final/corrected/highdim_smoke/README.md)，仍不进入效果表。
3. 用 `--dry_run --phase highdim_selection_M --limit 1` 检查命令后，在计划注册的 `_v3` 根目录运行完整 selection。`_v2` 与无后缀目录中的少量作业仍是旧快照诊断，不能续跑或进入选择统计；当前 `_v3` selection 已完成并通过 validator（378/378）。完整阶段完成后依次运行 `validate_training_phase.py`、`select_hyperparameters.py` 和 `attest_selection.py`；保存 manifest、validator JSON、selection JSON 和 sidecar。
4. 用 attested selection 启动 confirmatory。每个 job 必须有 immutable provenance、checkpoint、validation trajectory、`stream_metrics.json` 和 `stream_samples.npz`。确认性阶段先通过 validator，再运行 report/statistics 脚本。

Selection 完成后的固定命令（必须使用 `_v3` 路径）如下。三条命令按顺序执行，任何一步失败都不得启动 confirmatory：

```bash
conda run --no-capture-output -n mamba python scripts/validate_training_phase.py \
  --plan docs/research/highdim_training_plan.json \
  --phase highdim_selection_M \
  --manifest artifacts/paper_final/corrected/highdim_selection_M_v3/logs/job_manifest.jsonl \
  --provenance artifacts/paper_final/corrected/highdim_selection_M_v3/provenance.json \
  --output artifacts/paper_final/corrected/highdim_selection_M_v3/phase_validation.json

conda run --no-capture-output -n mamba python scripts/select_hyperparameters.py \
  --manifest artifacts/paper_final/corrected/highdim_selection_M_v3/logs/job_manifest.jsonl \
  --validation artifacts/paper_final/corrected/highdim_selection_M_v3/phase_validation.json \
  --plan docs/research/highdim_training_plan.json \
  --phase highdim_selection_M \
  --output artifacts/paper_final/corrected/highdim_selection_M_v3/selected_hyperparameters.json

conda run --no-capture-output -n mamba python scripts/attest_selection.py \
  --selection artifacts/paper_final/corrected/highdim_selection_M_v3/selected_hyperparameters.json \
  --validation artifacts/paper_final/corrected/highdim_selection_M_v3/phase_validation.json \
  --plan docs/research/highdim_training_plan.json \
  --manifest artifacts/paper_final/corrected/highdim_selection_M_v3/logs/job_manifest.jsonl \
  --output artifacts/paper_final/corrected/highdim_selection_M_v3/selected_hyperparameters.provenance.json
```

Confirmatory 实际启动还必须显式传入这两个绑定文件：

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/highdim_training_plan.json \
  --phase highdim_confirmatory_M \
  --selection_file artifacts/paper_final/corrected/highdim_selection_M_v3/selected_hyperparameters.json \
  --selection_provenance artifacts/paper_final/corrected/highdim_selection_M_v3/selected_hyperparameters.provenance.json
```

计划校验（只读，不启动训练）：

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/highdim_training_plan.json --validate_only
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/highdim_training_plan.json \
  --phase highdim_selection_M --dry_run --limit 1
conda run --no-capture-output -n mamba python scripts/validate_highdim_contract.py \
  --plan docs/research/highdim_training_plan.json --strict \
  --output artifacts/paper_final/corrected/highdim_data_quality/contract_validation.json
```

## 统计和风险边界

主比较仍是同一 dataset、horizon、seed 的配对差，例如 `Mamba2AutoCon - Mamba2` 的 AutoCon 增量；数据集之间不把通道或窗口数当作重复。确认性报告还必须运行 `scripts/summarize_channel_metrics.py`，同时给出整体微平均、逐通道宏平均、origin-stride 敏感性、参数量和显存/吞吐。该脚本输出的 seed 表和 group 表只把 seed 作为推断单位。

高维风险包括：CUDA-only Mamba kernel、batch/显存不足、长 horizon 的有效测试 origin 数较少、Traffic/ECL 通道尺度差异、当前低秩 channel-mixing 瓶颈，以及多重比较导致的功效下降。MAPE/MSPE 在含近零负荷的通道上可能极端，只作诊断，不参与主排序或假设检验。遇到 OOM 只能在未读测试集前按预注册 resource gate 调整 batch 或停止该配置，并记录变更；不能临时删掉失败 arm、seed 或 horizon。Synthetic resource 结果和 smoke 结果均不进入效果表。

## 当前工件状态（2026-08-31）

旧的无后缀目录有 4/378 个诊断预检，`_v2` 目录有 1/378 个旧 validator 快照预检；它们不用于选择或效果统计。计划注册的 `_v3` selection 已通过 `valid=true`、`expected_jobs=378`、`manifest_unique_jobs=378`；其 confirmatory 阶段也已通过 `valid=true`、`expected_jobs=270`、`manifest_unique_jobs=270`、`status_counts.ok=270`。confirmatory manifest 的 280 行包含 10 次因路径名过长而失败的早期尝试，随后同一 job ID 均成功，validator 按最终记录去重后得到 270/270。

这 270/270 是完整的可复核结果工件，但不是原始 launch-time 预注册确认：confirmatory provenance 的绑定模式为 `prelaunch_replay_of_posthoc_attestation`，selection sidecar 为 `attestation_mode=posthoc_verified`、`publication_ready=false`。因此报告可以使用这批后验 replay 的冻结读数，但必须保留该证据等级限制。机制消融、容量公平性及新 MambaACF 结构的独立多 seed 验证仍待完成。

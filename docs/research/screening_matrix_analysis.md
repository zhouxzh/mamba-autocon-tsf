# Screening matrix analysis

[`scripts/analyze_screening_matrix.py`](../../scripts/analyze_screening_matrix.py) 是一个只读的结果整理工具，用于把多个 phase 的紧凑指标和 `resource_v2` 测量放到同一份可审计输出中。它不读取预测数组、不重跑模型，也不做模型或超参数选择。

## 输入与输出

训练结果根目录通过重复的 `--results-root` 指定。每个 `config.json` 必须提供 `data`、`features`、`seq_len`、`pred_len` 和 seed（`run_seed` 或 `seed`）；指标按以下优先级读取：

1. `stream_metrics.json` 的 `scaled.mse`；
2. 旧格式 `metrics.npy` 的第二个元素（MSE）；
3. 少数旧 JSON 指标格式。

资源可以通过 `--resource-root` 递归发现 `model_resources.json`，或用 `--resource-json` 显式指定。`resource_records[*].gpu_inference` 的 latency/throughput，以及 `parameters` 的参数量会被汇总；显存优先使用 `steady_state_peak_allocated_mb`，缺失时回退到 reserved 字段。资源行标记为 `measurement_only=true`，因为 synthetic forward 测量不等价于端到端训练速度。

输出目录必须显式给出 `--output-dir`，默认拒绝覆盖已有文件，也拒绝把输出写在任何输入根目录下。需要重建同一分析目录时必须显式传 `--overwrite`。输出包括：

- `raw_results.csv`、`seed_summary.csv`、`model_summary.csv`；
- `paired_deltas.csv`、`paired_summary.csv`；
- `resource_rows.csv`、`resource_summary.csv`；
- `screening_matrix.json`、`screening_matrix.md` 和 `issues.csv`。

## 配对与证据边界

配对键是 `dataset × features × seq_len × horizon × seed`。对每个 `--baseline`，`paired_delta` 定义为：

`candidate scaled MSE - baseline scaled MSE`

因此负值表示候选误差较低，`relative_delta` 是该差值除以基线 MSE。缺基线、重复坐标或重复坐标的 MSE 不一致时，行会保留但状态分别标为 `missing_baseline`、`ambiguous_*` 或 `invalid_candidate`，不会静默取最小值/最新值。

聚合行带有以下证据标签：

- `one_seed=true`：只有一个有效 seed；
- `short_budget=true`：配置中的训练 epoch 不超过 `--short-budget-epochs`（默认 3）；
- `exploratory=true`：满足上述任一条件，或配置明确标记为 pilot/screen/smoke。

带 `exploratory=true` 的结果只用于筛查和后续实验设计，不能单独支撑确认性精度结论。不同 phase 的切分、优化器、宽度或来源不一致时，工具仍会并列展示，但人工复核必须依据 `source_roots`、`phases` 和原始 `config_paths` 决定是否可以合并。

示例：

```bash
conda run --no-capture-output -n mamba python scripts/analyze_screening_matrix.py \
  --results-root artifacts/fair_pilot_20260831/results \
  --results-root artifacts/extension_pilot_20260831/traffic_h96 \
  --resource-root artifacts/resource_v2_screen \
  --baseline PatchTST --baseline DLinear --baseline Mamba2 \
  --output-dir artifacts/screening_analysis_20260831
```

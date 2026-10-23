# 资源公平性采集

[`scripts/benchmark_model_resources.py`](../../scripts/benchmark_model_resources.py) 是确认性结果完成后的独立分析工具。它不调用 `run.py`、不读取数据集窗口；从已通过 validator 的 `config.json` 重建模型，并优先使用该阶段 provenance 中的只读 source snapshot。也可以用明确的 CLI 架构参数做未验证的资源预览。

## 输出

- `model_resources.json` / `model_resources.csv`：模型参数总量、可训练参数、参数/ buffer 大小，以及可选的 GPU 推理延迟、p95、吞吐和 steady-state peak memory。
- `training_elapsed_summary.csv`：从 manifest 按 `dataset × features × horizon × model × 配置` 汇总成功 job 的总耗时、每 job 的均值/中位数和 seed 数。
- 每条记录保留 `config_sha256`、`provenance_id`、source snapshot 和资源签名，便于人工复核。

本次环境记录为 Python `3.9.25`、PyTorch `2.8.0+cu128`、CUDA `12.8`、`mamba-ssm 2.3.1`、`causal-conv1d 1.6.2.post1`、Triton `3.4.0`，GPU 为 NVIDIA GeForce RTX 5090 D。Mamba-1 和 Mamba-2 都通过了 CUDA 前向/反向 smoke；CPU-only 环境应明确报 CUDA kernel 不可用，而不是静默切换实现。

吞吐/延迟使用固定 seed、固定 batch 的合成 device-resident 输入；预热轮次用于排除 Mamba/Triton 首次初始化。窗口、通道和计时迭代只表示负载，不是预测效果的独立统计样本；论文中的显著性检验仍以 seed 为单位。

默认基准是 prediction-only：对声明 `supports_return_repr_control` 的模型传入
`return_repr=false`，因此不把 AutoCon 辅助表示计入部署预测负载。若要审计
自监督训练期的前向表示开销，可显式加入 `--include-representation`；工具会传入
`return_repr=true`，检查并记录表示形状，同时测量预测和表示共同产生的延迟/显存。
该模式仍不包含 relation loss、反向传播、优化器或数据加载，输出中的
`forward_path`、`auxiliary_representation_*` 和 `relation_loss_included=false` 必须
与 prediction-only 记录分开比较。没有表示输出的模型在该模式下会直接报错，避免
把不完整的负载静默标记为表示基准。

资源基准会复用训练器的任务输出合同：`features=MS` 时，模型内部可能先返回全部输入通道，基准在测量前执行 `prediction[:, :, -1:]`，并在 JSON 中写入 `prediction_contract=trainer_MS_target_last_channel`。Mamba-1 的 `effective_mamba_architecture.mamba_headdim` 为 `null`（该实现不使用 headdim）；Mamba-2 才记录实际 headdim。

## 推荐流程

确认性阶段完成并通过 validator 后，在空闲 GPU 上执行。例如 S 阶段：

```bash
PHASE=corrected_confirmatory_S
ROOT=artifacts/paper_final/corrected/$PHASE

conda run --no-capture-output -n mamba python scripts/validate_training_phase.py \
  --plan docs/research/corrected_training_plan.json --phase "$PHASE" \
  --manifest "$ROOT/logs/job_manifest.jsonl" \
  --output "$ROOT/phase_validation.json"

conda run --no-capture-output -n mamba python scripts/benchmark_model_resources.py \
  --phase-validation "$ROOT/phase_validation.json" --all-validated \
  --output-dir "$ROOT/resource_benchmark" \
  --run-benchmark --device cuda:0 --benchmark-batch-size 32 \
  --warmup-iters 25 --timed-iters 100 --input-seed 2718
```

`--all-validated` 会枚举该阶段全部配置，再按资源签名去重；因此不同 seed 的相同架构只测一次，而 manifest 的 seed-level elapsed 仍完整汇总。若只需要检查参数量和 manifest 耗时，省略 `--run-benchmark`，不会执行 GPU 前向。

显式架构预览（未绑定验证结果）示例：

```bash
conda run --no-capture-output -n mamba python scripts/benchmark_model_resources.py \
  --model Mamba2AutoCon --seq-len 96 --label-len 48 --pred-len 96 \
  --enc-in 1 --dec-in 1 --c-out 1 --freq h --output-dir /tmp/mamba2-resource
```

未验证模式的输出只能用于开发期容量检查，不能替代确认性阶段的 provenance 资源表。训练仍在运行时不要加 `--run-benchmark`，避免与训练争用 GPU。

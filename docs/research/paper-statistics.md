# 论文统计输出

`scripts/compute_paper_statistics.py` 面向一个 corrected confirmatory phase，直接读取每个结果目录的 `config.json` 与 `metrics.npy`。它不读取或改写训练源快照，也不把 `raw_results.csv` 当作数值来源。

配对单位是相同 dataset、feature、预测长度和 seed 的 plain/AutoCon 两个模型。默认分析标准化空间的 MSE，并输出每个 cell 的均值 delta（`AutoCon - plain`）、相对变化、Student-t 95% CI、双侧配对 t 检验，以及 Holm-Bonferroni 和 Benjamini-Hochberg FDR 校正。负 delta 表示误差下降；跨 cell 的 backbone 汇总仅作描述，不作为额外独立检验。

corrected S/M 阶段完成并通过 phase validator 后分别运行：

```bash
conda run --no-capture-output -n sci-agent python scripts/compute_paper_statistics.py \
  --results_dir artifacts/paper_final/corrected/confirmatory_S \
  --plan docs/research/corrected_training_plan.json \
  --phase corrected_confirmatory_S \
  --validation artifacts/paper_final/corrected/confirmatory_S/phase_validation.json \
  --output_dir artifacts/paper_final/corrected/confirmatory_S/statistics

conda run --no-capture-output -n sci-agent python scripts/compute_paper_statistics.py \
  --results_dir artifacts/paper_final/corrected/confirmatory_M \
  --plan docs/research/corrected_training_plan.json \
  --phase corrected_confirmatory_M \
  --validation artifacts/paper_final/corrected/confirmatory_M/phase_validation.json \
  --output_dir artifacts/paper_final/corrected/confirmatory_M/statistics
```

输出为 `paper_statistics.json`（含 seed-level delta、artifact 相对路径和哈希，并记录统计脚本自身 SHA-256）与 `paper_statistics.md`（人工复核表）。脚本会校验 phase validator 保存的 manifest 和 artifact 哈希；正式论文统计不能绕过该校验。对于含 `confirmation_contract` 的计划，统计入口还必须验证 phase provenance 中与计划哈希一致的已批准 launch sidecar，且拒绝 `--allow-incomplete`/`--allow-unvalidated`。分析脚本 SHA-256 是统计代码指纹，不替代训练 provenance 的源码快照。只有在有意检查中间结果时才使用 `--allow-incomplete` 或 `--allow-unvalidated`，这两种模式都不应进入论文主表。

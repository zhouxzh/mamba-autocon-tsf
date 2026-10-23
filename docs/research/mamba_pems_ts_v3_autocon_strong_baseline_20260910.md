# Calibrated MambaTS v3 AutoCon vs TimesNet（2026-09-10）

## 目的与边界

本轮固定前一轮校准得到的 `MambaTSv3AutoCon, AutoCon_lambda=0.001`，在 PEMS03/04/07/08、H=12/96、seed 2022/2023 上重新训练 16 个 validation-only 作业，并与已经冻结的 TimesNet seed-2022 strong-baseline rows 与 seed-2023 fresh-seed rows 做精确坐标配对。测试集保持关闭；这是 exploratory cross-plan pairing，不是 launch-time preregistered confirmation。

计划和工件：

- [`mamba_pems_ts_v3_autocon_strong_baseline_20260910.json`](mamba_pems_ts_v3_autocon_strong_baseline_20260910.json)
- [`timesnet_readout.md`](../../artifacts/exploratory/mamba_pems_ts_v3_autocon_strong_baseline_20260910/readout/timesnet_readout.md)
- [`phase_validation.json`](../../artifacts/exploratory/mamba_pems_ts_v3_autocon_strong_baseline_20260910/pems_mamba_ts_v3_autocon_strong_baseline_h12_h96/phase_validation.json)

冻结阶段为 `valid=true`、`16/16`、无畸形 manifest 行、`test_selection_allowed=false`。TimesNet 参考来自：

- [`mamba_pems_ts_v3_strong_baseline_screen_20260909.md`](mamba_pems_ts_v3_strong_baseline_screen_20260909.md)，seed 2022；
- [`mamba_pems_ts_v3_timesnet_fresh_seed_20260909.md`](mamba_pems_ts_v3_timesnet_fresh_seed_20260909.md)，seed 2023。

配对脚本会拒绝缺失或重复坐标：[`analyze_mamba_ts_autocon_strong_baseline.py`](../../scripts/analyze_mamba_ts_autocon_strong_baseline.py)。

## 汇总结果

负值表示 calibrated AutoCon candidate 的 validation MSE 更低。

| 分层 | 配对数 | candidate 胜出 | 平均相对变化 | 中位数 |
|---|---:|---:|---:|---:|
| 全部 | 16 | 10/16 | -6.3818% | -5.3461% |
| H=12 | 8 | 8/8 | -19.5021% | -20.1806% |
| H=96 | 8 | 2/8 | +6.7385% | +6.8886% |

逐数据集的 4 个坐标（两 seed、两 horizon）平均相对变化：PEMS03 `+4.56%`、PEMS04 `+2.67%`、PEMS07 `-16.65%`、PEMS08 `-16.11%`。因此候选优势主要来自 PEMS07/08 和短期 H=12；PEMS03/04 的 H=96 是稳定失败模式。

## 解释

1. 低权重 AutoCon 没有破坏 v3 的短期预测能力：H=12 对 TimesNet 为 `8/8`、平均低 `19.5%`。
2. 结果不能写成普遍优于 TimesNet：H=96 只有 `2/8`，平均高 `6.74%`；PEMS03/04 H=96 的 TimesNet 优势约 `12--20%`。
3. 这批结果支持的论文定位是“低干扰 AutoCon 表示正则化的 MambaTS v3 短期交通预测候选”，不是全 horizon SOTA。长期预测要么需要专门的 horizon-aware memory/decoder，要么应明确作为失败/限制分析。
4. AutoCon 的独立收益仍未被证明：`lambda=0.001` 相对同结构零权重仅有接近零的平均变化；本轮只证明保留表示目标后，v3 与 TimesNet 的短期竞争力仍在。

## 复核命令

```bash
/home/zhong/anaconda3/envs/mamba/bin/python scripts/run_training_plan.py \
  --plan docs/research/mamba_pems_ts_v3_autocon_strong_baseline_20260910.json \
  --validate_only
/home/zhong/anaconda3/envs/mamba/bin/python scripts/analyze_mamba_ts_autocon_strong_baseline.py \
  --candidate-csv artifacts/exploratory/mamba_pems_ts_v3_autocon_strong_baseline_20260910/readout/validation_rows.csv \
  --reference-csv artifacts/exploratory/mamba_pems_ts_v3_strong_baseline_screen_20260909/readout/validation_rows.csv \
  --reference-csv artifacts/exploratory/mamba_pems_ts_v3_timesnet_fresh_seed_20260909/readout/validation_rows.csv \
  --output-dir artifacts/exploratory/mamba_pems_ts_v3_autocon_strong_baseline_20260910/readout
```

下一步若要进入论文主实验，必须预先冻结更长训练预算、至少 5 个 seed、完整测试释放和非劣效阈值；不能依据本轮验证结果直接选择 H=12-only 的论文主表。

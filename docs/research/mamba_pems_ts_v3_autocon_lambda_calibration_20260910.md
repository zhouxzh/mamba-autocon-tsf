# MambaTS v3 + AutoCon 权重校准（2026-09-10）

## 目的

当前 `MambaTSv3AutoCon` 把 legacy AutoCon 表示损失直接回传到共享 Mamba 编码器。此前 `AutoCon_lambda=0.03` 在 PEMS 四数据集、H=12/96 的 seed 2022 screen 中仅 1/8 坐标改善，平均验证 MSE 变化为 `+1.64%`。本轮只回答一个问题：是否存在一个足够小的正权重，使 AutoCon 表示学习保持启用而基本不扰动预测分支。

## 固定合同

- 数据：PEMS03/04/07/08，NPZ `data` 的 feature 0；文件 hash 与数组 shape 见 [`pems_dataset_audit_20260909.md`](pems_dataset_audit_20260909.md)。
- 输入/目标：`features=M`、`seq_len=96`、`label_len=48`、H=12/96、MeanStd、相同四 epoch 预算和优化器。
- 目标：仅 legacy AutoCon；未启用 MambaACF memory 或 relation-loss 分支。
- 评价：每个数据集×horizon×seed 的最小 validation MSE；测试集保持关闭。
- 控制：`MambaTSv3AutoCon` 的 `lambda=0`，因此控制与正权重共享同一模型接口和初始化路径，仅关闭辅助损失梯度。

## 工件

第一轮（seed 2022，40 jobs）计划与工件：

- [`mamba_pems_ts_v3_autocon_lambda_calibration_20260910.json`](mamba_pems_ts_v3_autocon_lambda_calibration_20260910.json)
- [`lambda_readout.md`](../../artifacts/exploratory/mamba_pems_ts_v3_autocon_lambda_calibration_20260910/readout/lambda_readout.md)
- [`phase_validation.json`](../../artifacts/exploratory/mamba_pems_ts_v3_autocon_lambda_calibration_20260910/pems_mamba_ts_v3_autocon_lambda_h12_h96/phase_validation.json)

第二轮（seed 2023，24 jobs）计划与工件：

- [`mamba_pems_ts_v3_autocon_lambda_fresh_seed_20260910.json`](mamba_pems_ts_v3_autocon_lambda_fresh_seed_20260910.json)
- [`lambda_readout.md`](../../artifacts/exploratory/mamba_pems_ts_v3_autocon_lambda_fresh_seed_20260910/readout/lambda_readout.md)
- [`phase_validation.json`](../../artifacts/exploratory/mamba_pems_ts_v3_autocon_lambda_fresh_seed_20260910/pems_mamba_ts_v3_autocon_lambda_fresh_seed_h12_h96/phase_validation.json)

两个冻结阶段均为 `valid=true`、分别 `40/40` 与 `24/24`、无畸形 manifest 行、`test_selection_allowed=false`。

## 结果（相对各自 lambda=0 控制）

| lambda | 坐标数 | 改善数 | 平均相对变化 | 中位数 | 最小/最大变化 |
|---:|---:|---:|---:|---:|---:|
| 0.001（seed 2022） | 8 | 5/8 | -0.0025% | -0.0076% | -0.0647% / +0.0350% |
| 0.003（seed 2022） | 8 | 4/8 | +0.0227% | +0.0013% | -0.0691% / +0.1205% |
| 0.010（seed 2022） | 8 | 3/8 | +0.0976% | +0.0370% | -0.0673% / +0.3929% |
| 0.030（seed 2022） | 8 | 2/8 | +0.3142% | +0.2804% | -0.1380% / +0.8706% |
| 0.001（seed 2023） | 8 | 6/8 | -0.0164% | -0.0100% | -0.0466% / +0.0081% |
| 0.003（seed 2023） | 8 | 4/8 | -0.0144% | -0.0059% | -0.1179% / +0.0947% |

两轮合并的描述性读数（不是把时间窗口或通道当作独立样本）：

- `lambda=0.001`：16 个精确坐标中 11 个改善，平均相对变化 `-0.0095%`，中位数 `-0.0086%`；H=12 为 `-0.0008%`（5/8），H=96 为 `-0.0181%`（6/8）。
- `lambda=0.003`：16 个坐标中 8 个改善，平均 `+0.0042%`；H=12 为 `+0.0174%`，H=96 为 `-0.0090%`。
- AutoCon loss 在正权重候选中保持非零（约 `2.15--2.21`），所以这不是“表示分支未执行”的假象；但是四 epoch 短预算下预测变化非常小。

## 人工复核结论

1. `lambda=0.03` 不应继续作为 MambaTS v3 的默认值；它在本矩阵中表现出随权重增大而加剧的目标干扰。
2. `lambda=0.001` 是当前最合理的探索性候选：AutoCon 梯度实际启用，且两 seed 的平均预测变化接近零。它只能描述为“近似保持预测性能”，不能描述为 AutoCon 带来精度提升或统计显著性。
3. 现有结果仍只有两个 seed、四 epoch、验证集；不能替代 launch-time preregistered selection，也不能支撑高水平论文主表。正式确认前至少需要更多 seed、完整训练预算、测试释放、容量/资源公平性和预先冻结的 lambda 选择规则。
4. 下一轮若继续，应固定 `lambda=0.001` 与同结构 `lambda=0`，并在独立确认计划中预注册非劣效界限；不要再根据测试集或单一数据集选择权重。

## 复跑

```bash
/home/zhong/anaconda3/envs/mamba/bin/python scripts/run_frozen_exploratory_plan.py \
  --plan docs/research/mamba_pems_ts_v3_autocon_lambda_calibration_20260910.json
/home/zhong/anaconda3/envs/mamba/bin/python scripts/analyze_mamba_ts_autocon_lambda_calibration.py \
  --rows-csv artifacts/exploratory/mamba_pems_ts_v3_autocon_lambda_calibration_20260910/readout/validation_rows.csv \
  --output-dir artifacts/exploratory/mamba_pems_ts_v3_autocon_lambda_calibration_20260910/readout
```

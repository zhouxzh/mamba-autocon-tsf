# MambaTS v4 AutoCon ACF Representation-Fusion Screen

**日期：** 2026-09-10  
**状态：** validation-only exploratory；测试集保持关闭。  
**目的：** 检验 train-only raw ACF 是否能直接改善 AutoCon 的时间表示，而不是只作为 forecast residual。

## 结构

`MambaTSv4AutoCon` 保持原有预测 backbone 和 ACF forecast memory 不变，只在 representation 分支加入一个四维 ACF summary（signed seasonal mean、absolute magnitude、RMS、available-lag reliability），再经 `4 -> d_model` projection 注入 AutoCon temporal grid。新 projection 权重正常初始化，但 ReZero 的 `ts_acf_repr_scale` 从零开始；plain forecast 不读取该分支。该开关要求 `mamba_use_acf_memory=true` 和 `AutoCon_acf_source=raw`，以保证先验来源与 forecast branch 一致。

## 合同与工件

- 计划：[`mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910.json`](mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910.json)
- 工件：[`mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910`](../../artifacts/exploratory/mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910/)
- 读出：[`readout.md`](../../artifacts/exploratory/mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910/readout/readout.md)
- 精确配对 CSV：[`mambatsv4autocon_deltas.csv`](../../artifacts/exploratory/mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910/readout/mambatsv4autocon_deltas.csv)

矩阵为 PEMS03/04/07/08、H=12/96、seed=2022/2023，共 16 个作业。候选与原 v4 screen 使用同一数据划分、模型超参、AutoCon `lambda=0.001` 和 train-only ACF prior；16/16 成功，phase validation 通过，测试指标未生成。

## 结果

| 比较 | H=12 | H=96 |
|---|---:|---:|
| 表示融合 v4 vs 原 v4 | 3/8 胜，平均 `+0.075%` | 4/8 胜，平均 `-0.244%` |
| 表示融合 v4 vs TimesNet | 8/8 胜，平均 `-19.450%` | 2/8 胜，平均 `+6.379%` |

`ts_acf_repr_scale` 在 16/16 个 checkpoint 中为非零，平均绝对值为 H=12 的 `0.281518`、H=96 的 `0.280137`；forecast ACF gate 也在 16/16 中非零。这证明表示分支实际参与了 AutoCon 优化，但 H=12 没有改善，H=96 只有轻微、未稳定的 validation 下降，且仍明显落后 TimesNet。

候选平均训练 wall-time 为 `18.77s`，原 v4 为 `18.09s`，短预算下约增加 `3.7%`。这是训练作业 wall-time，不是独立吞吐或端到端效率结论。

## 人工复核结论

1. **AutoCon 机制路径成立。** ACF summary 进入 representation、ReZero gate 离开零点，且预测 head 没有被替换。
2. **精度增益不足。** 两个 horizon 都没有达到稳定的 exact-coordinate improvement；H=96 的相对 TimesNet gap 仍是主要失败模式。
3. **当前不晋级主线。** 该分支应保留为机制消融，不应写成“ACF 提升 AutoCon”或“解决长期预测”。
4. **证据等级仍为 exploratory。** 仅 4 epochs、每坐标两个 seed、validation-only，尚未完成 `lambda=0` 同结构控制的正式确认性配对。

## 后续门槛

- 若要继续，必须预注册同结构 `lambda=0` 与 representation-fusion `lambda=0.001` 的多 seed/full-budget paired test；不能只挑选 H=96 的有利坐标。
- 论文中应同时报告 gate 激活、AutoCon loss、表示范数和 wall-time，避免把非零 gate 当作精度证明。
- 在当前结果下，推荐将该分支作为失败/中性消融归档，不增加更多结构复杂度。


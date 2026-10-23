# MambaTSv3 d24 高维 H12 迁移：失败复核

## 结论

这是一轮 validation-only 的独立迁移验证，不产生也不读取测试指标。冻结了此前在 PEMS H12 上使用的轻量配置：`d_model=24`、`d_ff=48`、`patch_len=stride=24`、单尺度 MambaTSv3，以及同一训练预算和三个新 seed（`2301--2303`）。Electricity 和 Traffic 各包含 DLinear、NLinear、PatchTST、TimesNet、TimeMixer、iTransformer、Mamba2 七类外部基线，以及同容量 MambaTSv3 的 `lambda=0` null 和 `lambda=.001` AutoCon active。

严格 reader 读取到 54/54 个成功作业，source snapshot 一致，测试工件为空。两个数据集均未通过预注册的“active 不高于最强外部基线 2%”门槛，因此停止该配置的高维扩展与测试释放。

| 数据集 | Active MSE | Null MSE | Active 相对 Null | Null 胜出 | 最强外部基线 | Active 相对基线 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Electricity | 0.109479 | 0.109536 | -0.052% | 1/3 | TimeMixer 0.106383 | +2.910% |
| Traffic | 0.341378 | 0.341351 | +0.008% | 0/3 | iTransformer 0.324793 | +5.106% |

## 人工复核判断

- AutoCon 在 Electricity 上只有微小变化，在 Traffic 上略有退化；两个数据集都没有达到“均值改善且至少 2/3 seed 胜出”的独立增益标准。
- 候选相对各数据集的最强外部基线分别落后 2.910% 和 5.106%，所以不能声称 PEMS H12 的 d24 结果可迁移到高维 Electricity/Traffic。
- 这不是测试集失败，而是更早的 validation gate 失败；不应补跑测试来挽救该分支。

完整机器读数见 [`strict readout`](mambatsv3_d24_highdim_h12_transfer_20260927_readout/readout.md)，实验根目录为 `artifacts/exploratory/mambatsv3_d24_highdim_h12_transfer_20260927/`。

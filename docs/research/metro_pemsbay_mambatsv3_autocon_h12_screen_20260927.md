# METR-LA / PEMS-BAY MambaTSv3 AutoCon H12 迁移复核

## 结论

新增两个独立的交通速度基准后，冻结的 d24 `MambaTSv3AutoCon` 在 `L=96,H=12` 的 32-job validation-only screen 通过了预注册的外部精度和 exact-null 非劣门槛。全部 32 个作业成功，三个 phase 的 source snapshot 一致，时间戳首列 CSV 数值合同、5 分钟采样间隔和传感器数量均已验证；没有生成或读取 held-out test 指标。

| 数据集 | Active | Null | Plain MambaTSv3 | Active vs null | wins/null | 最强基线 | Active vs 最强 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| METR-LA | 0.301471 | 0.301482 | 0.302461 | -0.004% | 1/2 | TimeMixer 0.296230 | +1.769% |
| PEMS-BAY | 0.328045 | 0.327987 | 0.329126 | +0.018% | 1/2 | MambaTSv3 0.329126 | -0.329% |

两处 active 都处于最强外部基线的 2% 内，也都处于 exact lambda-zero null 的 1% 内。因此可以注册新的冻结 held-out confirmation；但 active 未出现两个 seed 一致改善，不能声称 AutoCon 对比损失本身带来独立精度增益。

## 资源边界

同源 RTX 5090 D、FP32、device-resident `B=32,L=96,H=12`、50 次预热和 200 次前向计时下，AutoCon candidate 的 prediction-only median latency 为 METR-LA `.244 ms`、PEMS-BAY `.266 ms`。它在两处均快于 PatchTST（`.363/.497 ms`）和 TimeMixer（`.980/1.330 ms`）；METR-LA 慢于 iTransformer（`.221 ms`），因此不能使用全面加速表述。线性 DLinear/NLinear 仍显著更快，是强制保留的速度控制。

这些资源数据排除了数据加载、训练和对比损失，且不是精度重复次数。完整机器读数位于 `artifacts/exploratory/metro_pemsbay_resource_preflight_20260927/`。

## 解释与下一步

这轮提高了 PEMS H12 结果的外部有效性，因为 METR-LA 和 PEMS-BAY 与原 PEMS03/04/07/08 的输入文件和时间跨度独立。但它仍只覆盖短期预测、两个种子和非图模型；不能与使用道路图的 DCRNN/STGCN 类方法直接排名，也不能推进为高水平论文的完整主结论。

下一步是保持候选、数据划分、训练合同、主要比较器和资源协议全部冻结，使用五个新 seed 做一次 held-out confirmation。严格机器读数见 [`readout`](metro_pemsbay_mambatsv3_autocon_h12_screen_20260927_readout/readout.md)，不可变筛选计划见 [`plan`](metro_pemsbay_mambatsv3_autocon_h12_screen_20260927.json)。

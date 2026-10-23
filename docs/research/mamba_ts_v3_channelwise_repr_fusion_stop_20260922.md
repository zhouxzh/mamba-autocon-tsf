# MambaTS v3 通道保留表示融合的效率停止记录（2026-09-22）

该候选的目标是修复全局通道平均在 ETTh1/ETTm1 上的退化。冻结计划 [`mamba_ts_v3_channelwise_repr_fusion_screen_20260922.json`](mamba_ts_v3_channelwise_repr_fusion_screen_20260922.json) 启动后，第一个 Traffic/C862/H96/seed2022 的训练作业在 GPU 100%、约 `8.0 GB` 显存下持续超过 7 分钟，远高于全局融合结构约 50 秒的同预算作业。该进程由本轮主动中止，GPU 随后恢复空闲。

初版的完整时间网格通道表示被判定不满足项目“优于基线速度”的硬约束，未完成矩阵不产生或汇总精度结果。优化成 patch-level 表示后，独立 Traffic/C862/B8、FP32、10 warm-up/30 timed、prediction-plus-representation 资源 smoke 为：

| 结构 | 中位前向 | 峰值增量 | 参数 |
| --- | ---: | ---: | ---: |
| 全局融合 | 0.720 ms | 152.6 MB | 168,508 |
| patch-level 通道保留 | 1.415 ms | 343.7 MB | 111,644 |

参数更少不代表实际更快；通道保留激活和逐通道投影仍成为瓶颈。该候选停止，不应把中止作业、部分 manifest 或资源 smoke 写成预测精度结果。

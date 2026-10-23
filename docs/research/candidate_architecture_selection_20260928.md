# Candidate Architecture Selection (2026-09-28)

本文档用于人工复核当前架构路线，不新增训练结果，也不授权读取任何 held-out 测试集。

## 当前分层

| 层级 | 架构 | 证据 | 当前定位 |
| --- | --- | --- | --- |
| 主结果 | MambaTSv3AutoCon + AutoCon relation auxiliary objective | PEMS、METR-LA/PEMS-BAY、Weather 已冻结 held-out release；Electricity/Traffic H96 八臂、18-job mechanism 和 relation-auxiliary-only confirmation | 论文主模型；高维路线固定为 relation auxiliary-only，prediction-connected residual 不作为该路线的精度来源 |
| 高维速度 challenger | stride-24 relation-auxiliary MambaTSv3 | Electricity/Traffic H96 `42/42` 及 Weather/ETTh1 H96 `48/48` validation-only；完整强基线与 manifest-bound resource matrix | Electricity/Traffic/Weather 对登记基线均 `3/3` 胜出；ETTh1 落后 DLinear/RLinear/PatchTST/iTransformer。固定 shape 上快于 Mamba-1/TimesNet/Mamba-2，且在部分域快于 PatchTST；仍慢于线性和 iTransformer，未覆盖 held-out |
| 高维 challenger | MambaACFAutoCon + rank-4 channel-state mixer | Electricity/Traffic/ETTh1 各 27-job validation-only factorial | 高维局部结构证据；Electricity/Traffic 通过，ETTh1 失败，不能替代主结果 |
| 机制补充 | NLinear variable-axis Mamba + representation fusion | Solar H96 27-job validation-only matrix | 结构/表示融合补充；优于 iTransformer，但 relation-null 略好，不能归因于 relation loss |
| 停止分支 | selector、BiPatch、v12/v13、多尺度、线性 blend 等 | 已有 validation/resource 失败或边界记录 | 不再继续宽搜，也不进入主结论 |
| 图基线边界 | STGCN/GraphWaveNet/AGCRN/DCRNN validation-only matrices | AGCRN METR-LA 基本持平、PEMS-BAY 落后 `2.695%`；DCRNN 在 METR-LA/PEMS-BAY 分别落后候选 `5.287%/27.444%`；固定 shape 前向候选低于两类循环图模型 | 图基线覆盖主要 canonical-style 族，但所有新增图结果仍是 local validation，不能支持 held-out 或普适图模型优势 |

## 推荐论文叙事

1. 主模型使用已经完成 held-out release 的 MambaTSv3AutoCon；主表只放 PEMS、METR-LA/PEMS-BAY 和 Weather。
2. Electricity/Traffic H96 的 relation-auxiliary-only 48 行同源码强基线矩阵可作为验证集补充：候选对 DLinear、NLinear、RLinear、PatchTST、TimesNet、iTransformer、Mamba2 全部三 seed 胜出，较 prior fusion 候选提高 `1.798%/1.379%`；该表必须明确标为 validation-only。
3. Weather/ETTh1 的 36 行跨域复核应与高维矩阵一起报告：Weather 五类基线均三 seed 胜出，ETTh1 对 RLinear/PatchTST/iTransformer 分别落后 `1.50%/3.17%/3.93%`，作为强制失败边界。
4. 将 channel-state mixer 作为高维结构筛选和负边界证据：它在 Electricity/Traffic 有局部收益，但 ETTh1 失败，说明通道状态交互具有数据集依赖性。
5. 将 stride-24 candidate 作为速度导向 challenger：它在 Electricity/Traffic/Weather H96 对照中保持精度并缩短相对指定非线性基线的 forward 路径；ETTh1 失败边界已复现，且没有 held-out 证据，不能替代主结果。
6. 不将 AutoCon 表示门控 persistence blend 接入 stride-24 路径：它虽能避免第二次 Mamba scan，但 ETTh1 resource preflight 相对同形状 control 增加 `23.8%` latency，并慢于 PatchTST，未达到效率目标，故不启动训练筛选。
6. 将 AutoCon relation objective 写成与预测路径兼容的训练约束。最新 Electricity/Traffic 机制矩阵中 active 相对 lambda-zero 有 `0.376%/0.236%` validation 改善，但 active 相对 fusion-off 退化 `1.160%/1.081%`；不能声称 prediction-connected fusion 独立提升精度。
7. 将速度表述限制为固定 RTX 5090 D、固定 batch/窗口/预测长度下，相对指定非线性基线的 prediction-only forward latency；DLinear/NLinear 仍是更快的线性边界。

## 下一步边界

- 不再进行无注册的架构宽搜索或后验调参。
- GraphWaveNet held-out 草案仍需人工批准精确 SHA-256 后才能运行。
- 高维 relation-auxiliary-only 86-job held-out 草案已冻结并通过计划校验，但仍需人工批准精确 SHA-256 后才能运行。
- 草案的严格 reader 已覆盖授权绑定、source identity、候选/基线隔离、streaming metrics 和 1.02 非劣界；批准前不读取任何测试指标。
- 若要把 channel-state mixer 变成主结果，必须另行注册并批准独立 held-out release；当前 validation-only 数字不能写入测试主表。
- 在批准前，继续工作仅限文档、表格、源码快照和审计，不启动测试集训练。

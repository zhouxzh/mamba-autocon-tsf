# NLinearVariableMambaAutoCon Traffic 双向 ACF 相似性排序读出

日期：2026-09-26  
状态：探索性 validation-only；停止该结构搜索；所有测试目录为空

## 设计

该轮固定此前表现最好的 Traffic 双向变量轴 Mamba：LastVal-NLinear 预测基座、单层
Mamba-1 变量 token 扫描、预测连接的 AutoCon 表示融合，以及训练期 ACF relation objective
（`lambda=1e-4`）。唯一新增因素是 `mamba_acf_channel_order=similarity`：在训练 split 上的
signed ACF profile 计算一次确定性变量排序，预测和表示路径都在该排序下扫描、随后恢复原变量次序。
验证期间不根据窗口计算 ACF，也不访问验证或测试标签。

Traffic H336/H720、seed 2022/2023、8 epoch、batch 8。iTransformer 在完全相同的源码与
数据快照下重新训练；全体作业均 `skip_test=true`。

## 结果

| Horizon | 双向原候选均值 | 双向 + ACF 相似性排序 | 相对双向原候选 | 同源 iTransformer | 相对 iTransformer |
|---:|---:|---:|---:|---:|---:|
| 336 | 0.373792 | 0.373396 | -0.11% | 0.368431 | +1.35% |
| 720 | 0.399869 | 0.399875 | +0.00% | 0.397886 | +0.50% |

H336 两个 seed 的候选验证损失为 `0.374415`、`0.372377`；H720 为 `0.399413`、`0.400336`。
因此排序在 H336 有很小的方向性改善，但 H720 没有改善，且候选在四个 seed-horizon 坐标仍均高于
iTransformer。它没有通过“跨 horizon 非劣于强基线”的停止门槛。

## 结论与审计边界

- 停止 ACF similarity/strength 通道排序及其与双向变量扫描的组合搜索；不进入 Solar/Weather 复验或测试释放。
- 这不是 AutoCon relation loss 的因果消融：它改变了变量扫描的顺序，而 relation loss 保持开启；不能据此声称 ACF contrastive learning 提升精度。
- 双向扫描只作用于已观测历史窗口，仍不意味着全模型逐位置严格因果；LastVal 归一化与双向变量交互均不应写成严格左因果表示。
- 本轮仅验证集架构筛选，2 个 seed、单一数据集、8 epoch；不能用于显著性、泛化或论文主表结论。

计划及授权分别为
[`screen plan`](nlinear_variable_mamba_bidir_similarity_traffic_multihorizon_screen_20260926.json) 和
[`launch sidecar`](nlinear_variable_mamba_bidir_similarity_traffic_multihorizon_screen_20260926_launch_sidecar.json)。
两个 phase 的 source-tree SHA-256 均为
`0dc0ce70edf69c1e2c16a9b0f85fb685b0cfad90c4eaa797281dfa94e54c7d35`；计划 SHA-256 为
`5b0d03f817518a22964e9189593baba675ea5a9566151e7d944a96a1f997ee1c`；Traffic 数据 SHA-256 为
`cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16`。原始工件位于
[`bidir similarity screen`](../../artifacts/exploratory/nlinear_variable_mamba_bidir_similarity_traffic_multihorizon_screen_20260926/)。

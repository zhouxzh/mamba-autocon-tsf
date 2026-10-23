# NLinearVariableMambaAutoCon Traffic 长预测读出

日期：2026-09-25  
状态：探索性 validation-only；所有测试目录为空

## 冻结 Solar 候选的跨数据集复验

Traffic 是 `17,544 x 862` 的小时级多变量数据。固定 Solar 候选的全部架构与训练参数，仅改数据集和 batch size（Traffic 使用 8），在 H336/H720、seed 2022/2023 上重新训练 DLinear、iTransformer、PatchTST 和候选。

| Horizon | 候选均值 | iTransformer 均值 | PatchTST 均值 | DLinear 均值 | 候选相对 iTransformer |
|---:|---:|---:|---:|---:|---:|
| 336 | 0.376081 | 0.368431 | 0.372239 | 0.460595 | +2.08% |
| 720 | 0.402033 | 0.397886 | 0.399159 | 0.493251 | +1.04% |

候选 4/4 个 seed-horizon 坐标均落后 iTransformer，H336/H720 也都落后 PatchTST。Solar 的长预测优势因此没有迁移到 Traffic；该结构不能作为跨数据集论文主候选，也不开放测试集。

两个 phase 的 source-tree 哈希相同：`8a78949d9a14049552d74a1302179e82f2caecc065790e4814ba69969c9a021a`；计划哈希：`7bbf1f8f3f22571f8a88b8c0ac7c1fb910b12742718e592a1073b0d4a6dafef1`；16/16 作业完成、测试指标工件为零。

## 低秩跨变量 mixer 负消融

为检验 Traffic 的缺口是否来自单次顺序 Mamba 的跨变量交互不足，加入了一个 ReZero 低秩 `C -> r -> C` 残差（`r=8`）。它不创建 `C x C` 参数，预测和 AutoCon 表示仍使用同一 Mamba token。为保持同源比较，重新运行 iTransformer 和低秩候选，共 8 个 validation-only 作业。

| Horizon | 原候选均值 | 低秩 mixer 均值 | mixer 相对原候选 | iTransformer 均值 | mixer 相对 iTransformer |
|---:|---:|---:|---:|---:|---:|
| 336 | 0.376081 | 0.388541 | +3.31% | 0.368431 | +5.46% |
| 720 | 0.402033 | 0.409483 | +1.85% | 0.397886 | +2.91% |

低秩 mixer 在 4/4 坐标变差，按停止规则不再搜索 rank、初始化或额外层数。该模块保留为可复现负消融，不进入后续 test release。

## 资源测量

同一 RTX 5090 D、FP32、device-resident、B8、30 warm-up/100 timed 的 prediction-only 测量中，低秩 mixer 为 H336 `0.38200 ms`、H720 `0.45706 ms` median；同源 iTransformer 为 `0.96408/0.97909 ms`。低秩候选仍快于 iTransformer，但精度未达到非劣，不能用效率掩盖预测失败。资源工件见 [`model_resources.json`](../../artifacts/exploratory/nlinear_variable_mamba_lowrank_mixer_traffic_screen_20260925/resource_traffic_h336_h720/model_resources.json)。

本测量不包含数据加载、训练反向、ACF 拟合或 relation loss；它不是端到端吞吐和独立统计样本。


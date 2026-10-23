# NLinearVariableMambaAutoCon 多预测步长确认 screen 读出

日期：2026-09-25  
状态：探索性 validation-only；未开放测试集

## 目的

在冻结的 Solar 多变量坐标上，比较 prediction-connected `NLinearVariableMambaAutoCon` 与 DLinear、iTransformer、PatchTST。候选仍使用 AutoCon 表示和 train-only ACF relation objective：`AutoCon_lambda=1e-4`、`mamba_acf_relation_loss=true`、`scope=global`、`min_lag=96`、`target_mode=hard_selected`、`mamba_variable_repr_fusion=true`。本 screen 只回答“冻结后的候选是否在新 horizon/seed 上保留验证集信号”，不能回答 AutoCon 的因果贡献。

## 协议

- 数据：Solar AL，`52,560 x 137`，TXT loader，10 分钟间隔，`features=M`。
- 输入/标签：`seq_len=96`、`label_len=48`、预测长度 `H=336/720`。
- 训练：8 epoch，early stopping patience 3，batch 32，Adam，学习率 `0.001`，`weight_decay=0`，seed `2022/2023`。
- 所有 16 个作业均 `skip_test=true`；本文件的数值是最佳 validation MSE，不是测试集结果。
- 两个 phase 的 `source_tree_sha256` 均为 `fd7371cead7db17f132f2b5f07833432644e94e1fdd6ea0d99878e42609e2bf2`，计划哈希为 `4fa9c7883f3cf3c902e7489239c17a2f9030fe2594a9d31abb284dbee64dafaa`。

## 验证集结果

| Horizon | 候选 seed 2022 | 候选 seed 2023 | 候选均值 | iTransformer 均值 | 候选相对 iTransformer |
|---:|---:|---:|---:|---:|---:|
| 336 | 0.154008 | 0.154210 | 0.154109 | 0.179891 | -14.33% |
| 720 | 0.154132 | 0.154025 | 0.154078 | 0.180783 | -14.77% |

候选在 4/4 个 seed-horizon 单元优于 iTransformer。PatchTST 均值为 H336 `0.195373`、H720 `0.192491`；DLinear 均值为 H336 `0.229200`、H720 `0.225856`。因此在这个冻结 Solar 坐标上，候选保留了明显的 validation 精度优势，但该优势不能外推到尚未复核的数据集，也不能归因于 relation loss。

## GPU 预测前向资源

资源工件：[`model_resources.json`](../../artifacts/exploratory/nlinear_variable_mamba_repr_fusion_solar_multihorizon_confirmation_20260925/solar_h336_h720_prediction_resource_benchmark_20260925/model_resources.json)。协议为 RTX 5090 D、FP32、CUDA device-resident synthetic input、batch 32、30 warm-up/100 timed；不包含数据加载、反向、ACF 计算或 relation loss。

| Horizon | 候选 median / p95 (ms) | DLinear median (ms) | iTransformer median (ms) | PatchTST median (ms) |
|---:|---:|---:|---:|---:|
| 336 | 0.29779 / 0.31145 | 0.09218 | 0.31426 | 1.74606 |
| 720 | 0.34960 / 0.36043 | 0.11691 | 0.34021 | 1.85718 |

相对 iTransformer，候选 H336 快 `5.24%`，H720 慢 `2.76%`；相对 PatchTST 分别快 `82.94%`、`81.18%`；两种 horizon 都慢于 DLinear。因此只能作“在 H336 上略快、H720 上近似持平、显著快于 PatchTST”的表述，不能声称超过所有基线或普遍更快。

候选参数量为 H336 `108,643`、H720 `170,851`。iTransformer 对应为 `95,248/120,208`，DLinear 为 `65,184/139,680`。资源结果是单机预测路径测量，不等价于端到端训练吞吐。

## 结论与限制

1. 候选在新 seed 和长 horizon 上的 validation 精度信号是稳定的：4/4 单元击败 iTransformer，均值改善约 14%--15%。
2. 速度目标只部分满足：候选显著快于 PatchTST，但不快于 DLinear；对 iTransformer 只在 H336 有小幅优势。
3. 本 screen 没有 active/null 配对，因此不能证明 AutoCon contrastive/relation objective 带来精度增益。已有 H96 三 seed 配对 screen 显示 active 与 null 的差异接近零且不稳定，论文中应把 AutoCon 写成共享表示训练机制，并报告 null 消融。
4. 仍缺少冻结选择后的独立测试集 replay、多数据集复核和完整统计显著性分析；当前读数不能单独支撑“已达到高水平论文发表要求”的结论。


# Mamba variable-token representation fusion pilot

日期：2026-09-07  
状态：两数据集、单 seed、validation-only 工程 pilot；不是确认性结果，也不进入论文主表。

## 目的和单因素合同

该 pilot 检查 `mamba_variable_repr_fusion` 是否能让 AutoCon relation objective
直接监督已有的 variable-token Mamba 状态。控制组和处理组都启用
`mamba_variable_mixer=true`、`mamba_variable_group_size=1`；处理组唯一额外开关是
`mamba_variable_repr_fusion=true`。两组使用相同 seed、数据切分、ACF relation sampler、
`hard_selected + mamba_acf_relation_lag_mask=true`、`AutoCon_lambda=0.03`、
`AutoCon_wnorm=MeanStd`、`AutoCon_acf_source=residual`、4 epoch、`skip_test=true`。
本次命令的共同结构为 `mamba_d_state=64`、`mamba_scales=[1,2]`、
`mamba_causal_embedding=true`、`mamba_use_acf_memory=false`；没有读取 test split。

## Validation 结果

| 数据集 / H96 | fusion off | fusion on | on - off | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.722297 | 0.723860 | +0.001563 | +0.216% |
| Traffic | 0.380184 | 0.380549 | +0.000365 | +0.096% |

两组的 relation telemetry 完全一致，因为 sampler 和 ACF prior 没有改变。ETTh1 的
global contrastive-active anchor fraction 为 `86.6--87.1%`、degenerate-positive
fraction 为 `5.0--5.5%`；Traffic 分别为 `83.8--84.6%` 和 `7.9--9.1%`。因此处理组的
结果不能归因于 relation pair 暴露变化。两个数据集的 best validation loss 都略有退化，
没有支持继续扩展该融合分支的信号。

## 参数、训练和推理资源

处理组新增的 fusion adapter 为 `2,146` 个可训练参数：ETTh1 从 `78,607` 增至
`80,753`（`+2.73%`），Traffic 从 `188,902` 增至 `191,048`（`+1.14%`）。去掉首次
CUDA 初始化后，4 epoch 记录的稳定训练/验证时间上升约 `2.78%`（ETTh1）和
`2.01%`（Traffic）；首次 epoch 的差异受 kernel 初始化影响，不能用于速度结论。

prediction-only 的 GPU 测量使用 RTX 5090 D、`B=8`、10 warm-up、50 timed forwards、
FP32，输入驻留 device，不含数据加载、反向和优化器：

| 数据集 | off median / p95 (ms) | on median / p95 (ms) | 吞吐变化 | 峰值增量变化 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 1.695 / 1.783 | 1.768 / 1.777 | -2.88% | 0% |
| Traffic | 1.881 / 1.963 | 1.830 / 1.874 | +3.11% | 0% |

延迟方向在两个数据集上相反，属于单次短测量的噪声范围；不能写成加速或减速。
处理组在 prediction-only 路径不执行 representation fusion，但仍保留 adapter 参数。
带 `--include-representation` 的 Traffic 复核中，median 为 `1.802/1.805 ms`
（off/on），同样没有可重复的速度优势。

## 结论和停止规则

- 该 adapter 确实建立了 relation gradient 到 variable-token 分支的显式路径，但短训
  validation 没有改善，两个数据集均为轻微退化。
- 参数、稳定训练时间和 representation 路径都有额外开销；prediction-only 资源没有
  一致方向的收益。
- 因此不启动原 12-job variable-representation screen，不把该分支叠加到 patch、GDD
  或其它未确认开关，也不做“AutoCon 表示融合提高精度/速度”的机制或论文表述。

原始工件：[`artifacts/exploratory/mamba_variable_repr_fusion_pilot_20260907`](../../artifacts/exploratory/mamba_variable_repr_fusion_pilot_20260907)。
设计草案（仍未授权）见 [`mamba_variable_repr_fusion_screen_plan_20260903.json`](mamba_variable_repr_fusion_screen_plan_20260903.json)。

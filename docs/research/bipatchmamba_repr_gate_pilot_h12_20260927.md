# BiPatchMamba 有界表示门控 H12 复核

## 结论

在此前加性 representation correction 的四数据集失败后，本轮固定 BiPatchMamba 主干、数据划分、窗口、训练预算和 AutoCon 超参，只将 AutoCon 表示进入预测的方式替换为零初始化、有界的乘性门控。24/24 个 validation-only 作业成功，三臂 source snapshot 一致，且没有生成或读取 held-out test 指标。

| 数据集 | Active gate | Null gate | Plain | Active vs null | wins/null | Active vs plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.313071 | 0.312793 | 0.313147 | +0.089% | 1/2 | -0.024% |
| Electricity | 0.110758 | 0.111020 | 0.108040 | -0.236% | 2/2 | +2.516% |
| Traffic | 0.327827 | 0.328119 | 0.323963 | -0.089% | 1/2 | +1.193% |
| Weather | 0.223865 | 0.224445 | 0.219406 | -0.258% | 2/2 | +2.032% |

门控让 active 相对 exact null 在四个数据集都满足预注册的 1% 非劣界，其中 Electricity 和 Weather 为两个 seed 都改善。但它没有稳定超过 fresh plain BiPatch：ETTh1 基本持平，另外三个数据集仍落后 `1.19%--2.52%`。因此不能释放测试集，不能称 AutoCon representation gate 已带来可靠的预测增益，也不应继续围绕这一门控形式做事后调参。

## 与加性路径的关系

先前加性 active 相对 plain 的退化为 ETTh1 `+1.377%`、Electricity `+13.467%`、Traffic `+5.904%`、Weather `+12.924%`。门控 pilot 的 fresh seeds 与 source snapshot 不同，不能做正式配对差异检验；但作为方向性诊断，它说明“将对比表示直接以无界加性残差叠加到预测”是主要问题，而有界调制可显著降低该伤害。

该诊断不足以构成论文贡献。下一步应停止这条直接表示修正分支，把资源转回已经有 held-out 证据的 PEMS H12 主线，或设计一个预注册且能相对 plain 带来跨数据集改善的新机制。

严格机器读数见 [`readout`](bipatchmamba_repr_gate_pilot_h12_20260927_readout/readout.md)，不可变训练合同见 [`plan`](bipatchmamba_repr_gate_pilot_h12_20260927.json)。

# Native baseline pilot 跨数据集汇总

日期：2026-09-07  
状态：ETTh1、ETTh2、ETTm1、ETTm2、Traffic 五数据集，单 seed、H96、4 epoch、
validation-only；不是确认性论文结果。

## 共同合同

每个数据集运行 DLinear、NLinear、PatchTST、TimesNet、iTransformer、Mamba-1 和
Mamba-2。训练/验证切分、batch size=32、学习率、优化器、4 epoch 和 `skip_test=true`
保持一致；没有读取 test split。Mamba-1/2 使用当前共享 Mamba head、`d_state=16`、
`d_conv=4`、`expand=2`、`headdim=16`、单尺度和 `mamba_causal_embedding=true`。

这是 native baseline，不是 normalization-parity 实验：普通基线和 `AutoCon=false`
的 Mamba 不会走 MambaACF 的 `MeanStd` 窗口归一化。表中的 validation MSE 只在同一
数据集内作方向性比较，不能直接和 AutoCon/MeanStd 候选或历史 test 表合并。

## 输入文件指纹

| 数据集 | 相对路径 | SHA-256 |
| --- | --- | --- |
| ETTh1 | `dataset/ETT-small/ETTh1.csv` | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |
| ETTh2 | `dataset/ETT-small/ETTh2.csv` | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` |
| ETTm1 | `dataset/ETT-small/ETTm1.csv` | `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e` |
| ETTm2 | `dataset/ETT-small/ETTm2.csv` | `db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd` |
| Traffic | `dataset/traffic/traffic.csv` | `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16` |

## 最佳 validation MSE

| 数据集 | DLinear | NLinear | PatchTST | TimesNet | iTransformer | Mamba-1 | Mamba-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.653467 | 0.700956 | 0.690674 | 0.765748 | 0.693611 | 0.689483 | 0.707538 |
| ETTh2 | 0.222539 | 0.208738 | 0.209764 | 0.237469 | 0.222801 | 0.249931 | 0.242271 |
| ETTm1 | 0.397810 | 0.410208 | 0.408682 | 0.430037 | 0.395855 | 0.402181 | 0.406243 |
| ETTm2 | 0.131522 | 0.126379 | 0.126694 | 0.130115 | 0.130172 | 0.140575 | 0.133088 |
| Traffic | 0.513800 | 0.508901 | 0.438770 | 0.431119 | 0.442530 | 0.497026 | 0.494128 |

数据集之间的绝对值不应横向排名。ETTh1 的短预算最低值是 DLinear，ETTh2 和 ETTm2
是 NLinear，ETTm1 是 iTransformer，Traffic 是 TimesNet；Mamba-2 仅在 Traffic 比
Mamba-1 略低，在其余 ETT 数据集并未成为最低值。Traffic 上 Mamba-2 比 PatchTST 高
`12.6%`，比 TimesNet 高 `14.6%`；这仍是单 seed/4 epoch 观察。

## 稳定训练 wall time（4 epoch）

| 数据集 | DLinear | NLinear | PatchTST | TimesNet | iTransformer | Mamba-1 | Mamba-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.710 | 0.591 | 1.875 | 9.885 | 1.553 | 2.410 | 13.331 |
| ETTh2 | 0.663 | 0.816 | 1.851 | 9.885 | 1.195 | 1.947 | 13.543 |
| ETTm1 | 2.634 | 2.919 | 5.286 | 38.521 | 5.960 | 9.039 | 25.657 |
| ETTm2 | 3.479 | 3.013 | 7.076 | 51.910 | 6.263 | 7.100 | 25.806 |
| Traffic | 8.577 | 8.939 | 22.443 | 20.456 | 11.616 | 10.338 | 25.533 |

这里的 epoch time 受首次 kernel 初始化、数据加载和短训练预算影响；它不是部署
吞吐，也不是严格的优化器计算基准。Mamba-2 的首 epoch 初始化尤其明显，不能只看
单个 epoch 推断长期训练速度。

## Prediction-only GPU 资源

协议：RTX 5090 D、FP32、device-resident synthetic input、`B=8`、10 warm-up、
50 timed forwards；不包含数据加载、传输、反向、优化器或 relation loss。

| 数据集 | 模型 | 参数（可训练） | median ms | p95 ms | sequences/s | 峰值增量 MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | DLinear / NLinear | 18,624 / 9,312 | 0.067392 / 0.031728 | 0.087202 / 0.040725 | 116,827.9 / 245,497.6 | 0.103 / 0.062 |
| ETTh1 | PatchTST / iTransformer | 46,080 / 14,944 | 0.272352 / 0.159600 | 0.289677 / 0.171373 | 29,084.3 / 49,082.7 | 1.514 / 1.062 |
| ETTh1 | Mamba-1 / Mamba-2 | 31,975 / 29,907 | 0.387888 / 0.666976 | 0.405954 / 0.692976 | 20,466.2 / 11,938.2 | 1.812 / 3.303 |
| ETTh2 | DLinear / NLinear | 18,624 / 9,312 | 0.067936 / 0.032208 | 0.076923 / 0.037984 | 114,607.4 / 242,487.7 | 0.103 / 0.062 |
| ETTh2 | PatchTST / iTransformer | 46,080 / 14,944 | 0.274448 / 0.156784 | 0.298914 / 0.167350 | 28,847.4 / 49,841.1 | 1.514 / 1.062 |
| ETTh2 | Mamba-1 / Mamba-2 | 31,975 / 29,907 | 0.382368 / 0.675600 | 0.409862 / 0.702163 | 20,693.0 / 11,780.8 | 1.812 / 3.303 |
| ETTm1 | DLinear / NLinear | 18,624 / 9,312 | 0.068704 / 0.031136 | 0.075181 / 0.035128 | 114,505.6 / 247,603.2 | 0.103 / 0.062 |
| ETTm1 | PatchTST / iTransformer | 46,080 / 14,944 | 0.275248 / 0.160208 | 0.289987 / 0.175443 | 28,714.3 / 49,052.7 | 1.514 / 1.062 |
| ETTm1 | Mamba-1 / Mamba-2 | 32,007 / 29,939 | 0.385408 / 0.661248 | 0.409933 / 0.682218 | 20,609.6 / 12,063.1 | 1.812 / 3.303 |
| ETTm2 | DLinear / NLinear | 18,624 / 9,312 | 0.067840 / 0.031200 | 0.071710 / 0.037326 | 115,814.7 / 248,429.9 | 0.103 / 0.062 |
| ETTm2 | PatchTST / iTransformer | 46,080 / 14,944 | 0.277632 / 0.157168 | 0.293694 / 0.170690 | 28,523.9 / 49,840.3 | 1.514 / 1.062 |
| ETTm2 | Mamba-1 / Mamba-2 | 32,007 / 29,939 | 0.388112 / 0.672672 | 0.412499 / 0.699163 | 20,472.5 / 11,831.0 | 1.812 / 3.303 |
| Traffic | DLinear / NLinear | 18,624 / 9,312 | 0.078736 / 0.033632 | 0.087202 / 0.040725 | 100,032.8 / 229,690.7 | 13.258 / 7.576 |
| Traffic | PatchTST / iTransformer | 46,080 / 14,944 | 0.888256 / 0.342352 | 0.898965 / 0.352885 | 8,990.6 / 23,170.9 | 95.188 / 8.629 |
| Traffic | Mamba-1 / Mamba-2 | 169,630 / 167,562 | 0.427360 / 0.716608 | 0.440315 / 0.738003 | 18,677.0 / 11,116.7 | 23.104 / 23.104 |
| ETTh1 | TimesNet | 1,191,751 | 2.693472 | 2.706122 | 2,969.2 | 6.866 |
| ETTm1 | TimesNet | 1,191,783 | 2.692208 | 2.703104 | 2,970.3 | 7.053 |
| ETTh2 | TimesNet | 1,191,751 | 2.690544 | 2.708654 | 2,970.4 | 6.866 |
| ETTm2 | TimesNet | 1,191,783 | 2.690336 | 2.702794 | 2,972.5 | 7.053 |
| Traffic | TimesNet | 1,302,046 | 2.796272 | 2.813582 | 2,859.3 | 17.918 |

完整精确值同时保存在每个数据集 artifact 的 `resources/<model>/model_resources.json`
和 `.csv` 中。不要把 synthetic latency
直接解释为端到端训练或生产服务延迟。

## 当前结论和下一步

1. 代码已经具备五数据集、七模型的最小 baseline 运行覆盖；Mamba-1/2 的入口、配置和
   resource tool 均通过。
2. 现有 native pilot 没有支持“纯 Mamba 同时不掉精度且普遍更快”的结论。Traffic 上
   Mamba-1/2 比 PatchTST/TimesNet 更快，但验证误差更高；iTransformer 同时更快且更准。
3. 要形成论文主表，必须先注册并执行 normalization-parity（至少 `MeanStd`）、三到五
   个 seed、完整训练预算和固定 test-free selection；当前扩展计划仍未授权。

原始目录：

- [`baseline_etth1_pilot_20260907`](../../artifacts/exploratory/baseline_etth1_pilot_20260907)
- [`baseline_etth2_pilot_20260907`](../../artifacts/exploratory/baseline_etth2_pilot_20260907)
- [`baseline_ettm1_pilot_20260907`](../../artifacts/exploratory/baseline_ettm1_pilot_20260907)
- [`baseline_ettm2_pilot_20260907`](../../artifacts/exploratory/baseline_ettm2_pilot_20260907)
- [`baseline_traffic_pilot_20260907`](../../artifacts/exploratory/baseline_traffic_pilot_20260907)

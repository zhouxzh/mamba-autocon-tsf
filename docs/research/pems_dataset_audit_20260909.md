# PEMS 数据集审计（2026-09-09）

这份审计记录新增的四个交通流数组数据集是否真的能通过仓库的
`Dataset_Array`，不把“文件已下载”误写成“实验已完成”。原始文件来自
Hugging Face 数据集镜像 [`pkr7098/time-series-forecasting-datasets`](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets)，每个文件均以 SHA-256 固定。

## 当前数据合同

- 文件格式：NPZ，键 `data`；loader 支持 `[T,C]` 和 `[T,C,F]`。
- `freq=t`、采样间隔 5 分钟；由于原始数组没有时间列，loader 生成规则日历 marks。
- 固定 `data_feature_index=0`。PEMS03/07 的 feature 轴长度为 1，PEMS04/08 的长度为 3；本阶段不把 feature 轴当作额外变量。
- 实验只使用 dense `[time, channel]` 数值矩阵，不读取或推断图邻接矩阵；因此结果只能解释为无空间图结构的多变量预测基线。
- 审计窗口为 `[seq_len, label_len, pred_len]=[96,48,12]`，`features=M`，train ratio 与正式 loader 相同。

| 数据集 | 原始 shape | 选定后 shape | SHA-256 | train/val/test 窗口数（H12） |
| --- | ---: | ---: | --- | ---: |
| PEMS03 | `[26208,358,1]` | `[26208,358]` | `34859517425eded66a0014db63de54b95d0428c38890741a1710ed4d3b043de8` | `15617 / 5232 / 5230` |
| PEMS04 | `[16992,307,3]` | `[16992,307]` | `95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a` | `10088 / 3388 / 3387` |
| PEMS07 | `[28224,883,1]` | `[28224,883]` | `a07887e610aa986276209bc9384d14ed7858fdcad1ded67f53dca19eb828600b` | `16827 / 5635 / 5633` |
| PEMS08 | `[17856,170,3]` | `[17856,170]` | `e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075` | `10606 / 3561 / 3560` |

四个文件的原始值和 loader 输出均为有限浮点数；每个 split 的首个样本都满足
`x=[96,C]`、`y=[60,C]`、marks 分别为 `[96,5]` 与 `[60,5]`。审计 JSON 位于
[`artifacts/exploratory/pems_dataset_audit_20260909/array_audit.json`](../../artifacts/exploratory/pems_dataset_audit_20260909/array_audit.json)，可用以下命令重跑：

```bash
/home/zhong/anaconda3/envs/mamba/bin/python scripts/audit_array_datasets.py \
  --output artifacts/exploratory/pems_dataset_audit_20260909/array_audit.json
```

## 解释边界

这一步只证明数据 schema、hash、有限值和窗口切分可复现；它没有产生任何模型精度、速度或统计显著性证据。PEMS 是高维交通流数据，尤其 PEMS07 的 `C=883` 会放大共享低维 hidden state 的容量瓶颈。正式比较应至少配对 DLinear/NLinear、PatchTST、TimesNet、Mamba-1、Mamba-2、MambaTS 及其 AutoCon 变体，并固定 horizon、seed、归一化、预算和 `data_feature_index`。

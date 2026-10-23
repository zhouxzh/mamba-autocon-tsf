# 论文级实验边界与下一阶段清单

## 当前判断（2026-09-28 更新）

本文档的早期“缺少论文主结果”判断已经过时。当前可引用的主证据已集中在
[`paper_evidence_matrix_20260928.md`](paper_evidence_matrix_20260928.md)：PEMS03/04/07/08 H12、
METR-LA/PEMS-BAY H12 与 Weather H96/H336/H720 均已完成冻结 held-out 评估。它们支持在**命名的
注册比较器与固定硬件 shape**下的非劣/局部速度结论，但不支持“所有时间序列数据集都更准确、更快”的表述。

SolarEnergy 仍是关键的高维验证域而不是主测试域：`NLinearVariableMambaAutoCon` 在 Solar H336/H720
对 iTransformer 有强 validation 信号，但其精确 AutoCon active/null 对照未显示稳定增益，已经按停止规则
关闭。它不能替代 MambaTSv3AutoCon 的 held-out 主表，也不能用来宣称对比学习独立提高精度。

## 已完成与仍需补齐的比较

已冻结主表使用各自独立 source snapshot、固定配置和 fresh paired seeds；对应 comparator、split 与
统计边界均由证据矩阵中的 release readout 管理。它不应与 validation-only 的旧矩阵拼接成一张表。

若要继续扩展主结果，任何新增坐标仍应使用同一 source snapshot、同一归一化、同一训练预算和同一 seed
集合。最低可接受的新增矩阵为：

| 维度 | 固定范围 |
| --- | --- |
| 数据集 | ETTh1、ETTm1、Electricity、Traffic |
| 特征 | 全部使用 M；S 结果单独报告 |
| 预测长度 | 96、336、720 |
| seed | 5 个预先固定 seed |
| plain 基线 | DLinear、NLinear、PatchTST、TimesNet、Mamba-1、Mamba-2 |
| 表示控制 | AutoConNet、同 backbone 的 AutoCon null |
| ACF 候选 | MambaACF plain、MambaACF + AutoCon；只保留验证阶段预先选出的一个结构 |

新增主表必须包含当前强线性、patch、Transformer/Mamba 比较器，并保留同源 plain-v3 或 exact-null
控制来界定 AutoCon 表示的作用。ETTh1/ETTh2/ETTm2、Exchange 与 ILI 是失败边界，不能被只包含成功
数据集的结果替代。

## 不应继续投入的方向

- 不继续扩大 v6 residual 初始化网格；warm-init screen 已显示 Exchange 只有微小改善，ILI 反而退化。
- 不基于当前证据增加 Exchange 通道选择器；三 seed 的八通道平均误差没有稳定优势通道。
- 不把 validation-only 的资源 smoke 写成端到端训练速度；效率比较必须统一报告 forward、每 epoch wall time、峰值显存和参数量。
- 不在测试集上挑选 lambda、结构或 seed；测试释放只能在验证选择规则冻结后进行。

## 测试释放门槛

只有同时满足以下条件，才值得开放最终测试集：

1. 六个 plain baseline 和 proposed rows 在同一新 provenance 下完整运行；
2. NLinear 不再缺失，且所有 dataset-horizon 使用相同 seed 配对；
3. proposed row 在验证集上相对同 backbone null 有预先登记的改进门槛，而不是只相对某个最弱基线；
4. 资源合同包含同一 GPU、软件版本、batch、warm-up 和计时次数；
5. 测试集只运行一次，报告 seed 均值、标准差和逐 seed 结果，不再因测试结果追加结构。

若预算不足，应先减少数据集或 horizon，再保持基线集合和五 seed 不变。当前阶段更重要的是形成一张可审计的公平主表，而不是继续增加 Mamba 变体数量。

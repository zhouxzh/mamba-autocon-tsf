# 基线与数据集覆盖审计

日期：2026-09-07  
工具：[`scripts/audit_benchmark_matrix.py`](../../scripts/audit_benchmark_matrix.py)  
性质：只读库存审计，不合并不同训练协议，也不选择模型。

## 当前可用数据集

仓库中有 9 个已存在且通过表头/通道检查的数据集：ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Traffic、Weather、Exchange-Rate 和 National Illness。Traffic 为 862 通道，Electricity 为 321 通道，ETT 数据为 7 通道；这些高维/低维数据共同覆盖了跨变量压缩与周期预测两类压力。

PEMS03/04/07/08 和 Solar 只在注册适配器中出现，当前工作树没有对应原始文件，因此不能列入已完成数据集结果。要加入它们，必须先固定下载来源、文件哈希、时间频率、目标通道和 split 合同，再另建计划，不能把缺失文件当作空结果。

另外，Weather 与 National Illness 虽有本地文件，但暂不进入下一轮 native pilot：Weather
的主时间间隔实际为 10 分钟，而当前注册频率 `t` 表示 15 分钟；National Illness 的间隔
实际为 7 天，而当前注册频率为 `d`。这两个频率合同需要先修正并重新审计，之后才能解释
时间特征、lag 和 forecast horizon。

## 模型注册状态

当前审计可定位以下模型源文件：

- 参考基线：DLinear、NLinear、PatchTST、TimesNet、iTransformer，以及新增已注册的 Autoformer、FEDformer、ETSformer、Transformer、Crossformer；
- 普通 Mamba：Mamba1、Mamba2 及其 AutoCon 变体；
- ACF 分支：MambaACF、MambaACF1、MambaACFAutoCon、MambaACFVariable 及其 AutoCon 变体；
- TimeMachine 风格：MambaTM1、MambaTM2 及其 AutoCon 变体。

本次修复了 coverage audit 对 `MambaTM1/2` 源文件的漏映射；模型实现本身此前已存在，修复后审计能正确报告 `models/MambaTM1AutoCon.py` 和 `models/MambaTM2AutoCon.py`。

## 结果覆盖和缺口

审计扫描到 53 个计划、144 个 phase、3,009 条可用 metric artifact；其中 238 条 artifact 位于已登记 phase 根目录之外，只能按历史/探索性工件处理。新增的三个计划是 2026-09-07 的 MeanStd 核心基线 pilot、Weather/Exchange-Rate 经典模型扩展 pilot 和四核心数据集经典模型扩展 pilot；这些 validation-only 结果不进入正式论文主表。

本轮另新增了未注册到上述确认性矩阵的五数据集 native baseline pilot：ETTh1、ETTh2、
ETTm1、ETTm2 和 Traffic，各含七个模型的 H96、单 seed、4 epoch validation-only 运行
及独立 GPU resource 记录。它们用于入口覆盖和资源预算，不能减少下面注册计划的 `0/N`
缺口，也不能与 MeanStd/AutoCon 或历史 test 表直接拼接。

此外，`mamba_normalization_parity_pilot_20260907`（20 jobs）、
`mamba_variable_parity_pilot_20260907`（4 jobs）以及其 seed-2023 replication（4 jobs）
是启动前固定条件的独立 exploratory pilot。它们不属于已登记的确认性矩阵，没有 test 指标，
不改变下表中注册计划的缺口；结果只在各自的 paired validation 合同内解释。

新增的 `mamba_meanstd_baseline_pilot_20260907` 是另一份独立的 84-job pilot：四个数据集、七个模型、H96、三 seed、四 epoch，`MeanStd` 和 `skip_test=true`；84/84 作业及 28 个 prediction-only 资源记录均已完成，phase validation 通过。它修复了 DLinear/NLinear 未实现 MeanStd 时的公平比较缺口，但仍不减少下面正式 P2 计划的 `0/216` 缺口，也不能与 native normalization、AutoCon 或历史 test 表混合。

新增的 `mamba_classical_baseline_extension_pilot_20260907` 是独立的 72-job pilot：Weather/Exchange-Rate、12 个模型、H96、三 seed、四 epoch，`MeanStd` 和 `skip_test=true`；72/72 作业、phase validation 和 24 条 prediction-only 资源记录均已完成。它补齐了仓库内 Autoformer/FEDformer/ETSformer/Transformer/Crossformer 的 maintained runner 入口，并显示 Mamba 的精度方向明显依赖数据集；该 pilot 仍不减少正式计划缺口，也不与其它 normalization/source snapshot/test 表混合。
新增的 `mamba_classical_baseline_core_extension_pilot_20260907` 是独立的 60-job pilot：ETTh1/ETTm1/Electricity/Traffic、五个经典模型、H96、三 seed、四 epoch，`MeanStd` 和 `skip_test=true`；60/60 作业、phase validation 和 20 条 prediction-only 资源记录均已完成。它只补齐经典模型在四个核心数据集上的覆盖，没有 Mamba 臂，不能与已有 84-job MeanStd pilot 直接拼接排名。

当前最重要的缺口是：

| 位置 | 现状 | 解释 |
| --- | --- | --- |
| corrected/final confirmatory baseline phases | 已有 450-job 级别结果，但 baseline completeness 仍报告缺 `NLinear` | 不能把现有矩阵称为六模型完整基线矩阵 |
| highdim Electricity/Traffic extension | 180-job 注册计划，`0/180` metric artifact | 仍未训练，不能填入高维主表 |
| 四数据集 plain baseline expansion | 216-job 计划，`0/216` metric artifact | ETTh1/ETTm1/Electricity/Traffic 的 DLinear/NLinear/PatchTST/TimesNet/Mamba1/Mamba2 仍需按该注册计划的 MeanStd、三 seed、三 horizon 合同重跑；native pilot 不替代它 |
| MeanStd core baseline pilot | 独立 84-job exploratory pilot，`84/84` validation trajectory，test withheld | 已覆盖 H96、三 seed 和 iTransformer，但只有四 epoch；结果用于公平性/方向性审计，不填入正式 P2/P3 主表 |
| TimeMachine baseline screen | 165-job 计划，`0/165` metric artifact | 只完成实现 smoke 和资源预检，不能声称精度结果 |
| patch-token contrastive screen | 24-job 计划，`0/24` metric artifact | 只完成 ETTh1/Traffic 单 seed pilot 与四数据集资源预检，训练保持未授权 |

这里的 `0/N` 是“没有该计划下的可用 metric artifact”，不是失败率，也不是把其他计划的历史结果复制过来。不同计划的 normalization、source snapshot、optimizer、seed 和 test-release 合同不一致时，不能横向拼表。

## 建议的最小推进顺序

1. 先补齐同一冻结合同下的 DLinear、NLinear、PatchTST、TimesNet、Mamba1、Mamba2，至少覆盖 ETTh1、ETTm1、Electricity、Traffic 的 H96/H336/H720 和 3--5 个 seed；这是判断新结构是否有价值的必要条件。
2. 在 plain baseline 矩阵完成后，再选择一个 MambaACF 候选做 paired ablation。当前最值得复验的是有效 `hard_selected + lag_mask=true` 的 relation 合同和 channel-preserving variable/patch 分支；不要同时堆叠 GDD、memory、gate、temperature 等已出现负结果或交互不稳定的开关。
3. TimeMachine Mamba-1/2 先作为结构对照，不默认放进最终主模型。它目前结构资源开销高于普通 Mamba，且尚无本仓库精度结果；只有在统一 baseline 矩阵中显示独立价值时才进入论文主表。
4. 若论文需要交通/能源以外的外部泛化，再考虑补齐 PEMS 或 Solar；在原始文件和 provenance 未冻结前，不应下载后直接启动训练。

## 可复核命令

```bash
conda run --no-capture-output -n mamba python scripts/audit_benchmark_matrix.py \
  --plan-dir docs/research --artifact-root artifacts \
  --output-dir /tmp/autocon_benchmark_audit_20260907
```

这条命令只写指定输出目录，不改动现有计划、checkpoint 或 metric artifact。TimeMachine 单计划审计还应确认其模型 source 映射为真实文件，并且 phase 状态仍为 `authorized=false`、`training_started=false`。

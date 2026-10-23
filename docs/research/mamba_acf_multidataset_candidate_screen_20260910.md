# MambaACF 多数据集候选 screen（2026-09-10）

## 结论先行

这是一轮独立 fresh provenance、validation-only 的 H96 候选筛选，共 `48/48` 个作业完成。`MambaACFAutoCon` 使用 train-only ACF memory、global relation objective 和 `lambda=0.001`，在 Electricity 与 Traffic 上低于同一批次六个外部 baseline 的最佳均值，但在 ETTh1 与 ETTm1 上仍明显落后 PatchTST。因此它是值得继续做长 horizon 复验的候选，不是已经证明普遍优于基线的论文模型。

更重要的是，严格的同架构 null 对照显示 `lambda=0.001` 相对 `lambda=0` 的增量接近零：ETTh1 `-0.04%`、ETTm1 `-0.01%`、Electricity `-0.15%`、Traffic `+0.02%`。当前不能把相对 ACF-off 的整体改善归因于对比损失；主要变化来自 AutoCon forecast head/ACF memory 组合，而不是 relation loss 本身。

## 冻结合同

- 计划：[`mamba_acf_multidataset_candidate_screen_20260910.json`](mamba_acf_multidataset_candidate_screen_20260910.json)
- 数据集：ETTh1、ETTm1、Electricity、Traffic；`features=M`；`pred_len=96`
- seed：`2021/2022/2023`
- 预算：最多 8 epochs，patience 3；测试阶段显式关闭
- 归一化：所有 arm 使用 `AutoCon_wnorm=MeanStd`
- phases：`acf_off_h96`、`acf_memory_h96`、`autocon_memory_relation_lambda0_h96`、`autocon_memory_relation_lambda001_h96`
- source-tree SHA256：`e491e7014b2512e2d159c4460c97f879196c39b666cba94633a547148901e161`
- plan SHA256：`1fa724772c280bf3248577677c7d58028e4f156fca810e98ac27b4f7b66e3b30`
- 每个 phase 的 validator 均为 `valid=true`，sidecar hash 检查通过，测试指标和测试样本工件均不存在。
- 计划 immutable snapshot 保留 `execution.authorized=false`、`human_review_required=true`；因此这批工件只能作为未授权 exploratory screen，不能追溯性地升级为预注册确认性实验。独立 benchmark audit 记录 `0` 个 usable metric artifact、每 phase `12` 个 config-only validation-complete 记录，并标出 `artifacts_under_unauthorized_execution_snapshot`。

严格分析器和机器可读输出位于：[`scripts/analyze_mamba_acf_multidataset_candidate_screen.py`](../../scripts/analyze_mamba_acf_multidataset_candidate_screen.py)、[`artifacts/exploratory/mamba_acf_multidataset_candidate_screen_20260910/readout/`](../../artifacts/exploratory/mamba_acf_multidataset_candidate_screen_20260910/readout/)。

## 验证集结果

下表是每个数据集三个 seed 的 best pure-forecast `val_loss` 均值 +/- 样本标准差；只用于同数据集内的探索性比较。

| 数据集 | ACF-off | ACF memory | AutoCon `lambda=0` | AutoCon `lambda=0.001` |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.742608 +/- 0.003073 | 0.742074 +/- 0.003091 | 0.730038 +/- 0.009198 | 0.729716 +/- 0.009157 |
| ETTm1 | 0.434699 +/- 0.025216 | 0.433216 +/- 0.025302 | 0.428115 +/- 0.017110 | 0.428074 +/- 0.017713 |
| Electricity | 0.159070 +/- 0.002400 | 0.157515 +/- 0.002301 | 0.155258 +/- 0.000770 | 0.155025 +/- 0.000571 |
| Traffic | 0.400118 +/- 0.003668 | 0.396979 +/- 0.002524 | 0.394233 +/- 0.005025 | 0.394327 +/- 0.004937 |

### ACF memory 的单因素结果

相对 ACF-off 的 exact-seed 配对变化为：ETTh1 `-0.07%`（3/3）、ETTm1 `-0.34%`（3/3）、Electricity `-0.98%`（3/3）、Traffic `-0.78%`（1/3）。Traffic 的 95% CI 为 `[-0.018492, 0.012214]`，所以不能写成四个数据集都稳定受益。

### relation objective 的单因素结果

`lambda=0.001 - lambda=0` 的 paired mean relative change 为：ETTh1 `-0.04%`（3/3）、ETTm1 `-0.01%`（2/3）、Electricity `-0.15%`（3/3）、Traffic `+0.02%`（2/3），四个数据集的 95% CI 均跨过或接近零。AutoCon telemetry 确认目标确实运行：`lambda=0.001` phase 的平均 relation CE 为 `0.842604`，signed contribution 为 `0.001326`；这些是机制执行读数，不是精度增益证据。

## 与同坐标 plain baseline 的 H96 对照

下表来自已完成的 216-job plain matrix 中相同数据集、horizon 和三个 seed 的 validation rows；外部基线同样是短预算 validation-only。`MambaACFAutoCon` 一列为本 screen 的 `lambda=0.001` 均值。

| 数据集 | DLinear | NLinear | PatchTST | TimesNet | Mamba1 | Mamba2 | MambaACFAutoCon |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.695156 | 0.697799 | **0.685596** | 0.768010 | 0.723166 | 0.734823 | 0.729716 |
| ETTm1 | 0.410020 | 0.408890 | **0.399746** | 0.440102 | 0.419661 | 0.411823 | 0.428074 |
| Electricity | 0.201873 | 0.201958 | 0.194715 | **0.170231** | 0.180997 | 0.176997 | **0.155025** |
| Traffic | 0.506108 | 0.506352 | 0.420740 | 0.425349 | 0.492186 | 0.489107 | **0.394327** |

按 exact-seed paired relative change，候选相对 PatchTST 在 ETTh1/ETTm1 为 `+6.43%/+7.06%`，在 Electricity/Traffic 为 `-20.38%/-6.27%`；相对 TimesNet 在四个数据集分别为 `-4.98%/-2.72%/-8.93%/-7.27%`。这说明候选对高维数据有信号，但不能支持跨数据集普遍胜出，也不能用均值表替代配对统计。

## 解释边界

1. 这批数据只覆盖 H96、每数据集三个 seed 和 8 epoch 短预算；没有 H336/H720，也没有 test 指标。
2. 与 plain baseline 的结果可以做同坐标描述性比较，但不能直接写成显著性、非劣效或 SOTA 结论。
3. 训练 elapsed time 同时包含数据加载、初始化和 early stopping，不是速度证据。效率结论必须使用同步的 prediction、train-step、显存和端到端 wall-time 合同。
4. `lambda=0` 是保留相同 AutoCon/ACF 架构的 relation null，不等同于裸 Mamba baseline；因此必须同时报告 ACF-off、memory-only 和 null-to-relation 三个层次。
5. 这轮结果还不能支撑投稿。若 H96 结果经人工复核后值得推进，下一轮应先冻结 H336/H720 的同一四臂合同，并把 PatchTST、TimesNet、Mamba2 作为至少三类外部 comparator；若高维优势不能复现，应停止继续堆叠 ACF 组件。

## 人工复核清单

- [ ] 核对四个 phase 的 `phase_validation.json`、manifest 去重和 source hash 一致性。
- [ ] 核对每个 checkpoint 的 `config.json` 中 `AutoCon_wnorm=MeanStd`、dataset、horizon、seed 和 lambda。
- [ ] 核对 `acf_prior.npz` 的 SHA256 与 config sidecar identity；确认测试工件计数为零。
- [ ] 先复核 `lambda=0.001` 对 `lambda=0`，再复核 memory 对 ACF-off，不要把两种差异合并成“对比学习增益”。
- [ ] 在没有 H336/H720、完整测试集和端到端资源测量前，不把本 screen 写入论文主结果表。

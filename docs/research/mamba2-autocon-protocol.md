# Mamba1/Mamba2 + AutoCon 实验协议

本文档记录当前仓库中可复现的实验设置、数据边界和结果解释方式。所有路径均相对于仓库根目录。

> **版本边界：** 本文的详细旧矩阵和 20 epoch 配置描述的是已完成的 `artifacts/paper_final/strict/` 历史 pilot。当前复验以 [`corrected_training_plan.json`](corrected_training_plan.json) 为准（50 epochs、patience 10、cosine 学习率、保留训练尾 batch）；不要用本文的旧命令直接启动当前论文复验。

## 环境

主训练环境是 conda 环境 `mamba`：

```text
Python 3.9.25
PyTorch 2.8.0+cu128
mamba_ssm 2.3.1
causal_conv1d 1.6.2.post1
triton 3.4.0
GPU: NVIDIA GeForce RTX 5090 D
```

`mamba_ssm==2.3.1` 同时提供 `Mamba`（Mamba-1）和 `Mamba2`（Mamba-2）；本项目分别通过 `from mamba_ssm import Mamba` 和 `from mamba_ssm import Mamba2` 使用两种骨干。四个模型名 `Mamba1`、`Mamba1AutoCon`、`Mamba2`、`Mamba2AutoCon` 均由 `run.py` 显式路由。该安装是 CUDA 版本，CPU 前向不支持。

环境检查命令：

```bash
conda run --no-capture-output -n mamba python -c \
  "import sys,torch,mamba_ssm; print(sys.executable); print(torch.__version__, torch.cuda.is_available()); print(mamba_ssm.__version__)"
```

## 数据和划分

当前仓库实际包含 7 个数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Exchange-Rate 和 National Illness。目标列统一为 `OT`；正式计划分别注册单变量（`features=S`）和 ETT 多变量（`features=M`）阶段。M 阶段的 `enc_in/dec_in/c_out` 由 CSV schema 自动推导（ETT 为 7/7/7），不接受手工覆盖。ETT 数据按官方 12/4/4 月边界划分；其余数据按 60%/20%/20% 划分。标准化器只在训练段拟合，ACF 先验也只用训练段计算。历史 strict 允许训练 loader 丢弃最后不完整 batch；corrected 默认保留尾 batch，验证和测试 loader 始终保留全部窗口。

文件哈希、缺失值、重复时间戳和训练段 ACF 摘要见 [`dataset_manifest.json`](../../artifacts/data_analysis/dataset_manifest.json) 和 [`acf_summary.csv`](../../artifacts/data_analysis/acf_summary.csv)。ACF 可视化见 `../../artifacts/data_analysis/*_acf.png`。

## 模型和优化

Mamba1/Mamba2 模型结构为 `DataEmbedding -> pre-norm residual Mamba blocks -> AutoConNet 多尺度预测头`。正式计划的主设置如下：

代码入口为 [`exp/exp_long_term_forecasting_with_mamba.py`](../../exp/exp_long_term_forecasting_with_mamba.py)：`Mamba1`/`Mamba2` 使用普通预测损失，`Mamba1AutoCon`/`Mamba2AutoCon` 使用 AutoCon 训练器；`run.py` 会按模型名显式路由到对应实验类。Mamba-1 不使用 `headdim` 参数，Mamba-2 使用 `headdim` 约束。

```text
seq_len=96, label_len=48, d_model=32, d_ff=32, e_layers=1
mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_headdim=16
AutoCon_multiscales=[96], AutoCon_wnorm=LastVal
ACF source=residual, temperature=1.0, base_temperature=1.0
AutoCon_lambda 由 validation-only selection 冻结，learning_rate=1e-3
```

对比模型为 DLinear、PatchTST、TimesNet、AutoConNet、Mamba1、Mamba1AutoCon、Mamba2、Mamba2AutoCon 和 AutoConNetAutoCon。历史 strict selection/确认性阶段使用最多 20 epochs、patience=5；corrected 阶段改为 50 epochs、patience=10、cosine 学习率和 `drop_last_train=false`。两者均使用统一 `weight_decay=1e-5`，`autocon_warmup=0` 明确表示不使用 warm-up。S 与 M 阶段分别冻结 lambda，禁止跨特征设置复用。`seq_len=96`，因此不能解读为长上下文消融。参数量不同，结果必须同时报告资源摘要。PatchTST/TimesNet 当前只作为 plain 外部基线，未把它们未经验证的 representation 接入 AutoCon。M 版本的 AutoCon 使用跨通道平均 ACF prior，论文中应称为 multivariate/channel-mixing 版本，不应称为逐通道 CI 版本。

## 历史 pilot 矩阵

- ETTh1/ETTh2：5 个模型、horizon 96/336/720、3 seeds、3 epochs，共 90 个训练作业。
- ETTm1/ETTm2/Weather/Exchange-Rate：5 个模型、horizon 96/336、2 seeds、2 epochs，共 80 个训练作业。
- National Illness：5 个模型、horizon 96/192、3 seeds、3 epochs，共 30 个训练作业；horizon 192 由于验证/测试窗口很少，仅作探索性结果。
- AutoCon lambda 消融：ETTh1/ETTh2 的 h96/h336（lambda 0.1/1.0），ETTh1/ETTh2 的 h720（lambda 1.0），以及四个迁移数据集 h96/h336（lambda 1.0），共 46 个训练作业。

其中前 3 项的 200 个作业包含 40 个 AutoConNetAutoCon 解耦对照，另有 46 个 lambda 消融，合计 246 个训练作业；全部 manifest 状态为 `ok`。另有 52 条确定性的 NaiveLast/NaiveMean 基线记录。该节只描述历史 pilot，不是当前确认性矩阵。

## 结果和统计

主指标是训练标准化空间中的 MSE/MAE/RMSE。训练模型的均值和 95% Student-t 区间跨 seed 计算；朴素基线只有一个确定性运行，不应解读为跨 seed 区间。测试指标和朴素基线在所有测试窗口和元素上累计误差后计算，和模型测试路径的全局均值一致；early-stopping 的验证 loss 同样累计 validation batch 的平方误差后除以全部预测元素数，避免尾 batch 获得过高权重。滑动窗口高度重叠，区间单位是 seed 而不是窗口。标准化序列上的 MAPE/MSPE 不用于排名。

汇总文件：

- [`strict-evidence-status.md`](../reports/strict-evidence-status.md)：全部严格阶段的完成状态和可解释结论边界。
- [`confirmatory-core-fresh.md`](../reports/confirmatory-core-fresh.md)：严格 ETT-S 确认性结果。
- [`confirmatory-M-fresh.md`](../reports/confirmatory-M-fresh.md)：严格 ETT-M 确认性结果。
- [`secondary-transfer-fresh.md`](../reports/secondary-transfer-fresh.md)：严格 Weather/Exchange secondary transfer。
- [`exploratory-ili-fresh.md`](../reports/exploratory-ili-fresh.md)：严格 National Illness exploratory 结果。
- [`confirmatory-core-legacy.md`](../reports/archive/confirmatory-core-legacy.md)：历史核心矩阵的诊断汇总和 seed 配对差。
- [`dataset_manifest.json`](../../artifacts/data_analysis/dataset_manifest.json)：数据边界、哈希和质量审计。
- `artifacts/paper_final/strict/<phase>/report/`：冻结 strict 快照中 validator 通过的机器可读表格；新结果应写入独立的 corrected 根目录。

生成流程：

```bash
conda run --no-capture-output -n mamba python scripts/analyze_datasets.py
conda run --no-capture-output -n mamba python scripts/compute_naive_baselines.py
conda run --no-capture-output -n mamba python scripts/smoke_mamba2.py
conda run --no-capture-output -n mamba python scripts/smoke_mamba1.py
# Run only after a confirmatory phase has passed validate_training_phase.py.
conda run --no-capture-output -n sci-agent python scripts/collect_results.py \
  --results_dir artifacts/paper_final/strict/confirmatory_core/results \
  --output_dir artifacts/paper_final/strict/confirmatory_core/raw_report \
  --docs_dir artifacts/paper_final/strict/confirmatory_core/raw_report \
  --naive /nonexistent/naive.csv
```

严格阶段的完整参数、解释器路径、seed、lambda、配置哈希、源码快照哈希和 UTC 时间均写入 `artifacts/paper_final/strict/<phase>/logs/job_manifest.jsonl`；阶段 provenance 见 `provenance.json`。确认性阶段保存严格形状的 `pred.npy/true.npy`，同时写入标准化/原始量纲指标和逐通道误差；selection 阶段用 `--skip_test` 且只保存 validation trajectory。窗口和通道只用于误差聚合与异质性描述，不作为独立重复。

## 数据边界

仓库当前不包含 GLD CSV。旧的 GLD 日志、checkpoint 和图无法从当前仓库独立重跑，因此没有混入上述汇总表；需要 GLD 结果时应先固定并提交原始 CSV 快照及其 SHA256，再按相同协议重新运行。Weather、Exchange-Rate 和 ILI 使用本仓库的 60/20/20 custom split，不应直接与采用其他官方 split 的论文数字比较。当前严格结果是受控的 ablation comparison，不是充分收敛或 SOTA 声明。

## 历史 strict 计划与当前 corrected 入口

`artifacts/paper_*_v2` 是已完成的 controlled pilot。它使用 2--3 个 seed、2--3 个 epoch，且曾经查看测试指标后比较 lambda，因此不能用来选择最终超参数，也不能作为充分收敛的主结果。历史 strict 的注册入口是 [`final_training_plan.json`](final_training_plan.json)；当前 corrected 的注册入口是 [`corrected_training_plan.json`](corrected_training_plan.json)，输出写入独立的 `artifacts/paper_final/corrected/`。源码快照和 legacy 边界见 [`provenance-and-reproducibility.md`](provenance-and-reproducibility.md)。

### 估计对象和重复单位

主要估计对象是同一数据集、同一 horizon、同一 seed 下的配对差：

```text
MSE(Mamba2 + AutoCon) - MSE(Mamba2)
```

次要配对差为 Mamba1、AutoConNet backbone 的同类比较，以及 Mamba1 与 Mamba2 的 backbone 比较。滑动窗口有高度重叠，只是观测窗口，不是独立样本；置信区间和敏感性分析以 seed 为重复单位，必要时另报按 forecast-origin、stride=horizon 的结果。ETTh1/ETTm1 和 ETTh2/ETTm2 是同一序列族的不同采样率，跨数据集汇总时不把它们当作七个独立 domain。

### 阶段 A：validation-only 选择

正式计划有两个互相独立的 validation-only 选择阶段。`selection` 和 `selection_M` 都覆盖 ETTh1/ETTh2（96/336/720）及 ETTm1/ETTm2（96/336），分别在 S 与 M 特征空间使用独立 selection seeds 101/102/103。两者均对 `AutoConNetAutoCon`、`Mamba1AutoCon`、`Mamba2AutoCon` 评估预注册候选 `lambda ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}`。每个作业只建立训练和验证 loader，使用验证 MSE 的最佳 epoch；`--skip_test` 会写入 `training_complete.json`，并保证测试 loader 不被调用。选择器只接受已通过严格 validator 的阶段，并在每个 dataset/model/horizon/seed cell 内归一化最佳验证损失，再对候选 lambda 求平均相对损失，得到该特征空间的冻结 lambda。若选择阶段失败，不允许用测试集补选。

### 阶段 B：confirmatory core

`confirmatory_core`/`confirmatory_M` 是历史 strict 阶段；当前对应的 corrected 阶段名称为 `corrected_confirmatory_S`/`corrected_confirmatory_M`。四个核心阶段均在 ETTh1/ETTh2（96/336/720）和 ETTm1/ETTm2（96/336）上运行九个预注册 arms：`DLinear`、`PatchTST`、`TimesNet`、`AutoConNet`、`AutoConNetAutoCon`、`Mamba1`、`Mamba1AutoCon`、`Mamba2`、`Mamba2AutoCon`。每个 corrected cell 使用 seed 2021--2025、最多 50 epochs、validation early stopping patience 10、Adam、统一 `weight_decay=1e-5`、cosine `lr`，并保存 `pred.npy/true.npy`；历史 strict 参数仍见上文。确认性阶段只在全部训练决策冻结后读取 untouched test split。corrected 核心计划展开为 S selection 630、M selection 630、S confirmatory 450、M confirmatory 450；历史 strict 另有 secondary 180 和 exploratory ILI 180 个作业。

冻结 strict 快照的六个阶段记录为完成并通过当时的 validator，严格报告见 [`strict-evidence-status.md`](../reports/strict-evidence-status.md)。历史运行曾完成 350/350 个作业，冻结 lambda 为 `AutoConNetAutoCon=0.3`、`Mamba1AutoCon=0.01`、`Mamba2AutoCon=0.1`，结果见 [`confirmatory-core-legacy.md`](../../docs/reports/archive/confirmatory-core-legacy.md)。这些历史作业是在 provenance 快照加入前运行，且源码文件修改时间与作业时间重叠，因此只作为 legacy 诊断，不能作为已冻结源码的正式确认性证据。

Weather/Exchange 进入同样设置的 secondary transfer；National Illness 单独作为 exploratory。ILI 原始记录为 7 天间隔，当前实现的 horizon 是 step 数而不是天数，故不在主结论中使用；Weather 的重复时间戳和不规则间隔也必须在报告中标记，并进行敏感性处理后才能升级为主数据集。

### 记录和统计约束

每个阶段写入环境版本、GPU、git 状态、数据 SHA256、完整命令、耗时、return code、best epoch 和 validation trajectory。主表报告 seed-level mean、95% CI、配对差和效应量；不把窗口数当作样本量，不跨数据集直接平均原始 MSE。Mamba CUDA kernel 可能有设备相关非确定性，因此固定 seed 仍需多 seed 复制，并在论文中报告硬件和软件版本。当前参数量不匹配（DLinear、AutoConNet、Mamba 的容量不同），需同时报告参数量/FLOPs；不能把容量差解释成纯 backbone 效应。

阶段命令和恢复方式见 [`docs/README.md`](../README.md)。历史核心结果已经生成诊断报告 [`confirmatory-core-legacy.md`](../../docs/reports/archive/confirmatory-core-legacy.md)，但运行跨越了未冻结源码变更，只能标记为 `legacy/unknown code provenance`。冻结 strict 快照中的 core、M、secondary 与 exploratory 阶段均有当时 validator 的通过记录；live validator 后续已加固，不能把这些冻结记录理解为当前源码的重新验证。在补齐现代 baseline、容量匹配和独立复核前，本文档中的结论仍不能写成 SOTA 或因果结论。

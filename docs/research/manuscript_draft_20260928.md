# AutoCon-Connected Mamba Forecasting: Evidence-Bound Internal Draft

> **Status:** internal draft for human review; not submission-ready. Every numeric statement below is bound to a local readout. The draft deliberately does not infer universal accuracy, speed, or publication readiness from the current evidence.

## Working Title

**AutoCon-Connected Mamba Forecasting for High-Dimensional Time Series: Accuracy, Efficiency, and Failure-Boundary Evidence**

## Abstract (Evidence-Bound Draft)

This study evaluates a Mamba-based time-series forecaster whose patch-token encoder shares its state with an AutoCon contrastive representation objective and an optional prediction-connected representation residual. The implementation is an independent local realization of design ideas reviewed from public Mamba forecasting repositories; it is not a bitwise reproduction of any external codebase. Under frozen held-out protocols, MambaTSv3AutoCon is non-inferior at a ratio margin of 1.02 in 28/28 PEMS H12 comparison cells, 12/12 METR-LA/PEMS-BAY H12 cells, 12/12 Weather H96/H336/H720 cells, and 13/14 Electricity/Traffic H96 cells. The prediction-connected fusion release improves over the same-source plain MambaTSv3 by 2.257% on METR-LA and 1.816% on PEMS-BAY. Matched GPU forward measurements support lower latency than named nonlinear comparators on selected shapes, but not a universal speed advantage and not an advantage over DLinear or NLinear. Validation-only results also identify clear failures on low-dimensional ETT and related datasets, while a faster latent-temporal candidate fails its exact null and external-baseline accuracy gates. The evidence therefore supports a scoped claim about named traffic, Weather, and high-dimensional Electricity/Traffic protocols, with explicit negative and thermal-stop boundaries. [claim:C001,C002,C003,C008,C009,C010,C011,C012,C013] [evidence:E001,E002,E003,E004,E005,E059,E060,E061]
The frozen stride-24 relation-auxiliary configuration wins all seven registered baselines on both ETTm1 and ETTm2 H96 in validation-only paired screening, but an exact ETTm1 lambda-zero control changes MSE by only -0.004%, so the overall forecasting advantage must not be attributed to AutoCon alone. [claim:C004,C005,C009] [evidence:E029,E032,E033,E036]

**Evidence:** [PEMS readout](../../artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/readout/readout.md), [METR/PEMS-BAY readout](../../artifacts/paper_final/metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927/readout/readout.md), [Weather readout](../../artifacts/paper_final/mambatsv3_weather_test_release_20260922/readout/readout.md), [revalidated main table](../../artifacts/exploratory/paper_main_table_20260930/main_table.md).

## 1. Introduction

Long-context forecasting models must balance temporal memory, multivariate interaction, and computational cost. Selective state-space models provide a natural temporal encoder, but a Mamba encoder alone does not establish that its learned representation is useful for forecasting. This work therefore couples the forecasting path to AutoCon's representation learning objective and evaluates whether a prediction-connected representation residual improves forecasts under a fixed protocol.

The study has three deliberately bounded questions:

1. Does the candidate remain competitive with strong linear, patch, Transformer, mixing, and Mamba baselines on frozen held-out datasets?
2. Does connecting the AutoCon representation to the forecast produce an improvement over an exact same-source MambaTSv3 control?
3. Can the candidate's forward cost be lower than named nonlinear comparators under matched device-resident measurements?

The corresponding negative question is equally important: on which datasets does the candidate fail against strong linear or patch models?

## 2. Method

### 2.1 Candidate architecture

MambaTSv3 uses patch tokens and a single Mamba-1 scan. The AutoCon branch pools the encoded observed-window states into a representation used by the contrastive objective. In the fusion configuration, a zero-initialized prediction-connected residual projects this representation into the forecast horizon and is added to the forecasting path. The candidate-only AutoCon and fusion flags are disabled for external baselines. [claim:C004] [evidence:E002,E015]

The repository also contains independently implemented exploratory variants inspired by public Mamba forecasting structures, including variable-token, decomposition, multi-scale, time-conditioned, VPT/VAST-lite, Mamba-1, Mamba-2, and TimeMachine-style branches. These variants are treated as local implementations and are not reported as external bitwise reproductions. [claim:C010] [evidence:E006]

The external structure registry is maintained in [`github_mamba_reference_registry_20260928.md`](github_mamba_reference_registry_20260928.md). A 2026-09-30 repository-page audit verifies the main design signals and available license/citation metadata for MambaTS, TimeMachine, Mamba4Cast, and ms-Mamba. The corresponding local bibliography and verification notes are [`mamba_autocon_references.bib`](references/mamba_autocon_references.bib) and [`citation verification notes`](references/citation_verification_20260930.md). These projects are used as architectural references only; their public repository results are not substituted for the local frozen experiments, and exact commit/license verification remains a submission-stage author task. [claim:C010] [evidence:E006]

### 2.2 Data and protocols

The principal held-out releases cover PEMS03/04/07/08 at H12, METR-LA and PEMS-BAY at H12, and Weather at H96/H336/H720. Each release uses five fresh paired seeds and streaming test metrics. The release bundle audit verifies phase validity, authorization, plan hashes, summary/test row counts, and non-inferiority fields. [claim:C001,C002,C003] [evidence:E001,E002,E003,E004,E005]

Additional ETTh1/ETTh2/ETTm1/ETTm2, Electricity, Traffic, Exchange-Rate, ILI, and Solar experiments are validation-only or exploratory unless explicitly linked to a frozen held-out release. They are not interchangeable with the principal test results. The frozen stride-24 validation matrices have 24/24 successful jobs on each of ETTm1 and ETTm2 and three paired wins against each of DLinear, RLinear, PatchTST, iTransformer, Mamba-1, TimesNet, and Mamba-2. Candidate MSE is 0.384043 on ETTm1 and 0.125532 on ETTm2; matched prediction-only latency is 0.2855/0.2746 ms, faster than the named nonlinear comparators but slower than DLinear/RLinear. An exact ETTm1 active/null control uses the same architecture and prediction path: active MSE is 0.385314 versus 0.385329 for lambda-zero, a -0.004% change with 2/3 wins. Thus the ETT results support transfer of the frozen architecture, not an independently established AutoCon gain. In a current-source Solar H96 validation matrix, a compact variable-axis Mamba with prediction-connected AutoCon representation fusion improves over iTransformer by 9.66% across three paired seeds and has 9.4% lower matched forward latency; however, its exact contrastive relation-loss null is 0.19% better, so this result is architecture and representation-fusion evidence, not a causal relation-loss result. [claim:C004,C005,C006,C009] [evidence:E008,E017,E029,E032,E033,E036,E050,E051,E052]

An independent rank-4 low-rank channel-state mixer is retained as a separate challenger rather than merged into the principal MambaTSv3 release. In validation-only H96 matrices it is 8.082% below TimesNet on Electricity and 4.080% below iTransformer on Traffic, while remaining 4.498% above iTransformer on ETTh1. The same mixer-on/mixer-off contrast is therefore positive only in selected high-dimensional domains. Relation-active and relation-null controls differ by at most 0.013% across these three datasets, so the experiment supports a state-mixing interpretation and a non-inferior AutoCon training constraint, not an independently established contrastive-loss gain. [claim:C004,C005,C009] [evidence:E012,E013]

### 2.3 Baselines

The main release comparisons include DLinear, NLinear, PatchTST, TimesNet, TimeMixer, iTransformer, Mamba2, and same-source plain MambaTSv3 where registered. RLinear, STGCN-style, GraphWaveNet-style, AGCRN-style, DCRNN-style, and additional Mamba variants are documented as validation, resource, or partial evidence with their boundaries stated explicitly.

### 2.4 Metrics and efficiency protocol

The primary forecasting metric is scaled MSE as recorded by the frozen release readers. Non-inferiority is assessed using the registered one-sided upper ratio and a 1.02 margin. Efficiency claims use matched device-resident prediction-only measurements with explicit batch, sequence length, horizon, channel count, device, and precision. Training wall time is not treated as inference speed. [claim:C001,C008] [evidence:E001,E004,E005,E016,E021,E023,E030,E034,E041,E044,E051,E053,E054]

## 3. Results

### 3.1 Frozen held-out accuracy

The PEMS release contains 28 candidate-baseline cells; all 28 satisfy the registered non-inferiority criterion, with five paired seed wins in each cell. The METR-LA/PEMS-BAY fusion release contains 12 cells, all non-inferior, and the candidate wins all five paired seeds in each cell. The Weather release contains 12 horizon-baseline cells, all non-inferior, also with five paired seed wins in each cell. [claim:C001,C002,C003] [evidence:E001,E002,E003,E004,E005]

The same-source plain-Mamba comparison in the METR-LA/PEMS-BAY release shows a mean MSE reduction of 2.257% on METR-LA and 1.816% on PEMS-BAY. These are scoped same-release observations, not evidence that AutoCon contributes an independent improvement on every dataset. [claim:C002,C004] [evidence:E002,E032]

### 3.2 Validation-only boundaries

The current-source H96 validation matrix favors the candidate over all registered baselines on Weather, Electricity, Traffic, and the frozen ETTm2 transfer, but the candidate is worse than PatchTST on ETTh1 by 4.21%. ETTh2 remains a negative ETT transfer boundary; ETTm2 is also a long-horizon boundary, where the frozen candidate is 1.095%/1.913% above PatchTST at H336/H720 with no paired wins. A temporal-axis MambaTSv3 Solar H96 matrix is 3.23% above iTransformer and loses all three paired seeds, while a separately evaluated compact variable-axis Mamba is 9.66% below iTransformer with three wins. The latter also shows a 1.39% prediction-connected representation-fusion gain over its exact fusion-off control, whereas its relation-loss active arm is 0.19% worse than the exact relation-null control. A separate fresh-seed paired Solar extension places this variable-axis candidate against TimeMixer: MSE is 0.138005 versus 0.192720, a 28.385% reduction with 3/3 wins, and fixed-shape prediction latency is 0.2916 versus 0.6755 ms. The fixed-fusion training audit confirms a nonzero auxiliary objective but not an independent relation-loss effect. These successes and failures are retained as validation-only evidence, not selectively promoted into held-out conclusions. [claim:C005,C006,C009] [evidence:E008,E010,E013,E014,E017,E020,E022,E024,E029,E033,E038,E040,E043,E047,E050,E051]

The CPU-4 high-dimensional held-out release provides the final current-source Electricity/Traffic H96 comparison. Across 14 registered candidate-baseline cells and five paired seeds, 13 cells pass the 1.02 non-inferiority gate. The required failure is Traffic/iTransformer: candidate MSE is 2.154% higher and wins only one of five paired seeds. This result supports a scoped high-dimensional held-out claim, not all-baseline dominance. [claim:C011,C009] [evidence:E059]

The earlier 48-row validation matrix remains useful as a pre-release screening result, but it must not be substituted for the CPU-4 held-out release. [claim:C005,C009] [evidence:E010,E013]

A fast short-token latent-temporal Mamba was evaluated under an exact lambda-zero control and the same seven external baselines. Although its representation-inclusive forward path was faster than the stride-24 MambaTSv3 reference, active AutoCon degraded validation MSE by 10.186% on Electricity and 6.343% on Traffic relative to lambda-zero, with zero of three paired wins in both domains. It is retained as a negative architecture and AutoCon boundary, not as a paper candidate. [claim:C012,C009] [evidence:E060]

An exact AutoCon mechanism matrix then separated the prediction connection from the contrastive objective. On Electricity and Traffic H96, active AutoCon was worse than the same-graph fusion-off control by 1.160% and 1.081% (0/3 paired wins in both domains), so the prediction-connected residual is not supported as a high-dimensional accuracy mechanism. Relative to an otherwise identical lambda-zero control, active relation loss improved validation MSE by 0.376% and 0.236% (2/3 paired wins in both domains). This is a small validation-only signal, not a held-out causal claim, and the current paper position should treat AutoCon relation as an auxiliary training constraint rather than attribute the full forecast gain to representation fusion. [claim:C004,C009] [evidence:E012,E013]

A read-only training audit over fifteen relation-auxiliary-only candidate runs verifies that the train-time representation objective was active rather than nominal: every run used a finite nonzero contrastive loss at lambda 0.001, and each epoch satisfied the recorded loss identity `train_loss = mse_loss + lambda * autocon_loss` to at most 1.29e-09 residual. The epoch-average contrastive loss decreased by 1.84% to 6.03% across Electricity, Traffic, Weather, ETTh1, and Solar. This establishes implementation-level participation of AutoCon representation learning, while not treating a decreasing auxiliary loss as an accuracy or causal result. [claim:C004] [evidence:E015,E028,E031,E035,E039,E042,E045,E052]

Following this frozen mechanism choice, a six-job relation-auxiliary-only confirmation reused the immutable same-source baseline artifacts and paired seeds. The candidate retained three wins against each of DLinear, NLinear, RLinear, PatchTST, TimesNet, iTransformer, and Mamba2 on both Electricity and Traffic, with mean MSE 6.117% below TimesNet and 7.029% below iTransformer, respectively. It also improved by 1.798% and 1.379% over the earlier fused candidate. Matched forward measurements reduced median latency from 0.9231/2.3551 ms to 0.8011/2.2052 ms on Electricity/Traffic, but remained slower than linear models, iTransformer, and Mamba2. These are validation and fixed-shape resource results only. [claim:C005,C008] [evidence:E013,E016,E021,E022,E023]

An independent cross-domain validation matrix on Weather and ETTh1 retained the same candidate and retrained five strong baselines under the same source snapshot. Weather reproduced five paired wins per comparator, with candidate MSE 1.99% below TimesNet. ETTh1 reproduced the low-dimensional failure boundary: the candidate was 1.50% above RLinear, 3.17% above PatchTST, and 3.93% above iTransformer, while beating only TimesNet and Mamba2. This negative result is retained as evidence against a universal claim. [claim:C005,C009] [evidence:E014,E016,E024,E025]

In a manifest-bound matched RTX 5090 D forward protocol, the cross-domain candidate measured 0.2758 ms on Weather and 0.2751 ms on ETTh1. It was faster than PatchTST, TimesNet, and Mamba2 on both shapes, but slower than RLinear and iTransformer. Since ETTh1 fails its accuracy comparison, only Weather provides an additional scoped accuracy-latency Pareto observation; the resource measurement is synthetic prediction-only timing and not training throughput or an end-to-end deployment claim. [claim:C008,C009] [evidence:E016,E025]

Solar provides a further non-traffic boundary under a fresh-seed, same-source 24-job matrix. The relation-auxiliary-only candidate beats DLinear, NLinear, RLinear, PatchTST, TimesNet, and Mamba2 in all three paired seeds, but remains 5.839% worse than iTransformer (0/3 wins). We therefore report Solar as a complete transfer boundary rather than selecting it as another success domain. [claim:C005,C009] [evidence:E017]

To add a direct architecture comparator, we trained plain Mamba-1 on Electricity and Traffic H96 under the same source snapshot. The relation-auxiliary-only candidate has lower validation MSE than plain Mamba-1 by 13.648% and 23.822% in independent fresh-seed mean comparisons, respectively. Because the candidate and comparator use different seed sets, these numbers are descriptive rather than paired evidence; they are not included in the held-out main table and do not identify an independent causal contribution from the AutoCon loss. [claim:C005,C009] [evidence:E018]

The matched source-bound resource measurement reverses the efficiency ordering: candidate median forward latency is 0.8011/2.2067 ms versus 0.4469/0.6451 ms for plain Mamba-1 on Electricity/Traffic, corresponding to +79.2%/+242.0%. Peak allocated memory is also higher for the candidate (247.6/649.8 MB versus 54.0/128.2 MB). Thus plain Mamba-1 is an explicit efficiency negative boundary; the paper must not claim a universal speed advantage over Mamba architectures. [claim:C008,C009] [evidence:E019]

A separate speed-oriented stride-24 MambaTSv3 screen provides a narrower high-dimensional Pareto candidate. With four non-overlapping patch tokens and the same train-only AutoCon objective, it wins all three paired validation seeds against DLinear, PatchTST, iTransformer, and plain Mamba-1 on both Electricity and Traffic H96. Its median prediction-only latency is 0.3971/0.8379 ms: faster than PatchTST on both shapes, faster than Mamba-1 on Electricity, and faster than iTransformer on Traffic. It remains slower than DLinear and respectively iTransformer/Mamba-1 on the other two comparisons. This result is validation-only, does not include TimesNet or Mamba-2, and must be reported as a scoped accuracy-latency screen rather than a replacement for the principal held-out model. [claim:C005,C008,C009] [evidence:E020,E021]

After adding TimesNet and Mamba-2 under the same seeds, the complete stride-24 matrix contains 42 successful validation jobs and 12 paired cells. The candidate wins all three seeds against all six registered baselines on both Electricity and Traffic; its margins against the strongest comparator are -3.438% and -4.332%. The attested fixed-shape latency remains 0.3971/0.8379 ms, faster than PatchTST, TimesNet, and Mamba-2 on both datasets, but slower than DLinear on both and slower than iTransformer on Electricity or Mamba-1 on Traffic. This is the appropriate scoped Pareto claim; it is not a held-out result or universal efficiency claim. [claim:C005,C008,C009] [evidence:E022,E023]

The same stride-24 candidate has a direct high-dimensional AutoCon audit over its six Electricity/Traffic validation jobs: every run records a finite nonzero contrastive loss with effective lambda `.001`, and the maximum residual of `train_loss = mse_loss + lambda * autocon_loss` is `9.61e-10`. A deterministic CUDA comparison also confirms that the current default Mamba-1 path retains the frozen validation candidate: model-state and prediction hashes match exactly, while representation maximum absolute difference is `3.5763e-7`, within the predeclared `2e-6` FP32 interpolation-equivalence tolerance. These checks establish train-time objective participation and source-path continuity, not a causal AutoCon accuracy effect. [claim:C004,C005] [evidence:E055,E056,E057]

The same frozen stride-24 configuration was tested on a separate Weather/ETTh1 H96 validation matrix with seven baselines and three paired seeds. Weather remained a success domain: candidate MSE was 0.444975 and it won all three seeds against every comparator, including a 3.977% mean improvement over TimesNet. ETTh1 is a required counterexample: candidate MSE was 0.699699, marginally above DLinear/RLinear and 1.611%/2.425% above PatchTST/iTransformer, while only beating Mamba-1, TimesNet, and Mamba-2. Fixed-shape forward latency was 0.2755 ms on Weather and 0.2817 ms on ETTh1, lower than Mamba-1, TimesNet, and Mamba-2 but not linear or iTransformer baselines. These 48 validation-only jobs refine the claim to selected high-dimensional or periodic domains and explicitly rule out a cross-domain universal accuracy or speed statement. [claim:C005,C008,C009] [evidence:E024,E025]

The same frozen stride-24 configuration was then transferred without retuning to ETTm1 H96. Across 24 validation-only jobs, it won all three paired seeds against each of seven baselines, with candidate MSE 0.384043 and relative improvements of 2.500% to 13.007%. Its resource boundary remains explicit: candidate latency is lower than PatchTST, iTransformer, Mamba-1, TimesNet, and Mamba-2, but higher than DLinear and RLinear. The exact lambda-zero mechanism control described above shows that this transfer result cannot be interpreted as a standalone causal effect of the contrastive objective. [claim:C005,C008,C009] [evidence:E029,E030,E031,E032]

To directly compare with the public MambaTS design signal, a separate 24-job matrix paired the stride-24 AutoCon candidate with the local plain `MambaTS` v1 patch/variable-scan baseline on ETTh1, Weather, Electricity, and Traffic. The candidate improved validation MSE by 8.376%, 9.625%, and 3.490% on Weather, Electricity, and Traffic respectively, with three paired wins in each dataset. ETTh1 remained a failure boundary at +1.047% and 0/3 wins. The AutoCon training audit found finite nonzero losses and a maximum objective-identity residual of 9.65e-10 across all twelve candidate runs. However, fixed-shape latency was 14.4% to 28.1% higher than plain MambaTS, so the candidate is not universally faster than Mamba architectures. The local baseline is an architecture reference, not a bitwise reproduction of the official MambaTS repository. [claim:C004,C005,C008,C009,C010] [evidence:E026,E027,E028]

The channel-state mixer provides a second, more targeted architecture screen. It clears the external-baseline gate on Electricity and Traffic but fails it on ETTh1, and its relation-active/null contrast is effectively flat. This narrows the mechanism claim: the useful component is the bounded channel-state interaction path in selected high-dimensional domains, while the current evidence does not isolate the relation objective as the source of the accuracy change.

The ETTm1 active/null control reinforces the same interpretation. AutoCon's raw contrastive loss is finite and nonzero in the active arm, and the recorded objective identity is exact within the audit tolerance; the lambda-zero arm still records the diagnostic loss but assigns it zero effective weight. The validation difference is therefore an implementation-level participation result with near-zero standalone predictive effect, not evidence for a universal contrastive-learning improvement.

The same frozen transfer was evaluated on ETTm2 without dataset-specific retuning. Across 24 validation-only jobs, the candidate won all three paired seeds against each of the seven baselines, with mean MSE 0.125532 and relative improvements from 0.924% against DLinear to 10.106% against Mamba-2. Its fixed-shape latency was 0.2746 ms, again faster than PatchTST, iTransformer, Mamba-1, TimesNet, and Mamba-2 but slower than DLinear/RLinear. The ETTm2 training audit confirms finite nonzero AutoCon loss and a maximum objective-identity residual of 4.48e-10. An exact ETTm2 active/null control then measured 0.124977 versus 0.125106, a -0.102% change with 2/3 wins and a maximum degradation of 0.038%. This is a small local auxiliary-objective signal, not evidence for universal AutoCon causality or held-out generalization. [claim:C004,C005,C008,C009] [evidence:E033,E034,E035,E036]

The ETTm2 result does not extend unchanged to long horizons. In a 48-job H336/H720 validation matrix with the same frozen architecture and seven baselines, candidate MSE is 0.219028/0.293225 and is 1.095%/1.913% above PatchTST, with 0/3 paired wins at each horizon. The long-horizon AutoCon audit retains finite nonzero losses and a maximum objective-identity residual of 1.01e-09. This makes the boundary explicit: the frozen candidate is locally competitive under the 2% non-inferiority tolerance, but has not demonstrated a long-horizon win over the strongest patch comparator and must not be promoted to a held-out long-forecast claim. [claim:C004,C005,C009] [evidence:E038,E039]

A separate same-source TimeMixer extension provides an architecture-specific accuracy-latency comparison across all three ETTm2 horizons. In 18 validation-only jobs, candidate MSE is 0.125868/0.220017/0.290159 against TimeMixer 0.142550/0.266009/0.301022 at H96/H336/H720, corresponding to 11.319%, 15.356%, and 3.320% relative improvements with 3/3, 3/3, and 2/3 paired wins. A serial manifest-bound RTX 5090 D protocol yields candidate latency 0.2741/0.2773/0.2760 ms versus 0.4979/0.5035/0.4862 ms for TimeMixer, and candidate parameter counts are lower at every horizon. The paired candidate training audit confirms finite nonzero AutoCon losses and an objective-identity residual at most 1.14e-09. This supports a narrow validation-only Pareto observation against TimeMixer, not a universal speed claim, held-out conclusion, or independent causal claim for AutoCon. [claim:C004,C005,C008,C009] [evidence:E040,E041,E042]

The same frozen contract was additionally evaluated against TimeMixer on the ETTh2 H96 boundary. Across six validation-only jobs, the candidate MSE is 0.223213 versus 0.229752 for TimeMixer, a 2.837% improvement with 3/3 paired wins. Serial fixed-shape RTX 5090 D prediction latency is 0.2772 ms for the candidate versus 0.4977 ms for TimeMixer, with 39,544 versus 70,135 trainable parameters. This is useful evidence that the candidate's local accuracy-latency advantage is not restricted to the ETTm2 frequency, but ETTh2 still loses to RLinear, PatchTST, and iTransformer by 10.90%, 9.84%, and 5.91% in the stronger baseline matrix. The ETTh2 result therefore strengthens a bounded TimeMixer comparison while preserving the cross-model negative boundary. [claim:C005,C008,C009] [evidence:E043,E044,E045]

An exact ETTh2 H96 active/null control further isolates the train-time objective. With the architecture and prediction path fixed, the active arm (lambda=0.001) reaches MSE 0.221769 versus 0.221790 for lambda zero, a 0.010% improvement with 3/3 paired wins. The effect is measurable but negligible relative to the candidate's broader architectural difference, so it supports implementation and local-mechanism claims rather than a universal causal claim for AutoCon. [claim:C004,C009] [evidence:E046]

The ETTh2 TimeMixer advantage does not persist to long horizons. In 12 validation-only H336/H720 jobs, candidate MSE is 0.372600/0.618063 against TimeMixer 0.357728/0.609351, corresponding to 4.157% and 1.438% degradation and 0/3 paired wins at both horizons. Although the candidate retains lower fixed-shape forward latency at 0.2713/0.2816 ms versus 0.4969/0.4885 ms, the H336 degradation violates the predeclared 2% non-inferiority boundary. This branch is therefore stopped and reported as a clear separation between local efficiency and long-horizon forecasting quality. [claim:C005,C008,C009] [evidence:E047,E048,E049]

### 3.3 Efficiency

On selected METR-LA and PEMS-BAY shapes, matched RTX 5090 D prediction-only measurements show lower candidate latency than the named TimesNet, Mamba2, and STGCN-style comparators. The candidate is not faster than DLinear or NLinear, and the measured memory advantage is not universal. The result therefore supports only a comparator- and shape-bounded efficiency statement. [claim:C008,C009] [evidence:E002,E016,E053]

The AGCRN-style extension closes a previously missing adaptive-graph accuracy baseline. In 12 validation-only jobs, the candidate is essentially tied on METR-LA (0.273466 versus 0.273197, +0.103%, with 1/3 paired wins) but is worse on PEMS-BAY (0.296674 versus 0.288887, +2.695%, with 0/3 wins). The candidate jobs are much shorter under the registered training implementation (27.7/86.6 s versus 1076.6/1697.8 s), but these are full training-job times rather than forward-latency measurements. A separate manifest-bound METR-LA `B=32,L=96,H=12` prediction-only measurement is 0.3240 versus 18.2249 ms, with 29,430 versus 72,258 trainable parameters. This fixed-shape resource advantage does not reverse the PEMS-BAY accuracy boundary. AGCRN is a local canonical-style implementation, so the comparison is a reproducible baseline boundary rather than an external bitwise reproduction. [claim:C007,C008,C009] [evidence:E053]

The completed DCRNN-style extension provides a second diffusion-recurrent graph baseline. Under three paired seeds per dataset, the candidate improves validation MSE by 5.287% on METR-LA and 27.444% on PEMS-BAY, winning all paired seeds in both cells. The local DCRNN implementation uses a semantics-preserving cache for input diffusion terms and batched matrix multiplication; a manifest-bound fixed-shape METR-LA measurement is 9.8622 ms for DCRNN versus 0.3252 ms for the candidate. The result is a complete validation boundary, not a held-out graph-model ranking or an external bitwise reproduction. [claim:C007,C008,C009] [evidence:E054]

The aligned Electricity/Traffic H96 resource bundle further narrows this claim. On both shapes the candidate is faster than PatchTST and TimesNet, but slower than DLinear, NLinear, RLinear, iTransformer, and Mamba2. The measurement uses actual validation configurations and the fused prediction path, but synthetic device-resident inputs and no relation-loss computation. It is therefore a forward-latency comparison, not an end-to-end deployment or training-throughput claim. [claim:C008,C009] [evidence:E011,E016,E023]

## 4. Discussion

The strongest evidence supports a narrow conclusion: an AutoCon-connected single-scan Mamba forecaster can be competitive on high-dimensional traffic and Weather tasks under the named protocols, while retaining a lower forward cost than several nonlinear comparators on selected shapes. The evidence does not support a universal claim across low-dimensional ETT-like datasets, all horizons, all graph models, or all hardware.

The validation-only architecture screens provide useful negative evidence. Multi-scale and channel-preserving branches either failed cross-domain accuracy gates or exceeded the resource budget. This suggests that adding Mamba branches or preserving full channel-state tensors is not automatically beneficial under the project efficiency objective.

## 5. Limitations and Open Gates

- GraphWaveNet held-out extension was stopped for thermal safety at 75C/68% fan during the first baseline job; its paired comparator phases are incomplete and no candidate test metric from that partial run is reportable. [claim:C013,C009] [evidence:E061]
- AGCRN and DCRNN now have complete validation-only accuracy matrices; both remain local canonical-style implementations rather than external bitwise reproductions. Neither is a held-out graph-model ranking.
- External repository implementations were reviewed for structure, but local Mamba variants are not bitwise reproductions.
- Five seeds provide paired uncertainty evidence but do not establish universal generalization.
- The ETTm1 stride-24 matrix and active/null control are validation-only; they do not authorize a new held-out release. [claim:C009] [evidence:E029,E032]
- The ETTm1 and ETTm2 stride-24 matrices are validation-only; only ETTm1 has an exact active/null control, and neither authorizes a new held-out release. [claim:C009] [evidence:E029,E032,E033,E036]
- The ETTm1 and ETTm2 stride-24 matrices and active/null controls are validation-only; neither authorizes a new held-out release. [claim:C009] [evidence:E029,E032,E033,E036]
- Author list, target venue, citation verification, reporting checklist, data/code release policy, and human scientific approval are unresolved.

## 6. Human Review Checklist

- [ ] Confirm the paper's target venue and its current author instructions.
- [ ] Verify every numeric statement against the linked release readout and generated main table.
- [x] Decide the GraphWaveNet held-out plan's safety boundary; retain it as an incomplete thermal-stop record, not a result.
- [x] Add locally validated bibliography entries for AutoCon, MambaTS, TimeMachine, ms-Mamba, and Mamba4Cast; retain source notes for human review.
- [ ] Add verified S-Mamba/TimesMamba metadata and format all references for the selected venue.
- [ ] Resolve the remaining AutoCon conference-pagination/style warning and add verified S-Mamba/TimesMamba metadata before venue formatting.
- [ ] Human-verify repository versions, licenses, and exact paper metadata from the [GitHub reference registry](github_mamba_reference_registry_20260928.md). [claim:C010] [evidence:E006]
- [ ] Reconcile all tables, figures, seeds, denominators, and metric definitions before submission.
- [ ] Add authorship, CRediT, funding, conflicts, data/code availability, and AI-use disclosures based on actual project records.

## Evidence Registry (Local)

| ID | Evidence | Role |
|---|---|---|
| E001 | `artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/readout/readout.md` | PEMS held-out results |
| E002 | `artifacts/paper_final/metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927/readout/readout.md` | METR-LA/PEMS-BAY held-out results |
| E003 | `artifacts/paper_final/mambatsv3_weather_test_release_20260922/readout/readout.md` | Weather held-out results |
| E004 | `artifacts/exploratory/paper_release_bundle_audit_20260930/audit.md` | Revalidated release integrity audit |
| E005 | `artifacts/exploratory/paper_main_table_20260930/main_table.md` | Revalidated machine-generated 52-row main table |
| E006 | `docs/research/paper_evidence_matrix_20260928.md` | Evidence boundaries and negative results |
| E007 | `docs/research/research_goal_completion_audit_20260928.md` | Requirement-level completion audit |
| E008 | `artifacts/exploratory/nlinear_variable_mamba_solar_h96_mechanism_strong_baseline_20260928/readout/readout.md` | Solar variable-axis Mamba validation and mechanism controls |
| E009 | `docs/research/mamba_acf_channel_mixer_electricity_h96_20260927_readout/readout.md` and `docs/research/mamba_acf_channel_mixer_multidataset_h96_20260927.md` | Rank-4 channel-state mixer validation and cross-dataset boundary |
| E010 | `artifacts/exploratory/mambatsv3_highdim_full_baseline_matrix_h96_validation_20260928/readout/readout.md` | Electricity/Traffic full same-source eight-arm validation matrix |
| E011 | `docs/research/mambatsv3_highdim_full_baseline_matrix_h96_resource_20260928.md` | Aligned Electricity/Traffic prediction-only resource measurements |
| E012 | `artifacts/exploratory/mambatsv3_highdim_autocon_mechanism_h96_validation_20260928/readout/readout.md` | Electricity/Traffic AutoCon fusion and relation-loss mechanism controls |
| E013 | `artifacts/exploratory/mambatsv3_highdim_relation_auxiliary_only_h96_validation_20260928/readout/readout.md` | Electricity/Traffic relation-auxiliary-only high-dimensional confirmation |
| E014 | `artifacts/exploratory/mambatsv3_relation_auxiliary_crossdomain_h96_validation_20260929/readout/readout.md` | Weather success-domain and ETTh1 failure-boundary cross-domain confirmation |
| E015 | `artifacts/exploratory/relation_auxiliary_autocon_training_audit_20260929/readout/readout.md` | Train-only AutoCon representation-learning objective audit |
| E016 | `artifacts/exploratory/mambatsv3_relation_auxiliary_crossdomain_h96_validation_20260929/resources/readout/readout.md` | Cross-domain manifest-bound prediction-only resource attestation |
| E017 | `artifacts/exploratory/mambatsv3_relation_auxiliary_solar_h96_validation_20260929/readout/readout.md` | Fresh-seed Solar relation-auxiliary-only transfer and failure boundary |
| E018 | `artifacts/exploratory/mambatsv3_mamba1_baseline_highdim_h96_validation_20260929/readout/readout.md` | Independent fresh-seed plain Mamba-1 architecture comparator |
| E019 | `artifacts/exploratory/mambatsv3_relation_auxiliary_mamba1_resource_h96_20260929/readout/readout.md` | Source-bound candidate versus plain Mamba-1 resource boundary |
| E020 | `artifacts/exploratory/mambatsv3_stride24_relation_auxiliary_h96_validation_20260929/readout/readout.md` | Stride-24 accuracy screen versus DLinear/PatchTST/iTransformer/Mamba-1 |
| E021 | `artifacts/exploratory/mambatsv3_stride24_relation_auxiliary_h96_resource_20260929/readout/readout.md` | Manifest-bound stride-24 fixed-shape resource attestation |
| E022 | `artifacts/exploratory/mambatsv3_stride24_full_baseline_h96_validation_20260929/readout/readout.md` | Complete stride-24 six-baseline validation matrix |
| E023 | `artifacts/exploratory/mambatsv3_stride24_full_baseline_resource_20260929/readout/readout.md` | Complete stride-24 six-baseline resource attestation |
| E024 | `artifacts/exploratory/mambatsv3_stride24_crossdomain_h96_readout_20260929/readout.md` | Stride-24 Weather success and ETTh1 failure-boundary validation matrix |
| E025 | `artifacts/exploratory/mambatsv3_stride24_crossdomain_resource_20260929/readout/readout.md` | Stride-24 cross-domain fixed-shape resource attestation |
| E026 | `artifacts/exploratory/mambatsv3_stride24_official_mambats_baseline_h96_readout_20260929/readout.md` | Stride-24 AutoCon-v3 versus plain MambaTS validation matrix |
| E027 | `artifacts/exploratory/mambatsv3_stride24_official_mambats_baseline_resource_20260929/readout/readout.md` | AutoCon-v3 versus plain MambaTS fixed-shape resource attestation |
| E028 | `artifacts/exploratory/mambatsv3_stride24_official_mambats_autocon_training_audit_20260929/readout.md` | Stride-24 candidate AutoCon objective participation audit |
| E029 | `artifacts/exploratory/mambatsv3_stride24_ettm1_h96_readout_20260929/readout.md` | ETTm1 H96 stride-24 seven-baseline validation matrix |
| E030 | `artifacts/exploratory/mambatsv3_stride24_ettm1_resource_20260929/readout/readout.md` | ETTm1 fixed-shape resource attestation |
| E031 | `artifacts/exploratory/mambatsv3_stride24_ettm1_training_audit_20260929/readout.md` | ETTm1 train-time AutoCon objective audit |
| E032 | `artifacts/exploratory/mambatsv3_stride24_ettm1_autocon_mechanism_h96_readout_20260929/readout.md` | ETTm1 exact AutoCon active versus lambda-zero mechanism control |
| E033 | `artifacts/exploratory/mambatsv3_stride24_ettm2_h96_readout_20260929/readout.md` | ETTm2 H96 stride-24 seven-baseline validation matrix |
| E034 | `artifacts/exploratory/mambatsv3_stride24_ettm2_resource_20260929/readout/readout.md` | ETTm2 fixed-shape resource attestation |
| E035 | `artifacts/exploratory/mambatsv3_stride24_ettm2_training_audit_20260929/readout.md` | ETTm2 train-time AutoCon objective audit |
| E036 | `artifacts/exploratory/mambatsv3_stride24_ettm2_autocon_mechanism_h96_readout_20260929/readout.md` | ETTm2 exact AutoCon active versus lambda-zero mechanism control |
| E037 | `docs/research/mambatsv3_stride24_ettm2_d16_resource_preflight_20260929.md` | ETTm2 lightweight d16 resource stop boundary |
| E038 | `artifacts/exploratory/mambatsv3_stride24_ettm2_long_h336_h720_readout_20260929/readout.md` | ETTm2 H336/H720 stride-24 seven-baseline validation matrix |
| E039 | `artifacts/exploratory/mambatsv3_stride24_ettm2_long_h336_h720_training_audit_20260929/readout.md` | ETTm2 long-horizon AutoCon objective audit |
| E040 | `artifacts/exploratory/mambatsv3_stride24_ettm2_timemixer_h96_h336_h720_readout_20260929/readout.md` | ETTm2 H96/H336/H720 TimeMixer paired validation matrix |
| E041 | `artifacts/exploratory/mambatsv3_stride24_ettm2_timemixer_resource_20260929/readout/readout.md` | ETTm2 TimeMixer serial fixed-shape resource attestation |
| E042 | `artifacts/exploratory/mambatsv3_stride24_ettm2_timemixer_training_audit_20260929/readout.md` | ETTm2 TimeMixer matrix AutoCon training audit |
| E043 | `artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_h96_readout_20260929/readout.md` | ETTh2 H96 TimeMixer paired validation matrix |
| E044 | `artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_resource_20260929/readout/readout.md` | ETTh2 TimeMixer serial fixed-shape resource attestation |
| E045 | `artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_training_audit_20260929/readout/readout.md` | ETTh2 candidate AutoCon training audit |
| E046 | `artifacts/exploratory/mambatsv3_stride24_etth2_autocon_mechanism_h96_readout_20260929/readout.md` | ETTh2 exact AutoCon active versus lambda-zero mechanism control |
| E047 | `artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_long_h336_h720_readout_20260929/readout.md` | ETTh2 H336/H720 TimeMixer paired validation matrix |
| E048 | `artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_long_resource_20260929/readout/readout.md` | ETTh2 long TimeMixer serial fixed-shape resource attestation |
| E049 | `artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_long_training_audit_20260929/readout/readout.md` | ETTh2 long candidate AutoCon training audit |
| E050 | `artifacts/exploratory/nlinear_variable_mamba_solar_timemixer_h96_readout_20260929/readout.md` | Solar H96 variable-axis Mamba versus TimeMixer paired validation matrix |
| E051 | `artifacts/exploratory/nlinear_variable_mamba_solar_timemixer_resource_attestation_20260929/readout.md` | Solar H96 manifest-bound fixed-shape resource attestation |
| E052 | `artifacts/exploratory/nlinear_variable_mamba_solar_timemixer_h96_training_audit_20260929/readout.md` | Solar variable-axis Mamba AutoCon objective participation audit |
| E053 | `artifacts/exploratory/metro_pemsbay_agcrn_mambatsv3_h12_readout_20260930/readout.md` | METR-LA/PEMS-BAY AGCRN-style paired validation matrix |
| E054 | `artifacts/exploratory/metro_pemsbay_dcrnn_mambatsv3_h12_readout_20260930/readout.md` | METR-LA/PEMS-BAY optimized DCRNN paired validation matrix |
| E055 | `artifacts/exploratory/mambatsv3_stride24_highdim_autocon_training_audit_20261001/readout/readout.md` | Stride-24 Electricity/Traffic AutoCon participation and loss-identity audit |
| E056 | `artifacts/exploratory/mambatsv3_stride24_default_path_equivalence_20261001/readout.md` | Frozen-snapshot versus current default Mamba-1 stride-24 path equivalence |
| E057 | `docs/research/highdim_candidate_tradeoff_review_20261001.md` | Stride-8 versus stride-24 high-dimensional accuracy-latency tradeoff |
| E058 | `docs/research/mambatsv3_stride24_relation_auxiliary_highdim_test_release_draft_20261001.json` | Unauthorized stride-24 held-out draft and its frozen plan boundary |
| E059 | `artifacts/paper_final/mambatsv3_stride24_relation_auxiliary_highdim_cpu4_test_release_20261001/readout/readout.md` | CPU-4 Electricity/Traffic H96 held-out readout, including the Traffic/iTransformer failure |
| E060 | `artifacts/paper_final/latent_temporal_mamba_highdim_autocon_validation_20261001/readout/readout.md` | LatentTemporalMamba active/null/reference/baseline validation readout |
| E061 | `docs/research/metro_pemsbay_graphwavenet_h12_thermal_stop_20261001.md` | GraphWaveNet held-out thermal stop and incomplete comparator boundary |

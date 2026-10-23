# GitHub Mamba Time-Series Reference Registry

本清单只记录公开仓库的结构参考入口，不把外部仓库的实验数字当作本地结果。外部仓库代码、许可证和版本在正式投稿前仍需由作者逐项核验。

| Repository | Public design signal | Local treatment | Boundary |
|---|---|---|---|
| [MambaTS-pytorch](https://github.com/XiudingCai/MambaTS-pytorch) | MambaTS long-term forecasting implementation with VPT/VAST-related variable/point ordering ideas | 本地 `MambaTS` v1--v16、VPT/VAST-lite 分支参考其高层结构 | 本地实现不是逐位复制；外部结果未写入本地主表 |
| [TimeMachine](https://github.com/Atik-Ahamed/TimeMachine) | Four-Mamba multi-rate forecasting design built on a PatchTST-style pipeline | 本地 TimeMachine-style baseline 与资源/验证筛选 | 本地适配了 AutoCon 接口，不能称为官方训练复现 |
| [Mamba4Cast](https://github.com/automl/Mamba4Cast) | One-pass zero-shot horizon prediction and synthetic-data pretraining | 作为一次性 horizon head 与效率讨论的结构参考 | 本仓库没有 zero-shot 预训练权重或 synthetic pretraining，因此不报告 Mamba4Cast 结果 |
| [ms-Mamba](https://github.com/meric-karadag/ms-mamba) | Hierarchical/multi-scale Mamba with fixed or adaptive temporal scales | 本地 v12/v13 多尺度候选参考并完成 validation/resource gate | v12/v13 未通过跨域或速度门槛，不进入主模型 |
| [awesome-mamba-ts](https://github.com/tamlhp/awesome-mamba-ts) | Survey/index of Pure Mamba, bidirectional/multi-directional and hybrid TS architectures | 用于结构谱系和候选筛选，不作为结果来源 | 汇总仓库不是实验实现，不能替代原始论文或官方代码核验 |

## External Repository Verification (2026-09-30)

The following facts were checked against the public repository pages on 2026-09-30. This is repository metadata verification, not a claim that the local code reproduces those projects.

| Repository | Verified repository signal | License / citation metadata | Local interpretation |
|---|---|---|---|
| [MambaTS-pytorch](https://github.com/XiudingCai/MambaTS-pytorch) | The README identifies `models/MambaTS.py` as the main architecture, points to VPT shuffle/unshuffle, and identifies VAST's variable-aware scan implementation | Public page exposes an MIT license and cites arXiv:2405.16440 | Local MambaTS v1--v16 borrow the design vocabulary but use an independent training and AutoCon contract |
| [TimeMachine](https://github.com/Atik-Ahamed/TimeMachine) | README identifies a four-Mamba long-term forecaster built on PatchTST and exposes `models/TimeMachine.py` | Public page exposes Apache-2.0 and arXiv:2403.09898 | Local TimeMachine-style code is an adapter/baseline, not the official training reproduction |
| [Mamba4Cast](https://github.com/automl/Mamba4Cast) | README describes synthetic-data training, zero-shot forecasting, and one-pass full-horizon prediction | Public page exposes MIT; local project does not use its synthetic pretraining or released checkpoints | It is a structural efficiency reference only, not a local baseline result |
| [ms-Mamba](https://github.com/meric-karadag/ms-mamba) | README describes parallel Mamba branches with fixed, learnable, or dynamic SSM sampling scales; it lists `Hier_S_Mamba` and `Adaptive_Hier_S_Mamba` | Public README cites Neurocomputing 680 (2026), DOI `10.1016/j.neucom.2026.133226`; repository page should be rechecked for a declared license before redistribution | Local v12/v13 multi-scale probes were tested under the repository's own budget and stopped by cross-domain/resource gates |

The verified repository contracts reinforce the manuscript boundary: external projects motivate hypotheses and implementation structure, while all reported numbers come from this repository's frozen local protocols. Before submission, authors must still check the exact commit, license text, and citation metadata they intend to cite.

## Reproducibility Notes

- Official repositories expose different training contracts, data preprocessing, and model sizes; direct numerical ranking without harmonizing these contracts would be invalid.
- The current paper's primary evidence comes only from local frozen releases under the repository's own data, seed, and streaming-metric contracts.
- AutoCon is a local auxiliary representation-learning objective and prediction-connected fusion path. It is not claimed to be part of the external Mamba repositories.
- The external references motivate architectural hypotheses; the local validation/resource screens decide whether a branch is retained.

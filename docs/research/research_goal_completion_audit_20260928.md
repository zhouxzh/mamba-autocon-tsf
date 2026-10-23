# Research Goal Completion Audit

_人工复核入口，2026-10-01 汇总。它只链接既有不可变读数，不产生新训练或 held-out 结果。_

---

## Current decision

项目已经拥有可撰写、但范围必须严格限定的论文证据：MambaTSv3AutoCon 在四组命名冻结 release 上完成 66 个 held-out 比较单元，其中 65 个通过注册的非劣效门；固定 GPU 形状上对指定非线性比较器存在效率优势。研究目标仍不能被解释为普适 SOTA：Traffic/iTransformer 是明确失败单元，LatentTemporalMamba 的精度验证失败，GraphWaveNet held-out 扩展因热保护中止，且没有证据支持跨域 AutoCon 独立因果增益。

| Goal requirement | Current evidence | Audit status | Claim boundary |
| --- | --- | --- | --- |
| More datasets | PEMS03/04/07/08、METR-LA、PEMS-BAY、Weather held-out；ETT、Electricity、Traffic、Exchange-Rate、ILI、Solar validation/failure coverage | Scoped evidence complete | Main test claims only cover the three frozen release families |
| More baselines | Linear, PatchTST, TimesNet, TimeMixer, iTransformer, Mamba-1/2, STGCN, GraphWaveNet, AGCRN and DCRNN | Coverage complete at mixed evidence levels | New graph baselines are local canonical-style validation evidence, not external bitwise reproductions |
| Public Mamba designs | MambaTS/VPT/VAST, TimeMachine, Mamba4Cast, S-Mamba/TimesMamba and ms-Mamba design signals examined; local mechanism implementations retained | Complete as design reference | Do not describe local models as copies of external repositories |
| AutoCon contrastive/representation learning | Main candidate has a real train-time relation objective; fusion releases connect the learned representation to forecasting | Complete as implementation and scoped release evidence | Independent accuracy gain from the relation objective is small or inconsistent outside named releases |
| Accuracy at least comparable | PEMS `28/28`, METR-LA/PEMS-BAY `12/12`, Weather `12/12` frozen comparison cells pass the registered 1.02 gate | Proven only for named releases | ETT, Exchange-Rate, ILI, Solar and AGCRN include negative boundaries |
| Faster computation | Fixed-shape RTX 5090 D forward evidence is favorable against selected nonlinear and graph comparators | Proven only for named shapes/comparators | Not faster than DLinear/NLinear in general; not a training-throughput claim |
| High-level-paper readiness | 66-cell main table, four-release provenance, numerical consistency, claim/evidence bindings and `842` regression tests are audited | Draftable, not submission-ready | Requires venue/statistics/license/authorship review and final human approval |

## Frozen evidence

| Release | Result | Primary source |
| --- | --- | --- |
| PEMS H12 | `28/28` non-inferior cells and `140` paired candidate wins | [PEMS readout](../../artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/readout/readout.md) |
| METR-LA/PEMS-BAY H12 | `12/12` non-inferior cells and `60` paired candidate wins | [Fusion readout](../../artifacts/paper_final/metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927/readout/readout.md) |
| Weather H96/H336/H720 | `12/12` non-inferior cells and `60` paired candidate wins | [Weather readout](../../artifacts/paper_final/mambatsv3_weather_test_release_20260922/readout/readout.md) |
| Release integrity | Four bundles and the regenerated 66-row table pass read-only provenance checks; 65/66 cells are non-inferior | [Release audit](../../artifacts/exploratory/paper_release_bundle_audit_20261001/audit.md) |
| Code and manuscript integrity | `842 passed, 45 warnings`; six local submission checks pass and retain human approval status | [Submission audit](../../artifacts/exploratory/submission_readiness_audit_20261001/audit.json) |

The defensible manuscript claim is limited to: under the registered protocols, AutoCon-connected MambaTSv3 is competitive on the named traffic and Weather held-out releases, with efficiency advantages only against named comparators at measured hardware shapes.

## Negative evidence retained

| Boundary | What the evidence says | Consequence |
| --- | --- | --- |
| AGCRN graph baseline | Candidate is `+0.103%` on METR-LA and `+2.695%` on PEMS-BAY against local AGCRN | Do not claim superiority over adaptive graph models |
| DCRNN graph baseline | Candidate improves validation MSE by `5.287%` on METR-LA and `27.444%` on PEMS-BAY, but DCRNN remains local validation-only | Include only as a bounded graph-baseline extension |
| Low-dimensional/long horizon | ETTh1/ETTh2 and several long-horizon coordinates lose to strong baselines | Do not claim domain- or horizon-independent superiority |
| Alternative Mamba mechanisms | VPT, multi-Delta, lightweight multi-scale, S-Mamba and RaMambaLite screens have explicit stop decisions | Do not revive a stopped branch from a favorable isolated coordinate |
| AutoCon isolation | Exact active/null studies show near-zero or inconsistent standalone gains on several datasets | Do not attribute release gains solely to contrastive learning |

Detailed coverage, including all stopped branches, is in the [baseline coverage dashboard](baseline_coverage_dashboard_20261001.md), [architecture selection record](candidate_architecture_selection_20260928.md), and [publication readiness audit](publication_readiness_audit_20260930.md).

## Decisions required

| Decision | Frozen scope | Current preflight | What approval permits | What it cannot establish |
| --- | --- | --- | --- | --- |
| GraphWaveNet held-out extension | METR-LA/PEMS-BAY H12; 46-job plan | Stopped at 75C/68% fan during the first GraphWaveNet baseline; comparator phases incomplete | No claim; retain thermal-stop record | Do not read partial candidate test metrics |
| High-dimensional held-out extension | Electricity/Traffic H96; stride-24 candidate and seven baselines; 86 jobs | Complete, 13/14 non-inferior cells; Traffic/iTransformer is the failure cell | Scoped high-dimensional held-out evidence | No universal speed or cross-domain AutoCon causality |

The stride-24 high-dimensional release has a completed sidecar-bound run and thermal/fan telemetry. The GraphWaveNet extension is explicitly stopped and cannot be resumed as a complete release. The necessary scope and risk records are in the [GraphWaveNet thermal stop](metro_pemsbay_graphwavenet_h12_thermal_stop_20261001.md) and [CPU-4 held-out readout](stride24_highdim_cpu4_heldout_readout_20261001.md).

## Human review checklist

- [x] Freeze the paper scope to the four completed held-out release families and exclude incomplete GraphWaveNet/Latent runs
- [ ] Verify that every paper claim uses the approved scope in the [claim/evidence registry](claim_evidence_registry_20261001.md)
- [ ] Resolve the AutoCon bibliographic warning and independently check venue, license and authorship requirements
- [ ] Review statistical reporting and the distinction between held-out, validation-only and resource-only results
- [ ] Keep graph baselines labeled as local canonical-style implementations

## Evidence map

- [Current model and experiment status](mamba_current_status.md)
- [Publication readiness audit](publication_readiness_audit_20260930.md)
- [Claim/evidence registry](claim_evidence_registry_20261001.md)
- [Main manuscript draft](manuscript_draft_20260928.md)
- [High-dimensional held-out review](highdim_heldout_review_20260929.md)

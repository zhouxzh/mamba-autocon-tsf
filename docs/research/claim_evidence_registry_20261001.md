# Claim-Evidence Registry (2026-10-01)

This registry binds manuscript-level claims to local evidence. A claim may be used only within the scope stated here. `Held-out` and `validation-only` are not interchangeable.

| ID | Claim | Scope | Evidence | Status |
|---|---|---|---|---|
| C001 | MambaTSv3AutoCon is non-inferior to registered baselines on the PEMS H12 release | PEMS03/04/07/08, H12, five paired held-out seeds, 28 cells | `artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/readout/readout.md`; release audit | Allowed, held-out scoped |
| C002 | Prediction-connected AutoCon fusion improves over same-source plain MambaTSv3 on METR-LA and PEMS-BAY | H12, five paired held-out seeds, two non-graph traffic datasets | `artifacts/paper_final/metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927/readout/readout.md` | Allowed, release scoped |
| C003 | MambaTSv3AutoCon is non-inferior across Weather horizons | Weather H96/H336/H720, five paired held-out seeds | `artifacts/paper_final/mambatsv3_weather_test_release_20260922/readout/readout.md`; release audit | Allowed, Weather scoped |
| C004 | The train-only AutoCon relation objective participates in optimization | Registered candidate training logs with finite nonzero loss and objective identity | `artifacts/exploratory/relation_auxiliary_autocon_training_audit_20260929/readout/readout.md`; `artifacts/exploratory/mambatsv3_stride24_highdim_autocon_training_audit_20261001/readout/readout.md` | Allowed, implementation-level only |
| C005 | Candidate accuracy and efficiency can beat selected nonlinear baselines on high-dimensional validation domains | Electricity/Traffic/Weather validation matrices; fixed RTX 5090 D shapes | `artifacts/exploratory/mambatsv3_stride24_full_baseline_h96_validation_20260929/readout/readout.md`; resource attestations; `docs/research/highdim_candidate_tradeoff_review_20261001.md` | Allowed, validation/resource scoped |
| C006 | Solar variable-axis Mamba can outperform TimeMixer on a fresh paired validation matrix | Solar H96, three paired seeds, one source snapshot | `artifacts/exploratory/nlinear_variable_mamba_solar_timemixer_h96_readout_20260929/readout.md` | Allowed, validation-only |
| C007 | Candidate remains competitive against canonical-style graph baselines at validation level | METR-LA/PEMS-BAY H12; STGCN, GraphWaveNet, AGCRN, optimized DCRNN local implementations | `artifacts/exploratory/metro_pemsbay_stgcn_mambatsv3_h12_validation_20260928/readout/readout.md`; `artifacts/exploratory/metro_pemsbay_graphwavenet_mambatsv3_h12_validation_20260928/readout/readout.md`; `artifacts/exploratory/metro_pemsbay_agcrn_mambatsv3_h12_readout_20260930/readout.md`; `artifacts/exploratory/metro_pemsbay_dcrnn_mambatsv3_h12_readout_20260930/readout.md` | Allowed, local validation only |
| C008 | Candidate is faster than selected graph/nonlinear comparators on named fixed shapes | Synthetic device-resident RTX 5090 D prediction-only protocol | Resource attestations for METR-LA/PEMS-BAY, Solar and ETT matrices | Allowed, fixed-shape only |
| C009 | Candidate is not uniformly superior to all baselines or domains | ETTh1/ETTh2 long, Exchange-Rate, ILI, Solar and AGCRN boundaries | `docs/research/paper_evidence_matrix_20260928.md`; `docs/research/publication_readiness_audit_20260930.md` | Required limitation |
| C010 | Public Mamba repositories are architectural references, not bitwise local reproductions | MambaTS, TimeMachine, Mamba4Cast, ms-Mamba | `docs/research/github_mamba_reference_registry_20260928.md`; `docs/research/references/citation_verification_20260930.md` | Required limitation |
| C011 | CPU-4 high-dimensional held-out release is non-inferior in 13/14 registered cells | Electricity/Traffic H96, five paired held-out seeds; Traffic/iTransformer is the required failure cell | `artifacts/paper_final/mambatsv3_stride24_relation_auxiliary_highdim_cpu4_test_release_20261001/readout/readout.md`; `docs/research/stride24_highdim_cpu4_heldout_readout_20261001.md` | Allowed, release scoped |
| C012 | LatentTemporalMamba is a negative validation result despite lower fixed-shape latency | Electricity/Traffic H96, three paired validation seeds; active-vs-null and external-baseline failures | `artifacts/paper_final/latent_temporal_mamba_highdim_autocon_validation_20261001/readout/readout.md`; `docs/research/latent_temporal_mamba_highdim_validation_20261001.md` | Required negative boundary |
| C013 | GraphWaveNet held-out extension was stopped before paired comparator completion for thermal safety | METR-LA/PEMS-BAY H12; selection and candidate completed, graph baseline stopped at 75C/68% fan | `docs/research/metro_pemsbay_graphwavenet_h12_thermal_stop_20261001.md`; GPU telemetry | Required execution limitation |

## Prohibited Rephrasings

- Do not replace `validation-only` with `test`, `held-out`, or `confirmatory`.
- Do not replace “selected nonlinear comparators” with “all baselines”.
- Do not convert a nonzero AutoCon loss into a causal accuracy claim.
- Do not convert fixed-shape forward latency into training throughput, end-to-end latency, or hardware-independent speed.
- Do not call local STGCN, GraphWaveNet, AGCRN or DCRNN implementations official reproductions.

## Human Review Fields

- Human verifier: unresolved
- Verification date: unresolved
- Target venue and reporting style: unresolved
- Final author approval: unresolved
- Numeric-prose binding audit: [`manuscript_claim_binding_audit_20261001.md`](manuscript_claim_binding_audit_20261001.md)

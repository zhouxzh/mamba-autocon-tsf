# Corrected Evidence Status

**Updated:** 2026-09-01. This page is a concise historical review index; frozen artifacts remain the source of truth. The current MambaACF boundary is summarized in [`mamba_current_status.md`](../research/mamba_current_status.md); [`mamba_acf_core_status_20260901.md`](../research/mamba_acf_core_status_20260901.md) is a historical snapshot.

## Audit Boundary (2026-08-31)

The `corrected_*` artifacts are complete and reproducible under their immutable
snapshots, but those snapshots predate the current legacy ACF identity/out-of-
range masks, the AutoCon projection contract, and the CI channel-wise global
relation fix. Plain forecasting rows are numerically unaffected by these
auxiliary-loss changes. Every row that trains an AutoCon objective must be
reselected and retrained before it is used as a current-method paper claim;
the tables below remain historical diagnostic readouts until then.

## Decision Status

| Evidence | Status | What it supports |
|---|---|---|
| Corrected ETT-S | 450/450 validated | Descriptive paired ablation evidence only |
| Corrected ETT-M | 450/450 validated | Descriptive paired ablation evidence only |
| M resource benchmark | Complete | Architecture/resource measurements, not forecast effects |
| Electricity/Traffic quality gate | Passed | Data readiness, not forecast effects |
| Real-data GPU smoke | Complete | Implementation gate only; not forecast effects |
| High-dimensional selection (`_v3`) | 378/378 validated | Validation-only lambda selection; its attestation remains `posthoc_verified` and `publication_ready=false` |
| High-dimensional confirmatory (`_v3`) | 270/270 validated (deduplicated final records) | Frozen Electricity/Traffic readout; posterior replay evidence, not launch-time preregistered confirmation |

## Corrected M

[`phase_validation.json`](../../artifacts/paper_final/corrected/confirmatory_M/phase_validation.json) records `valid=true`, 450 expected and 450 unique/`ok` jobs, with no validator or malformed-manifest errors. [`provenance.json`](../../artifacts/paper_final/corrected/confirmatory_M/provenance.json) records `provenance_id=8bb6120ba4d85497` and selection-binding SHA256 `615b31637653cd0eb0ad7037a5adfb75071e0b5503e388b2ca063e45de5799d1`.

The selection sidecar is deliberately qualified: it is `posthoc_verified` and `publication_ready=false`. Its pre-launch replay binds the selected settings, but it is not evidence that the original selection was launch-time preregistered.

[`paper_statistics.json`](../../artifacts/paper_final/corrected/confirmatory_M/statistics/paper_statistics.json) covers 450 validated results and uses seed-level paired tests with Holm and BH-FDR correction. For MSE, AutoCon improved 8/10 Mamba-1 cells (mean relative change `-0.77%`; 0 Holm/FDR cells) and 7/10 Mamba-2 cells (`-2.38%`; 1 Holm and 1 FDR cell). These are heterogeneous descriptive results, not a universal-effect claim.

## High-Dimensional v3 Boundary

The high-dimensional artifacts are present and internally validated. [`highdim_selection_M_v3/phase_validation.json`](../../artifacts/paper_final/corrected/highdim_selection_M_v3/phase_validation.json) records `valid=true`, `expected_jobs=378`, `manifest_unique_jobs=378`, and `status_counts.ok=378`. [`highdim_confirmatory_M_v3/phase_validation.json`](../../artifacts/paper_final/corrected/highdim_confirmatory_M_v3/phase_validation.json) records `valid=true`, `expected_jobs=270`, `manifest_unique_jobs=270`, `status_counts.ok=270`, `malformed_manifest_lines=0`, and empty `errors`/`record_errors`. The confirmatory manifest has 280 attempt records because ten initial `Mamba2AutoCon` attempts failed with `OSError: [Errno 36] File name too long`; each of those ten job IDs has a later `ok` record, so the validator's final deduplicated coverage is 270/270.

The confirmatory provenance binds to the selection replay with `binding_mode=prelaunch_replay_of_posthoc_attestation`. The selection sidecar is explicitly `attestation_mode=posthoc_verified` and `publication_ready=false` ([`provenance.json`](../../artifacts/paper_final/corrected/highdim_confirmatory_M_v3/provenance.json), [`selected_hyperparameters.provenance.json`](../../artifacts/paper_final/corrected/highdim_selection_M_v3/selected_hyperparameters.provenance.json)). Thus the 270/270 table is a complete, reproducible posterior replay of the selected settings; it must not be described as an originally launch-time preregistered or publication-ready confirmatory result.

## Remaining Boundary

The resource audit has 45 de-duplicated architecture/resource records and 90 elapsed-time groups. The Electricity and Traffic quality artifacts pass their integrity/cadence checks. The old no-suffix and `_v2` high-dimensional directories remain runtime/pre-compatibility diagnostics; they are not the source of the completed `_v3` tables. The `_v3` selection and confirmatory artifacts are complete as files and pass the validator, but their posthoc selection provenance prevents a formal launch-time confirmatory claim. Mechanism ablations, capacity-fairness analyses, and (for any new MambaACF claim) an independently frozen multi-seed protocol remain required before final paper claims.

Primary review entry: [`../manual-review.md`](../manual-review.md). Historical `strict` evidence remains separate in [`strict-evidence-status.md`](strict-evidence-status.md).

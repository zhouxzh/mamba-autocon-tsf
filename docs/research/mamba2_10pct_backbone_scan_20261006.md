# Mamba-2 10% Few-Shot Backbone Scan

Status: completed validation-only backbone comparison. No new test reader was
run, and all target training uses the declared chronological 10% prefix.

## Fixed comparison

This scan relaxes the Mamba-1-only backbone choice while preserving the
successful transfer protocol: 24/8 patches, `d_model=24`, `d_ff=48`, one
shared channel-agnostic scan, source next-patch prediction plus train-only ACF
forecast-contrastive relations, frozen transferred target core, five target
epochs, and the existing official PatchTST validation checkpoints. Mamba-2
uses `headdim=12`; the only intended architecture change is the selective-scan
family.

PEMS-BAY retains its fixed PEMS07/PEMS08/PEMS04 source pool. METR-LA uses the
previously frozen 10%-prefix ACF source selector (PEMS-BAY/PEMS07/PEMS08).
The target selector did not read validation or test rows. Each domain uses the
same Mamba-1 seed coordinates and an exact Mamba-2 relation-null control.

| Target | Mamba-2 MSE | Mamba-1 MSE | Mamba-2 vs Mamba-1 | Wins | One-SE upper | Mamba-2 vs null | Mamba-2 vs official PatchTST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192, 10% | 2.753417 | 2.780617 | -0.978% | 3/3 | 0.99462 | -3.219%, 3/3 | -1.423%, 3/3 |
| METR-LA H192, 10% | 1.579254 | 1.587743 | -0.535% | 2/3 | 0.99772 | -0.541%, 3/3 | -4.509%, 3/3 |

All quoted one-standard-error bounds are ratios below one. The result is
therefore a repeatable validation signal that the Mamba-2 family can produce a
small low-label accuracy gain under this protocol.

## Decision

Do not replace the Mamba-1 release backbone. The accuracy margin is below 1%
on both coordinates, while the matched PEMS-BAY device-resident inference
benchmark is 0.766 ms and 354.8 MB for Mamba-2 versus 0.333 ms and 25.8 MB for
Mamba-1. Mamba-2 is a validation-only accuracy-oriented alternative, not the
main efficiency/accuracy submission candidate.

No Mamba-2 test release is justified: its head width and source-selection
rule were chosen through earlier exploratory validation, and the present scan
only resolves whether the broader backbone family merits consideration. The
frozen Mamba-1 test release remains the sole test-facing result.

## Audit and Resource Record

- Audit: `artifacts/exploratory/mambatsv3_mamba2_10pct_backbone_scan_20261006/audit.json`
- Registered plan: `artifacts/exploratory/mambatsv3_mamba2_10pct_backbone_scan_20261006/plan.json`
- Every stage was serialized under `run_gpu_guarded.py` with a 78C temperature
  and 85% fan limit. PEMS-BAY peaked at 63C/46%; the correctly paired METR-LA
  rerun peaked at 61C/40%. No guard triggered.

An earlier METR-LA exploratory subrun used PEMS seed IDs `5960--5962` rather
than METR-LA's frozen Mamba-1 IDs `5970--5972`. It is intentionally excluded
from `audit.json` and from all statements above; the paired rerun is the only
METR-LA result used here.

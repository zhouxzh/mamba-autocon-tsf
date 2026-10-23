# Mamba-2 Backbone Family Scan: Validation Readout

Status: exploratory validation only. No new test metric was produced or read.

## Fixed protocol

The scan retains the existing channel-count-agnostic shared patch encoder,
train-prefix ACF positive/negative self-supervision, frozen target core, and
single scan per channel patch sequence. The only backbone-family change is
`mamba_ts_mixer_type=mamba2`. The strongest screened Mamba-2 configuration has
`d_model=24`, `headdim=12` (two state-space heads), `patch_len=24`, and
`patch_stride=8`.

For every PEMS-BAY run, source pretraining uses PEMS07/PEMS08/PEMS04 and the
target is excluded from that source pool. The target is PEMS-BAY H192 at 25%
labels. Source-pretrained encoder tensors are copied into the target and remain
frozen; the target forecast head, NLinear head, and residual scale train. The
external comparator is the pinned official PatchTST commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9` under the same chronological split,
budget, and 24/8 patch geometry.

## PEMS-BAY results

| Mixer | Mamba-2 heads | Active MSE | Null MSE | Active vs null | Official PatchTST MSE | Active vs PatchTST | Active wins vs PatchTST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mamba-2 | 1 (`headdim=24`) | 2.356788 | 2.445844 | -3.641% | 2.357927 | -0.048% | 2/3 |
| Mamba-2 | 2 (`headdim=12`) | 2.352223 | 2.431258 | -3.251% | 2.377603 | -1.067% | 2/3 |

The two-head branch has a paired mean active/PatchTST ratio of `0.989512`; its
one-standard-error upper ratio is `0.999726`. It therefore clears the local
one-SE noninferiority screen. All three active/null pairs favor the active arm.

This does not establish an external SOTA claim. `headdim=12` was chosen after
viewing validation evidence, and two additional seeds were then added after
the first positive seed. The result is a candidate-selection signal, not an
independently preregistered confirmation.

Detailed auditable artifacts:

- `artifacts/exploratory/mambatsv3_pems_mamba2_twohead_probe_20261006/official_patchtst_mamba2_validation_audit.json`
- `artifacts/exploratory/official_patchtst_mamba2_patch24_comparator_20261006/`

## Robustness readout

The selected two-head PEMS-BAY checkpoints were evaluated without training on
the chronological validation block under six deterministic stresses. Every
cell is paired by seed with the matched null checkpoint. All values remain
validation-only and are still selection-contaminated by the earlier backbone
search.

| Condition | Active vs null | Active wins | One-SE ratio upper bound |
| --- | ---: | ---: | ---: |
| In-distribution | -3.251% | 3/3 | 0.9692 |
| Later validation origins | -2.884% | 3/3 | 0.9728 |
| 10% masked channels | -2.960% | 3/3 | 0.9720 |
| 30% masked channels | -2.321% | 3/3 | 0.9780 |
| Relative noise 0.05 | -3.258% | 3/3 | 0.9691 |
| Relative noise 0.10 | -3.277% | 3/3 | 0.9689 |

Zero-masking is applied after target-train normalization, so absolute MSE may
occasionally decline under masking. The valid comparison is the paired active
versus null contrast in each condition, not a claim that missing sensors make
forecasting easier. The complete machine-readable audit is
`artifacts/exploratory/mambatsv3_pems_mamba2_twohead_probe_20261006/robustness_validation/audit.json`.

## Inference path

On device-resident synthetic PEMS-BAY-shaped input (batch 4, 50 warm-up and
200 timed CUDA iterations), the active checkpoint ran with one Mamba encoder
block. Runtime hooks observed exactly 250 block calls for 250 prediction calls;
no AutoCon representation or relation loss was constructed. Median latency was
`0.766 ms`, with 104,100 parameters and 354.8 MB peak allocated memory.

Under the same batch and timing budget, same-geometry official PatchTST had a
`0.254 ms` median latency, 56,617 parameters, and 25.2 MB peak allocated
memory. Therefore the evidence proves that AutoCon adds no inference-time
scan/loss to this Mamba-2 candidate, but it does **not** support a claim of
latency or memory advantage over PatchTST.

## Cross-dataset source selection

The original METR-LA H192, 25%-label probe used PEMS-BAY/PEMS07/PEMS08 as its
source pool and failed internally: active MSE `1.189902` versus matched-null
`1.188965` (`+0.079%`). This negative result was retained rather than hidden.
Its train-only target ACF signature had a different peak structure from PEMS07.

The replacement source-selection procedure uses only the METR-LA few-shot
training prefix, never target validation/test rows. It ranks source training
prefixes by cosine similarity of the mean signed ACF profile and selects
PEMS-BAY (`0.793`), Weather (`0.737`), and PEMS08 (`0.592`). Replacing only
PEMS07 with Weather changed the three-seed result to:

| Target | Labels | Active MSE | Null MSE | Active vs null | Official PatchTST MSE | Active vs PatchTST | Wins vs PatchTST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| METR-LA H192 | 25% | 1.184482 | 1.192603 | -0.681% | 1.255006 | -5.619% | 3/3 |
| Weather H192 | 10% | 4.500666 | 4.522286 | -0.478% | 4.636977 | -2.940% | 3/3 |

METR-LA's paired active/PatchTST one-standard-error ratio upper bound is
`0.94785`; Weather's is `0.97826`. Both satisfy the local validation
noninferiority screen. Source selection and later seeds were chosen after
validation evidence, so these remain exploratory selection results and cannot
justify a held-out test or a general transfer claim.

METR-LA stress results also retain active/null improvements in all six
validation conditions: `-0.49%` under later origins, `-0.61%/-0.47%` under
10%/30% masked channels, and `-0.68%/-0.69%` under 0.05/0.10 relative noise;
all are 3/3 active wins. The audit is
`artifacts/exploratory/mambatsv3_metr_mamba2_selected_sources_probe_20261006/robustness_validation/audit.json`.

## Resource safety

Every newly run training stage used `run_gpu_guarded.py` with a 78C temperature
limit and an 85% fan limit, with `AUTOCON_CPU_THREAD_LIMIT=4`. No guard event
triggered. Observed maxima were:

| Workload | Max temperature | Max fan |
| --- | ---: | ---: |
| PEMS-BAY Mamba-2 single-head scan | 60C | 41% |
| PEMS-BAY Mamba-2 two-head scan | 60C | 40% |
| METR-LA Mamba-2 two-head probe | 57C | 33% |

## Decision

Treat the fixed PEMS-only source pool as an explicit METR-LA negative boundary.
The next valid step is a separately registered, source-selection-frozen
replication across PEMS-BAY, METR-LA, and Weather. Do not open any additional
test partition for this adaptive search.

# Independent Weather External Robustness Replication

Status: independently registered five-seed, validation-only replication. It
uses seeds `5991--5995`, which are not pooled with the prior three-seed
extension. No test row was read.

The frozen Mamba-1 channel-agnostic ACF-pretraining protocol and locked
official PatchTST adapter were retrained under the same Weather H192, 10%
chronological protocol. After validation selection, both checkpoint sets were
evaluated under the same late-origin, missing-channel, and signal-relative
noise perturbations.

| Condition | Active vs null | Active vs official PatchTST | PatchTST upper ratio | Active wins vs PatchTST |
| --- | ---: | ---: | ---: | ---: |
| In distribution | -0.580%, 5/5 | -2.343% | 0.98549 | 4/5 |
| Late shift | -0.634%, 5/5 | -2.682% | 0.98184 | 5/5 |
| Missing channels 10% | -0.412%, 5/5 | -1.710% | 0.98997 | 4/5 |
| Missing channels 30% | -0.249%, 5/5 | -0.984% | 0.99443 | 4/5 |
| Noise 0.05 | -0.585%, 5/5 | -2.347% | 0.98553 | 4/5 |
| Noise 0.10 | -0.589%, 5/5 | -2.355% | 0.98553 | 4/5 |

All six cells satisfy the one-standard-error noninferiority criterion against
official PatchTST. This resolves the prior three-seed Weather extension's
near-threshold upper-bound misses with a separately registered replication,
not a pooled or post-hoc combined estimate.

The evidence remains validation-only and model-selection-free: it does not
authorize a universal robustness claim, a new test read, or a claim of
hardware-independent performance.

Artifacts:

- `artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006/audit.json`
- `artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006/robustness_validation/external_patchtst_audit.json`
- `scripts/run_mambatsv3_weather_mamba1_10pct_independent_replication.py`
- `scripts/run_weather_mamba1_10pct_independent_robustness.py`
- `scripts/run_official_patchtst_weather_mamba1_10pct_independent_replication.py`
- `scripts/run_official_patchtst_weather_mamba1_10pct_independent_robustness.py`

All training and read-only stress stages used the 78C/85% guard. Candidate
evaluation peaked at 54C; official PatchTST stress evaluation peaked at 43C.

# External Robustness Against Official PatchTST

Status: post-hoc validation-only external robustness extension. Candidate and
official PatchTST checkpoints were trained and validation-selected before this
read-only stress evaluation. No test rows were opened.

The locked official PatchTST checkpoints were source-hash checked and evaluated
with the same chronological validation origins, few-shot normalization, seeds,
channel masking, and signal-relative noise protocol as the frozen Mamba-1
candidate. Each cell contains three paired training seeds. Lower MSE is better.

| Target | Late shift | Missing 10% | Missing 30% | Noise 0.05 | Noise 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192, 10% | -0.502%, upper .99843 | -0.406%, upper .99930 | -0.332%, upper .99948 | -0.502%, upper .99879 | -0.655%, upper .99719 |
| METR-LA H192, 10% | -3.934%, upper .96817 | -3.594%, upper .97103 | -2.782%, upper .97757 | -4.025%, upper .96747 | -4.108%, upper .96665 |
| Weather H192, 10% | -1.709%, upper .99574 | -0.823%, upper 1.00110 | -0.544%, upper 1.00076 | -1.188%, upper 1.00044 | -1.188%, upper 1.00046 |

PEMS-BAY and METR-LA satisfy the declared one-standard-error noninferiority
criterion in every listed condition, including in-distribution evaluation.
METR-LA wins all three seeds in every cell. The initial PEMS-BAY three-seed
extension was close to the threshold, but a separately registered five-seed
replication passes every cell with five wins; see
`mamba1_pems_independent_external_robustness_20261006.md`.

The initial three-seed Weather extension had lower mean MSE in every condition
but four marginal upper-bound misses. A separately registered five-seed Weather
replication now passes the strict external noninferiority gate in all six
conditions; see `mamba1_weather_independent_external_robustness_20261006.md`.
The two seed sets remain separate rather than being pooled post hoc.

This extension strengthens the robustness section for the two traffic targets,
but it is not a test release, does not establish universal robustness, and did
not select the model or its hyperparameters.

Artifacts:

- `artifacts/exploratory/official_patchtst_mamba1_10pct_robustness_20261006/audit.json`
- `scripts/evaluate_official_patchtst_robustness.py`
- `scripts/run_official_patchtst_mamba1_10pct_robustness.py`
- `scripts/audit_mambatsv3_mamba1_10pct_external_robustness.py`

All read-only evaluations ran under the 78C/85% GPU guard. Observed maxima
were 54C with no fan escalation above 32%.

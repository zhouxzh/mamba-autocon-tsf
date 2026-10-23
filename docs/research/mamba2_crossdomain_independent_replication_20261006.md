# Independent Cross-Domain Replication: Mamba-2 AutoCon Pretraining

Status: frozen validation replication. No test metric was produced or read.

After the exploratory architecture and source-pool search, the following were
frozen: Mamba-2 `headdim=12`, one encoder block, 24/8 patch geometry,
channel-count-agnostic shared scan, next-patch plus 10% ACF forecast-
contrastive pretraining, frozen target core, and official PatchTST commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9`. New seeds `5910--5912` were then
run without changing any target, baseline, or source rule.

## Accuracy Replication

| Target | Labels | Active MSE | Null MSE | Active vs null | Official PatchTST | Active vs PatchTST | PatchTST wins | One-SE ratio upper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192 | 25% | 2.363911 | 2.448366 | -3.449% | 2.364131 | -0.009% | 1/3 | 1.00421 |
| METR-LA H192 | 25% | 1.186842 | 1.193368 | -0.547% | 1.257184 | -5.595% | 3/3 | 0.94904 |
| Weather H192 | 10% | 4.497464 | 4.529186 | -0.700% | 4.553164 | -1.223% | 3/3 | 0.99177 |

The active arm beats the matched null in all nine seed pairs. METR-LA and
Weather are independently non-inferior to official PatchTST at the local
one-SE gate. PEMS-BAY retains a strong active/null effect but is statistically
indistinguishable from PatchTST under this strict small-sample gate; it must not
be presented as a confirmed PatchTST win.

The METR-LA source pool was frozen from a target few-shot train-prefix ACF
selector before this replication: PEMS-BAY, Weather, PEMS08. PEMS-BAY and
Weather retain their previously registered fixed source pools. No validation or
test target rows were used to modify these rules during replication.

Artifacts:

- `artifacts/exploratory/mambatsv3_mamba2_crossdomain_replication_20261006/`
- `artifacts/exploratory/official_patchtst_mamba2_crossdomain_replication_20261006/`

## Interpretation

The independent evidence supports a narrower, defensible claim: train-only ACF
relations can select transferable source dynamics, and the resulting frozen
shared Mamba representation improves low-label target adaptation over a matched
null on three domains. It does not establish universal SOTA, because PEMS-BAY
does not clear the strict external non-inferiority upper bound and no test
partition has been opened for this replication.

The reported effect is not attributable to an extra inference-time loss: the
deployed path returns prediction only, uses one Mamba scan, and computes no
AutoCon relation objective. The candidate remains slower than official
PatchTST in the named batch-four benchmark, so speed should be reported as a
single-scan/inference-path property rather than a universal latency advantage.

## Independent Robustness

The same new seeds were evaluated on the validation block under later origins,
10%/30% channel masking, and relative noise 0.05/0.10. All 18 active/null
cells are complete, validation-only, and have 3/3 active wins. Representative
active-versus-null changes are:

| Target | Later origins | 10% mask | 30% mask | Noise 0.05 | Noise 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY | -3.11% | -3.13% | -2.49% | -3.45% | -3.46% |
| METR-LA | -0.37% | -0.49% | -0.38% | -0.55% | -0.54% |
| Weather | -0.74% | -0.53% | -0.30% | -0.70% | -0.70% |

Every cell's one-standard-error ratio upper bound is below 1.0. The complete
machine-readable audit is
`artifacts/exploratory/mambatsv3_mamba2_crossdomain_replication_20261006/robustness_validation/audit.json`.

This is the strongest current evidence for the stated research question, but it
does not convert validation results into a publication claim: the test
partition remains closed and the source-selection rule was designed from an
earlier exploratory validation stage.

The parallel Mamba-1 efficiency arm was independently screened under the same
rules. It achieved PEMS-BAY active/null `-2.90%` and METR-LA active/null
`-0.75%` across three new seeds, with a measured PEMS-BAY median latency of
`0.333 ms` versus Mamba-2's `0.766 ms`. This supports reporting the Mamba-1 /
Mamba-2 accuracy-efficiency trade-off rather than silently replacing the
pre-registered Mamba-2 accuracy results.

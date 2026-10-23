# ACF Transfer Compatibility Analysis

The frozen transfer evidence suggests that target/source compatibility is a
more useful research variable than another Mamba architectural variant. The
score below is the mean similarity of the three sources selected from the
target's declared few-shot train prefix. It is computed before validation/test
access and is not a test-set feature.

| Target | Labels | Mean source similarity | Active vs null |
| --- | ---: | ---: | ---: |
| PEMS-BAY | 10% | 0.866 | -1.425% |
| METR-LA | 10% | 0.811 | -0.169% |
| Weather | 10% | 0.566 | -0.211% |
| Electricity | 25% | 0.373 | +2.146% |
| Traffic | 25% | 0.343 | +0.263% |

Across these five target-level points, the descriptive Pearson correlation
between compatibility and active-vs-null gain is `r=-0.765`. This is not an
inferential result: the sample is tiny, labels/horizons are not fully matched,
and the negative domains were intentionally retained as boundaries. The value
of this analysis is a testable paper hypothesis:

> train-prefix ACF/source compatibility predicts whether contrastive
> pretraining transfers, while low compatibility predicts null or negative
> transfer.

PEMS03 is a subsequent external high-compatibility gate: its selected-source
mean is 0.993, but the fixed three-seed validation gate is mixed (two active
wins and one loss), so its test partition remains closed. This falsifies a
deterministic-threshold interpretation. Compatibility should be treated as a
risk signal whose incremental predictive value needs a larger preregistered
domain set, not as a sufficient condition for guaranteed transfer.

The next experiment should freeze a compatibility threshold and evaluate new
domains or new chronological slices without changing the model after seeing
the outcome. No further Mamba block, channel mixer, or auxiliary inference
branch is justified by the current evidence.

Machine-readable artifact: [`compatibility.json`](../../artifacts/exploratory/acf_transfer_compatibility_analysis_20261003/compatibility.json).

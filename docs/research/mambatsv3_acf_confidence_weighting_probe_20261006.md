# ACF-Confidence-Weighted Positive Pair Probe

Status: completed validation-only negative mechanism result. No target test
reader was executed.

The normal forecast-level InfoNCE treats each train-prefix ACF-selected
positive pair equally. This probe retained its same signed ACF labels, source
windows, Mamba-1 24/8 shared encoder, frozen target core, 10% target prefix,
and `relation_loss_fraction=0.10`, but weighted each positive log-probability
by its train-only `|ACF(lag)|`. The per-anchor normalization preserves the
overall loss scale; this is a pair-reliability change, not a larger auxiliary
weight. It creates no inference parameter, representation branch, or Mamba
scan.

| Target | Null MSE | Plain InfoNCE | Confidence weighted | Weighted vs plain | Weighted vs null |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192, 10% | 2.847047 | 2.765934 | 2.766164 | +0.008% | -2.841% |
| METR-LA H192, 10% | 1.572933 | 1.571584 | 1.572428 | +0.054% | -0.032% |

The relation anchor fraction was at least 0.8 in every active source epoch.
Although the weighted arm remains no worse than its exact null, it loses to
the ordinary active arm on both targets. The cross-domain selection gate
therefore fails, and no new seed replication or test evaluation is authorized.

The result rejects the simple hypothesis that raw ACF magnitude is a reliable
cross-domain confidence score for positive-pair weighting. Future mechanisms
must use a distinct pre-registered notion of relation reliability rather than
iterating magnitude transforms.

Artifacts:

- `artifacts/exploratory/mambatsv3_mamba1_acf_confidence_probe_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_mamba1_acf_confidence_probe_20261006/audit.json`
- `scripts/run_mambatsv3_mamba1_acf_confidence_probe.py`
- `scripts/audit_mambatsv3_mamba1_acf_confidence_probe.py`

All stages ran serially under the 78C/85% guard. Peak temperature was 54C on
PEMS-BAY and 51C on METR-LA; no guard event triggered.

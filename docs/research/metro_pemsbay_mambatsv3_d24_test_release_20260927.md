# METR-LA / PEMS-BAY MambaTSv3 AutoCon H12 Held-Out Release

## Protocol

This is a frozen `L=96,H=12` release of the d24 causal MambaTSv3 AutoCon candidate on two timestamped, non-graph traffic-speed datasets. The selection phase used three seeds and no test metrics; candidate and six-baseline test phases used five fresh paired seeds. All `76/76` jobs completed, all three strict phase validators passed, and the test metrics were read only after validation.

## Held-out accuracy

Mean streaming scaled MSE and one-sided 95% paired upper MSE-ratio bounds are shown below. The registered non-inferiority margin is `1.02`.

| Dataset | Candidate | PatchTST | TimeMixer | Plain MambaTSv3 | iTransformer | Primary result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| METR-LA | 0.432489 | 0.427167 | 0.420330 | 0.430351 | 0.438699 | Fails PatchTST/TimeMixer non-inferiority |
| PEMS-BAY | 0.344955 | 0.403969 | 0.375111 | 0.347258 | 0.358730 | Passes all registered comparators |

On METR-LA, candidate versus PatchTST is `+1.248%`, 0/5 wins, upper ratio `1.0203`; versus TimeMixer it is `+2.900%`, 0/5 wins, upper ratio `1.0394`. The candidate remains non-inferior to DLinear, NLinear, iTransformer and plain MambaTSv3, but that is insufficient for the primary claim.

On PEMS-BAY, candidate improves over PatchTST by `14.602%`, TimeMixer by `7.999%`, iTransformer by `3.840%`, and plain MambaTSv3 by `0.665%`; the candidate wins 4/5 against plain MambaTSv3 and 5/5 against the other listed baselines. All registered one-sided upper bounds are below `1.02`.

## Matched prediction-only resources

RTX 5090 D, FP32, device-resident `B=32,L=96,H=12`, 50 warm-up and 200 timed forwards:

| Dataset | Candidate | PatchTST | TimeMixer | Plain MambaTSv3 | iTransformer |
| --- | ---: | ---: | ---: | ---: | ---: |
| METR-LA | 0.245 ms | 0.363 ms | 0.996 ms | 0.246 ms | 0.220 ms |
| PEMS-BAY | 0.268 ms | 0.498 ms | 1.327 ms | 0.270 ms | 0.331 ms |

The candidate is faster than PatchTST and TimeMixer on both datasets, and faster than iTransformer only on PEMS-BAY. It is not faster than DLinear/NLinear, and METR-LA is slightly slower than iTransformer. These are named-comparator forward measurements, not universal efficiency evidence.

## Final interpretation

The independent-data confirmation supports a narrow PEMS-BAY H12 result and a mixed METR-LA result. It does not support a two-dataset primary claim, universal Mamba superiority, long-horizon generalization, graph-aware traffic forecasting, or an independent AutoCon accuracy effect. AutoCon active/null validation evidence was non-inferior but inconsistent; the held-out candidate gain must be attributed to the frozen MambaTSv3 architecture plus its training protocol, not to the contrastive loss alone.

The high-level-paper objective is therefore not yet achieved. The next valid research step is a new pre-registered mechanism that improves the METR-LA strong-baseline gap or a narrower paper explicitly centered on PEMS-BAY/short-horizon efficiency, rather than releasing more selective test comparisons.

Machine-readable paired results are in [`paired_test_summary.csv`](metro_pemsbay_mambatsv3_d24_test_release_20260927_readout/paired_test_summary.csv); the immutable confirmation plan is [`metro_pemsbay_mambatsv3_d24_test_release_20260927.json`](metro_pemsbay_mambatsv3_d24_test_release_20260927.json).

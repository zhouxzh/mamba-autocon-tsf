# MambaTS v3 Weather Held-Out Test Release (2026-09-22)

## Frozen Protocol

This single-dataset held-out confirmation was registered before execution in
[`mambatsv3_weather_test_release_20260922.json`](mambatsv3_weather_test_release_20260922.json).
MambaTS v3 AutoCon (`lambda=0.001`) was fixed before test access. The matrix
contains Weather H96/H336/H720, five fresh seeds (2031--2035), and four fixed
nonlinear baselines: PatchTST, TimesNet, iTransformer, and Mamba-2. All arms
use MeanStd, sequence length 96, batch size 32, a shared 20-epoch maximum,
validation-only early stopping, and streaming held-out metrics.

Both strict phase validators pass: candidate `15/15` and baseline `60/60`
jobs, all `ok`. Each phase has an enforced launch-sidecar binding, immutable
source snapshot, dataset hash, config/checkpoint/test-artifact hashes, and no
missing seed-horizon cell. The machine-readable readout is
[`readout.md`](../../artifacts/paper_final/mambatsv3_weather_test_release_20260922/readout/readout.md).

## Held-Out Accuracy

The table gives mean streaming scaled MSE across five paired seeds. The upper
ratio is the frozen one-sided 95% paired-t upper bound for candidate/baseline
MSE; it must be at most `1.02` for the registered Weather non-inferiority gate.
Every cell passes and the candidate wins all five seed pairs.

| H | Baseline | Candidate MSE | Baseline MSE | Change | Upper ratio | Gate |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 96 | PatchTST | `0.189144` | `0.207067` | `-8.650%` | `0.9254` | pass |
| 96 | TimesNet | `0.189144` | `0.205104` | `-7.697%` | `0.9531` | pass |
| 96 | iTransformer | `0.189144` | `0.208245` | `-9.164%` | `0.9218` | pass |
| 96 | Mamba-2 | `0.189144` | `0.212480` | `-10.748%` | `0.9431` | pass |
| 336 | PatchTST | `0.317257` | `0.323388` | `-1.896%` | `0.9830` | pass |
| 336 | TimesNet | `0.317257` | `0.370926` | `-14.225%` | `0.9080` | pass |
| 336 | iTransformer | `0.317257` | `0.327574` | `-3.149%` | `0.9725` | pass |
| 336 | Mamba-2 | `0.317257` | `0.358467` | `-11.429%` | `0.9129` | pass |
| 720 | PatchTST | `0.416618` | `0.419095` | `-0.591%` | `0.9963` | pass |
| 720 | TimesNet | `0.416618` | `0.445624` | `-6.471%` | `0.9559` | pass |
| 720 | iTransformer | `0.416618` | `0.423124` | `-1.537%` | `0.9884` | pass |
| 720 | Mamba-2 | `0.416618` | `0.461379` | `-9.674%` | `0.9215` | pass |

This supports a narrow conclusion: under this frozen Weather protocol,
MambaTS v3 AutoCon is non-inferior to and descriptively lower-MSE than the
four registered nonlinear baselines. It is not a cross-dataset generalization
claim. The `lambda=0` null remains validation-only mechanism evidence: it
improved H96 but did not improve H336/H720 consistently, so this test release
does not claim a stable independent AutoCon treatment effect.

## Efficiency Evidence

Prediction-only measurements use matched Weather shapes, FP32 device-resident
inputs, batch 32, 50 warm-ups, and 200 timed forwards. Candidate median latency
is `0.269/0.266/0.272 ms` for H96/H336/H720, versus PatchTST
`0.278/0.280/0.280 ms`, TimesNet `2.795/4.174/8.474 ms`, Mamba-2
`0.693/0.684/0.694 ms`, and iTransformer `0.150/0.152/0.150 ms`.

The candidate is faster than PatchTST, TimesNet, and Mamba-2 at all registered
shapes, but slower than iTransformer. Its trainable parameters are
`62,053/170,053/342,853` for H96/H336/H720, more than PatchTST and Mamba-2.
The supported claim is therefore latency against named slower nonlinear
comparators, not parameter efficiency or universal speed.

Manifest end-to-end job wall time under the same 20-epoch maximum/test protocol
is candidate `34.17/32.23/34.29 s`; PatchTST `40.94/51.59/55.74 s`; TimesNet
`108.48/143.36/221.16 s`; iTransformer `38.39/42.63/49.72 s`; and Mamba-2
`39.92/39.91/44.39 s`. This includes loading, early stopping, and testing on
one hardware setup per seed, not a hardware-independent throughput estimate.

## Publication Gap

The Weather evidence is stronger than the prior validation-only screen but is
not enough for a high-level general forecasting paper. The next confirmation
must use new data families, fresh seeds, a frozen cross-dataset matrix, and
multiplicity-aware inference. A paper that claims contrastive representation
learning itself improves accuracy also needs a stable active-versus-null effect.

# Mamba-1 / Mamba-2 Accuracy-Efficiency Trade-off

The project now has two explicitly separated backbone arms under the same
channel-agnostic ACF-pretraining and frozen-core transfer protocol.

| Backbone | PEMS-BAY active/null | METR-LA active/null | PEMS-BAY median latency | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: |
| Mamba-1, 24/8 | -2.90%, 3/3 | -0.75%, 3/3 | 0.333 ms | 25.8 MB |
| Mamba-2, `headdim=12`, 24/8 | -3.45%, 3/3 | -0.55%, 3/3 | 0.766 ms | 354.8 MB |

The Mamba-1 PEMS-BAY independent replication also beats official PatchTST by
`1.064%` (3/3; one-SE ratio upper `0.9942`). The corresponding Mamba-2
independent PEMS-BAY replication is statistically tied with PatchTST under the
strict one-SE gate (`-0.009%`, 1/3 wins, upper `1.0042`). Mamba-2 remains the
stronger external-accuracy arm on METR-LA and Weather, while Mamba-1 is the
more defensible low-latency arm.

Both arms run one Mamba encoder scan at inference and construct no AutoCon
loss. The correct paper framing is therefore an accuracy-efficiency trade-off,
not a claim that one mixer dominates every criterion. All numbers here are
validation-only; the test partition remains closed.

A later matched 10%-label backbone scan confirms the same trade-off on the
few-shot coordinates: Mamba-2 improves MSE over Mamba-1 by `0.978%` (3/3) on
PEMS-BAY and `0.535%` (2/3) on METR-LA, while retaining active/null and
official-PatchTST validation gains. These sub-1% gains do not justify replacing
the Mamba-1 release backbone given the measured latency and memory gap. The
scan is validation-only and has no Mamba-2 test release; see
`mamba2_10pct_backbone_scan_20261006.md`.

The later frozen Mamba-1 test release supersedes the last sentence for the
Mamba-1 arm: PEMS-BAY, METR-LA, and Weather all have read-only active/PatchTST
test results with 3/3 candidate wins. The Mamba-2 arm remains validation-only.
See `mambatsv3_mamba1_frozen_test_release_20261006.md` for the scoped release
and its historical-test-access caveat.

Electricity remains a negative boundary even for the optimized Mamba-1 arm:
the overlap-patch, Traffic-source Electricity probe obtained validation MSE
8.24315 versus 8.14374 for its exact null (+1.22%). No Electricity test metric
was opened. Traffic likewise remains a historical validation failure domain.

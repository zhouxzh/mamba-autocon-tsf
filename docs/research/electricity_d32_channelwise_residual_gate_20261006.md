# Electricity d32/24 Channelwise Residual Gate

Status: independently replicated validation improvement with a near-tie,
but external noninferiority failure. No test rows were read.

## Mechanism

The shared channel-count-agnostic MambaTS v3 core remains frozen. The target
hybrid normally uses one scalar to blend the NLinear primary forecast and the
existing Mamba correction. Electricity has 321 heterogeneous variables, so
the scalar was replaced only during target adaptation by a zero-initialized
target-channel scale of shape `[1, 1, C]`. It does not introduce a second
Mamba scan, channel identity into source pretraining, or an inference-time
AutoCon loss.

## Replication

The d32/24 Electricity bridge uses Traffic source pretraining, Electricity
25% calibration, H96, frozen shared core, `d_model=32`, `d_ff=64`, and one
Mamba-1 scan. Across fresh seeds `6047--6049`, the channelwise gate improves
the exact scalar-gate control by `-3.362%`, winning all three pairs with a
one-SE ratio upper of `0.96746`.

| Seed | Scalar gate | Channelwise gate |
| ---: | ---: | ---: |
| 6047 | 8.054045 | 7.800090 |
| 6048 | 8.154670 | 7.875600 |
| 6049 | 8.111496 | 7.826887 |

## External Comparison

Against same-seed official PatchTST (`d_model=32`, `d_ff=64`, patch `24/24`,
same label and target-window budget), the initial three seeds are nearly tied
but not noninferior: channelwise gate `7.834192`, PatchTST `7.814428`,
`+0.253%`, 1/3 wins, one-SE upper `1.00489`.

A pre-registered two-seed expansion (`6050--6051`) gives five-seed means of
`7.830562` versus `7.825424` (`+0.066%`, 2/5 wins, upper `1.00255`). This
remains an external noninferiority failure. It demonstrates that the gate
substantially closes the prior high-dimensional gap, but does not justify an
external-SOTA claim.

A final capped five-seed expansion yields ten matched pairs: channelwise gate
`7.845240` versus PatchTST `7.823808` (`+0.274%`, 3/10 wins, upper `1.00509`).
The external noninferiority failure therefore persists with more precision; no
further gate seed expansion is allowed.

## Closed Follow-ups

- A horizon/channel-specific additive output bias degrades the gate by
  `+0.142%` on its registered seed.
- A horizon/channel-specific correction scale degrades the channelwise gate
  by `+0.207%`.
- A fair 10-epoch screen selects the channelwise gate's epoch-five MSE
  `7.800090`; later epochs overfit while official PatchTST continues improving
  to `7.407932`. More training is not a remedy.

The target-channel residual scale is retained as the strongest high-dimensional
adaptation mechanism. The bias, horizon-specific gate, and extended-budget
variants are stopped. High-dimensional robustness and test work remain closed
until a separately registered external validation gain is obtained.

## Shared Statistic Gate Boundary

A one-seed registered screen replaced all channel-indexed gate parameters with
one shared rank-eight MLP. For every channel it consumes only the current
normalized window's last value, mean, standard deviation, and endpoint trend.
The output layer is zero-initialized, so the initial path is exactly the same
linear-primary forecast; it remains one Mamba scan and has no channel IDs.

The gate became active (final absolute output `0.08103`) but produced MSE
`8.266289`, versus `7.800090` for the matching learned channelwise gate and
`8.054045` for the scalar gate. This is `+5.977%` versus the channelwise gate
and fails the pre-registered `-0.25%` expansion threshold. It is therefore a
closed high-dimensional architecture branch, with no seed expansion, external
comparator, robustness run, or test read.

## External-Domain Boundary

Applying the same target-channel gate to the PEMS-BAY 10% channel-patch
checkpoint degrades validation MSE from `2.708263` to `2.775566` (`+2.485%`).
The learned gate mean is negative on PEMS-BAY, whereas it is positive on the
Electricity bridge. Thus this is not a new cross-domain default: retain it as
an Electricity-specific target calibration mechanism and keep the scalar gate
for the shared PEMS/METR/Weather transfer protocol.

## Traffic Boundary

The symmetric Traffic d32/24 bridge remains relation-negative after applying
the target-channel gate to both its active and null pretrained cores. Active
gate MSE is `0.814971` versus null gate `0.811806` (`+0.390%`), although the
gate slightly improves the plain active bridge (`0.816451`). The gate therefore
improves target fitting in both arms but does not recover the AutoCon causal
effect. No Traffic replication, external comparator, robustness matrix, or
test reader is justified from this failed gate.

Artifacts:

- `artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006/audit.json`
- `artifacts/exploratory/electricity_d32_channelwise_gate_expansion_20261006/five_seed_audit.json`
- `artifacts/exploratory/electricity_d32_channelwise_gate_epoch10_screen_20261006/audit.json`

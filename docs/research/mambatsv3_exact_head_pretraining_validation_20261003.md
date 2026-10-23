# Exact-Head AutoCon Pretraining Validation

## Mechanism

The retained candidate uses the existing single MambaTSv3 Mamba-1 block once
per channel patch sequence. Channel embeddings and cross-channel context are
bypassed, making the shared encoder permutation-equivariant and independent of
sensor count. Source pretraining predicts the next 96-point segment through a
shared `P*d_model -> 96` head. Train-only ACF lag-pair consistency is applied
directly to paired predicted segments, not to an auxiliary hidden feature.

The source head is saved and restored into the target H96 forecast head only
when its geometry is exact. This avoids the rejected H24-to-H96 tiled-head
transfer. Target adaptation uses 10% or 25% of the target chronological
training prefix, then selects on the existing chronological validation block.
Neither target test partition is opened.

## Initial Validation Evidence

| Target | Source exclusion | Target fraction | Initial MSE | Exact-head pretraining MSE | Change |
| --- | --- | ---: | ---: | ---: |
| Traffic | Traffic | 10% | .893987 | .880236 | -1.539% |
| Weather | Weather | 10% | 4.447756 | 4.302628 | -3.263% |
| Traffic | Traffic | 25% | .888374 | .881089 | -0.820% |
| Weather | Weather | 25% | 2.820773 | 2.740773 | -2.836% |

The first row is the development condition. Weather was evaluated using a
fresh leave-Weather-out seven-source pool, so it is an independent validation
gate. All rows currently use one registered seed per target and are
validation-only; they are insufficient for a publication claim until fresh
paired seed replications and strong baselines are completed.

## Validation Stress Evidence

The saved 10% target checkpoints were evaluated only on their validation
blocks. The same fixed channel mask/noise seed was used within each target
pair. In both Traffic and Weather, the pretrained checkpoint had lower MSE in
the later validation half, 10%/30% channel masking, and 5%/10%
signal-relative Gaussian noise. These are one-seed validation observations,
not robustness claims.

## Prediction-Only Inference

The deployed benchmark explicitly calls `return_repr=False`, returns no
AutoCon representation, constructs no relation loss, and uses one Mamba
block. On RTX 5090 D, FP32, device-resident `B=8,L=96,H=96`, 25 warm-up and
100 timed forwards, median latency was `.5839 ms` for Traffic (`C=862`) and
`.2463 ms` for Weather (`C=21`). These are fixed-shape forward measurements,
not end-to-end throughput claims.

## Next Gate

Run a fresh second paired source/target seed for the same two 10% conditions.
Only if both targets retain non-inferiority to their initial-core controls may
the candidate advance to a three-seed validation matrix and strong-baseline
comparison. Target test splits remain closed throughout this stage.

## Three-Seed Update And Mechanism Boundary

The exact-head transfer itself replicated across three seeds. Relative to a
new target MambaTSv3 head, Traffic improved on two of three paired seeds and
averaged `-0.916%`; Weather improved on all three and averaged `-2.259%`.
The one-standard-error paired ratio upper estimates were `.9958` and `.9826`,
respectively, below the predeclared 1.02 non-inferiority bound.

However, this does not establish an AutoCon effect. Exact lambda-zero source
pretraining was compared with forecast-patch AutoCon at fixed architecture,
source pools, head geometry, source epochs, target split, and target seed.
With the original fixed `lambda=.1`, active-minus-null was `+.015%` on
Traffic and `-.002%` on Weather. A calibrated version made the weighted
forecast-pair loss roughly `8.3--9.9%` of next-segment prediction loss with
`93.8--96.3%` pair coverage; it changed Traffic by `-.014%` and worsened
Weather by `+.150%`. Therefore the current low-data gain is attributable to
cross-domain predictive head/core pretraining, not specifically to the ACF
relation objective.

A DLinear-primary, zero-gated one-scan hybrid was also tested to meet strong
low-data baselines. It beat DLinear on all Traffic seeds by `-.290%` mean,
but lost to DLinear/NLinear on Weather by `+.424%/+1.184%`. The hybrid is a
useful local mechanism result, not a cross-domain publication candidate.

## ACF-Selected Sources And Strong Baselines

The all-source affine relation treatment improved Traffic but degraded Weather.
A frozen train-prefix ACF signature rule selected the top three source domains
for each target: Electricity/Weather/METR-LA for Traffic, and
METR-LA/PEMS-BAY/Electricity for Weather. The selected-source affine hybrid
beat DLinear on two fresh Traffic seeds by `-.753%/-.958%` and on two fresh
Weather seeds by `-.244%/-.209%`. This is still
validation-only and source selection uses no target validation/test values.

The broader three-seed low-data baseline matrix shows the current boundary:
the DLinear-primary hybrid beats DLinear on Traffic by `-.257%` mean, but the
same all-source hybrid loses to Weather DLinear/NLinear by `+.424%/+1.184%`.
iTransformer is weaker in these local low-data runs. These are comparator
context, not a held-out release.

## Final Source-Selection Boundary

Train-prefix-only ACF source selection found a stable primary linear choice in
three original seeds: DLinear for Traffic and NLinear for Weather. The
top-three selected-source affine hybrid beat DLinear on the two fresh Traffic
seeds and on the two fresh Weather seeds. Yet a further fresh end-to-end
validation, where the primary was first selected by an inner calibration tail
and then the full few-shot prefix was used for main validation, produced only
`-.072%` on Traffic and `-.016%` on Weather. The Mamba residual gates were
near zero (`+.0097` and `-.0025`).

This fails the material-effect requirement. It means the selected-source
signal is not stable enough to claim a general AutoCon pretraining mechanism.
No 25% selected-source matrix, broader stress matrix, or held-out target test
is authorized for this candidate. The correct interpretation is a bounded
validation finding, not a SCI-Q1-ready method.

Artifacts:

- `artifacts/exploratory/mambatsv3_exact_head_traffic_20261003`
- `artifacts/exploratory/mambatsv3_exact_head_weather_gate_20261003`
- `artifacts/exploratory/mambatsv3_exact_head_25pct_20261003`
- `artifacts/exploratory/mambatsv3_exact_head_robustness_20261003`
- `artifacts/exploratory/mambatsv3_exact_head_inference_retry_20261003`

# MambaTSv3 Global Channel Mixer H12 Screen

## Decision

Stop the global channel-state mixer branch.  It fails the pre-registered METR-LA accuracy gate on all three paired seeds, so it must not be tuned further or promoted to a held-out test release.

## Scope and Audit Status

This is a frozen validation-only screen over METR-LA and PEMS-BAY at `L=96`, `H=12`, with seeds `2411--2413`.  The off/on arms use the same fast MambaTSv3 AutoCon backbone (`d_model=24`, `d_ff=48`, `patch_len=stride=24`, `d_state=8`, `d_conv=2`, `expand=1`, `AutoCon_lambda=.001`); the only architecture difference is the rank-4 global channel-state mixer.  It derives a mean/RMS descriptor from the observed window, maps it through a low-rank adapter, and applies a ReZero FiLM residual to the shared Mamba state before both the forecast and AutoCon representation heads.

All phase validators passed: control `6/6`, treatment `6/6`, external baselines `24/24`.  There are `36` `training_complete.json` artifacts and zero `metrics.npy` files, so no test result was generated.  The treatment phase is bound to source snapshot `921bb2eecccb5fc631d6d3e3849bb11ea3525a8e8e93bfc6d487b51afd0b5da1`; see the immutable phase validation and manifest at [treatment validation](../../artifacts/exploratory/metro_pemsbay_mambatsv3_global_channel_mixer_screen_20260927/mambatsv3_global_channel_mixer_on_h12/phase_validation.json).

## Paired Validation Result

| Dataset | Control MSE | GCM MSE | GCM vs control | Paired wins | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| METR-LA | .281822 | .288517 | +2.376% | 0/3 | Fail |
| PEMS-BAY | .305140 | .304363 | -0.255% | 3/3 | Pass |

METR-LA per-seed control/treatment MSE is `.283785/.288073`, `.281284/.291744`, and `.280395/.285734`.  The treatment violates the required `>=1%` improvement, `>=2/3` paired wins, and `<=2%` per-seed degradation thresholds.  Its learned ReZero scales are `-.1249`, `-.1229`, and `.0983`, which confirms that the zero-initialized mixer received gradient and changed the executed forecast/representation path; the negative result is architectural, not a dormant-module artifact.

PEMS-BAY improves slightly in all three paired seeds, but this cannot compensate for the METR-LA failure or establish a cross-dataset claim.

## Same-Budget Context

| Dataset | Strongest baseline | MSE | GCM gap |
| --- | --- | ---: | ---: |
| METR-LA | MambaTSv3 | .282963 | +1.963% |
| PEMS-BAY | MambaTSv3 | .305095 | -0.240% |

The GCM stays inside the exploratory 2% external-context boundary on METR-LA only narrowly, but it fails the more important exact off/on causal comparison.  PatchTST, TimeMixer, and iTransformer results were retained as same-budget descriptive rows and must not be mixed with the earlier held-out release.

## Resource Boundary

The exact-shape RTX 5090 D, FP32, prediction-only preflight (`B=32`, 50 warm-up, 200 timed forwards; representation return disabled) measured:

| Dataset | MambaTSv3 control | GCM | GCM overhead |
| --- | ---: | ---: | ---: |
| METR-LA | .249 ms | .342 ms | +37.3% |
| PEMS-BAY | .263 ms | .336 ms | +28.0% |

The GCM remains faster than the earlier exact-shape PatchTST and TimeMixer measurements, but slower than ordinary MambaTSv3.  Since it also fails METR-LA accuracy, no speed claim should be made for this branch.  Resource artifacts are in [the preflight directory](../../artifacts/exploratory/metro_pemsbay_mambatsv3_global_mixer_preflight_20260927/).

## Consequence

Do not launch held-out test, rank/lambda follow-ups, or larger-horizon runs for this mixer.  The remaining publication path still requires a new mechanism that improves METR-LA under paired validation without sacrificing the fast shared-scan MambaTSv3 route, plus a separately demonstrated AutoCon effect.

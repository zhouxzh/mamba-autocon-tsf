# Staged Core Adaptation on Electricity

Status: validation-only negative adaptation probe. No target test reader ran.

The Electricity H96 strong-calibration protocol previously freezes the
transferred channel-agnostic core during target adaptation. This probe retains
the same one-scan Mamba-1 24/8 architecture, source/calibration data,
active/null pretraining checkpoints, target split, and five target epochs. It
changes only target adaptation: one frozen warm-up epoch followed by a shared
core learning rate of `1e-4` against target parameter learning rate `1e-3`.

| Arm | Staged-core validation MSE | Frozen-core validation MSE | Change |
| --- | ---: | ---: | ---: |
| Active | 8.256705 | 8.205705 | +0.621% |
| Null | 8.267606 | 8.249465 | +0.220% |

Active remains marginally better than its staged null (`-0.132%`), but both
staged models are worse in absolute validation MSE than the fully frozen
counterparts. Thus low-rate post-warm-up core adaptation does not close the
substantial external PatchTST gap and is not extended to additional learning
rates, seeds, or test evaluation.

The run preserves one Mamba scan and does not modify the source AutoCon
positive/negative construction. Guarded adaptation peaked at 46C with no fan
escalation.

Artifacts:

- `artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_staged_core_probe_20261006/audit.json`
- `scripts/run_mambatsv3_electricity_strong_calibration_staged_core_probe.py`
- `scripts/audit_mambatsv3_electricity_strong_calibration_staged_core_probe.py`

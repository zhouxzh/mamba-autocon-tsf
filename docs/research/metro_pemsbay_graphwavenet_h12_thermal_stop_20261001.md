# GraphWaveNet H12 Held-out Thermal Stop

_The registered 46-job GraphWaveNet extension was stopped by proactive thermal protection before a paired comparison could be completed._

---

## 🌡️ Stop event

The release started under the GPU guard with hard limits of `78C` and `85%` fan. Selection completed `6/6` and candidate held-out jobs completed `10/10`. The first GraphWaveNet baseline job then sustained `74--75C`, `67--68%` fan, approximately `510--528W`, `98%` GPU utilization, and `3269MB` memory. The process group was proactively terminated at `75C`, before the hard guard threshold.

GPU utilization subsequently returned to zero and temperature fell to `49C`. The guard event records a user-safety termination return code, not a model failure.

## 📋 Reporting boundary

GraphWaveNet `0/10` and TimesNet/Mamba2 context baselines `0/20` are incomplete. Candidate test artifacts must not be read, reported, or combined with any graph baseline result because all paired comparator cells are absent. The release plan is now marked `stopped_gpu_thermal_protection` and unauthorized.

The prior three-seed validation-only GraphWaveNet result remains the only usable graph-wave baseline evidence. A future attempt would need a separately registered low-power/low-batch protocol and new authorization; it cannot resume this held-out release as if it were complete.

## 🔗 Evidence

- [Stopped plan](metro_pemsbay_graphwavenet_mambatsv3_h12_test_release_draft_20260928.json)
- [GPU guard event](../../artifacts/paper_final/metro_pemsbay_graphwavenet_mambatsv3_h12_test_release_20260928/gpu_guard/event.json)
- [GPU telemetry](../../artifacts/paper_final/metro_pemsbay_graphwavenet_mambatsv3_h12_test_release_20260928/gpu_guard/telemetry.csv)

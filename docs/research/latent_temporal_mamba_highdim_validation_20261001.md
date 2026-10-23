# Latent Temporal Mamba High-Dimensional Validation: Stop

_Completed CPU-4, validation-only Electricity/Traffic H96 matrix; no test split was read._

---

## 📋 Result

The registered 60-job matrix completed with four valid phase validators: latent-temporal lambda-zero `6/6`, latent-temporal active AutoCon `6/6`, stride-24 MambaTSv3 reference `6/6`, and seven strong baselines `42/42`. The strict reader verified 60 unique rows, one immutable source snapshot, and zero test metrics or prediction artifacts.

| Dataset | Contrast | Active MSE | Comparator MSE | Active change | Active wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Electricity H96 | Active vs lambda-zero | .201894 | .183284 | +10.186% | 0/3 |
| Traffic H96 | Active vs lambda-zero | .506252 | .476057 | +6.343% | 0/3 |
| Electricity H96 | Active vs stride-24 MambaTSv3 | .201894 | .165645 | +21.891% | 0/3 |
| Traffic H96 | Active vs stride-24 MambaTSv3 | .506252 | .377784 | +34.014% | 0/3 |

Against the strongest external baseline, the active model is `+19.407%` above Electricity/TimesNet and `+26.131%` above Traffic/iTransformer. Both registered external 2% gates fail.

## ⚠️ Interpretation

The latent temporal candidate is faster on the fixed representation-inclusive forward benchmark, but it does not retain competitive validation accuracy. The active AutoCon objective was live: its final logged loss is finite and nonzero across all six active jobs (`1.960--1.989` on Electricity and `1.768--1.780` on Traffic, effective lambda `.001`). The negative result is therefore not attributable to an inactive contrastive branch.

This architecture is stopped. Do not run a held-out release, change lambda, increase width, alter the rank-one decoder, or select a dataset-specific repair from these validation numbers. It may remain as a negative efficiency/accuracy ablation.

## 🌡️ Execution boundary

All 60 jobs ran serially with CPU thread limit 4. The GPU guard completed normally with no trigger: training temperature peak `57C`, fan peak `35%`, power peak `300.8W`, utilization peak `97%`, and memory peak `1993MB`.

## 🔗 Evidence

- [Strict machine readout](../../artifacts/paper_final/latent_temporal_mamba_highdim_autocon_validation_20261001/readout/readout.md)
- [Row-level results](../../artifacts/paper_final/latent_temporal_mamba_highdim_autocon_validation_20261001/readout/validation_rows.csv)
- [Phase artifacts](../../artifacts/paper_final/latent_temporal_mamba_highdim_autocon_validation_20261001/)
- [GPU guard event](../../artifacts/paper_final/latent_temporal_mamba_highdim_autocon_validation_20261001/gpu_guard/event.json)

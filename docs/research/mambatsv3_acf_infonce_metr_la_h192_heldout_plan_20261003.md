# Frozen METR-LA H192 Held-Out Plan: 2026-10-03

This release opens the METR-LA chronological test partition once for six
validation-frozen coordinates (seeds 5440--5442). The active signed-InfoNCE
Hybrid, NLinear, PatchTST, iTransformer, and TimesNet use the same 10% target
budget and five-epoch validation protocol. The source selector and NLinear
primary selector were completed before the main validation block; no test
metrics were available during selection.

Validation gate before opening test:

| Comparison | Active relative change | Wins |
| --- | ---: | ---: |
| NLinear | -5.204% | 3/3 |
| PatchTST | -5.352% | 3/3 |
| iTransformer | -8.890% | 3/3 |
| TimesNet | -8.700% | 3/3 |

The test reader is read-only and records `training_performed=false`. This is a
separate expansion release; it must not be merged into the earlier PEMS-BAY
confirmatory audit without an explicit new audit binding.

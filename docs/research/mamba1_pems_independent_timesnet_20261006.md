# Independent PEMS-BAY Official TimesNet Extension

Status: independently registered five-seed, validation-only official-code
baseline extension. No target test rows were read.

The pinned official TSLib TimesNet implementation was trained on the same
PEMS-BAY H192/10% chronological split, seeds `6001--6005`, target-window
budget, epoch budget, and width family as the independent Mamba-1 replication.
TimesNet uses the official adapter's memory-safe batch size of two; the data
split and number of sampled training windows are unchanged.

| Mamba-1 active | Official TimesNet | Active vs TimesNet | Wins | One-SE upper ratio |
| ---: | ---: | ---: | ---: | ---: |
| 2.774439 | 3.095302 | -10.366% | 5/5 | 0.91665 |

This is a validation-only official-model extension and was not used to select
the Mamba architecture, source pool, relation weight, or target protocol. It
reinforces external baseline coverage for the independently replicated PEMS
few-shot coordinate but does not establish universal superiority over TimesNet.

Artifacts:

- `artifacts/exploratory/official_timesnet_pems_mamba1_10pct_independent_replication_20261006/audit.json`
- `scripts/run_official_timesnet_pems_mamba1_10pct_independent_replication.py`
- `scripts/audit_pems_mamba1_10pct_independent_timesnet.py`

All five guarded official baseline runs completed below 54C and 32% fan.

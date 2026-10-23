# Weather Full-Pool Source Selection Check

After the Weather target-prefix calibration screens, the train-prefix source
selector was rerun against the full available non-Weather pool. It uses only
the earliest Weather 10%-label training prefix (3,161 rows), source training
prefixes, and the existing train-only ACF signature; no validation or test
window is opened.

| Rank | Source | Similarity |
| ---: | --- | ---: |
| 1 | METR-LA | 0.9274 |
| 2 | PEMS-BAY | 0.4721 |
| 3 | Electricity | 0.2971 |
| 4 | PEMS07 | 0.1489 |
| 5 | PEMS08 | 0.1453 |
| 6 | PEMS04 | 0.1343 |
| 7 | PEMS03 | 0.1125 |
| 8 | Traffic | 0.0752 |

The registered Weather source pool already equals the three highest-ranked
sources. No source replacement is justified by this selection rule, so no
post-hoc source-pool retraining was launched.

Machine-readable selection artifact:
`artifacts/exploratory/weather_full_pool_source_selection_20261007/selection.json`

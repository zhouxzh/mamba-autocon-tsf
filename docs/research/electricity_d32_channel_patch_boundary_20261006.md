# d32/24 Channel-Patch High-Dimensional Boundary

Status: validation-only negative replication. No external comparator or test
reader was started after this gate failed.

The d32/24 bridge was the strongest existing high-dimensional transfer
coordinate: a single Mamba-1 scan, patch `24/24`, `d_model=32`, `d_ff=64`,
Traffic source plus Electricity 25% calibration, and a frozen target core. A
one-seed gate showed channel-patch MSE `8.101177` versus forecast-only
`8.107439` (`-0.077%`).

Three fresh seeds reverse that weak signal:

| Seed | Forecast-only | Channel-patch | Relative change |
| ---: | ---: | ---: | ---: |
| 6044 | 8.155280 | 8.174291 | +0.233% |
| 6045 | 8.029091 | 8.033841 | +0.059% |
| 6046 | 7.998998 | 8.036307 | +0.467% |
| Mean | 8.061123 | 8.081480 | +0.253% |

The candidate loses all three pairs and has one-SE ratio upper `1.00371`.
All channel-patch source epochs had active ACF anchors, so the failure is not
an inactive-objective artifact. Stop this d32/24 channel-patch branch before
external-baseline expansion or test reading. The result supports retaining
the high-dimensional limitation in the manuscript rather than claiming that
the representation objective universally rescues Electricity.

Artifacts:

- `artifacts/exploratory/electricity_d32_channel_patch_probe_20261006/audit.json`
- `artifacts/exploratory/electricity_d32_channel_patch_replication_20261006/audit.json`

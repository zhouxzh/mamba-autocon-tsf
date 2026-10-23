# Electricity D32 Channelwise Gate Final Boundary

The d32/24 target-channel residual gate is the strongest high-dimensional
target-head adjustment found for the frozen Channel-Patch encoder. The initial
five matched Electricity H96/25% pairs narrowly miss official PatchTST
noninferiority. A final pre-registered five-seed expansion uses exactly the
same Mamba-1 d32/24 core, channelwise gate, batch four, five epochs, 64 target
windows, official PatchTST geometry, and validation-only boundary.

| Metric | Channelwise gate | Official PatchTST |
| --- | ---: | ---: |
| Mean validation MSE, 10 seeds | 7.845240 | 7.823808 |
| Relative mean | +0.274% | -- |
| Paired gate wins | 3/10 | -- |
| One-SE MSE ratio upper | 1.00509 | -- |

The expanded result is less favorable than the five-seed near tie and remains
externally inferior. This is a conclusive stopped branch: no additional gate
seeds, robustness matrix, or test reader is justified. All ten runs are
validation-only and GPU-guarded; fresh expansion telemetry peaks at 50°C and
0% fan.

Machine-readable audit:
`artifacts/exploratory/electricity_d32_channelwise_gate_final_expansion_20261007/audit.json`

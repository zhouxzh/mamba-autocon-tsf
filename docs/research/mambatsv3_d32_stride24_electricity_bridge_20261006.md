# d32/24 Electricity Transfer Bridge

Status: validation-only architecture bridge; no test rows were read.

The existing high-dimensional held-out release used a larger
`d_model=32,d_ff=64,patch_stride=24` MambaTSv3 geometry, while the current
cross-domain few-shot SSL release used `d_model=24,patch_stride=8`. To isolate
that capacity/geometry difference, this probe reused compatible PEMS d32/24
MambaTSv3 initial cores and ran the current Traffic-source plus Electricity
calibration-prefix SSL protocol: next-patch prediction, train-only ACF pairs,
25% target labels, frozen target core, and one Mamba scan.

Three fresh paired seeds gave:

| Quantity | Result |
| --- | ---: |
| d32/24 active mean validation MSE | 8.063889 |
| d32/24 null mean validation MSE | 8.128540 |
| active vs null | -0.795%, 3/3, one-SE upper 0.99359 |
| previous frozen d24/8 active | 8.205705 |
| d32/24 vs d24/8 | -1.728% |
| same-geometry official PatchTST | 7.794073 |
| d32/24 vs PatchTST | +3.462%, 0/3, one-SE upper 1.03743 |

The larger/less-overlapping geometry materially improves the candidate over the
previous d24/8 transfer point and preserves an ACF active/null effect, but it
does not reach the matched external baseline. It is therefore a useful
capacity/geometry diagnosis, not a publishable high-dimensional SOTA result.
No further d32 learning-rate, patch, or test search is authorized under this
line; the external gap must remain an explicit limitation.

Artifacts:

- `artifacts/exploratory/mambatsv3_electricity_mamba1_d32_stride24_transfer_replication_20261006/audit.json`
- `artifacts/exploratory/official_patchtst_electricity_d32_stride24_comparator_20261006/audit.json`
- `scripts/run_mambatsv3_electricity_d32_stride24_transfer_replication.py`
- `scripts/audit_mambatsv3_electricity_d32_stride24_transfer_replication.py`
- `scripts/run_official_patchtst_electricity_d32_stride24_comparator.py`
- `scripts/audit_mambatsv3_electricity_d32_stride24_external_comparator.py`

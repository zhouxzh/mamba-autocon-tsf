# Six-Token Full-Coverage MambaTS v3 PEMS07 H96 Readout

The `patch_len=16`, `patch_stride=16`, `d_model=16` graph has six non-overlapping patches that cover every input point. Its same-shape PEMS07 H96 resource preflight was favorable: 0.683 ms median and 149.6 MB peak allocated, compared with TimeMixer 2.630 ms and 1454.6 MB under `B=32,L=96,H=96`.

Fresh paired validation seeds `2151/2152/2153` completed successfully. Both phases are `valid=true`, have 3/3 jobs, and generated no test metrics. Null MSE was `0.233869/0.227777/0.233582` (mean `0.231743`); active MSE was `0.234296/0.227680/0.233629` (mean `0.231868`). Active is `+0.054%` versus null and wins only one seed.

The active mean is `+7.424%` above the historical PEMS07 H96 MambaTS-v3 reference (`0.215844`), exceeding the registered 2% stopping threshold. The architecture is therefore stopped before four-PEMS expansion or test release. The result indicates that adding patches at fixed low width does not repair the long-horizon capacity gap.

- [Frozen plan](pems07_mambatsv3_fullcover_p16_s16_d16_autocon_h96_probe_20260926.json)
- [Null validation](../../artifacts/exploratory/pems07_mambatsv3_fullcover_p16_s16_d16_autocon_h96_probe_20260926/null/phase_validation.json)
- [Active validation](../../artifacts/exploratory/pems07_mambatsv3_fullcover_p16_s16_d16_autocon_h96_probe_20260926/active/phase_validation.json)

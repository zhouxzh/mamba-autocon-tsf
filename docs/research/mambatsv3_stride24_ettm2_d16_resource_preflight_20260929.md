# ETTm2 H96 Stride-24 MambaTSv3 d16 Resource Preflight

This is a resource-only preflight for a lightweight AutoCon-connected MambaTSv3 configuration. It uses the same ETTm2 H96 shape as the frozen d32 candidate, with `d_model=16`, `d_ff=32`, `d_state=8`, `d_conv=2`, and `expand=1`; prediction-only timing disables returned representations and relation loss.

| Configuration | Trainable parameters | Median latency | Peak allocated memory |
| --- | ---: | ---: | ---: |
| d16 AutoCon MambaTSv3 | 19,272 | 0.2835 ms | 9.78 MB |
| d32 frozen candidate | 39,544 | 0.2746 ms | 10.74 MB |

The narrower model reduces parameters and memory, but is 3.2% slower on the matched RTX 5090 D `B=32,L=96,H=96` prediction-only protocol. This is a resource preflight only; no training or test artifact was produced. Because it fails the speed-improvement gate before accuracy evaluation, no d16 validation matrix is authorized.

Raw record: `artifacts/exploratory/mambatsv3_stride24_ettm2_d16_resource_preflight_20260929/model_resources.json`.

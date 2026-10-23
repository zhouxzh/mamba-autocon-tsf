# PEMS-BAY H336 Inference Boundary

On the same RTX 5090 D, device-resident batch-four, 25-warmup/100-timed
prediction-only protocol, deployed active uses one Mamba scan and no AutoCon
relation loss or returned representation. Its median latency is 0.315024 ms,
compared with 0.253328 ms for locked official PatchTST. Active is therefore
24.36% slower on this shape, while using 20.84 MB rather than 26.69 MB peak
allocated memory (21.93% less).

This supports a clean single-scan inference claim and a memory advantage on the
stated shape. It does not support an inference-latency advantage over official
PatchTST. The performance comparison remains valid independently: active is
1.934% lower MSE at this H336 coordinate, 3/3 seeds.

Audit: [`inference_boundary.json`](../../artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_pems_h336_extension_20261003/inference_boundary.json).

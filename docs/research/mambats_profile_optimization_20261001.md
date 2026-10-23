# MambaTS Representation-Grid GPU Optimization (2026-10-01)

This note records a semantics-preserving, prediction-only optimization on the local MambaTS implementation. It does not alter historical frozen training artifacts or held-out release numbers.

## Profile

On the METR-LA H12 candidate shape (`B=32,L=96,H=12,C=207,d_model=24`, RTX 5090 D, FP32), the synchronized PyTorch profiler identified `native_layer_norm`, small matrix multiplications, patch-to-time `upsample_linear1d`, and Mamba selective scan among the main GPU regions. The selective scan itself was not the dominant region in this short sequence shape.

## Change

The AutoCon representation restores patch states to the observed time grid using `F.interpolate(..., mode="linear", align_corners=False)`. Because the patch count and sequence length are fixed by the config, the implementation now precomputes the exact interpolation weight matrix once and uses a device-resident GEMM. The channelwise representation path preserves its original transpose contract. The matrix is a non-persistent derived buffer: it moves with the model but does not change historical checkpoint state dictionaries.

## Verification

- Microbenchmark `P=4 -> T=96`: interpolation and precomputed GEMM agree within `7.2e-7` forward absolute error; backward gradients agree within `2e-6` absolute / `1e-5` relative FP32 tolerance.
- Fixed METR-LA prediction-only benchmark before optimization: median `0.3252 ms`.
- Current-source optimized benchmark: median `0.3208 ms`.
- Approximate latency change on this shape: `-1.4%`.
- Regression tests cover ordinary and channelwise representation layouts.
- The interpolation matrix is absent from `state_dict()`, preserving checkpoint compatibility.

This is a local inference-path optimization only. It is not a new accuracy result, does not modify frozen checkpoints, and does not establish universal speed or end-to-end training throughput.

## Torch Compile Preflight

`torch.compile(mode="reduce-overhead")` was tested on the same frozen METR-LA shape. A single `no_grad` forward produced an output with maximum absolute difference `2.38e-7`, but the Mamba selective-scan extension caused a Dynamo graph break, and repeated CUDA-Graph execution failed with an overwritten-output error. The compile path is therefore rejected as a production optimization for this model until the custom operator and output-lifetime contract are explicitly fixed.

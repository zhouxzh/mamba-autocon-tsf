# MambaACF Memory-Gate Mode Resource Check

## Scope

This is a structural and synthetic device-resident forward check for the new
`mamba_acf_memory_gate_mode` option. It does not train a model and it is not a
forecasting result. Both modes use the same `MambaACF` configuration,
`d_model=32`, one Mamba block, `seq_len=96`, `pred_len=96`, and variable-token
mixing. A deterministic synthetic ACF prior is attached only to exercise the
memory path; no dataset or train split is loaded (`AutoCon_acf_source` remains
the CLI default `residual`, and `source_attestation.verified=false`). The GPU
measurements use batch size 32 and 5 warm-up plus 20 timed forwards on an RTX
5090 D.

## Canonical paired measurements

| Dataset | Mode | Parameters | Gate output / item | Gate MACs / item | Median ms | P95 ms | Peak increment MB |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1 (C=7) | `hidden` | 58,947 | 96 | 3,072 | 1.4372 | 1.4660 | 13.129 |
| ETTh1 (C=7) | `variable_token` | 58,947 | 672 | 21,504 | 1.4395 | 1.4684 | 13.129 |
| traffic (C=862) | `hidden` | 169,242 | 96 | 3,072 | 1.5957 | 1.6147 | 125.347 |
| traffic (C=862) | `variable_token` | 169,242 | 82,752 | 2,648,064 | 1.6704 | 1.6950 | 125.347 |

Relative to `hidden`, the canonical `variable_token` timing is about `+0.16%`
on ETTh1 and `+4.68%` on traffic. The same `Linear(d_model, H)` parameters
are applied per token, so parameter count is unchanged, while output elements,
MACs, and activation bytes scale with the number of channels. The allocator
peak increment happened to match in this run; it must not be read as equal
memory cost (Traffic profile activation is 384 -> 331,008 FP32 bytes per item).
These timings exclude data loading and host/device transfer and are not
end-to-end training throughput.

The canonical records are under
`../../artifacts/exploratory/mamba_acf_memory_gate_mode_resources_20260902/`:
`ETTh1_hidden_gpu_canonical`, `ETTh1_variable_token_gpu_canonical`,
`traffic_hidden_gpu`, and `traffic_variable_token_gpu`. Two earlier ETTh1
records (`ETTh1_*_gpu`) were run with implicit `scales=[1,2]` and
non-causal embedding; they are retained for audit history but excluded from
the paired table.

## Interpretation boundary

No accuracy experiment has been run for this mode. The token gate also uses
the variable-token representation as its gate source, so it changes the
optimization path of that branch in addition to changing the gate axis. A
future accuracy screen must therefore compare `hidden` and `variable_token`
under the same sidecar, seed, optimizer, and variable-token trunk, and report
the mode as a functional-capacity ablation rather than a pure ACF-effect test.

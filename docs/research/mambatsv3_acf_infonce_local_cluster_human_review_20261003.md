# Local-Cluster Human Review Packet

## Evidence Eligible for Results

The following are three-seed held-out active versus exact-null confirmations:

| Coordinate | Active vs null | Evidence |
| --- | ---: | --- |
| PEMS-BAY H192, 10% | -1.425%, 3/3 | `local_cluster_pems_confirmation/audit.json` |
| PEMS-BAY H192, 25% | -1.115%, 3/3 | `local_cluster_pems_25_confirmation/audit.json` |
| PEMS-BAY H336, 10% | -0.908%, 3/3 | `local_cluster_pems_h336_confirmation/audit.json` |
| METR-LA H192, 10% | -0.169%, 3/3 | `local_cluster_metr_confirmation/audit.json` |
| METR-LA H192, 25% | -0.304%, 3/3 | `local_cluster_metr_25_confirmation/audit.json` |
| Weather H192, 10% | -0.211%, 3/3 | `local_cluster_weather_confirmation/audit.json` |

The active and exact-null arms share source pool, source next-patch objective,
local pair geometry, seed, initialization, head transfer, target tuning, and
chronological split. They differ only in the forecast-level ACF InfoNCE
weights. Inference in every active checkpoint reports one Mamba scan, no
returned AutoCon representation, and no relation loss.

## Extensions

Validation-only robustness extensions give active 3/3 wins in six input
conditions for PEMS-BAY H192, PEMS-BAY H336, METR-LA H192, and Weather H192 at
10% labels. These are post-confirmation validation extensions, not separately
preregistered held-out claims.

Official PatchTST extensions use pinned external source code and a local
chronological adapter. They show active 3/3 wins at PEMS-BAY H192/25%
(-1.024%), PEMS-BAY H336/10% (-1.934%), and METR-LA H192/25% (-5.070%). They
are external-code post-confirmation extensions, not members of the active/null
causal matrix.

A same-machine independent-interpreter replication gives PEMS-BAY H192/10%
-1.105%. It supports execution reproducibility but is not an independent
machine result and is not pooled with the primary estimates.

A separate clean-install replication creates a new Python/CUDA environment,
checks locked cp39 binary artifacts and CUDA Mamba forward, then obtains
-1.438% at PEMS-BAY H192/10% in a new seed. It is stronger environment
evidence than an environment clone, but remains same-machine and uses locked
local binary artifacts rather than a from-source Mamba binary build.

The portable independent-machine launcher was exercised end-to-end on the
same clean-install host (seed 5612, -1.340%), including checksum, preflight,
validation-gate, test-reader, and GPU-guard paths. This is a launcher
rehearsal, not an independent-machine replication.

An independent-machine PEMS-BAY H192/10% replication has now completed on a
reviewer-verified distinct RTX 3060 Ti host with Torch `2.6.0+cu126` and
Mamba `2.2.4`. The remote staging manifest matched all 14 tracked code, data,
sampler, and initialization-checkpoint hashes. Fresh seeds `6130--6132` gave
active/null `2.638441/2.670296` (`-1.193%`), with 3/3 test wins and a one-SE
ratio upper bound of `.99151`. All 18 guarded stages completed without a
trigger; peak temperature was `63C` and peak fan was `31%`. This is separate
from the primary local estimates and supports execution reproducibility across
host, GPU, CUDA, Torch, and Mamba versions. The audited artifact is
`artifacts/reproducibility/mambatsv3_local_cluster_independent_machine_20261008/independent_audit.json`.

The same reviewed host independently replicated a second target domain,
METR-LA H192/10%, with a different held-out sensor dataset and the fixed source
pool PEMS-BAY/PEMS07/PEMS08. Its fresh seeds `6140--6142` gave active/null
`2.622863/2.630788` (`-0.301%`), 3/3 wins and one-SE upper ratio `.99762`.
All 18 guarded stages completed without a trigger, with peak `60C` and `30%`
fan. This confirms the direction across two independent target domains without
pooling either replication into the original local effect estimates. The audit
is `artifacts/reproducibility/mambatsv3_local_cluster_metr_independent_machine_20261008/independent_audit.json`.

Weather H192/10% was independently replicated on the same reviewed host with
its distinct source pool METR-LA/PEMS-BAY/Electricity. Fresh seeds
`6150--6152` gave active/null `1.016437/1.017655` (`-0.120%`), 3/3 wins and
one-SE upper ratio `.99903`. The effect is smaller than PEMS-BAY and METR-LA
but directionally consistent across all three fresh seeds. All 18 guarded
stages completed without a trigger, with peak `60C` and `30%` fan. This is
cross-domain support evidence only and is not pooled with the original local
estimates. Audit:
`artifacts/reproducibility/mambatsv3_local_cluster_weather_independent_machine_20261008/independent_audit.json`.

The same three independent-host target coordinates now also have a pinned
official PatchTST extension. The external code is `yuqinie98/PatchTST` commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9`; validation selected each
checkpoint before a separate read-only test reader opened the target test
block. Active wins all three matching seeds in each target: PEMS-BAY
`2.638441` versus `2.649355` (`-0.412%`, one-SE upper `.99762`), METR-LA
`2.622863` versus `2.727006` (`-3.819%`, `.96717`), and Weather `1.016437`
versus `1.056680` (`-3.809%`, `.96653`). This is a post-release external-code
extension, not a replacement for or a pooled addition to the frozen
active/null mechanism matrix. All 18 guarded stages completed without a
trigger; validation-stage peaks were at most `63C` and `32%` fan. Audit:
`artifacts/reproducibility/mambatsv3_local_cluster_independent_official_patchtst_20261008/independent_patchtst_audit.json`.

## Mandatory Negative Results

- Electricity H96/25%: active loses all validation pairs (+2.146% mean).
- Traffic H96/25%: active loses all validation pairs (+0.263% mean).
- Weather H192/25%: all-seed gate failed; held-out test remains closed.
- PEMS03 H192/10% high-compatibility gate: one of three validation pairs lost;
  held-out test remains closed.
- PEMS-BAY H336 matched inference: active is 24.35% slower than official
  PatchTST, but uses 21.93% less peak allocated memory.

## Claim Boundaries

The supported claim is conditional low-label transfer for named periodic
sensor targets. It is not a universal Mamba advantage, universal AutoCon gain,
guaranteed compatibility-threshold effect, or PatchTST latency advantage.

The observed compatibility/gain correlation is descriptive and not suitable
for causal or predictive claims without a new preregistered domain set.

## Human Gates

Before submission, an accountable author must review every evidence file,
check the final source/code release state, select a venue and verify policies,
and complete declarations/authorship/data availability. The independent-machine
replication requirement is now met; the clean-install replication remains
same-machine support evidence. This packet is a review aid, not a
submission-ready manuscript.

The claim-by-claim worksheet is
`mambatsv3_acf_infonce_local_cluster_human_verification_worksheet_20261003.csv`.
It must be completed by an accountable human; the accompanying checker only
reports pending fields and never changes a verification status.

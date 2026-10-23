# Local-Cluster ACF-InfoNCE Evidence Matrix

## Confirmed Effect Versus Exact-Null Predictive Pretraining

| Target | H | Labels | Active vs null | Seeds | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| PEMS-BAY | 192 | 10% | -1.425% | 3/3 | Held-out confirmation |
| PEMS-BAY | 192 | 25% | -1.115% | 3/3 | Held-out confirmation |
| PEMS-BAY | 336 | 10% | -0.908% | 3/3 | Held-out confirmation |
| METR-LA | 192 | 10% | -0.169% | 3/3 | Held-out confirmation |
| METR-LA | 192 | 25% | -0.304% | 3/3 | Held-out confirmation |
| Weather | 192 | 10% | -0.211% | 3/3 | Held-out confirmation |
| Weather | 192 | 25% | mixed validation (+0.022%, +0.026%, -0.034%) | 1/3 | Failed gate; test closed |
| Electricity | 96 | 25% | +2.146% | 0/3 | Validation-only negative boundary; test closed |
| Traffic | 96 | 25% | +0.263% | 0/3 | Validation-only negative boundary; test closed |

Negative is better. Exact-null preserves source prediction task, pair geometry,
seed, initialization, target head transfer, and tuning, while setting both
InfoNCE weights to zero.

## Robustness Extension

For PEMS-BAY, METR-LA, and Weather at 10% labels, all six closed validation
conditions (ordinary validation, late origins, 10%/30% missing channels, and
0.05/0.10 input noise) favor active in all three seeds. The effect range is
-1.18% to -1.71% on PEMS-BAY, -0.33% to -0.48% on METR-LA, and -0.09% to
-0.26% on Weather. These are post-confirmation validation-only extensions.

PEMS-BAY H336/10% has an independent validation-only robustness extension:
ordinary validation, late origins, 10%/30% missing channels, and 0.05/0.10
noise all favor active in 3/3 seeds, with relative improvements from -0.746%
to -1.095%.

## Official-Code Baseline Extension

On PEMS-BAY H192 with 25% labels, the same frozen active checkpoints obtain
2.273256 test MSE versus 2.296786 for pinned official PatchTST code
(`yuqinie98/PatchTST`, commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9`),
a -1.024% difference with 3/3 wins and one-SE ratio upper bound 0.9923. The
official model and backbone source files are SHA-256 locked. This is an
external-code post-confirmation extension, not a component of the causal
active/null matrix.

METR-LA H192 with 25% labels adds a second external-code coordinate: active
1.931659 versus official PatchTST 2.034834 (-5.070%, 3/3 wins, one-SE ratio
upper bound 0.9518), using the same pinned commit and source-hash locks.

PEMS-BAY H336 with 10% labels adds a long-horizon external-code coordinate:
active 2.289701 versus official PatchTST 2.334861 (-1.934%, 3/3 wins, one-SE
ratio upper bound 0.9837), with the same source locks and chronological
adapter lifecycle.

PEMS-BAY H192 with 25% labels also compares to locked official iTransformer
code: active 2.273256 versus 2.478354 (-8.276%, 3/3 wins, one-SE ratio upper
bound 0.9256). It is a post-confirmation external-code extension with four
source-file SHA-256 locks, not a component of the active/null causal matrix.

## Inference and Scope

All positive confirmation checkpoints retain one Mamba scan, no returned
AutoCon representation, and no relation loss at inference. PEMS-BAY 10%/25%
batch-four median latency is about 0.316 ms; METR-LA is 0.304 ms; Weather is
0.309 ms on the guarded RTX 5090 D measurement setup.

The independent PEMS-BAY H336 confirmation has active mean test MSE 2.289701
versus 2.310690 for exact-null (-0.908%, 3/3; paired interval
[-0.024776, -0.017200]). It remains one scan at 0.315 ms median, with 20.84
MB peak allocated memory at batch size four.

The same-geometry H336 comparison with official PatchTST establishes an
important latency boundary: active is 24.35% slower (0.315024 versus 0.253328
ms) but uses 21.93% less allocated peak memory (20.84 versus 26.69 MB). Thus
the evidence supports clean single-scan, no-AutoCon inference and the stated
memory advantage, not a PatchTST latency advantage.

METR-LA's independent 25% confirmation retains one scan at 0.311 ms median,
with active mean test MSE 1.931659 versus 1.937556 for exact-null. The three
paired test differences have a 95% interval of [-0.007939, -0.003857].

The defensible claim is conditional low-data transfer on the named compatible
targets. Electricity and Traffic must be reported as failure domains. The
five-point source-compatibility association (`r=-0.765`) is exploratory and
motivates a preregistered follow-up, not a universal compatibility theorem.

## Same-Machine Independent-Interpreter Replication

A separately cloned interpreter (Python 3.9.25, PyTorch 2.8.0+cu128,
mamba-ssm 2.3.1) reran PEMS-BAY H192/10% with a new seed and recorded the
current source/data hashes. Active test MSE was 2.620800 versus 2.650094 for
exact-null (-1.105%). This supports execution reproducibility but remains a
same-machine replication and is not pooled with the three-seed confirmation.

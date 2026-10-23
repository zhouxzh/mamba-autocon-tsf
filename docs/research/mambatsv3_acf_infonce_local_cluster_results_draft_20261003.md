# Local-Cluster ACF-InfoNCE Results Draft

_Evidence-bound internal draft for human review. Not for submission._

## Method Summary

The method pretrains a channel-count-agnostic patch encoder with one temporal
Mamba scan. Source windows are paired using signed autocorrelation peaks fitted
only on chronological source-training prefixes. The active arm optimizes a
forecast-level signed InfoNCE objective in addition to residual next-patch
prediction; the exact-null arm preserves the same source windows, pair
geometry, initialization, prediction target, exact horizon-matched head
transfer, target tuning, and seeds, but sets both relation weights to zero.
Prediction-only deployment returns no contrastive representation and computes
no relation loss. [claim:C101] [evidence:E101]

## Held-Out Results

The primary local-cluster evidence contains six three-seed held-out active/null
confirmations. [claim:C102] [evidence:E101]

| Target | Horizon | Labels | Active MSE | Exact-null MSE | Active vs null | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY | 192 | 10% | 2.635171 | 2.673272 | -1.425% | 3/3 |
| PEMS-BAY | 192 | 25% | 2.273256 | 2.298880 | -1.115% | 3/3 |
| PEMS-BAY | 336 | 10% | 2.289701 | 2.310690 | -0.908% | 3/3 |
| METR-LA | 192 | 10% | 2.618048 | 2.622484 | -0.169% | 3/3 |
| METR-LA | 192 | 25% | 1.931659 | 1.937556 | -0.304% | 3/3 |
| Weather | 192 | 10% | 1.017754 | 1.019903 | -0.211% | 3/3 |

The PEMS-BAY H336 coordinate provides an independently confirmed long-horizon [claim:C103] [evidence:E101]
mechanism effect, rather than extrapolation from the H192 result. [claim:C103] [evidence:E101]
[claim:C103] [evidence:E101]

## Robustness and Deployment

Closed validation extensions evaluate late validation origins, missing input
channels, and signal-relative input noise after checkpoints are frozen. The
active arm wins all conditions in all three seeds for PEMS-BAY H192, PEMS-BAY [claim:C104] [evidence:E101]
H336, METR-LA H192, and Weather H192 at the reported label fraction. [claim:C104] [evidence:E101]
[claim:C104] [evidence:E102]

The H336 active checkpoint has one Mamba scan, no relation loss, and no [claim:C105] [evidence:E101]
returned AutoCon representation during prediction-only inference. In the
matched batch-four device-resident comparison, it uses less peak allocated
memory than official PatchTST but has higher median latency. [claim:C105] [evidence:E101]

## External-Code Comparisons

Pinned official PatchTST extensions use a local chronological adapter with
validation-selected checkpoints and source/data hash verification. The active
checkpoint wins all three seeds at PEMS-BAY H192 with 25% labels, PEMS-BAY [claim:C106] [evidence:E101]
H336 with 10% labels, and METR-LA H192 with 25% labels. These are [claim:C106] [evidence:E101]
post-confirmation external-code extensions and should be labelled separately
from the active/null causal matrix. [claim:C106] [evidence:E101]

An independent-host extension also reran pinned official PatchTST on the
three independently replicated H192/10% targets. Active won 3/3 paired seeds [claim:C106] [evidence:E101]
in PEMS-BAY (`-0.412%`), METR-LA (`-3.819%`), and Weather (`-3.809%`); these [claim:C106] [evidence:E101]
domain-specific effects are not pooled and do not alter the active/null
mechanism claim. [claim:C106] [evidence:E101]

## Boundaries

Electricity H96 with 25% labels and Traffic H96 with 25% labels are retained [claim:C107] [evidence:E101]
negative transfer boundaries. Weather H192 with 25% labels and the PEMS03 [claim:C107] [evidence:E101]
external high-compatibility check fail their all-seed validation gates, so no
held-out tests were read for those coordinates. [claim:C107] [evidence:E101]

The train-prefix source-compatibility analysis is exploratory. It should be
presented as a risk-signal hypothesis rather than a deterministic source
selection rule. [claim:C108] [evidence:E102]

## Review Requirements

An accountable author must verify every local evidence locator, decide the
final target venue, add human-verified literature context, select the final
external-code table scope, and complete declarations before submission. The
independent-machine replication requirement is now met; author evidence review,
venue selection, literature, table-scope, and declaration decisions remain
pending. [claim:C109] [evidence:E103] [evidence:E104]

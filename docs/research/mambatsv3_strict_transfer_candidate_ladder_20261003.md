# Strict Transfer Candidate Ladder

This decision record separates mechanisms that demonstrate an AutoCon
representation-transfer effect from mechanisms that meet a same-seed strong
external-baseline requirement. All entries are validation-only unless stated
otherwise. No negative result is omitted.

| Candidate | PEMS-BAY H192, 25% finding | Decision |
| --- | --- | --- |
| Frozen core, 24/24 patches | Active beats exact-null 3/3 (-1.736%), but loses same-seed official PatchTST 0/3 (+3.824%). | Retain causal transfer evidence; insufficient external accuracy. |
| Target channel adapter | Active beats exact-null at seed 5840 (-1.205%), but does not close PatchTST gap. | Closed. |
| Target temporal MLP adapter | Active beats exact-null at seed 5840 (-1.179%), MSE 2.42319. | Closed; no strong-baseline comparison opened. |
| Staged core adaptation | Active beats exact-null 3/3 (-1.576%); same-seed PatchTST mean is narrowly lower (-0.354% for staged active) but one-SE ratio upper bound is 1.0041. | Closed as noninferior claim fails. |
| Longer source pretraining | Active beats exact-null at seed 5850 (-1.084%), MSE 2.41972. | Closed; source budget alone is insufficient. |
| Overlapping 24/8 patches | Initial seeds 5860--62: 2/3 wins versus PatchTST, mean +0.307%, one-SE upper 1.0124. Independent seeds 5863--65: 0/3 wins, mean +0.825%, upper 1.0097. | Closed; the apparent initial benefit did not replicate. |
| Discriminative core LR | Seed 5866 used 1e-4 for transferred core after warm-up and 1e-3 for target parameters; active beats null by -1.745%, but no strong-baseline parity was established. | Closed as an optimizer-only probe. |
| PatchTST-matched 16/8 geometry | Seed 5867 uses 11 overlapping tokens, active 2.37278 versus null 2.42206 (-2.035%). This is worse than the 24/8 active probe (2.36301) and does not justify another official-baseline comparison. | Closed. |

The source ACF pretraining objective remains valid for the restricted
representation-transfer claim: it has paired active/null gains on PEMS-BAY,
METR-LA, and Weather and retained validation robustness on PEMS-BAY and
Weather. However, the current MambaTSv3 architecture and target adaptation do
not meet the same-seed official PatchTST requirement at the key PEMS-BAY 25%
coordinate. This is the primary empirical barrier to a credible SCI-Q1
submission claim.

Any future architecture should first be evaluated under a registered
same-seed official PatchTST validation gate before test access. A single-seed
win, an internal exact-null gain, or a latency result is insufficient.

The current strongest path is therefore the conditional frozen-core transfer
protocol with an explicit applicability boundary, not a universal accuracy
claim. Further work should target a genuinely stronger shared representation or
an apples-to-apples external-baseline protocol before any new held-out test
reading.

## Search Stop Rule

All candidate modifications above were evaluated on the same public
PEMS-BAY H192 25% validation coordinate. The evidence now includes an
independent three-seed replication in which the best single-scan structural
candidate remained inferior to official PatchTST. Continuing to alter the
current MambaTSv3 trunk, decoder, patch geometry, or local optimization based
on this coordinate would be post hoc architecture selection. The current
search is therefore closed. A future direction must either introduce a new
pre-declared backbone family or change the scientific question to conditional
representation transfer and robustness, with a new validation plan before any
test read.

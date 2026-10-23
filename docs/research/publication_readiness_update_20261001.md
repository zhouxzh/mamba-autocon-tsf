# Publication Readiness Update (2026-10-01)

This update supersedes only the numeric release inventory in the 2026-09-30 readiness audit. The earlier audit remains a historical record of its checks.

## Current Evidence

| Requirement | Current evidence | Status | Boundary |
| --- | --- | --- | --- |
| Diverse datasets | Four held-out release families: PEMS03/04/07/08, METR-LA/PEMS-BAY, Weather, Electricity/Traffic; additional ETT, Solar, Exchange-Rate and ILI validation boundaries | Partially satisfied | Positive results are not universal across ETT, Solar, Exchange/ILI or long horizons |
| Strong baselines | Linear, patch, Transformer, Mamba-1/2, TimeMixer and local graph baselines are registered across the releases | Partially satisfied | Graph competitors are validation-only local implementations |
| Published Mamba design signals | MambaTS, TimeMachine, multi-scale and Mamba-2 directions were implemented or evaluated under local contracts | Satisfied with scope | Local implementations are not bitwise external reproductions |
| AutoCon representation learning | Candidate releases use AutoCon; traffic fusion release has a prediction-connected representation path; active/null controls exist | Satisfied with limitations | Standalone relation-loss gains are not stable across all domains |
| Accuracy against registered baselines | PEMS `28/28`, METR-LA/PEMS-BAY `12/12`, Weather `12/12`, Electricity/Traffic CPU-4 `13/14` non-inferior cells | Satisfied for named releases | Traffic/iTransformer is a required failure cell; results cannot support universal superiority |
| Fixed-shape efficiency | Candidate is faster than named nonlinear comparators on several registered shapes | Partially satisfied | Linear and selected Transformer/Mamba comparators remain faster on other shapes |
| High-level paper | Four source-bound held-out releases, `66` paired baseline cells (`65` non-inferior), negative boundaries and a full `842`-test regression suite | Draftable, not proven publishable | Venue fit, independent reproduction, statistical review and human author approval remain open |

## CPU-4 High-Dimensional Release

The Electricity/Traffic H96 CPU-4 release completed `86/86` jobs and all three phase validators. The five-seed strict reader reported `13/14` non-inferior cells. The candidate loses to Traffic/iTransformer by `2.154%` on mean MSE, with one win in five paired seeds. This result strengthens cross-domain held-out coverage but also prevents an all-baseline dominance claim.

The fixed-shape candidate latency is `.3913 ms` on Electricity and `.8366 ms` on Traffic. It is not a universal speed winner: DLinear/NLinear/RLinear and Electricity iTransformer are faster. Training time is not used as an inference-speed claim.

## Pending Candidate

`LatentTemporalMambaAutoCon` passed the representation-inclusive efficiency preflight (`.3763/.5038 ms` on Electricity/Traffic versus stride-24 `.4255/.8727 ms`) but failed its completed 60-job validation-only matrix. Active AutoCon is `+10.186%/+6.343%` versus its exact lambda-zero control and `+19.407%/+26.131%` versus the strongest external baseline. It is stopped and cannot contribute a positive accuracy, baseline, or held-out claim.

## Claims Allowed Now

The strongest defensible claim is that AutoCon-connected MambaTSv3 is competitive on the named held-out traffic, Weather and high-dimensional Electricity/Traffic protocols, with comparator-specific fixed-shape efficiency advantages and explicit counterexamples. It is not support for universal SOTA, universal acceleration, a general causal benefit of the AutoCon relation loss, or a complete graph-model ranking.

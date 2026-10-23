# Official PatchTST H336 External-Code Extension Plan

The official PatchTST network is pinned to `yuqinie98/PatchTST` commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9` and runs through the same local
chronological boundary adapter used for H192. Coordinates are PEMS-BAY H336,
seeds 5413--5415, 10% target prefix, five epochs, and 64 training windows.

Validation result: active `2.386473`, official PatchTST `2.417429`, `-1.281%`.
Two of three seeds favor active; this weaker stability is retained. The test
extension reads each validation-selected official checkpoint once and must not
trigger candidate retraining or changed official/model parameters.

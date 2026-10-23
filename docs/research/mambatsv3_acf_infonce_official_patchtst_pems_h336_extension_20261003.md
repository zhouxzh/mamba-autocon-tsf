# Official PatchTST PEMS-BAY H336 Extension

The PEMS-BAY H336/10%-label local-cluster active/null checkpoints for seeds
5600--5602 are frozen. This post-confirmation extension runs official PatchTST
commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9` through the established local
chronological adapter: 60/20/20 split, 10% target-prefix supervision, H336,
five epochs, 64 target windows, and validation-selected checkpoint.

All official baseline runs are validation-only. A separate data/source-hash
bound evaluator reads every test block once only after all three baseline
checkpoints are frozen. The comparison does not alter the H336 active/null
causal matrix.

# Official PatchTST PEMS-BAY 25% Extension

The validation-selected PEMS-BAY H192 25%-label local-cluster checkpoints
(seeds 5540--5542) are frozen. This post-confirmation external-code extension
runs the pinned official PatchTST implementation from commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9` under the same local chronological
adapter contract: 60/20/20 split, 25% train-prefix supervision, H192, 5
epochs, 64 target windows, and validation-selected checkpoint.

All three official baseline training jobs complete before any baseline test
reader runs. The separate reader validates both dataset and upstream commit
hashes and never trains or selects. This extension remains distinct from the
candidate active/null confirmation matrix.

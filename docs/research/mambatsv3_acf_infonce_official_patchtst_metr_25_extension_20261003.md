# Official PatchTST METR-LA 25% Extension

The METR-LA H192 25%-label active/null checkpoints for seeds 5580--5582 are
frozen. This post-confirmation extension evaluates pinned official PatchTST
commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9` with the same local adapter:
chronological 60/20/20 split, 25% target train-prefix supervision, H192,
five epochs, 64 target windows, and validation-selected checkpoint.

All three official training jobs are validation-only. A separate source-hash
and dataset-hash-bound reader opens each test partition once only after the
three baseline checkpoints are frozen. This comparison is separate from the
active/null causal result.

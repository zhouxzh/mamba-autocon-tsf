# Pooled-Patch Representation Probe

This fresh-seed validation-only probe replaces the failed channelwise
forecast-level InfoNCE relation view with `pooled_patch_contrastive`.

For each window, the shared MambaTSv3 encoder produces channelwise patch
states. Their per-patch channel mean and standard deviation are concatenated
and sent to a shared, pretraining-only projection head. Signed global ACF
relations from the source training prefix define the InfoNCE positive and
negative pairs. The head is not transferred or used by prediction inference.

Electricity uses Traffic and Traffic uses Electricity, selected from their 25%
target-train-prefix rankings. Each source contributes 192 windows, matching the
previous three-source 64-window budget. Fresh seeds are 5750 and 5760. Active
and null share all settings except the relation weights. This probe reads no
test rows and cannot authorize a test run; it can only justify or reject a
separate three-seed validation matrix.

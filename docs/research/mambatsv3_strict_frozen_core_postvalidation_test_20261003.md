# Strict Frozen-Core Post-Validation Test Read

The frozen-core active/null checkpoint matrices were selected and locked using
validation-only results before this document and plan were created. This
protocol reads their chronological test partitions once using a read-only
checkpoint evaluator. It covers PEMS-BAY H192 at 10% and 25% labels, METR-LA
H192 at 10% labels, and Weather H192 at 10% labels, each with three seeds.

No checkpoint is trained, selected, or mutated during the test read. Because
earlier historical experiments in the shared workspace have already accessed
these public dataset test partitions under other protocols, any result is
labelled protocol-specific post-validation evidence rather than globally
pristine held-out evidence. It must not be used to hide the retained
Electricity/Traffic negative boundary.

## Completed Readout

All active checkpoints beat their paired exact-null checkpoints in all three
seeds for each coordinate: PEMS-BAY 10% (-1.196%), PEMS-BAY 25% (-1.519%),
METR-LA 10% (-0.286%), and Weather 10% (-0.233%). The audit artifact retains
per-seed values, read-only status, and the post-validation scope statement.

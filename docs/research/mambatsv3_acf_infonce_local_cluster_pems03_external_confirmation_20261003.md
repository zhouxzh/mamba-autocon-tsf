# PEMS03 External Compatibility Confirmation

## Registered Question

Does the frozen local-cluster ACF-InfoNCE protocol transfer to a new,
high-compatibility PEMS03 target when PEMS03 is excluded both from the source
pool and from the historical initialization checkpoints?

## Frozen Protocol

| Item | Value |
| --- | --- |
| Target | PEMS03, chronological 60/20/20 split, H=192, 10% labels |
| Source selection | PEMS03 10% train-prefix ACF signature only |
| Selected sources | PEMS04, PEMS08, PEMS07; mean similarity 0.993 |
| Initialization | PEMS04 and PEMS08 historical checkpoints only; PEMS03 excluded |
| Pretraining | Three epochs, 64 windows/source, source residual next-patch prediction |
| Pairing | Train-only channel ACF, support >= 0.5, local `contrastive_cluster` |
| Active/null | 0.1/0.1 forecast InfoNCE versus exact zero relation weights |
| Target tuning | NLinear residual hybrid, exact H192 head transfer, 5 epochs, 64 windows |
| Seeds | 5570, 5571, 5572 |

All six source and all six validation-only target jobs complete before any
test reader. Held-out test is read only if all active source anchor coverage is
at least 0.8 and each active validation MSE is no larger than its null pair.

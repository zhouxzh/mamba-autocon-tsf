# Frozen-Core Target-Head Probe

The primary low-data hybrid protocol froze all Mamba parameters only in epoch
one, then fine-tuned the full transferred encoder. This probe instead freezes
the transferable shared core throughout adaptation while allowing the target
forecast head, NLinear primary head, and residual scale to train. It therefore
tests the stated representation-transfer mechanism more directly.

It reuses the completed Traffic seed 5770 pooled-patch anti-collapse active
and exact-null pretraining checkpoints. Only validation MSE is produced. The
probe cannot open test evaluation and only decides whether frozen-core target
head adaptation deserves a fresh multi-seed validation matrix.

# PEMS03 External Compatibility Gate Readout

PEMS03 was selected as a new high-compatibility target before validation: its
10%-prefix source selector chose PEMS04, PEMS08, and PEMS07 with mean
similarity 0.993. PEMS03 was excluded from sources and from both historical
initialization checkpoints.

| Seed | Active validation | Exact-null validation | Active vs null |
| --- | ---: | ---: | ---: |
| 5570 | 1.230312 | 1.236331 | -0.487% |
| 5571 | 1.224815 | 1.223822 | +0.081% |
| 5572 | 1.229973 | 1.233437 | -0.281% |

The registered all-seed validation gate failed because seed 5571 is worse.
Therefore no PEMS03 test reader was launched and no PEMS03 test metric exists.
This is retained as a mixed high-compatibility boundary: high ACF signature
similarity appears associated with transfer direction in the current evidence,
but is not sufficient to guarantee a three-seed win.

The implication for the manuscript is stricter: source compatibility is a
descriptive risk signal, not a deterministic selector or a claim of guaranteed
benefit. The frozen local-cluster configuration is not retuned after this
failed gate.

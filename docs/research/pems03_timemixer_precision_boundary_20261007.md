# PEMS03 TimeMixer Precision Boundary

Pinned official TimeMixer is a high-variance but material external comparator
at PEMS03 H192/25%. On the initial same-seed comparison, Channel-Patch has a
favorable mean but does not pass one-standard-error noninferiority. A
pre-registered expansion adds fresh seeds `6065` and `6066` and combines them
only with the original coordinate seeds `6060--6062`.

| Metric | Channel-Patch | Official TimeMixer |
| --- | ---: | ---: |
| Mean validation MSE | 1.182786 | 1.237506 |
| Paired wins for Channel-Patch | 3/5 | -- |
| Relative mean | -4.422% | -- |
| One-SE MSE ratio upper | 1.02625 | -- |

The candidate's mean is favorable, but the required noninferiority gate fails.
TimeMixer is therefore explicitly excluded from passed-baseline claims. No test
partition is opened; all five runs are validation-only and GPU-guarded.

## Update-Matched Follow-up

The candidate used batch four (16 updates/epoch), whereas TimeMixer used batch
two (32 updates/epoch). A pre-registered update-matched follow-up retains every
candidate checkpoint, batch size, model parameter, and target window but runs
10 epochs, for 160 updates matching TimeMixer's five epochs. The candidate
mean improves to `1.175540` versus TimeMixer `1.237506` (`-5.007%`), but still
has only 3/5 wins and one-SE ratio upper `1.01643`. Thus the noninferiority
failure persists after matching update count; it cannot be attributed solely
to the batch-size budget asymmetry.

Update-matched audit:
`artifacts/exploratory/pems03_channel_patch_update_matched_timemixer_20261007/audit.json`

## Final Batch-Matched Expansion

The update-matched follow-up still differed in minibatch geometry. A final,
pre-registered five-seed expansion uses batch two, five epochs, 64 target
windows, and 160 updates for both Channel-Patch and official TimeMixer. It is
combined with the fixed ten batch-matched pairs and capped by an explicit stop
rule.

Across 15 paired seeds, Channel-Patch reaches `1.159693` versus TimeMixer
`1.199889` (`-3.350%`), with 7/15 wins and one-SE ratio upper `0.99978`.
The batch/update-matched PEMS03 comparison therefore passes noninferiority.
No further TimeMixer seed expansion is permitted by the registered protocol.

Final audit:
`artifacts/exploratory/pems03_channel_patch_batch2_timemixer_final_expansion_20261007/audit.json`

Machine-readable audit:
`artifacts/exploratory/pems03_channel_patch_timemixer_precision_expansion_20261007/audit.json`

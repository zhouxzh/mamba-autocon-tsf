# Source-Compatible Pretraining Candidate

This validation-only candidate addresses the observed concentration of ACF
compatibility in the high-dimensional transfer boundary. For Electricity at
25% labels, Traffic scores 0.986 on the target train-prefix signature whereas
the next source scores 0.200. For Traffic, Electricity scores 0.940 whereas
the next source scores 0.059.

The candidate retains only the first source when its similarity exceeds the
second-ranked source by a preregistered factor of three. It uses 192 source
windows, matching the former 64 windows across each of three sources per
epoch. Active and exact-null arms use the same selected source, source budget,
initialization, chronological partitions, prediction head, and fine-tuning
recipe; only the relation objective differs.

This matrix is validation-only with Electricity seeds 5650--5652 and Traffic
seeds 5660--5662. It cannot open test evaluation. All three seeds for a target
must favor active before any separately registered held-out confirmation.

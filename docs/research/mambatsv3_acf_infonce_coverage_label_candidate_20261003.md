# Strict Positive-Label Coverage Candidate

This is a validation-only mechanism screen for the two retained negative
transfer domains: Electricity and Traffic at H96 with 25% target labels.

The existing `channel_coverage` recipe used source-window lags with adequate
cross-channel support, but its InfoNCE numerator still treated all per-channel
ACF peaks as positives. The candidate restricts that numerator to the same
support-qualified lag bank. It preserves the frozen single-scan MambaTSv3
backbone, source pools, chronological train prefixes, seeds, initialization,
training budget, prediction-head transfer, and historical exact-null
comparators.

The registered matrix contains Electricity seeds 5550--5552 and Traffic seeds
5560--5562. It produces validation metrics only. Each target must improve on
its exact-null comparator for all three seeds before any new held-out test
protocol may be registered. A failed gate remains a retained applicability
boundary rather than a reason to search the test set.

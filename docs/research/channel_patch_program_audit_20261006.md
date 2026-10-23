# Channel-Patch Program Audit

Channel-patch ACF representation pretraining is a real training mechanism. It
uses `[window, channel, patch, embedding]` encoder states with per-channel,
train-only ACF positives, and it adds neither a second Mamba scan nor an
inference-time contrastive computation.

A source-train representation geometry audit directly confirms the intended
effect: source-specific ACF-positive patch trajectories gain `+0.0748` mean
alignment margin over low-ACF controls after channel-patch pretraining. See
[`representation geometry`](channel_patch_representation_geometry_20261007.md).

| Budget | PEMS-BAY vs forecast | METR-LA vs forecast | External stress result |
| --- | ---: | ---: | --- |
| 10% | -1.702%, 3/3 | -1.070%, 3/3 | all six PatchTST cells noninferior |
| 25% | -1.820%, 3/3 | -0.951%, 3/3 | all six PatchTST cells noninferior |

PEMS-BAY H336 provides an independent long-horizon extension under the current
`24/8` geometry: channel-patch improves forecast-only by `-1.167%` and
official PatchTST by `-2.568%`, both 3/3, with all six stress cells externally
noninferior. See
[`H336 replication`](pems_h336_channel_patch_replication_20261006.md).

The strictly target-excluded PEMS03 replication adds a fourth traffic-network
target under H192/10% labels. PEMS03 is absent from both the three source
datasets and the two initialization checkpoints. Channel-patch reaches MSE
`1.2056`, improving matched forecast pretraining by `-0.333%` (3/3) and the
locked official PatchTST adapter by `-3.980%` (3/3). This is validation-only;
it does not repair the historical test-access or Weather-test limitations. See
[`PEMS03 external replication`](pems03_channel_patch_external_replication_20261007.md).
Its registered six-condition validation stress matrix also clears every
same-seed PatchTST noninferiority cell; its smallest margin is `-2.902%` at
30% masked channels.

The same target-excluded PEMS03 protocol also passes at 25% labels: it improves
forecast-only pretraining by `-0.455%` and official PatchTST by `-14.253%`,
both 3/3. Its six-condition matrix also passes every same-seed PatchTST cell;
the smallest PatchTST margin is `-10.750%` at 30% missing channels. The PEMS03
deployment checkpoint is one scan with no AutoCon inference computation, but
is `+34.631%` slower than matched PatchTST at batch four. It also exceeds
pinned official iTransformer by `-17.395%` (3/3; upper ratio `0.8313`). This
25% extension is validation-only; see
[`PEMS03 25% replication`](pems03_channel_patch_25pct_external_replication_20261007.md).

The strict PEMS03/25% causal gain also reproduces on an independently verified
RTX 3060 Ti host running Torch 2.6/CUDA 12.6/Mamba 2.2.4: `-0.406%` versus
forecast-only, 3/3, one-SE ratio upper `0.99662`. Source data, initialization
checkpoints, and code paths are hash-recorded; no test rows were opened. See
[`independent-machine packet`](pems03_independent_machine_replication_packet_20261007.md).

Official TimeMixer is a material PEMS03/25% external boundary. Across the
pre-registered five-seed precision expansion, Channel-Patch has a favorable
mean (`-4.422%`) but only 3/5 wins and a one-SE upper ratio `1.02625`; it fails
noninferiority. A capped 15-seed batch/update-matched expansion subsequently
resolves the budget asymmetry: `-3.350%`, 7/15 wins, one-SE upper `0.99978`.
PEMS03 TimeMixer is therefore a passed noninferiority comparison under that
matched training geometry.

Weather supplies validation external-domain support but not a confirmed causal
test gain. Its five-seed validation result is `-0.148%` versus forecast-only,
whereas the frozen checkpoint test extension is `+0.277%`. Electricity gains
against its own strong transfer baseline but remains externally inferior to
official PatchTST; the d32/24 Electricity bridge also fails to reproduce a
channel-patch benefit.

A separate post-registration Weather target-prefix calibration protocol repairs
the validation mechanism comparison: fresh Channel-Patch is `-0.280%` versus
same-calibration forecast-only and `-2.381%` versus official PatchTST, both
3/3, with all six validation stress cells externally noninferior. It is not a
replacement for the historically exposed unadapted Weather test extension;
see [`calibrated Weather replication`](weather_calibrated_channel_patch_replication_20261007.md).
The same protocol is also favorable versus iTransformer but fails official
TimeMixer noninferiority (`+0.034%`, 1/3; upper `1.00889`), so it cannot be
used as a universal strong-baseline claim.

The deployed checkpoint retains a clean one-scan path. It returns no AutoCon
representation and computes no relation loss. Its current batch-four latency
is not faster than official PatchTST. A fixed-shape CUDA Graph device-only
path preserves outputs exactly and lowers PEMS03 H192 batch-four latency by
`24.60%` to `0.25824 ms`, within `+1.52%` of PatchTST; this is not an
end-to-end or variable-shape speed claim.

This is not Q1-ready: pristine test governance, universal strong-baseline
competitiveness, and high-dimensional external competitiveness remain open.

Increasing the prediction-scaled Channel-Patch representation loss from `0.05`
to `0.10` under a strict PEMS03/25% target-excluded screen changes the weighted
objective as intended but improves validation MSE by only `-0.123%`, below its
registered `-0.25%` expansion threshold. The current `0.05` setting is not
retained merely because the representation loss was too weak.

Replacing flattened patch-trajectory cosine with same-position patch cosine on
strict PEMS03/25% changes MSE by only `-0.0013%`, below the `-0.25%` expansion
threshold. Local patch-position alignment is not the missing mechanism.

Adding a shared train-only `D -> D` MLP projection head before the same
Channel-Patch InfoNCE worsens the strict PEMS03/25% validation MSE by
`+0.430%` versus direct encoder states. The head is channel-count agnostic,
adds no scan, and is absent at inference, but does not improve transfer; the
identity path remains the deployed default. See
[`shared-projector boundary`](channel_patch_shared_projector_boundary_20261007.md).

Channel-Patch also does not repair the high-dimensional Electricity-to-Traffic
d32/24 bridge: it is `+0.320%` above the paired forecast-calibration control
despite active source and target-calibration ACF anchors. This validation-only
result closes the interpretation that the earlier Traffic failure was solely a
forecast-level relation-loss problem. See
[`Traffic d32 boundary`](traffic_d32_channel_patch_boundary_20261007.md).

Reweighting those high-dimensional Channel-Patch losses by normalized,
train-prefix per-channel ACF strength is also not the missing mechanism:
it changes Traffic validation MSE by `+0.023%` relative to equal aggregation.
The weight ranges are nontrivial and all anchors remain active, so this is not
a degenerate equal-weight control. See
[`Traffic weighting boundary`](traffic_d32_channel_patch_weighting_boundary_20261007.md).

Restricting high-dimensional Channel-Patch to the Electricity source while
retaining Traffic forecast calibration gives only `-0.043%` against the same
source-and-target Channel-Patch protocol. This is below the registered
`-0.25%` expansion threshold, so target-prefix representation interference is
not a sufficient explanation for the Traffic limitation. See
[`Traffic locality boundary`](traffic_d32_channel_patch_locality_boundary_20261007.md).

Replacing full patch-trajectory cosine with the forecast-origin final patch
state also changes high-dimensional Traffic validation MSE by only `-0.014%`.
The loss remains active, but this is far below the `-0.25%` expansion gate; it
does not support a claim that historical trajectory context is the limiting
factor. See
[`Traffic final-patch boundary`](traffic_d32_channel_patch_last_patch_boundary_20261007.md).

A loss-level channel-count-independent alternative also fails: four balanced,
train-prefix ACF groups with group-pooled patch states are `+0.806%` above the
same Traffic forecast-calibration control. Its grouped ACF anchors are active,
so this rules out a simple heterogeneous-channel aggregation explanation for
the high-dimensional limitation. See
[`Traffic grouped-patch boundary`](traffic_d32_grouped_patch_boundary_20261007.md).

Injecting a role-local, train-prefix ACF descriptor into the shared patch
encoder reduces Traffic's ordinary Channel-Patch regression (`-0.113%` versus
ordinary Channel-Patch) but remains `+0.135%` above forecast calibration. The
nonzero trained gate verifies that the path is active, yet it fails the two
registered expansion gates. See
[`Traffic ACF-descriptor boundary`](traffic_d32_acf_descriptor_boundary_20261007.md).

Allowing only the descriptor's existing scalar gate to adapt on target labels
does not repair this result. The gate changes materially while the shared core
remains frozen, yet its validation effect is `-0.00025%` versus the frozen
descriptor and `+1.345%` versus forecast calibration. See
[`Traffic target-gate boundary`](traffic_d32_acf_descriptor_target_scale_boundary_20261007.md).

Weather target-prefix calibration is a positive transfer-control result but not
yet a Channel-Patch promotion: on a validation-only screen it improves the
uncalibrated forecast control by `-2.778%`, while Channel-Patch adds `-0.178%`
relative to calibrated forecast-only, below the registered `-0.25%` expansion
threshold.

Machine-readable audit:

- `artifacts/paper_final/channel_patch_mechanism_evidence_20261006/audit.json`

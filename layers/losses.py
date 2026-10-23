import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def _strict_nonnegative_integer(value, label):
    """Parse relation geometry without bool coercion or decimal truncation."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a non-negative integer".format(label))
    if isinstance(value, (float, np.floating)) and (
        not np.isfinite(value) or not float(value).is_integer()
    ):
        raise ValueError("{} must be a non-negative integer".format(label))
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a non-negative integer".format(label)) from exc
    if not isinstance(value, (str, bytes)) and parsed != value:
        raise ValueError("{} must be a non-negative integer".format(label))
    if parsed < 0:
        raise ValueError("{} must be a non-negative integer".format(label))
    return parsed


def _strict_selected_lags(values):
    """Normalize selected relation peaks without changing the caller's lags."""

    if isinstance(values, (str, bytes)):
        raise ValueError("selected_lags must be a sequence of positive integers")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(
            "selected_lags must be a sequence of positive integers"
        ) from exc
    if not raw:
        raise ValueError("selected_lags must contain at least one positive lag")
    parsed = []
    for value in raw:
        lag = _strict_nonnegative_integer(value, "selected_lags")
        if lag <= 0:
            raise ValueError("selected_lags must contain at least one positive lag")
        parsed.append(lag)
    if len(set(parsed)) != len(parsed):
        raise ValueError("selected_lags must contain unique positive lags")
    return sorted(parsed)


class ACFRelationLoss(nn.Module):
    """Leakage-safe all-pair relation loss for signed ACF supervision.

    This objective is intentionally separate from :class:`AutoCon`, whose
    top-1 positive selection is retained for historical reproducibility.  It
    maps the signed train-split ACF relation between window origins to a soft
    distribution over candidate representations and adds a bounded sign
    penalty.  ``relation_scope='both'`` computes a global window objective and
    a local within-window objective for backward compatibility; ``'global'``
    is useful when dense local supervision would flatten a slowly decaying ACF,
    while ``'local'`` isolates that component for an ablation.  In every mode,
    one representation is compared per candidate window and channel-specific
    ACF values are reduced only after lag lookup.  It is not a
    channel-independent loss.  Exact identity pairs are removed by position,
    never by testing whether an ACF value happens to equal one.

    Magnitude ranking uses ``abs(cosine)`` and the sign penalty uses the raw
    cosine.  This keeps a strongly anti-correlated pair close in magnitude
    while still teaching the sign, instead of asking one scalar similarity to
    be both large and negative.

    Parameters
    ----------
    acf_values:
        ``[lag]`` or ``[channel, lag]`` signed ACF table.  Channel reduction is
        performed *after* lag lookup, preserving heterogeneous channel
        periodicity until the relation is formed.  The resulting relation is
        reduced to one value per window pair because the public feature
        contract is ``[batch, steps, dimension]``.
    channel_reduce:
        ``'mean'`` (default) averages signed channel relations after lookup.
        ``'max_abs'`` selects the signed relation with the largest magnitude
        for each pair, avoiding cancellation when channels have opposite
        phase.  Neither mode is a substitute for a true per-channel
        representation objective.
    relation_scope:
        ``'both'`` (default), ``'global'``, or ``'local'``.  The trainer
        averages only the active components, so selecting one scope does not
        silently halve the requested AutoCon weight.
    selected_lags:
        Optional relation-peak lag list.  In ``'hard_selected'`` mode, those
        lags (within ``lag_tolerance``) define positive relation targets.  In
        ``'soft_multi_peak'`` mode, they receive extra target weight while
        other sufficiently correlated train-only lags remain valid targets.
        In both modes, in-range non-identity pairs remain denominator
        candidates.
    lag_tolerance:
        Non-negative integer radius around each selected lag.
    target_mode:
        ``'hard_selected'`` preserves the selected-peak objective.  The
        ``'soft_multi_peak'`` mechanism-control mode uses all candidates with
        ``abs(ACF) > soft_min_abs`` as a soft target distribution, avoiding
        hard false negatives for unselected but strongly correlated lags.  It
        is not a general accuracy default; performance must be established by
        a separately registered comparison.
        ``'soft_full'`` is a sensitivity control that uses every non-zero ACF
        candidate and ignores ``soft_min_abs``.
    soft_min_abs:
        Magnitude threshold used only by ``'soft_multi_peak'``.
    diagnostic_min_abs:
        Independent absolute-ACF threshold used only for high-ACF telemetry.
        Keeping this separate from ``soft_min_abs`` prevents soft-target
        coverage from becoming a tautological zero-exclusion diagnostic.
    selected_lag_boost:
        Non-negative additional multiplicative weight for selected lags in
        ``'soft_multi_peak'``.  A value of one doubles their soft target mass.
    min_lag:
        Lower temporal-separation bound for both target and denominator
        candidates.  Setting it to the input length creates a global
        non-overlap relation control.
    normalize_global_active_anchors:
        Opt-in normalization for the global component.  When enabled, a
        later batch mean is taken over anchors with at least one positive
        relation; the default ``False`` preserves historical zero-filled
        batch averaging.

    Notes
    -----
    ``last_global_valid_fraction`` and
    ``last_global_active_anchor_fraction`` are detached runtime diagnostics
    for the active target support.  The corresponding
    ``last_global_candidate_*`` fields describe the denominator candidate set.
    ``last_global_contrastive_active_anchor_fraction`` requires both a positive
    target and at least one denominator negative; it distinguishes meaningful
    contrastive rows from the one-candidate case, where the magnitude softmax
    cross-entropy is identically zero.  The accompanying
    ``last_global_degenerate_positive_anchor_fraction`` is conditioned on
    anchors that have a positive target.  The remaining
    while ``last_global_high_acf_excluded_fraction`` catches a hard-target
    false-negative risk using the historical candidate-pair denominator.
    ``last_global_high_acf_candidate_count``,
    ``last_global_high_acf_excluded_count``, and
    ``last_global_high_acf_conditional_excluded_fraction`` expose the same
    risk with an explicit high-ACF denominator.  Sign accuracy and hinge
    diagnostics are weighted by the target ACF magnitude and never affect the
    loss value.  They are useful because the training DataLoader may shuffle
    window origins, making the global ACF relation sparse in a batch.
    ``last_{global,local}_magnitude_ce`` and
    ``last_{global,local}_signed_hinge_contribution`` are detached finite
    means of the two terms that reconstruct the corresponding returned
    relation loss.  The latter already includes ``sign_weight``.
    """

    objective_version = 3

    def __init__(self, acf_values, temperature=0.2, relation_temperature=0.25,
                 sign_weight=0.25, sign_margin=0.1, channel_reduce="mean",
                 relation_scope="both", selected_lags=None, lag_tolerance=0,
                 target_mode="hard_selected", soft_min_abs=0.1,
                 diagnostic_min_abs=0.1, selected_lag_boost=0.0, min_lag=0,
                 normalize_global_active_anchors=False, channel_selected_lags=None):
        super().__init__()
        # Clone caller-owned arrays so a non-writable NumPy view cannot leak
        # undefined mutation semantics into the registered buffer.
        values = torch.as_tensor(np.array(acf_values, dtype=np.float32, copy=True)).clone()
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("acf_values must have shape [lag] or [channel, lag]")
        if not torch.isfinite(values).all():
            raise ValueError("acf_values must be finite")
        try:
            temperature = float(temperature)
            relation_temperature = float(relation_temperature)
            sign_weight = float(sign_weight)
            sign_margin = float(sign_margin)
        except (TypeError, ValueError) as exc:
            raise ValueError("contrastive hyperparameters must be finite") from exc
        if (
            not np.isfinite(temperature)
            or not np.isfinite(relation_temperature)
            or temperature <= 0
            or relation_temperature <= 0
        ):
            raise ValueError("contrastive temperatures must be positive")
        if (
            not np.isfinite(sign_weight)
            or not np.isfinite(sign_margin)
            or sign_weight < 0
            or sign_margin < 0
        ):
            raise ValueError("sign_weight and sign_margin must be non-negative")
        if channel_reduce not in {"mean", "max_abs"}:
            raise ValueError("channel_reduce must be 'mean' or 'max_abs'")
        relation_scope = str(relation_scope).lower()
        if relation_scope not in {"both", "global", "local"}:
            raise ValueError("relation_scope must be 'both', 'global', or 'local'")
        lag_tolerance = _strict_nonnegative_integer(
            lag_tolerance, "lag_tolerance"
        )
        target_mode = str(target_mode).lower()
        if target_mode not in {"hard_selected", "soft_multi_peak", "soft_full"}:
            raise ValueError(
                "target_mode must be 'hard_selected', 'soft_multi_peak', or 'soft_full'"
            )
        if not np.isfinite(soft_min_abs) or not 0.0 <= float(soft_min_abs) <= 1.0:
            raise ValueError("soft_min_abs must be finite and in [0, 1]")
        if not np.isfinite(diagnostic_min_abs) or not 0.0 <= float(diagnostic_min_abs) <= 1.0:
            raise ValueError("diagnostic_min_abs must be finite and in [0, 1]")
        if not np.isfinite(selected_lag_boost) or float(selected_lag_boost) < 0.0:
            raise ValueError("selected_lag_boost must be finite and non-negative")
        min_lag = _strict_nonnegative_integer(min_lag, "min_lag")
        if min_lag > values.shape[1] - 1:
            raise ValueError(
                "min_lag={} exceeds the measured ACF range [0, {}]".format(
                    min_lag, values.shape[1] - 1
                )
            )
        if selected_lags is None:
            parsed_lags = []
        else:
            parsed_lags = _strict_selected_lags(selected_lags)
        channel_lags = None
        if channel_selected_lags is not None:
            channel_lags = torch.as_tensor(channel_selected_lags, dtype=torch.long)
            if channel_lags.ndim != 2 or channel_lags.shape[0] != values.shape[0]:
                raise ValueError(
                    "channel_selected_lags must have shape [acf_channels, peak_slots]"
                )
            if channel_lags.shape[1] < 1:
                raise ValueError("channel_selected_lags must reserve at least one peak slot")
            if torch.any(channel_lags < 0) or torch.any(channel_lags > values.shape[1] - 1):
                raise ValueError("channel_selected_lags must contain lags in [0, max_lag]")
        self.register_buffer("acf_values", values.clamp(-1.0, 1.0))
        self.register_buffer(
            "selected_lags", torch.as_tensor(parsed_lags, dtype=torch.long)
        )
        self.register_buffer(
            "channel_selected_lags",
            torch.empty(0, 0, dtype=torch.long) if channel_lags is None else channel_lags,
        )
        self.temperature = temperature
        self.relation_temperature = relation_temperature
        self.sign_weight = sign_weight
        self.sign_margin = sign_margin
        self.channel_reduce = channel_reduce
        self.relation_scope = relation_scope
        self.lag_tolerance = lag_tolerance
        self.target_mode = target_mode
        self.soft_min_abs = float(soft_min_abs)
        self.diagnostic_min_abs = float(diagnostic_min_abs)
        self.selected_lag_boost = float(selected_lag_boost)
        self.min_lag = min_lag
        # The historical global objective averages over every batch element,
        # including anchors without an observed selected-lag positive.  Keep
        # that behavior by default.  The explicit opt-in instead makes the
        # batch mean equal the mean over active global anchors, avoiding an
        # incidental dependence on relation-pair coverage.
        self.normalize_global_active_anchors = bool(
            normalize_global_active_anchors
        )
        # Local relation geometry depends only on the number of representation
        # steps and the immutable train-only ACF table.  Cache that geometry
        # outside the autograd graph; global geometry remains batch/origin
        # dependent and is intentionally rebuilt on every call.
        self._local_lookup_cache = {}
        # These are deliberately non-persistent: they describe the most
        # recent batch, not model state or a training parameter.
        self.register_buffer(
            "last_global_valid_fraction", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_global_active_anchor_fraction", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_global_candidate_fraction", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_global_candidate_active_anchor_fraction",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_candidate_count_mean", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_global_contrastive_active_anchor_fraction",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_degenerate_positive_anchor_fraction",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_positive_count_mean", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_global_high_acf_excluded_fraction",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_high_acf_candidate_count",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_high_acf_excluded_count",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_high_acf_conditional_excluded_fraction",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_sign_target_count",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_sign_accuracy",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_sign_hinge",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_magnitude_ce",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_global_signed_hinge_contribution",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_local_target_fraction", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_local_active_position_fraction",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_local_sign_target_count",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_local_sign_accuracy",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_local_sign_hinge",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_local_magnitude_ce",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_local_signed_hinge_contribution",
            torch.tensor(0.0),
            persistent=False,
        )

    def _lookup_components(self, distances):
        """Look up relations and return candidate/selected lag masks.

        The ACF table is only measured for lags ``[0, max_lag]``.  A lower
        bound is applied here, before identity removal, so it constrains both
        positive targets and denominator negatives.  ``selected`` is kept as
        a separate mask because soft targets must be able to include strong
        unselected peaks as well.
        """
        acf_values = self.acf_values.to(distances.device)
        max_lag = acf_values.shape[1] - 1
        in_range = (distances >= self.min_lag) & (distances <= max_lag)
        if self.selected_lags.numel():
            selected = self.selected_lags.to(distances.device)
            distance_to_peak = (distances.unsqueeze(-1) - selected).abs()
            selected = in_range & (distance_to_peak <= self.lag_tolerance).any(dim=-1)
        else:
            # No explicit peak set: every in-range pair can be a target, but
            # none receives the selected-lag boost.
            selected = torch.zeros_like(in_range)
        safe = distances.clamp(min=0, max=max_lag)
        # [C,*] -> [*,C], then reduce only after indexing each channel.
        gathered = acf_values[:, safe].movedim(0, -1)
        if self.channel_reduce == "mean":
            relation = gathered.mean(dim=-1)
        else:
            # Keep the sign of the strongest channel rather than averaging
            # opposite phases into an artificial zero relation.
            strongest = gathered.abs().argmax(dim=-1, keepdim=True)
            relation = gathered.gather(-1, strongest).squeeze(-1)
        return relation, in_range, selected

    def _lookup(self, distances):
        """Return channel-averaged relations and the legacy target mask.

        Existing callers used the second value as the selected-lag mask.  The
        training path uses :meth:`_lookup_components` so candidate and target
        semantics remain explicit.
        """
        relation, in_range, selected = self._lookup_components(distances)
        return relation, selected if self.selected_lags.numel() else in_range

    @staticmethod
    def _tensor_version(tensor):
        """Return a mutation token when PyTorch exposes one."""

        try:
            return int(tensor._version)
        except (AttributeError, RuntimeError):
            return None

    def _cached_local_lookup(self, steps, device):
        """Return the fixed local ACF map for ``steps`` representation tokens.

        The cache is deliberately bounded and keyed by buffer identity/version
        so an in-place prior update or a module device move cannot reuse stale
        relations.  A cache miss still follows ``_lookup_components`` exactly.
        """

        steps = int(steps)
        device = torch.device(device)
        acf_version = self._tensor_version(self.acf_values)
        selected_version = self._tensor_version(self.selected_lags)
        key = (
            device.type,
            device.index,
            steps,
            id(self.acf_values),
            acf_version,
            id(self.selected_lags),
            selected_version,
            str(self.channel_reduce),
            int(self.min_lag),
            int(self.lag_tolerance),
        )
        if acf_version is not None and selected_version is not None:
            cached = self._local_lookup_cache.get(key)
            if cached is not None:
                return cached

        positions = torch.arange(steps, device=device, dtype=torch.long)
        distances = (positions[:, None] - positions[None, :]).abs()
        relation, in_range, selected = self._lookup_components(distances)
        result = (
            relation.detach(),
            in_range.detach(),
            selected.detach(),
        )
        if acf_version is not None and selected_version is not None:
            # Representation lengths are normally fixed; retaining a handful
            # of entries covers multiscale callers without unbounded growth.
            if len(self._local_lookup_cache) >= 8:
                self._local_lookup_cache.pop(next(iter(self._local_lookup_cache)))
            self._local_lookup_cache[key] = result
        return result

    def _target_specification(self, relation, candidate, selected):
        """Build target support and non-negative target weights."""
        candidate = candidate.to(dtype=torch.bool, device=relation.device)
        selected = selected.to(dtype=torch.bool, device=relation.device)
        magnitude = relation.abs()
        if self.target_mode in {"soft_multi_peak", "soft_full"}:
            # Strict thresholding keeps weak/zero ACF entries from becoming
            # artificial uniform positives.  Strong unselected peaks remain
            # targets, which is the key difference from hard selected mode.
            threshold = self.soft_min_abs if self.target_mode == "soft_multi_peak" else 0.0
            target = candidate & (magnitude > threshold)
            weights = magnitude
            if self.selected_lags.numel() and self.selected_lag_boost:
                boost = torch.where(
                    selected,
                    relation.new_tensor(1.0 + self.selected_lag_boost),
                    relation.new_tensor(1.0),
                )
                weights = weights * boost
        elif self.selected_lags.numel():
            target = candidate & selected
            weights = magnitude
        else:
            target = candidate
            weights = magnitude
        # A soft target with an explicitly zero weight is not a supervised
        # pair.  Hard selected mode retains its historical uniform behavior
        # when a selected lag happens to have zero ACF.
        if self.target_mode in {"soft_multi_peak", "soft_full"}:
            target = target & (weights > 0)
        return target, weights

    def _pair_loss(self, similarities, relation, valid, candidate_mask=None,
                   target_weights=None, return_components=False):
        """Compute relation cross-entropy plus a sign-aware hinge term.

        ``return_components`` is an opt-in telemetry path.  The default return
        value remains the historical per-anchor loss tensor, so existing
        callers and gradients are unchanged.  When requested, the two extra
        tensors are the magnitude cross-entropy and the *weighted* signed
        hinge contribution after inactive-anchor masking.
        """
        # Relation magnitude is phase-invariant; the separate sign term below
        # carries the direction.  Using raw similarities for both terms would
        # make a negative ACF pair simultaneously attractive and repulsive.
        logits = similarities.abs() / self.temperature
        if candidate_mask is None:
            candidate_mask = valid
        candidate_mask = candidate_mask.to(dtype=torch.bool, device=similarities.device)
        valid = valid.to(dtype=torch.bool, device=similarities.device)
        has_candidate = candidate_mask.any(dim=-1)
        has_positive = valid.any(dim=-1)
        # ``-inf`` is safe only for rows with at least one candidate.  Empty
        # rows are explicitly zeroed below instead of manufacturing a relation
        # from an out-of-range lag.
        masked_logits = logits.masked_fill(~candidate_mask, float("-inf"))
        safe_logits = torch.where(
            has_candidate.unsqueeze(-1), masked_logits, torch.zeros_like(masked_logits)
        )
        log_q = torch.log_softmax(safe_logits, dim=-1)

        if target_weights is None:
            target_weights = relation.abs()
        target_weights = target_weights.to(relation.device, dtype=relation.dtype)
        magnitude = relation.abs()
        target_logits = magnitude / self.relation_temperature
        # Multiplicative soft-target boosts are represented as an additive log
        # offset, preserving the temperature-controlled ACF ranking.  Keep a
        # zero-magnitude hard target finite; soft mode masks zero weights
        # before reaching this path.
        boost_ratio = target_weights / magnitude.clamp_min(1e-12)
        boost_log = torch.log(boost_ratio.clamp_min(1e-12))
        target_logits = target_logits + torch.where(
            magnitude > 1e-12, boost_log, torch.zeros_like(boost_log)
        )
        target_logits = target_logits.masked_fill(~valid, float("-inf"))
        safe_target_logits = torch.where(
            has_positive.unsqueeze(-1), target_logits, torch.zeros_like(target_logits)
        )
        target = torch.softmax(safe_target_logits, dim=-1)
        relation_ce = -torch.where(
            valid, target * log_q, torch.zeros_like(target)
        ).sum(dim=-1)

        weighted = target_weights * valid.to(relation.dtype)
        positive = relation > 0
        sign_penalty = torch.where(
            positive,
            F.relu(self.sign_margin - similarities),
            F.relu(self.sign_margin + similarities),
        )
        sign_penalty = sign_penalty * weighted
        sign_loss = sign_penalty.sum(dim=-1) / weighted.sum(dim=-1).clamp_min(1e-6)
        signed_hinge_contribution = self.sign_weight * sign_loss
        loss = relation_ce + signed_hinge_contribution
        # A denominator candidate without a selected positive is not a
        # supervised anchor; leave it at zero while retaining it as a negative
        # for anchors that do have a positive target.
        active_loss = torch.where(has_positive, loss, torch.zeros_like(loss))
        if return_components:
            active_relation_ce = torch.where(
                has_positive, relation_ce, torch.zeros_like(relation_ce)
            )
            active_signed_hinge = torch.where(
                has_positive,
                signed_hinge_contribution,
                torch.zeros_like(signed_hinge_contribution),
            )
            return active_loss, active_relation_ce, active_signed_hinge
        return active_loss

    @staticmethod
    def _finite_component_mean(values):
        """Return a finite detached scalar for telemetry only.

        The sanitization is deliberately outside the returned loss graph.  A
        malformed/non-finite batch should not make JSON logging fail, while
        the actual optimization tensor retains its original behavior.
        """

        values = values.detach().to(dtype=torch.float32)
        return torch.nan_to_num(values.mean(), nan=0.0, posinf=0.0, neginf=0.0)

    def _record_pair_components(self, scope, magnitude_ce, signed_hinge):
        """Persist detached component means for one relation scope."""

        if scope == "global":
            self.last_global_magnitude_ce = self._finite_component_mean(magnitude_ce)
            self.last_global_signed_hinge_contribution = (
                self._finite_component_mean(signed_hinge)
            )
        elif scope == "local":
            self.last_local_magnitude_ce = self._finite_component_mean(magnitude_ce)
            self.last_local_signed_hinge_contribution = (
                self._finite_component_mean(signed_hinge)
            )
        else:
            raise ValueError("scope must be 'global' or 'local'")

    def _clear_pair_components(self, scope, device):
        """Reset a scope's telemetry before an inactive/empty branch."""

        zero = torch.zeros((), device=device, dtype=torch.float32)
        self._record_pair_components(scope, zero, zero)

    def _sign_diagnostics(self, similarities, relation, valid, target_weights):
        """Return target count, weighted sign accuracy, and weighted hinge.

        ``valid`` identifies supervised relation pairs while ``target_weights``
        carries their non-negative ACF magnitude (and any configured boost).
        Broadcasting is intentional: local similarities have a batch axis,
        whereas the local relation table is shared by all windows.
        """
        # Diagnostics are telemetry only; avoid retaining a second autograd
        # graph for the (potentially large) local similarity matrix.
        similarities = similarities.detach()
        relation = relation.detach()
        valid = valid.detach()
        target_weights = target_weights.detach()
        similarities, relation, valid, target_weights = torch.broadcast_tensors(
            similarities,
            relation.to(device=similarities.device, dtype=similarities.dtype),
            valid.to(device=similarities.device, dtype=torch.bool),
            target_weights.to(device=similarities.device, dtype=similarities.dtype),
        )
        valid_float = valid.to(similarities.dtype)
        weights = target_weights.clamp_min(0.0) * valid_float
        weight_sum = weights.sum(dtype=torch.float32)
        target_count = valid.sum(dtype=torch.float32)
        same_sign = (relation * similarities > 0).to(similarities.dtype)
        sign_accuracy = (weights * same_sign).sum(dtype=torch.float32) / weight_sum.clamp_min(1e-6)
        sign_penalty = torch.where(
            relation > 0,
            F.relu(self.sign_margin - similarities),
            F.relu(self.sign_margin + similarities),
        )
        sign_hinge = (weights * sign_penalty).sum(dtype=torch.float32) / weight_sum.clamp_min(1e-6)
        # With no positive target (or only zero-weight targets), report zeros
        # rather than a misleading perfect score from the clamped denominator.
        active = weight_sum > 0
        sign_accuracy = torch.where(active, sign_accuracy, sign_accuracy.new_zeros(()))
        sign_hinge = torch.where(active, sign_hinge, sign_hinge.new_zeros(()))
        return target_count, sign_accuracy, sign_hinge

    def _record_global_coverage(
        self,
        valid,
        candidate_mask=None,
        relation=None,
        similarities=None,
        target_weights=None,
    ):
        """Save relation coverage and optional sign diagnostics."""

        if candidate_mask is None:
            candidate_mask = valid
        batch = valid.shape[0]
        if batch < 2:
            valid_fraction = valid.new_zeros((), dtype=torch.float32)
            active_fraction = valid.new_zeros((), dtype=torch.float32)
            candidate_fraction = valid.new_zeros((), dtype=torch.float32)
            candidate_active_fraction = valid.new_zeros((), dtype=torch.float32)
            candidate_count_mean = valid.new_zeros((), dtype=torch.float32)
            contrastive_active_fraction = valid.new_zeros((), dtype=torch.float32)
            degenerate_positive_fraction = valid.new_zeros(
                (), dtype=torch.float32
            )
            positive_count_mean = valid.new_zeros((), dtype=torch.float32)
            high_acf_excluded_fraction = valid.new_zeros((), dtype=torch.float32)
            high_acf_candidate_count = valid.new_zeros((), dtype=torch.float32)
            high_acf_excluded_count = valid.new_zeros((), dtype=torch.float32)
            high_acf_conditional_excluded_fraction = valid.new_zeros(
                (), dtype=torch.float32
            )
        else:
            pair_count = batch * (batch - 1)
            valid_fraction = valid.sum(dtype=torch.float32) / float(pair_count)
            active_fraction = valid.any(dim=-1).to(torch.float32).mean()
            candidate_fraction = candidate_mask.sum(dtype=torch.float32) / float(pair_count)
            candidate_active_fraction = candidate_mask.any(dim=-1).to(torch.float32).mean()
            candidate_count = candidate_mask.sum(dim=-1, dtype=torch.float32)
            positive_active = valid.any(dim=-1)
            negative = candidate_mask & ~valid
            has_negative = negative.any(dim=-1)
            candidate_count_mean = candidate_count.mean()
            contrastive_active_fraction = (
                positive_active & has_negative
            ).to(torch.float32).mean()
            degenerate_positive_fraction = (
                (positive_active & ~has_negative).sum(dtype=torch.float32)
                / positive_active.sum(dtype=torch.float32).clamp_min(1.0)
            )
            positive_count_mean = valid.sum(dim=-1, dtype=torch.float32).mean()
            if relation is None:
                high_acf_excluded_fraction = valid.new_zeros((), dtype=torch.float32)
                high_acf_candidate_count = valid.new_zeros((), dtype=torch.float32)
                high_acf_excluded_count = valid.new_zeros((), dtype=torch.float32)
                high_acf_conditional_excluded_fraction = valid.new_zeros(
                    (), dtype=torch.float32
                )
            else:
                # This threshold is deliberately independent of the target
                # mode.  In soft_multi_peak mode, reusing soft_min_abs would
                # make high_acf_excluded identically zero by construction.
                high_acf = candidate_mask & (relation.abs() > self.diagnostic_min_abs)
                high_acf_candidate_count = high_acf.sum(dtype=torch.float32)
                high_acf_excluded_count = (high_acf & ~valid).sum(dtype=torch.float32)
                high_acf_excluded_fraction = (
                    high_acf_excluded_count
                    / candidate_mask.sum(dtype=torch.float32).clamp_min(1.0)
                )
                high_acf_conditional_excluded_fraction = (
                    high_acf_excluded_count
                    / high_acf_candidate_count.clamp_min(1.0)
                )
        if (
            similarities is not None
            and relation is not None
            and target_weights is not None
        ):
            sign_target_count, sign_accuracy, sign_hinge = self._sign_diagnostics(
                similarities, relation, valid, target_weights
            )
        else:
            sign_target_count = valid.new_zeros((), dtype=torch.float32)
            sign_accuracy = valid.new_zeros((), dtype=torch.float32)
            sign_hinge = valid.new_zeros((), dtype=torch.float32)
        # Assignment retains their buffer registration while allowing this
        # standalone loss object to follow the feature device lazily.
        self.last_global_valid_fraction = valid_fraction.detach()
        self.last_global_active_anchor_fraction = active_fraction.detach()
        self.last_global_candidate_fraction = candidate_fraction.detach()
        self.last_global_candidate_active_anchor_fraction = candidate_active_fraction.detach()
        self.last_global_candidate_count_mean = candidate_count_mean.detach()
        self.last_global_contrastive_active_anchor_fraction = (
            contrastive_active_fraction.detach()
        )
        self.last_global_degenerate_positive_anchor_fraction = (
            degenerate_positive_fraction.detach()
        )
        self.last_global_positive_count_mean = positive_count_mean.detach()
        self.last_global_high_acf_excluded_fraction = high_acf_excluded_fraction.detach()
        self.last_global_high_acf_candidate_count = high_acf_candidate_count.detach()
        self.last_global_high_acf_excluded_count = high_acf_excluded_count.detach()
        self.last_global_high_acf_conditional_excluded_fraction = (
            high_acf_conditional_excluded_fraction.detach()
        )
        self.last_global_sign_target_count = sign_target_count.detach()
        self.last_global_sign_accuracy = sign_accuracy.detach()
        self.last_global_sign_hinge = sign_hinge.detach()

    def _record_local_coverage(
        self,
        target,
        candidate,
        similarities=None,
        relation=None,
        target_weights=None,
    ):
        """Record pair/anchor coverage and optional local sign diagnostics."""
        if target.numel() == 0 or candidate.numel() == 0:
            self.last_local_target_fraction = candidate.new_zeros((), dtype=torch.float32)
            self.last_local_active_position_fraction = candidate.new_zeros((), dtype=torch.float32)
            self.last_local_sign_target_count = candidate.new_zeros((), dtype=torch.float32)
            self.last_local_sign_accuracy = candidate.new_zeros((), dtype=torch.float32)
            self.last_local_sign_hinge = candidate.new_zeros((), dtype=torch.float32)
            return
        candidate_count = candidate.sum(dtype=torch.float32)
        target_count = target.sum(dtype=torch.float32)
        self.last_local_target_fraction = (
            target_count / candidate_count.clamp_min(1.0)
        ).detach()
        self.last_local_active_position_fraction = (
            target.any(dim=-1).to(torch.float32).mean()
        ).detach()
        if (
            similarities is not None
            and relation is not None
            and target_weights is not None
        ):
            sign_target_count, sign_accuracy, sign_hinge = self._sign_diagnostics(
                similarities, relation, target, target_weights
            )
            self.last_local_sign_target_count = sign_target_count.detach()
            self.last_local_sign_accuracy = sign_accuracy.detach()
            self.last_local_sign_hinge = sign_hinge.detach()
        else:
            self.last_local_sign_target_count = candidate.new_zeros((), dtype=torch.float32)
            self.last_local_sign_accuracy = candidate.new_zeros((), dtype=torch.float32)
            self.last_local_sign_hinge = candidate.new_zeros((), dtype=torch.float32)

    def local_contrastive_loss(self, features):
        batch, steps, _ = features.shape
        if self.relation_scope == "global":
            self._clear_pair_components("local", features.device)
            self._record_local_coverage(
                torch.zeros((steps, steps), device=features.device, dtype=torch.bool),
                torch.zeros((steps, steps), device=features.device, dtype=torch.bool),
            )
            return features.new_zeros((batch, steps))
        if steps < 2:
            self._clear_pair_components("local", features.device)
            self._record_local_coverage(
                torch.zeros((steps, steps), device=features.device, dtype=torch.bool),
                torch.zeros((steps, steps), device=features.device, dtype=torch.bool),
            )
            return features.new_zeros((batch, steps))
        relation, in_range, selected = self._cached_local_lookup(
            steps, features.device
        )
        eye = torch.eye(steps, device=features.device, dtype=torch.bool)
        candidate = in_range & ~eye
        positive, target_weights = self._target_specification(
            relation, candidate, selected
        )
        similarities = torch.matmul(features, features.transpose(1, 2))
        relation = relation.to(features.device)
        positive = positive.to(features.device)
        candidate = candidate.to(features.device)
        self._record_local_coverage(
            positive,
            candidate,
            similarities=similarities,
            relation=relation,
            target_weights=target_weights,
        )
        loss, magnitude_ce, signed_hinge = self._pair_loss(
            similarities,
            relation,
            positive,
            candidate_mask=candidate,
            target_weights=target_weights,
            return_components=True,
        )
        active_positions = positive.any(dim=-1).to(loss.dtype).sum(dim=-1).clamp_min(1.0)
        # Normalize each temporal anchor over its active relation positions;
        # inactive positions stay zero and do not dilute the auxiliary loss.
        scale = (float(steps) / active_positions).view(1, -1)
        loss = loss * scale
        magnitude_ce = magnitude_ce * scale
        signed_hinge = signed_hinge * scale
        self._record_pair_components("local", magnitude_ce, signed_hinge)
        return loss

    def global_contrastive_loss(self, features, origins):
        batch, steps, _ = features.shape
        if self.relation_scope == "local":
            self._clear_pair_components("global", features.device)
            self._record_global_coverage(
                torch.zeros((batch, batch), device=features.device, dtype=torch.bool),
                torch.zeros((batch, batch), device=features.device, dtype=torch.bool),
            )
            self.last_global_high_acf_excluded_fraction = features.new_zeros(())
            return features.new_zeros((batch,))
        if batch < 2:
            self._clear_pair_components("global", features.device)
            self._record_global_coverage(
                torch.zeros((batch, batch), device=features.device, dtype=torch.bool),
                torch.zeros((batch, batch), device=features.device, dtype=torch.bool),
            )
            return features.new_zeros((batch,))
        if origins is None:
            origins = torch.arange(batch, device=features.device, dtype=torch.long)
        else:
            origins = origins.to(features.device).long().reshape(-1)
            if origins.numel() != batch:
                raise ValueError("origins must have one value per feature window")
        pooled = F.normalize(features.mean(dim=1), dim=-1)
        similarities = torch.matmul(pooled, pooled.transpose(0, 1))
        distances = (origins[:, None] - origins[None, :]).abs()
        relation, in_range, selected = self._lookup_components(distances)
        eye = torch.eye(batch, device=features.device, dtype=torch.bool)
        candidate = in_range & ~eye
        positive, target_weights = self._target_specification(
            relation, candidate, selected
        )
        positive = positive.to(features.device)
        candidate = candidate.to(features.device)
        relation = relation.to(features.device)
        self._record_global_coverage(
            positive,
            candidate,
            relation,
            similarities=similarities,
            target_weights=target_weights,
        )
        loss, magnitude_ce, signed_hinge = self._pair_loss(
            similarities,
            relation,
            positive,
            candidate_mask=candidate,
            target_weights=target_weights,
            return_components=True,
        )
        if self.normalize_global_active_anchors:
            active_anchors = positive.any(dim=-1)
            active_count = active_anchors.sum().to(loss.dtype)
            # The inactive rows are already zero.  Scaling active rows makes
            # a later ``loss.mean()`` an active-anchor mean without creating
            # a gradient when this mini-batch has no supervised pair.
            scale = loss.new_tensor(float(batch)) / active_count.clamp_min(1.0)
            loss = torch.where(active_anchors, loss * scale, loss)
            magnitude_ce = torch.where(
                active_anchors, magnitude_ce * scale, magnitude_ce
            )
            signed_hinge = torch.where(
                active_anchors, signed_hinge * scale, signed_hinge
            )
        self._record_pair_components("global", magnitude_ce, signed_hinge)
        return loss

    def channelwise_global_contrastive_loss(self, features, origins):
        """Apply each channel's train-only ACF to its own window-state relation.

        ``features`` has shape ``[batch, channels, dimension]``.  This is a
        source-pretraining-only objective for channel-count agnostic encoders:
        the representation projection is shared, while every channel retains
        its signed ACF instead of collapsing incompatible phases by averaging.
        It has no forecasting-time path or extra temporal scan.
        """
        if features.ndim != 3:
            raise ValueError("channelwise features must have shape [batch, channels, dimension]")
        batch, channels, _ = features.shape
        if channels != self.acf_values.shape[0]:
            raise ValueError(
                "channelwise features require {} ACF channels, got {}".format(
                    self.acf_values.shape[0], channels
                )
            )
        if batch < 2:
            self._clear_pair_components("global", features.device)
            self._record_global_coverage(
                torch.zeros((batch, batch), device=features.device, dtype=torch.bool),
                torch.zeros((batch, batch), device=features.device, dtype=torch.bool),
            )
            return features.new_zeros((batch, channels))
        if origins is None:
            origins = torch.arange(batch, device=features.device, dtype=torch.long)
        else:
            origins = origins.to(features.device).long().reshape(-1)
            if origins.numel() != batch:
                raise ValueError("origins must have one value per feature window")
        states = F.normalize(features.transpose(0, 1), dim=-1)
        similarities = torch.matmul(states, states.transpose(1, 2))
        distances = (origins[:, None] - origins[None, :]).abs()
        _, in_range, selected = self._lookup_components(distances)
        safe = distances.clamp(min=0, max=self.acf_values.shape[1] - 1)
        relation = self.acf_values.to(features.device)[:, safe]
        eye = torch.eye(batch, device=features.device, dtype=torch.bool)
        candidate = (in_range & ~eye).unsqueeze(0).expand(channels, -1, -1)
        if self.channel_selected_lags.numel():
            lags = self.channel_selected_lags.to(features.device)
            # Zero is padding, never a cross-window positive because the
            # diagonal is excluded below. Every nonzero entry is a channel's
            # train-prefix ACF peak, with no source-wide union.
            channel_selected = (
                distances.unsqueeze(0).unsqueeze(-1) == lags[:, None, None, :]
            ).any(dim=-1)
            selected = channel_selected & in_range.unsqueeze(0) & ~eye.unsqueeze(0)
        else:
            selected = selected.unsqueeze(0).expand_as(candidate)
        positive, target_weights = self._target_specification(
            relation, candidate, selected
        )
        loss, magnitude_ce, signed_hinge = self._pair_loss(
            similarities, relation, positive, candidate_mask=candidate,
            target_weights=target_weights, return_components=True,
        )
        # Reuse the aggregate coverage fields for compatibility, then replace
        # the active-anchor fraction with the exact channelwise denominator.
        self._record_global_coverage(
            positive.any(dim=0), candidate.any(dim=0), relation.mean(dim=0),
            similarities=similarities.mean(dim=0),
            target_weights=target_weights.mean(dim=0),
        )
        active = positive.any(dim=-1) & (candidate & ~positive).any(dim=-1)
        self.last_global_contrastive_active_anchor_fraction = active.float().mean().detach()
        if self.normalize_global_active_anchors:
            # A channel-specific peak can be rare in a heterogeneous sensor
            # pool. Normalize by active channel-window anchors so an ACF term
            # is not silently shrunk as channel count grows.
            scale = loss.new_tensor(float(batch * channels)) / active.sum().to(loss.dtype).clamp_min(1.0)
            loss = torch.where(active, loss * scale, loss)
            magnitude_ce = torch.where(active, magnitude_ce * scale, magnitude_ce)
            signed_hinge = torch.where(active, signed_hinge * scale, signed_hinge)
        self._record_pair_components("global", magnitude_ce, signed_hinge)
        return loss.transpose(0, 1)

    def forward(self, features, labels=None):
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, steps, dimension]")
        normalized = F.normalize(features, dim=-1)
        local = self.local_contrastive_loss(normalized)
        global_ = self.global_contrastive_loss(normalized, labels)
        return local, global_


class ForecastACFConsistencyLoss(nn.Module):
    """Match forecast temporal correlations to a train-only signed ACF prior.

    Unlike :class:`ACFRelationLoss`, this objective consumes the forecast
    itself.  It has no trainable parameters and is evaluated only while
    training.  The ACF table is fixed from the training split; neither future
    labels nor validation/test samples are used to select lags or targets.
    """

    def __init__(self, acf_values, horizon, top_k=4, min_lag=2,
                 min_abs=0.10, eps=1e-6):
        super().__init__()
        values = torch.as_tensor(acf_values, dtype=torch.float32).clone()
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("acf_values must have shape [channel, lag>=2]")
        if not torch.isfinite(values).all():
            raise ValueError("acf_values must be finite")
        horizon = _strict_nonnegative_integer(horizon, "horizon")
        top_k = _strict_nonnegative_integer(top_k, "top_k")
        min_lag = _strict_nonnegative_integer(min_lag, "min_lag")
        if horizon < 2 or top_k < 1:
            raise ValueError("horizon must be >= 2 and top_k must be >= 1")
        if min_lag >= horizon:
            raise ValueError("min_lag must be smaller than horizon")
        if not np.isfinite(min_abs) or not 0.0 <= float(min_abs) <= 1.0:
            raise ValueError("min_abs must be finite and in [0, 1]")

        max_lag = min(values.shape[1] - 1, horizon - 1)
        candidates = torch.arange(min_lag, max_lag + 1, dtype=torch.long)
        if candidates.numel() == 0:
            raise ValueError("no ACF lag is visible in the forecast horizon")
        strength = values.index_select(1, candidates).abs().mean(dim=0)
        eligible = candidates[strength >= float(min_abs)]
        if eligible.numel() == 0:
            # Do not silently select an almost-zero prior: a zero loss is a
            # more truthful outcome for an unsupported data/horizon pair.
            selected = torch.empty(0, dtype=torch.long)
        else:
            eligible_strength = values.index_select(1, eligible).abs().mean(dim=0)
            count = min(int(top_k), int(eligible.numel()))
            selected = eligible[torch.topk(eligible_strength, count).indices]
            selected = selected.sort().values
        targets = values.index_select(1, selected) if selected.numel() else values[:, :0]
        weights = targets.abs().mean(dim=0)
        if weights.numel():
            weights = weights / weights.sum().clamp_min(float(eps))
        self.register_buffer("selected_lags", selected)
        self.register_buffer("targets", targets.clamp(-1.0, 1.0))
        self.register_buffer("weights", weights)
        self.eps = float(eps)

    def forward(self, forecast):
        if forecast.ndim != 3:
            raise ValueError("forecast must have shape [batch, horizon, channel]")
        if forecast.shape[1] < 2:
            raise ValueError("forecast horizon must be >= 2")
        if forecast.shape[-1] != self.targets.shape[0]:
            raise ValueError(
                "forecast channel count does not match the train-only ACF prior"
            )
        if self.selected_lags.numel() == 0:
            return forecast.new_zeros(forecast.shape[0])

        centered = forecast - forecast.mean(dim=1, keepdim=True)
        per_lag = []
        for lag in self.selected_lags.tolist():
            left = centered[:, :-lag, :]
            right = centered[:, lag:, :]
            correlation = (left * right).sum(dim=1)
            correlation = correlation / (
                left.square().sum(dim=1).sqrt()
                * right.square().sum(dim=1).sqrt()
            ).clamp_min(self.eps)
            per_lag.append(correlation)
        predicted = torch.stack(per_lag, dim=-1)
        target = self.targets.to(dtype=forecast.dtype).unsqueeze(0)
        squared_error = (predicted - target).square()
        return (squared_error * self.weights.to(dtype=forecast.dtype).view(1, 1, -1)).sum(dim=-1).mean(dim=-1)


class ForecastLagPairConsistencyLoss(nn.Module):
    """Supervise forecasts from ACF-lag paired input windows.

    The relation target is the signed, train-only ACF at the distance between
    two window origins.  This is intentionally different from
    :class:`ForecastACFConsistencyLoss`, which correlates positions *within*
    one forecast.  Here, the prediction head receives gradients from two
    distinct forecasts.  Pair construction itself remains outside this loss
    (``ACFLagPairBatchSampler``), so inference has no additional branch,
    state, or cost.
    """

    def __init__(self, acf_values, selected_lags, lag_tolerance=0, eps=1e-6):
        super().__init__()
        values = torch.as_tensor(np.array(acf_values, dtype=np.float32, copy=True)).clone()
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("acf_values must have shape [channel, lag>=2]")
        if not torch.isfinite(values).all():
            raise ValueError("acf_values must be finite")
        lags = _strict_selected_lags(selected_lags)
        if max(lags) >= values.shape[1]:
            raise ValueError("selected_lags must be inside the ACF table")
        tolerance = _strict_nonnegative_integer(lag_tolerance, "lag_tolerance")
        if not np.isfinite(eps) or float(eps) <= 0:
            raise ValueError("eps must be finite and positive")

        self.register_buffer("acf_values", values.clamp(-1.0, 1.0))
        self.register_buffer("selected_lags", torch.tensor(lags, dtype=torch.long))
        self.lag_tolerance = tolerance
        self.eps = float(eps)
        # Batch telemetry is deliberately detached.  It exposes whether the
        # auxiliary objective actually had sampled pairs without becoming
        # optimization state or a checkpoint parameter.
        self.last_pair_fraction = None
        self.last_active_anchor_fraction = None
        self.last_pair_count = None
        self.last_target_abs_mean = None

    def forward(self, forecast, origins):
        if forecast.ndim != 3:
            raise ValueError("forecast must have shape [batch, horizon, channel]")
        if forecast.shape[-1] != self.acf_values.shape[0]:
            raise ValueError(
                "forecast channel count does not match the train-only ACF prior"
            )
        origins = torch.as_tensor(origins, device=forecast.device)
        if origins.ndim != 1 or origins.numel() != forecast.shape[0]:
            raise ValueError("origins must have shape [batch]")
        if origins.dtype.is_floating_point and not torch.equal(origins, origins.round()):
            raise ValueError("origins must contain integer window offsets")

        batch = forecast.shape[0]
        zero = forecast.sum(dim=(1, 2)) * 0.0
        if batch < 2:
            self.last_pair_fraction = 0.0
            self.last_active_anchor_fraction = 0.0
            self.last_pair_count = 0.0
            self.last_target_abs_mean = 0.0
            return zero

        origins = origins.long()
        distances = (origins[:, None] - origins[None, :]).abs()
        lag_delta = (distances[..., None] - self.selected_lags.view(1, 1, -1)).abs()
        selected_pair = lag_delta.le(self.lag_tolerance).any(dim=-1)
        in_range = distances.lt(self.acf_values.shape[1])
        diagonal = torch.eye(batch, dtype=torch.bool, device=forecast.device)
        safe_distances = distances.clamp(max=self.acf_values.shape[1] - 1)
        target = self.acf_values[:, safe_distances].permute(1, 2, 0)
        target = target.to(dtype=forecast.dtype)
        channel_weight = target.abs()
        valid = (
            selected_pair
            & in_range
            & ~diagonal
            & channel_weight.sum(dim=-1).gt(self.eps)
        )

        centered = forecast - forecast.mean(dim=1, keepdim=True)
        normalized = F.normalize(centered, p=2, dim=1, eps=self.eps)
        similarity = torch.einsum("bhc,khc->bkc", normalized, normalized)
        weighted_error = (similarity - target).square() * channel_weight
        per_pair = weighted_error.sum(dim=-1) / channel_weight.sum(dim=-1).clamp_min(
            self.eps
        )
        valid_float = valid.to(dtype=forecast.dtype)
        per_anchor = (per_pair * valid_float).sum(dim=-1) / valid_float.sum(
            dim=-1
        ).clamp_min(1.0)
        active = valid.any(dim=-1)
        # Preserve a stable loss scale when a residual batch has unmatched
        # origins: ``per_anchor.mean`` is the active-anchor mean, while
        # unmatched forecasts receive zero gradient from this auxiliary term.
        scale = forecast.new_tensor(float(batch)) / active.sum().to(
            dtype=forecast.dtype
        ).clamp_min(1.0)
        result = torch.where(active, per_anchor * scale, zero)

        valid_count = valid.sum().detach()
        self.last_pair_fraction = float(
            valid_count.to(dtype=torch.float32).cpu()
            / max(batch * (batch - 1), 1)
        )
        self.last_active_anchor_fraction = float(
            active.to(dtype=torch.float32).mean().detach().cpu()
        )
        self.last_pair_count = float(valid_count.to(dtype=torch.float32).cpu() / 2.0)
        self.last_target_abs_mean = float(
            (channel_weight * valid_float.unsqueeze(-1)).sum().detach().cpu()
            / (valid_float.sum().detach().cpu() * forecast.shape[-1]).clamp_min(1.0)
        )
        return result


class ForecastLagPairAffineLoss(ForecastLagPairConsistencyLoss):
    """Direct signed ACF alignment of two predicted future segments.

    Unlike correlation matching, this keeps every future step in the loss.
    For a pair of source-window origins separated by lag ``l``, the centered
    prediction for one window is supervised against ``acf[l]`` times the
    centered prediction from its paired window. The ordinary next-segment loss
    remains the amplitude anchor; this relation term only supplies train-only
    temporal structure and is never present at inference.
    """

    def forward(self, forecast, origins):
        if forecast.ndim != 3:
            raise ValueError("forecast must have shape [batch, horizon, channel]")
        if forecast.shape[-1] != self.acf_values.shape[0]:
            raise ValueError("forecast channel count does not match the train-only ACF prior")
        origins = torch.as_tensor(origins, device=forecast.device).long().reshape(-1)
        if origins.numel() != forecast.shape[0]:
            raise ValueError("origins must have shape [batch]")
        batch = forecast.shape[0]
        zero = forecast.sum(dim=(1, 2)) * 0.0
        if batch < 2:
            self.last_pair_fraction = self.last_active_anchor_fraction = self.last_pair_count = self.last_target_abs_mean = 0.0
            return zero
        distances = (origins[:, None] - origins[None, :]).abs()
        lag_delta = (distances[..., None] - self.selected_lags.view(1, 1, -1)).abs()
        selected = lag_delta.le(self.lag_tolerance).any(dim=-1)
        in_range = distances.lt(self.acf_values.shape[1])
        diagonal = torch.eye(batch, dtype=torch.bool, device=forecast.device)
        target = self.acf_values[:, distances.clamp(max=self.acf_values.shape[1] - 1)].permute(1, 2, 0).to(forecast)
        channel_weight = target.abs()
        valid = selected & in_range & ~diagonal & channel_weight.sum(dim=-1).gt(self.eps)
        centered = forecast - forecast.mean(dim=1, keepdim=True)
        # Pair dimension order is [anchor, paired, horizon, channel].
        residual = centered[None, :, :, :] - target[:, :, None, :] * centered[:, None, :, :]
        weighted = residual.square() * channel_weight[:, :, None, :]
        per_pair = weighted.sum(dim=(2, 3)) / (channel_weight.sum(dim=-1) * forecast.shape[1]).clamp_min(self.eps)
        valid_float = valid.to(forecast.dtype)
        per_anchor = (per_pair * valid_float).sum(dim=-1) / valid_float.sum(dim=-1).clamp_min(1.0)
        active = valid.any(dim=-1)
        scale = forecast.new_tensor(float(batch)) / active.sum().to(forecast.dtype).clamp_min(1.0)
        result = torch.where(active, per_anchor * scale, zero)
        valid_count = valid.sum().detach()
        self.last_pair_fraction = float(valid_count.float().cpu() / max(batch * (batch - 1), 1))
        self.last_active_anchor_fraction = float(active.float().mean().detach().cpu())
        self.last_pair_count = float(valid_count.float().cpu() / 2.0)
        self.last_target_abs_mean = float((channel_weight * valid_float.unsqueeze(-1)).sum().detach().cpu() / (valid_float.sum().detach().cpu() * forecast.shape[-1]).clamp_min(1.0))
        return result


class ACFPeakPairInfoNCELoss(nn.Module):
    """Instance-level CPC using train-only ACF peaks without hard negatives.

    Selected lag-separated windows are positives. Other high-absolute-ACF
    lags are excluded from the denominator, so a periodic but unselected
    relation is not treated as a negative instance.
    """

    def __init__(self, acf_values, selected_lags, temperature=.2,
                 exclude_abs_acf=.1, lag_tolerance=0, eps=1e-6,
                 positive_mode="selected"):
        super().__init__()
        values = torch.as_tensor(np.asarray(acf_values, dtype=np.float32)).reshape(-1)
        if values.numel() < 2 or not torch.isfinite(values).all():
            raise ValueError("acf_values must be finite with at least two lags")
        self.register_buffer("acf_values", values.clamp(-1., 1.))
        self.register_buffer("selected_lags", torch.tensor(_strict_selected_lags(selected_lags), dtype=torch.long))
        if not 0 < float(temperature): raise ValueError("temperature must be positive")
        if not 0 <= float(exclude_abs_acf) <= 1: raise ValueError("exclude_abs_acf must be in [0,1]")
        self.positive_mode = str(positive_mode).lower()
        if self.positive_mode not in {"selected", "high_acf"}:
            raise ValueError("positive_mode must be 'selected' or 'high_acf'")
        self.temperature, self.exclude_abs_acf = float(temperature), float(exclude_abs_acf)
        self.lag_tolerance, self.eps = _strict_nonnegative_integer(lag_tolerance, "lag_tolerance"), float(eps)
        self.last_active_anchor_fraction = 0.0
        self.last_excluded_high_acf_fraction = 0.0

    def forward(self, features, origins):
        if features.ndim != 3: raise ValueError("features must have shape [batch, steps, dimension]")
        batch = features.shape[0]
        zero = features.sum(dim=(1, 2)) * 0.
        if batch < 3:
            self.last_active_anchor_fraction = self.last_excluded_high_acf_fraction = 0.0
            return zero
        origins = torch.as_tensor(origins, device=features.device).long().reshape(-1)
        if origins.numel() != batch: raise ValueError("origins must have one value per feature window")
        distances = (origins[:, None] - origins[None, :]).abs()
        safe = distances.clamp(max=self.acf_values.numel() - 1)
        relation = self.acf_values[safe].to(features)
        eye = torch.eye(batch, device=features.device, dtype=torch.bool)
        in_range = distances.lt(self.acf_values.numel()) & ~eye
        selected = (distances[..., None] - self.selected_lags.to(features.device)).abs().le(self.lag_tolerance).any(dim=-1)
        positive = in_range & (selected if self.positive_mode == "selected" else relation.abs().ge(self.exclude_abs_acf))
        high_acf = in_range & relation.abs().ge(self.exclude_abs_acf)
        candidate = in_range if self.positive_mode == "high_acf" else in_range & (~high_acf | positive)
        pooled = F.normalize(features.mean(dim=1), dim=-1, eps=self.eps)
        cosine = torch.matmul(pooled, pooled.transpose(0, 1))
        signed = relation.sign().masked_fill(relation.eq(0), 1.) * cosine
        logits = (signed / self.temperature).masked_fill(~candidate, -torch.inf)
        log_den = torch.logsumexp(logits, dim=-1)
        positive_log_prob = (logits - log_den[:, None]).masked_fill(~positive, 0.)
        positive_count = positive.sum(dim=-1)
        active = positive_count.gt(0) & candidate.sum(dim=-1).gt(1)
        per_anchor = -positive_log_prob.sum(dim=-1) / positive_count.clamp_min(1)
        scale = features.new_tensor(float(batch)) / active.sum().to(features.dtype).clamp_min(1)
        self.last_active_anchor_fraction = float(active.float().mean().detach().cpu())
        self.last_excluded_high_acf_fraction = float((high_acf & ~positive).float().sum().detach().cpu() / in_range.float().sum().detach().cpu().clamp_min(1))
        return torch.where(active, per_anchor * scale, zero)


class ForecastLagPairContrastiveLoss(ForecastLagPairConsistencyLoss):
    """Signed forecast-level InfoNCE over ACF-selected window pairs.

    Each channel of a predicted future segment is an individual contrastive
    item.  A train-only ACF peak at the temporal distance between two source
    windows supplies a positive pair; every other in-range, non-identical
    window in the mini-batch remains a denominator negative.  For a negative
    ACF peak, the positive numerator uses the sign-flipped cosine, teaching
    anti-phase alignment without treating the pair as a false negative.

    This differs materially from the affine objective: the positive must win
    against competing forecasts, so it is a representation-discrimination
    objective rather than a regression penalty on one paired trajectory.
    """

    def __init__(self, acf_values, selected_lags, channel_selected_lags=None,
                 lag_tolerance=0, temperature=0.2, eps=1e-6,
                 acf_confidence_weighting=False, channel_positive_weights=None):
        super().__init__(acf_values, selected_lags, lag_tolerance=lag_tolerance, eps=eps)
        if not np.isfinite(temperature) or float(temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)
        self.acf_confidence_weighting = bool(acf_confidence_weighting)
        if channel_selected_lags is None:
            self.register_buffer("channel_selected_lags", torch.empty(0, dtype=torch.long))
        else:
            peaks = torch.as_tensor(
                np.array(channel_selected_lags, dtype=np.int64, copy=True)
            ).clone()
            if peaks.ndim != 2 or peaks.shape[0] != self.acf_values.shape[0]:
                raise ValueError("channel_selected_lags must have shape [channel, peaks]")
            if ((peaks < 0) | (peaks >= self.acf_values.shape[1])).any():
                raise ValueError("channel_selected_lags must be inside the ACF table")
            self.register_buffer("channel_selected_lags", peaks)
        if channel_positive_weights is None:
            self.register_buffer("channel_positive_weights", torch.empty(0, dtype=torch.float32))
        else:
            weights = torch.as_tensor(
                np.array(channel_positive_weights, dtype=np.float32, copy=True)
            ).clone()
            if weights.shape != self.channel_selected_lags.shape:
                raise ValueError("channel_positive_weights must match channel_selected_lags")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("channel_positive_weights must be finite and non-negative")
            self.register_buffer("channel_positive_weights", weights)

    def forward(self, forecast, origins):
        if forecast.ndim != 3:
            raise ValueError("forecast must have shape [batch, horizon, channel]")
        if forecast.shape[1] < 2:
            raise ValueError("forecast horizon must be >= 2")
        if forecast.shape[-1] != self.acf_values.shape[0]:
            raise ValueError("forecast channel count does not match the train-only ACF prior")
        origins = torch.as_tensor(origins, device=forecast.device).long().reshape(-1)
        if origins.numel() != forecast.shape[0]:
            raise ValueError("origins must have shape [batch]")
        batch = forecast.shape[0]
        zero = forecast.sum(dim=(1, 2)) * 0.0
        if batch < 3:
            self.last_pair_fraction = self.last_active_anchor_fraction = self.last_pair_count = self.last_target_abs_mean = 0.0
            return zero

        distances = (origins[:, None] - origins[None, :]).abs()
        in_range = distances.lt(self.acf_values.shape[1])
        diagonal = torch.eye(batch, dtype=torch.bool, device=forecast.device)
        safe = distances.clamp(max=self.acf_values.shape[1] - 1)
        target = self.acf_values[:, safe].permute(1, 2, 0).to(forecast)
        if self.channel_selected_lags.numel():
            peaks = self.channel_selected_lags.to(device=forecast.device)
            peak_match = (distances[:, :, None, None] - peaks[None, None, :, :]).abs().le(
                self.lag_tolerance
            ) & peaks[None, None, :, :].gt(0)
            selected = peak_match.any(dim=-1)
        else:
            selected = (distances[..., None] - self.selected_lags.view(1, 1, -1)).abs().le(
                self.lag_tolerance
            ).any(dim=-1).unsqueeze(-1).expand_as(target)
        positive = selected & in_range.unsqueeze(-1) & ~diagonal.unsqueeze(-1) & target.abs().gt(self.eps)
        candidate = in_range & ~diagonal

        centered = forecast - forecast.mean(dim=1, keepdim=True)
        normalized = F.normalize(centered, p=2, dim=1, eps=self.eps)
        cosine = torch.einsum("bhc,khc->bkc", normalized, normalized)
        # Every candidate uses the same signed score as a positive numerator.
        # Otherwise a negative-ACF numerator could exceed a denominator built
        # from unsigned scores and no longer represent a log probability.
        signed_cosine = target.sign().masked_fill(target.eq(0), 1.0) * cosine
        logits = signed_cosine / self.temperature
        logits = logits.masked_fill(~candidate.unsqueeze(-1), -torch.inf)
        log_denominator = torch.logsumexp(logits, dim=1)
        positive_log_prob = (logits - log_denominator[:, None, :]).masked_fill(
            ~positive, 0.0
        )
        positive_float = positive.to(forecast.dtype)
        if self.channel_positive_weights.numel():
            stability = self.channel_positive_weights.to(device=forecast.device, dtype=forecast.dtype)
            selected_weight = (peak_match.to(forecast.dtype) * stability[None, None, :, :]).amax(dim=-1)
            positive_weight = selected_weight * positive_float
        elif self.acf_confidence_weighting:
            # Strong train-only ACF peaks carry more reliable relation labels.
            # Normalize per anchor/channel so the option changes pair emphasis,
            # not the overall relation-loss scale.
            positive_weight = target.abs() * positive_float
            positive_weight = positive_weight / positive_weight.sum(dim=1, keepdim=True).clamp_min(self.eps)
            positive_weight = positive_weight * positive_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            positive_weight = positive_float
        positive_count = positive_weight.sum(dim=1)
        # A nonzero denominator negative is required for an informative row.
        active = positive_count.gt(0) & candidate.sum(dim=1, keepdim=True).gt(1)
        per_anchor_channel = -(positive_log_prob * positive_weight).sum(dim=1) / positive_count.clamp_min(1.0)
        per_anchor = (per_anchor_channel * active.to(forecast.dtype)).sum(dim=-1) / active.sum(dim=-1).clamp_min(1.0)
        active_anchor = active.any(dim=-1)
        scale = forecast.new_tensor(float(batch)) / active_anchor.sum().to(forecast.dtype).clamp_min(1.0)
        result = torch.where(active_anchor, per_anchor * scale, zero)
        valid_count = positive.sum().detach()
        self.last_pair_fraction = float(valid_count.float().cpu() / max(batch * (batch - 1) * forecast.shape[-1], 1))
        self.last_active_anchor_fraction = float(active_anchor.float().mean().detach().cpu())
        self.last_pair_count = float(valid_count.float().cpu() / (2.0 * forecast.shape[-1]))
        self.last_target_abs_mean = float((target.abs() * positive_float).sum().detach().cpu() / positive_float.sum().detach().cpu().clamp_min(1.0))
        return result


class ChannelPatchLagContrastiveLoss(ForecastLagPairContrastiveLoss):
    """ACF InfoNCE on the full channelwise patch-state trajectory.

    The forecast-level objective uses predicted target segments as its
    contrastive items.  This pretraining-only objective instead consumes the
    shared encoder's ``[batch, channel, patch, embedding]`` states, retaining
    each channel's own train-prefix ACF peaks.  It therefore supervises a
    transferable representation without channel pooling, sensor identifiers,
    an additional Mamba scan, or an inference-time dependency.
    """

    def __init__(self, *args, patch_similarity="trajectory", channel_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_similarity = str(patch_similarity).lower()
        if self.patch_similarity not in {"trajectory", "positionwise", "last_patch"}:
            raise ValueError("patch_similarity must be 'trajectory', 'positionwise', or 'last_patch'")
        if channel_weights is None:
            weights = torch.ones(self.acf_values.shape[0], dtype=torch.float32)
        else:
            weights = torch.as_tensor(channel_weights, dtype=torch.float32).reshape(-1)
            if weights.numel() != self.acf_values.shape[0]:
                raise ValueError("channel_weights must match the ACF channel count")
            if not torch.isfinite(weights).all() or (weights < 0).any() or not bool(weights.sum() > 0):
                raise ValueError("channel_weights must be finite, non-negative, and have positive mass")
        self.register_buffer("channel_weights", weights)

    def forward(self, representation, origins):
        if representation.ndim != 4:
            raise ValueError(
                "channel patch representation must have shape [batch, channels, patches, embedding]"
            )
        batch, channels, patches, embedding = representation.shape
        if channels != self.acf_values.shape[0]:
            raise ValueError("representation channel count does not match the train-only ACF prior")
        if patches < 1 or embedding < 1:
            raise ValueError("channel patch representation must have nonempty patch features")
        origins = torch.as_tensor(origins, device=representation.device).long().reshape(-1)
        if origins.numel() != batch:
            raise ValueError("origins must have shape [batch]")
        zero = representation.sum(dim=(1, 2, 3)) * 0.0
        if batch < 3:
            self.last_pair_fraction = self.last_active_anchor_fraction = self.last_pair_count = self.last_target_abs_mean = 0.0
            return zero

        distances = (origins[:, None] - origins[None, :]).abs()
        in_range = distances.lt(self.acf_values.shape[1])
        diagonal = torch.eye(batch, dtype=torch.bool, device=representation.device)
        safe = distances.clamp(max=self.acf_values.shape[1] - 1)
        target = self.acf_values[:, safe].permute(1, 2, 0).to(representation)
        peak_match = None
        if self.channel_selected_lags.numel():
            peaks = self.channel_selected_lags.to(device=representation.device)
            peak_match = (
                (distances[:, :, None, None] - peaks[None, None, :, :]).abs().le(self.lag_tolerance)
                & peaks[None, None, :, :].gt(0)
            )
            selected = peak_match.any(dim=-1)
        else:
            selected = (
                (distances[..., None] - self.selected_lags.to(representation.device)).abs().le(self.lag_tolerance)
            ).any(dim=-1).unsqueeze(-1).expand_as(target)
        positive = selected & in_range.unsqueeze(-1) & ~diagonal.unsqueeze(-1) & target.abs().gt(self.eps)
        candidate = in_range & ~diagonal

        if self.patch_similarity == "trajectory":
            trajectories = representation.reshape(batch, channels, patches * embedding)
            states = F.normalize(trajectories.transpose(0, 1), p=2, dim=-1, eps=self.eps)
            cosine = torch.matmul(states, states.transpose(1, 2)).permute(1, 2, 0)
        elif self.patch_similarity == "positionwise":
            # Average same-position similarities instead of allowing one
            # displaced local pattern to dominate a flattened trajectory.
            states = F.normalize(representation, p=2, dim=-1, eps=self.eps)
            cosine = torch.einsum("bcpd,kcpd->bkcp", states, states).mean(dim=-1)
        else:
            # This is the final causal state consumed by the transferable
            # next-patch decoder, so it most directly represents the forecast
            # origin under the same train-only ACF pairing contract.
            states = F.normalize(
                representation[:, :, -1, :].transpose(0, 1), p=2, dim=-1, eps=self.eps
            )
            cosine = torch.matmul(states, states.transpose(1, 2)).permute(1, 2, 0)
        signed_logits = target.sign().masked_fill(target.eq(0), 1.0) * cosine / self.temperature
        logits = signed_logits.masked_fill(~candidate.unsqueeze(-1), -torch.inf)
        log_denominator = torch.logsumexp(logits, dim=1)
        positive_float = positive.to(representation.dtype)
        positive_log_prob = (logits - log_denominator[:, None, :]).masked_fill(~positive, 0.0)
        if self.channel_positive_weights.numel():
            if peak_match is None:
                raise RuntimeError("channel positive weights require channel-selected lag peaks")
            stability = self.channel_positive_weights.to(
                device=representation.device, dtype=representation.dtype
            )
            selected_weight = (
                peak_match.to(representation.dtype) * stability[None, None, :, :]
            ).amax(dim=-1)
            positive_weight = selected_weight * positive_float
        elif self.acf_confidence_weighting:
            positive_weight = target.abs() * positive_float
            positive_weight = positive_weight / positive_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(self.eps)
            positive_weight = positive_weight * positive_float.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        else:
            positive_weight = positive_float
        count = positive_weight.sum(dim=1)
        active = count.gt(0) & candidate.sum(dim=1, keepdim=True).gt(1)
        per_anchor_channel = -(positive_log_prob * positive_weight).sum(dim=1) / count.clamp_min(1.0)
        channel_weight = self.channel_weights.to(device=representation.device, dtype=representation.dtype)
        active_weight = active.to(representation.dtype) * channel_weight.unsqueeze(0)
        per_anchor = (per_anchor_channel * active_weight).sum(dim=-1) / active_weight.sum(dim=-1).clamp_min(1.0)
        active_anchor = active.any(dim=-1)
        scale = representation.new_tensor(float(batch)) / active_anchor.sum().to(representation.dtype).clamp_min(1.0)
        result = torch.where(active_anchor, per_anchor * scale, zero)
        valid_count = positive.sum().detach()
        self.last_pair_fraction = float(valid_count.float().cpu() / max(batch * (batch - 1) * channels, 1))
        self.last_active_anchor_fraction = float(active_anchor.float().mean().detach().cpu())
        self.last_pair_count = float(valid_count.float().cpu() / (2.0 * channels))
        self.last_target_abs_mean = float(
            (target.abs() * positive_float).sum().detach().cpu()
            / positive_float.sum().detach().cpu().clamp_min(1.0)
        )
        return result


class PooledPatchLagContrastiveLoss(nn.Module):
    """Signed lag-InfoNCE for one channel-count-invariant patch representation.

    Unlike :class:`ForecastLagPairContrastiveLoss`, this objective receives one
    representation per source window, not one predicted trajectory per raw
    sensor.  The representation may therefore be formed by a shared encoder's
    permutation-invariant channel pooling and transferred across domains with
    different channel counts.  A signed *global* training-prefix ACF supplies
    positives; all other in-range, non-identity windows remain denominator
    candidates.  This module is pretraining-only and does not participate in
    deployment inference.
    """

    def __init__(self, acf_values, selected_lags, lag_tolerance=0,
                 temperature=0.2, eps=1e-6):
        super().__init__()
        values = torch.as_tensor(np.asarray(acf_values, dtype=np.float32)).reshape(-1)
        if values.numel() < 2 or not torch.isfinite(values).all():
            raise ValueError("acf_values must be a finite one-dimensional lag table")
        if not np.isfinite(temperature) or float(temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        self.selected_lags = torch.as_tensor(_strict_selected_lags(selected_lags), dtype=torch.long)
        self.lag_tolerance = _strict_nonnegative_integer(lag_tolerance, "lag_tolerance")
        self.temperature, self.eps = float(temperature), float(eps)
        self.register_buffer("acf_values", values)
        self.last_pair_fraction = 0.0
        self.last_active_anchor_fraction = 0.0
        self.last_pair_count = 0.0
        self.last_target_abs_mean = 0.0

    def forward(self, representation, origins):
        if representation.ndim != 3:
            raise ValueError("representation must have shape [batch, patches, embedding]")
        if representation.shape[1] < 2:
            raise ValueError("pooled patch representation requires at least two patches")
        origins = torch.as_tensor(origins, device=representation.device).long().reshape(-1)
        if origins.numel() != representation.shape[0]:
            raise ValueError("origins must have shape [batch]")
        batch = representation.shape[0]
        zero = representation.sum(dim=(1, 2)) * 0.0
        if batch < 3:
            self.last_pair_fraction = self.last_active_anchor_fraction = self.last_pair_count = self.last_target_abs_mean = 0.0
            return zero
        distances = (origins[:, None] - origins[None, :]).abs()
        in_range = distances.lt(self.acf_values.numel())
        diagonal = torch.eye(batch, dtype=torch.bool, device=representation.device)
        target = self.acf_values[distances.clamp(max=self.acf_values.numel() - 1)].to(representation)
        selected = (distances[..., None] - self.selected_lags.to(representation.device)).abs().le(
            self.lag_tolerance
        ).any(dim=-1)
        positive = selected & in_range & ~diagonal & target.abs().gt(self.eps)
        candidate = in_range & ~diagonal
        centered = representation - representation.mean(dim=1, keepdim=True)
        normalized = F.normalize(centered.flatten(start_dim=1), dim=-1, eps=self.eps)
        cosine = normalized @ normalized.transpose(0, 1)
        signed_logits = target.sign().masked_fill(target.eq(0), 1.0) * cosine / self.temperature
        logits = signed_logits.masked_fill(~candidate, -torch.inf)
        log_denominator = torch.logsumexp(logits, dim=1)
        positive_float = positive.to(representation.dtype)
        positive_log_prob = (logits - log_denominator[:, None]).masked_fill(~positive, 0.0)
        count = positive_float.sum(dim=1)
        active = count.gt(0) & candidate.sum(dim=1).gt(1)
        per_anchor = -(positive_log_prob * positive_float).sum(dim=1) / count.clamp_min(1.0)
        scale = representation.new_tensor(float(batch)) / active.sum().to(representation.dtype).clamp_min(1.0)
        result = torch.where(active, per_anchor * scale, zero)
        valid_count = positive.sum().detach()
        self.last_pair_fraction = float(valid_count.float().cpu() / max(batch * (batch - 1), 1))
        self.last_active_anchor_fraction = float(active.float().mean().detach().cpu())
        self.last_pair_count = float(valid_count.float().cpu() / 2.0)
        self.last_target_abs_mean = float((target.abs() * positive_float).sum().detach().cpu() / positive_float.sum().detach().cpu().clamp_min(1.0))
        return result


class PooledPatchVarianceLoss(nn.Module):
    """Keep pooled contrastive representations from collapsing across windows.

    This is the variance term of a contrastive representation regularizer. It
    operates over the mini-batch for every pooled patch feature and is only
    meaningful during self-supervised pretraining. It has no forecast-time
    state or computation.
    """

    def __init__(self, target_std=0.2, eps=1e-4):
        super().__init__()
        if not np.isfinite(target_std) or float(target_std) <= 0:
            raise ValueError("target_std must be finite and positive")
        if not np.isfinite(eps) or float(eps) <= 0:
            raise ValueError("eps must be finite and positive")
        self.target_std, self.eps = float(target_std), float(eps)
        self.last_mean_std = 0.0

    def forward(self, representation):
        if representation.ndim != 3:
            raise ValueError("representation must have shape [batch, patches, embedding]")
        if representation.shape[0] < 2:
            self.last_mean_std = 0.0
            return representation.sum() * 0.0
        std = torch.sqrt(representation.var(dim=0, unbiased=False) + self.eps)
        self.last_mean_std = float(std.mean().detach().cpu())
        return F.relu(self.target_std - std).mean()

def temporal_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    if reduction:
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    else:
        loss = (logits[:, t, T + t - 1].mean(dim=1) + logits[:, T + t, t].mean(dim=1)) / 2
    return loss#, a_loss


def instance_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    if reduction:
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    else:
        loss = (logits[:, i, B + i - 1].mean(dim=0) + logits[:, B + i, i].mean(dim=0)) / 2

    return loss#, a_loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, reduction=True):
    B = z1.size(0)
    if reduction:
        loss = torch.tensor(0., device=z1.device)
    else:
        loss = torch.zeros(B, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, reduction)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        d += 1
    return loss / d


def relative_mask(distance_matrix):
    same_label_mask = (distance_matrix == 0.0)
    relative_matrix = distance_matrix.masked_fill(same_label_mask, np.inf) # remove same label
    min_vals, _ = torch.min(relative_matrix, dim=1, keepdim=True)
    pos_mask = (relative_matrix == min_vals).float()
    neg_mask = torch.ones_like(relative_matrix) - same_label_mask.float()
    return pos_mask, neg_mask


def get_circle_embedding(N):
    index = np.arange(N)
    interval = 2 * np.pi / N
    theta = index * interval
    x = np.cos(theta)
    y = np.sin(theta)
    embeds = np.stack([x, y], axis=1)
    return embeds


def autocorr_mask_with_CI(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask

def autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=1, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask


def local_autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask


def _legacy_acf_lookup(acf_values, distances):
    """Look up legacy ACF relations without turning unknown lags into data.

    The original AutoCon implementation indexed the final ACF entry for every
    distance beyond the fitted table.  That makes an unmeasured pair appear to
    have the same relation as the largest measured lag.  We still use a
    clamped index internally to keep indexing safe, but return an explicit
    validity mask so callers can remove those entries from both the candidate
    denominator and the positive set.
    """

    values = torch.as_tensor(acf_values)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("legacy ACF values must be a non-empty 1-D table")
    values = values.to(distances.device)
    indices = distances.to(dtype=torch.long)
    valid = (indices >= 0) & (indices < values.numel())
    safe_indices = indices.clamp(min=0, max=values.numel() - 1)
    relation = values[safe_indices].float()
    # The returned value for an invalid index is only a finite placeholder;
    # callers must use ``valid`` when constructing contrastive candidates.
    return relation, valid


def _legacy_relation_map(acf_values, distances, self_mask):
    """Build an ACF map and candidate mask for a legacy AutoCon objective."""

    relation, in_range = _legacy_acf_lookup(acf_values, distances)
    # Identity is a matrix-position property.  A non-diagonal relation that
    # happens to equal one is still a legitimate positive candidate.
    candidate_mask = in_range & ~self_mask
    relation = relation.masked_fill(~in_range, 0.0)
    return relation, candidate_mask


class AutoCon(nn.Module):
    def __init__(self, batch_size, seq_len, acf_values, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(AutoCon, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.register_buffer('acf_values', torch.as_tensor(acf_values, dtype=torch.float32))
        self.seq_len = seq_len

    def local_contrastive_loss(self, features, labels):
        B, T, D = features.shape
        if T < 2:
            return features.new_zeros((B, T))
        local_features = features.clone()  # Feature shape: (B, T, C)

        anchor_dot_contrast = torch.div(
            torch.bmm(local_features, local_features.transpose(2, 1)),  # Shape: (B, T, C) X (C, T, B) > (B, T, T)
            self.temperature)

        # Local Distance map shape: (B, T, T).  Identity is defined by the
        # matrix diagonal, not by an ACF value (a non-zero lag can also equal
        # one).  Distances outside the fitted ACF table are not candidates.
        labels = labels.to(features.device)
        distances = (labels.unsqueeze(1) - labels.unsqueeze(2)).abs()
        self_mask = torch.eye(T, device=features.device, dtype=torch.bool)
        self_mask = self_mask.unsqueeze(0).expand(B, -1, -1)
        local_distmap, candidate_mask = _legacy_relation_map(
            self.acf_values.to(features.device), distances, self_mask
        )
        neg_mask = candidate_mask.to(local_distmap.dtype)

        # Mask identity and unavailable relations before selecting the
        # highest-ACF positive; equality with one is not a self-test.
        pos_mask = local_autocorr_mask(local_distmap, ~candidate_mask)
        pos_mask = pos_mask * neg_mask

        # Stabilize over the actual candidate axis.  Empty rows use a zero
        # fallback and remain zero-valued after the positive-count guard.
        candidate_logits = anchor_dot_contrast.masked_fill(~candidate_mask, -np.inf)
        has_candidate = candidate_mask.any(dim=2, keepdim=True)
        logits_max = candidate_logits.max(dim=2, keepdim=True).values
        logits_max = torch.where(has_candidate, logits_max, torch.zeros_like(logits_max))
        local_logits = anchor_dot_contrast - logits_max.detach()

        exp_local_logits = torch.exp(local_logits) * neg_mask  # denominator

        denominator = exp_local_logits.sum(2, keepdim=True).clamp_min(torch.finfo(features.dtype).tiny)
        log_local_prob = local_logits - torch.log(denominator)  # (B, T, T) > (B, T ,1)

        pos_count = pos_mask.sum(2).clamp_min(1.0)
        mean_log_local_prob_pos = (local_distmap * pos_mask * log_local_prob).sum(2) / pos_count

        local_loss = -(self.temperature / self.base_temperature) * mean_log_local_prob_pos

        return local_loss


    def avg_global_contrastive_loss(self, features, labels):
        device = features.device
        B, T, C = features.shape
        if B < 2:
            return features.new_zeros((B,))

        pooled_features = F.max_pool1d(features.permute(0, 2, 1), kernel_size=T).squeeze(-1)

        global_features = pooled_features.clone()

        anchor_dot_contrast = torch.div(
            torch.matmul(global_features, global_features.transpose(1, 0)),  # (B, C) X (C, B) > (B, B)
            self.temperature)

        # Global Distance map shape: (B, B).  As in the local objective,
        # remove identity by position and discard unmeasured ACF distances.
        if labels is None:
            labels = torch.arange(B, device=device, dtype=torch.long)
        else:
            labels = labels.to(device).reshape(-1)
            if labels.numel() != B:
                raise ValueError("AutoCon labels must contain one origin per batch")
        distances = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()
        self_mask = torch.eye(B, device=device, dtype=torch.bool)
        global_distmap, candidate_mask = _legacy_relation_map(
            self.acf_values.to(device), distances, self_mask
        )
        neg_mask = candidate_mask.to(global_distmap.dtype)
        pos_mask = autocorr_mask(global_distmap, ~candidate_mask) * neg_mask

        candidate_logits = anchor_dot_contrast.masked_fill(~candidate_mask, -np.inf)
        has_candidate = candidate_mask.any(dim=1, keepdim=True)
        logits_max = candidate_logits.max(dim=1, keepdim=True).values
        logits_max = torch.where(has_candidate, logits_max, torch.zeros_like(logits_max))
        global_logits = anchor_dot_contrast - logits_max.detach()

        exp_global_logits = torch.exp(global_logits) * neg_mask  # denominator

        denominator = exp_global_logits.sum(1, keepdim=True).clamp_min(torch.finfo(features.dtype).tiny)
        log_global_prob = global_logits - torch.log(denominator)  # (B, B) > (B ,1)

        pos_count = pos_mask.sum(1).clamp_min(1.0)
        mean_log_global_prob_pos = (global_distmap * pos_mask * log_global_prob).sum(1) / pos_count

        global_loss = - (self.temperature / self.base_temperature) * mean_log_global_prob_pos

        return global_loss

    def forward(self, features, labels=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            print('features shape > 3')

        B, I, D = features.shape
        if I < 2:
            return features.new_zeros((B, I)), features.new_zeros((B,))
        sample_len = max(2, I // 3)
        feature_idxs = torch.rand(B, I, device=features.device).argsort(-1)[:, :sample_len]
        selected_features = torch.gather(features, 1, feature_idxs.unsqueeze(-1).repeat(1, 1, D))

        local_loss = self.local_contrastive_loss(selected_features, feature_idxs)
        global_loss = self.avg_global_contrastive_loss(features, labels)

        return local_loss, global_loss


class AutoConCI(nn.Module):  # AutoCon Channel Independence (CI) version for Multivariate
    def __init__(self, batch_size, seq_len, acf_values, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(AutoConCI, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.register_buffer('acf_values', torch.as_tensor(acf_values, dtype=torch.float32))
        self.seq_len = seq_len

    def _stream_acf(self, stream_count, device):
        """Return one ACF row for each flattened channel stream.

        ``AutoConCI`` receives streams in batch-major order ``[b0c0, ...,
        b0cC, b1c0, ...]``.  A per-channel ACF table therefore repeats with
        period ``C``; a one-row table is treated as a shared relation.
        """
        values = self.acf_values.to(device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("AutoConCI ACF values must have shape [lag] or [channel, lag]")
        channel_count = int(values.shape[0])
        if channel_count == 1:
            return values.expand(stream_count, -1)
        if stream_count % channel_count != 0:
            raise ValueError(
                "AutoConCI flattened stream count {} is not divisible by ACF channel count {}".format(
                    stream_count, channel_count
                )
            )
        stream_channels = torch.arange(stream_count, device=device) % channel_count
        return values.index_select(0, stream_channels)

    def _batch_acf(self, channels, device):
        """Return one ACF row for each channel in the global CI objective."""
        values = self.acf_values.to(device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("AutoConCI ACF values must have shape [lag] or [channel, lag]")
        if values.shape[0] == 1:
            return values.expand(channels, -1)
        if values.shape[0] != channels:
            raise ValueError(
                "AutoConCI ACF channel count {} does not match flattened input channels {}".format(
                    values.shape[0], channels
                )
            )
        return values

    def local_contrastive_loss(self, features, labels):
        BC, T, D = features.shape
        if T < 2:
            return features.new_zeros((BC, T))

        # Compute local representation similarities
        local_features = features.clone()  # BC, T, D

        anchor_dot_contrast = torch.div(
            torch.bmm(local_features, local_features.transpose(1, 2)),  # (BC, T, D) X (BC, D, T) > (BC, T, T)
            self.temperature)
        # for numerical stability

        # Each flattened stream uses the ACF row of its channel.  This preserves
        # the channel-independent objective while masking unavailable lags.
        labels = labels.to(features.device)
        distances = (labels.unsqueeze(1) - labels.unsqueeze(2)).abs()
        acf_values = self._stream_acf(BC, features.device)
        max_lag = acf_values.shape[1] - 1
        in_range = distances < acf_values.shape[1]
        safe_distances = distances.clamp(min=0, max=max_lag).long()
        local_distmap = torch.gather(
            acf_values,
            1,
            safe_distances.reshape(BC, -1),
        ).reshape(BC, T, T).float()
        local_distmap = local_distmap.masked_fill(~in_range, 0.0)
        self_mask = torch.eye(T, device=features.device, dtype=torch.bool)
        self_mask = self_mask.unsqueeze(0).expand(BC, -1, -1)
        candidate_mask = in_range & ~self_mask
        neg_mask = candidate_mask.to(local_distmap.dtype)
        pos_mask = local_autocorr_mask(local_distmap, ~candidate_mask) * neg_mask

        candidate_logits = anchor_dot_contrast.masked_fill(~candidate_mask, -np.inf)
        has_candidate = candidate_mask.any(dim=2, keepdim=True)
        logits_max = candidate_logits.max(dim=2, keepdim=True).values
        logits_max = torch.where(has_candidate, logits_max, torch.zeros_like(logits_max))
        local_logits = anchor_dot_contrast - logits_max.detach()

        exp_local_logits = torch.exp(local_logits) * neg_mask  # denominator

        denominator = exp_local_logits.sum(2, keepdim=True).clamp_min(torch.finfo(features.dtype).tiny)
        log_local_prob = local_logits - torch.log(denominator)  # (B, T, T) > (B, T ,1)

        mean_log_local_prob_pos = (local_distmap * pos_mask * log_local_prob).sum(2) / pos_mask.sum(2).clamp_min(1.0)

        local_loss = - (self.temperature / self.base_temperature) * mean_log_local_prob_pos

        return local_loss

    def avg_global_contrastive_loss(self, features, labels):
        device = features.device

        BC, T, D = features.shape
        if labels is None:
            base_B = min(self.batch_size, BC)
        else:
            labels = labels.to(device).reshape(-1)
            base_B = int(labels.numel())
        if base_B < 1 or BC % base_B != 0:
            raise ValueError(
                "AutoConCI requires flattened feature count {} divisible by batch size {}".format(
                    BC, base_B
                )
            )
        channels = max(1, BC // base_B)

        pooled_features = F.max_pool1d(features.permute(0, 2, 1), kernel_size=T).squeeze(-1) # (BC, D)

        # Compute one global representation and one relation table per channel.
        global_features = pooled_features.reshape(base_B, channels, D).permute(1, 0, 2).clone()
        C, B, D = global_features.shape

        anchor_dot_contrast = torch.div(
            torch.bmm(global_features, global_features.transpose(1, 2)),  # (C, B, D) X (C, D, B) > (C, B, B)
            self.temperature)

        # Remove identity and unavailable distances by position/mask while
        # retaining channel-specific ACF relations.
        if labels is None:
            labels = torch.arange(base_B, device=device, dtype=torch.long)
        distances = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()
        acf_values = self._batch_acf(channels, device)
        max_lag = acf_values.shape[1] - 1
        in_range = distances < acf_values.shape[1]
        safe_distances = distances.clamp(min=0, max=max_lag).long()
        global_distmap = acf_values[:, safe_distances].float()
        global_distmap = global_distmap.masked_fill(~in_range.unsqueeze(0), 0.0)
        self_mask = torch.eye(B, device=device, dtype=torch.bool)
        candidate_mask = (in_range.unsqueeze(0) & ~self_mask.unsqueeze(0)).expand(C, -1, -1)
        neg_mask = candidate_mask.to(global_distmap.dtype)
        pos_mask = autocorr_mask_with_CI(global_distmap, ~candidate_mask) * neg_mask

        candidate_logits = anchor_dot_contrast.masked_fill(~candidate_mask, -np.inf)
        has_candidate = candidate_mask.any(dim=2, keepdim=True)
        logits_max = candidate_logits.max(dim=2, keepdim=True).values
        logits_max = torch.where(has_candidate, logits_max, torch.zeros_like(logits_max))
        global_logits = anchor_dot_contrast - logits_max.detach()

        exp_global_logits = torch.exp(global_logits) * neg_mask  # denominator

        denominator = exp_global_logits.sum(2, keepdim=True).clamp_min(torch.finfo(features.dtype).tiny)
        log_global_prob = global_logits - torch.log(denominator)  # (C, B, B) > (C, B ,1)


        mean_log_global_prob_pos = (global_distmap * pos_mask * log_global_prob).sum(2) / pos_mask.sum(2).clamp_min(1.0)


        global_loss = - (self.temperature / self.base_temperature) * mean_log_global_prob_pos

        return global_loss

    def forward(self, features, labels=None):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            print('features shape > 3')

        B, I, D = features.shape
        if I < 2:
            return features.new_zeros((B, I)), features.new_zeros((B,))
        sample_len = max(2, I // 3)
        feature_idxs = torch.rand(B, I, device=features.device).argsort(-1)[:, :sample_len]
        selected_features = torch.gather(features, 1, feature_idxs.unsqueeze(-1).repeat(1, 1, D))

        local_loss = self.local_contrastive_loss(selected_features, feature_idxs)
        global_loss = self.avg_global_contrastive_loss(features, labels)

        return local_loss, global_loss


if __name__ == '__main__':
    batch_size = 5
    n_view = 2
    seq_len = 4
    dim = 3
    # features = torch.rand((batch_size, n_view, seq_len, dim)) # (Batch_size, N_view, T, Dim)
    features = torch.rand((batch_size, seq_len, dim)) # (Batch_size, T, Dim)
    labels = torch.tensor([0, 2, 3, 4, 6])
    distmap = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()

    acf_values = np.array([1.0, 0.3, 0.1, -0.3, 0.2, 0.4, 0.1, 0.0])
    print(acf_values[distmap])
    supcreg = AutoCon(acf_values, contrast_mode='all')#.cuda()
    local_loss, global_loss = supcreg(features.cuda(), labels.cuda())
    print(local_loss.shape, global_loss.shape)

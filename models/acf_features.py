"""Model-facing import surface for leakage-safe ACF guidance.

The implementation lives in :mod:`utils.acf_guidance` so it remains usable by
data preparation and evaluation code.  This module intentionally only
re-exports the small public API, making future Mamba models able to import
``models.acf_features`` without coupling themselves to the current loss code.
"""

from utils.acf_guidance import (
    ACFPeak,
    SignedACFPrior,
    build_acf_relation_targets,
    causal_lag_features,
    fit_signed_acf_prior,
)

__all__ = [
    "ACFPeak",
    "SignedACFPrior",
    "fit_signed_acf_prior",
    "causal_lag_features",
    "build_acf_relation_targets",
]

# Official TimesNet H192 External-Code Extension Plan

- Repository: `thuml/Time-Series-Library`
- Immutable commit: `4e938a1767106324dd753b2a44832bf870a0252e`
- License: MIT
- Imported network: `models/TimesNet.py`

The TSLib runner custom loader fixes 70/10/20 and has its own training/test
lifecycle. The adapter imports only the pinned official TimesNet network and
enforces the audited 60/20/20 chronology, 10% target prefix, validation-first
checkpoint selection, and a separate test read.

PEMS-BAY H192 uses seeds 5410--5412, `seq_len=96`, five epochs, 64 windows,
one layer, `d_model=24`, `d_ff=48`, `top_k=3`, and three Inception kernels.
Validation result: active `2.744908`, official TimesNet `3.018545`, `-9.065%`,
3/3. No test result may change candidate or adapter settings.

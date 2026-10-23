# Official TimesNet H336 External-Code Extension Plan

Pinned source: `thuml/Time-Series-Library` commit
`4e938a1767106324dd753b2a44832bf870a0252e` (MIT), imported from
`models/TimesNet.py` through the local chronological adapter. PEMS-BAY H336
uses seeds 5413--5415, `seq_len=96`, 10% target prefix, five epochs, 64
windows, one layer, `d_model=24`, `d_ff=48`, `top_k=3`, and three kernels.

Validation: active `2.386473`, official TimesNet `2.658152`, `-10.221%`, 3/3.
Test reads each validation-selected official checkpoint once without changing
candidate, source, or adapter configuration.

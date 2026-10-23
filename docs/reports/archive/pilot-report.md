# Forecasting comparison

Metrics are MSE/MAE in the train-standardized space used by the repository.
Each trained row is a seed; `ci95` is a Student-t 95% interval across seeds.

| dataset          | model     |   pred_len |   n_runs |   mse_mean |   mse_ci95 |   mae_mean |   mae_ci95 |   rank_mse |
|:-----------------|:----------|-----------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| ETTh1            | DLinear   |         96 |        1 |   0.060413 |          0 |   0.184226 |          0 |          1 |
| ETTh1            | NaiveMean |         96 |        1 |   0.063562 |          0 |   0.193225 |          0 |          2 |
| ETTh1            | NaiveLast |         96 |        1 |   0.067996 |          0 |   0.20174  |          0 |          3 |
| ETTh1            | NaiveMean |        192 |        1 |   0.077128 |          0 |   0.213979 |          0 |          1 |
| ETTh1            | NaiveLast |        192 |        1 |   0.086344 |          0 |   0.227446 |          0 |          2 |
| ETTh1            | NaiveMean |        336 |        1 |   0.097769 |          0 |   0.248192 |          0 |          1 |
| ETTh1            | NaiveLast |        336 |        1 |   0.11336  |          0 |   0.26531  |          0 |          2 |
| ETTh1            | NaiveMean |        720 |        1 |   0.111888 |          0 |   0.264446 |          0 |          1 |
| ETTh1            | NaiveLast |        720 |        1 |   0.129598 |          0 |   0.284156 |          0 |          2 |
| ETTh2            | NaiveMean |         96 |        1 |   0.206266 |          0 |   0.357476 |          0 |          1 |
| ETTh2            | NaiveLast |         96 |        1 |   0.299331 |          0 |   0.42623  |          0 |          2 |
| ETTh2            | NaiveMean |        192 |        1 |   0.252757 |          0 |   0.400392 |          0 |          1 |
| ETTh2            | NaiveLast |        192 |        1 |   0.337911 |          0 |   0.462628 |          0 |          2 |
| ETTh2            | NaiveMean |        336 |        1 |   0.285991 |          0 |   0.434755 |          0 |          1 |
| ETTh2            | NaiveLast |        336 |        1 |   0.390143 |          0 |   0.502447 |          0 |          2 |
| ETTh2            | NaiveMean |        720 |        1 |   0.338936 |          0 |   0.474021 |          0 |          1 |
| ETTh2            | NaiveLast |        720 |        1 |   0.439899 |          0 |   0.535573 |          0 |          2 |
| ETTm1            | NaiveLast |         96 |        1 |   0.033209 |          0 |   0.136496 |          0 |          1 |
| ETTm1            | NaiveMean |         96 |        1 |   0.037502 |          0 |   0.147496 |          0 |          2 |
| ETTm1            | NaiveLast |        192 |        1 |   0.0488   |          0 |   0.168735 |          0 |          1 |
| ETTm1            | NaiveMean |        192 |        1 |   0.049351 |          0 |   0.170624 |          0 |          2 |
| ETTm1            | NaiveMean |        336 |        1 |   0.061811 |          0 |   0.193737 |          0 |          1 |
| ETTm1            | NaiveLast |        336 |        1 |   0.065099 |          0 |   0.196352 |          0 |          2 |
| ETTm1            | NaiveMean |        720 |        1 |   0.081518 |          0 |   0.221144 |          0 |          1 |
| ETTm1            | NaiveLast |        720 |        1 |   0.08727  |          0 |   0.228868 |          0 |          2 |
| ETTm2            | NaiveMean |         96 |        1 |   0.148754 |          0 |   0.300364 |          0 |          1 |
| ETTm2            | NaiveLast |         96 |        1 |   0.221559 |          0 |   0.346626 |          0 |          2 |
| ETTm2            | NaiveMean |        192 |        1 |   0.171589 |          0 |   0.322973 |          0 |          1 |
| ETTm2            | NaiveLast |        192 |        1 |   0.248866 |          0 |   0.377983 |          0 |          2 |
| ETTm2            | NaiveMean |        336 |        1 |   0.196677 |          0 |   0.346835 |          0 |          1 |
| ETTm2            | NaiveLast |        336 |        1 |   0.286904 |          0 |   0.414009 |          0 |          2 |
| ETTm2            | NaiveMean |        720 |        1 |   0.23639  |          0 |   0.383785 |          0 |          1 |
| ETTm2            | NaiveLast |        720 |        1 |   0.340393 |          0 |   0.463834 |          0 |          2 |
| exchange_rate    | NaiveLast |         96 |        1 |   0.093704 |          0 |   0.22867  |          0 |          1 |
| exchange_rate    | NaiveMean |         96 |        1 |   0.155261 |          0 |   0.309912 |          0 |          2 |
| exchange_rate    | NaiveLast |        192 |        1 |   0.193988 |          0 |   0.338811 |          0 |          1 |
| exchange_rate    | NaiveMean |        192 |        1 |   0.260912 |          0 |   0.407848 |          0 |          2 |
| exchange_rate    | NaiveLast |        336 |        1 |   0.366161 |          0 |   0.469178 |          0 |          1 |
| exchange_rate    | NaiveMean |        336 |        1 |   0.427782 |          0 |   0.507434 |          0 |          2 |
| exchange_rate    | NaiveLast |        720 |        1 |   1.30588  |          0 |   0.900002 |          0 |          1 |
| exchange_rate    | NaiveMean |        720 |        1 |   1.39864  |          0 |   0.948939 |          0 |          2 |
| national_illness | NaiveLast |         96 |        1 |   1.96511  |          0 |   1.14623  |          0 |          1 |
| national_illness | NaiveMean |         96 |        1 |   3.12776  |          0 |   1.54774  |          0 |          2 |
| national_illness | NaiveMean |        192 |        1 |   4.91338  |          0 |   1.90002  |          0 |          1 |
| national_illness | NaiveLast |        192 |        1 |   8.05044  |          0 |   2.5632   |          0 |          2 |
| weather          | NaiveLast |         96 |        1 |   0.643179 |          0 |   0.562078 |          0 |          1 |
| weather          | NaiveMean |         96 |        1 |   0.647351 |          0 |   0.595564 |          0 |          2 |
| weather          | NaiveMean |        192 |        1 |   0.742401 |          0 |   0.63642  |          0 |          1 |
| weather          | NaiveLast |        192 |        1 |   0.808837 |          0 |   0.639652 |          0 |          2 |
| weather          | NaiveMean |        336 |        1 |   0.824396 |          0 |   0.676982 |          0 |          1 |
| weather          | NaiveLast |        336 |        1 |   0.953766 |          0 |   0.711743 |          0 |          2 |
| weather          | NaiveMean |        720 |        1 |   1.01616  |          0 |   0.744884 |          0 |          1 |
| weather          | NaiveLast |        720 |        1 |   1.18071  |          0 |   0.795769 |          0 |          2 |

## Interpretation limits

- Existing GLD logs/checkpoints are not included: the source CSV is absent from this repository.
- Overlapping forecast windows are aggregated as individual windows; confidence intervals are across seeds, not windows.
- MAPE/MSPE on standardized series are not used for ranking.

# Forecasting comparison

Metrics are MSE/MAE in the train-standardized space used by the repository.
Each trained row is a seed; `ci95` is a Student-t 95% interval across seeds.
The summary includes exploratory AutoCon lambda variants; use [paper tables](paper_tables.md) and [rank_summary.csv](../../artifacts/exploratory/mambatsv7_multihorizon_fresh_seed_confirmation_20260925/strong_baselines_fresh_long/readout/rank_summary.csv) for the fixed-lambda primary comparison.

         dataset     model  autocon_lambda  pred_len  n_runs mse_mean mse_ci95 mae_mean mae_ci95  rank_mse
           ETTh1 NaiveMean             0.0        96       1 0.066707 0.000000 0.198483 0.000000       1.0
           ETTh1 NaiveLast             0.0        96       1 0.069264 0.000000 0.203283 0.000000       2.0
           ETTh1 NaiveMean             0.0       192       1 0.083446 0.000000 0.224839 0.000000       1.0
           ETTh1 NaiveLast             0.0       192       1 0.091963 0.000000 0.235683 0.000000       2.0
           ETTh1 NaiveMean             0.0       336       1 0.097566 0.000000 0.247927 0.000000       1.0
           ETTh1 NaiveLast             0.0       336       1 0.113274 0.000000 0.265204 0.000000       2.0
           ETTh1 NaiveMean             0.0       720       1 0.109283 0.000000 0.261016 0.000000       1.0
           ETTh1 NaiveLast             0.0       720       1 0.129179 0.000000 0.283409 0.000000       2.0
           ETTh2 NaiveMean             0.0        96       1 0.206345 0.000000 0.358638 0.000000       1.0
           ETTh2 NaiveLast             0.0        96       1 0.295477 0.000000 0.423248 0.000000       2.0
           ETTh2 NaiveMean             0.0       192       1 0.249971 0.000000 0.400720 0.000000       1.0
           ETTh2 NaiveLast             0.0       192       1 0.337361 0.000000 0.462296 0.000000       2.0
           ETTh2 NaiveMean             0.0       336       1 0.285151 0.000000 0.434129 0.000000       1.0
           ETTh2 NaiveLast             0.0       336       1 0.389879 0.000000 0.502270 0.000000       2.0
           ETTh2 NaiveMean             0.0       720       1 0.316667 0.000000 0.458079 0.000000       1.0
           ETTh2 NaiveLast             0.0       720       1 0.436553 0.000000 0.531468 0.000000       2.0
           ETTm1 NaiveLast             0.0        96       1 0.033262 0.000000 0.136473 0.000000       1.0
           ETTm1 NaiveMean             0.0        96       1 0.038143 0.000000 0.148925 0.000000       2.0
           ETTm1 NaiveLast             0.0       192       1 0.049428 0.000000 0.169380 0.000000       1.0
           ETTm1 NaiveMean             0.0       192       1 0.050476 0.000000 0.173190 0.000000       2.0
           ETTm1 NaiveMean             0.0       336       1 0.061870 0.000000 0.193743 0.000000       1.0
           ETTm1 NaiveLast             0.0       336       1 0.065185 0.000000 0.196430 0.000000       2.0
           ETTm1 NaiveMean             0.0       720       1 0.082975 0.000000 0.223577 0.000000       1.0
           ETTm1 NaiveLast             0.0       720       1 0.089155 0.000000 0.231534 0.000000       2.0
           ETTm2 NaiveMean             0.0        96       1 0.152065 0.000000 0.304617 0.000000       1.0
           ETTm2 NaiveLast             0.0        96       1 0.227659 0.000000 0.353669 0.000000       2.0
           ETTm2 NaiveMean             0.0       192       1 0.175306 0.000000 0.325584 0.000000       1.0
           ETTm2 NaiveLast             0.0       192       1 0.257412 0.000000 0.387242 0.000000       2.0
           ETTm2 NaiveMean             0.0       336       1 0.196935 0.000000 0.347117 0.000000       1.0
           ETTm2 NaiveLast             0.0       336       1 0.286970 0.000000 0.414195 0.000000       2.0
           ETTm2 NaiveMean             0.0       720       1 0.238190 0.000000 0.387534 0.000000       1.0
           ETTm2 NaiveLast             0.0       720       1 0.332223 0.000000 0.457133 0.000000       2.0
   exchange_rate NaiveLast             0.0        96       1 0.096801 0.000000 0.231849 0.000000       1.0
   exchange_rate NaiveMean             0.0        96       1 0.162244 0.000000 0.317483 0.000000       2.0
   exchange_rate NaiveLast             0.0       192       1 0.207049 0.000000 0.348831 0.000000       1.0
   exchange_rate NaiveMean             0.0       192       1 0.281840 0.000000 0.425055 0.000000       2.0
   exchange_rate NaiveLast             0.0       336       1 0.409036 0.000000 0.492139 0.000000       1.0
   exchange_rate NaiveMean             0.0       336       1 0.496174 0.000000 0.549709 0.000000       2.0
   exchange_rate NaiveLast             0.0       720       1 1.107007 0.000000 0.804896 0.000000       1.0
   exchange_rate NaiveMean             0.0       720       1 1.156837 0.000000 0.834588 0.000000       2.0
national_illness NaiveLast             0.0        96       1 1.965108 0.000000 1.146234 0.000000       1.0
national_illness NaiveMean             0.0        96       1 3.127761 0.000000 1.547738 0.000000       2.0
national_illness NaiveMean             0.0       192       1 4.913376 0.000000 1.900022 0.000000       1.0
national_illness NaiveLast             0.0       192       1 8.050441 0.000000 2.563205 0.000000       2.0
         weather NaiveLast             0.0        96       1 0.659808 0.000000 0.571950 0.000000       1.0
         weather NaiveMean             0.0        96       1 0.665035 0.000000 0.607930 0.000000       2.0
         weather NaiveMean             0.0       192       1 0.769269 0.000000 0.654039 0.000000       1.0
         weather NaiveLast             0.0       192       1 0.836926 0.000000 0.655160 0.000000       2.0
         weather NaiveMean             0.0       336       1 0.826805 0.000000 0.678360 0.000000       1.0
         weather NaiveLast             0.0       336       1 0.956498 0.000000 0.713127 0.000000       2.0
         weather NaiveMean             0.0       720       1 1.056438 0.000000 0.767651 0.000000       1.0
         weather NaiveLast             0.0       720       1 1.225466 0.000000 0.817850 0.000000       2.0

## Interpretation limits

- Existing GLD logs/checkpoints are not included: the source CSV is absent from this repository.
- Overlapping forecast windows are aggregated as individual windows; confidence intervals are across seeds, not windows.
- MAPE/MSPE on standardized series are not used for ranking.
- `rank_mse` is a descriptive rank over all rows (including naive and exploratory lambda variants); it is not the primary statistical ranking.

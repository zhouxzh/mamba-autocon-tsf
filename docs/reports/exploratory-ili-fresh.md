# Exploratory Ili Report

The primary rows use lambdas selected from the validation-only phase; no test metric was used for selection.
The inferential replication unit is the random seed. Sliding windows are not treated as independent samples.

## Frozen Lambdas

```json
{
  "AutoConNetAutoCon": 0.01,
  "Mamba1AutoCon": 0.3,
  "Mamba2AutoCon": 0.3
}
```

## Primary Metrics

| dataset          | model             | features   |   enc_in |   c_out |   pred_len |   n_runs |   mse_mean |   mse_ci95 |   mae_mean |   mae_ci95 |   original_mse_mean |   original_mae_mean |
|:-----------------|:------------------|:-----------|---------:|--------:|-----------:|---------:|-----------:|-----------:|-----------:|-----------:|--------------------:|--------------------:|
| national_illness | TimesNet          | S          |        1 |       1 |         24 |        5 |    1.04568 |   0.124225 |   0.817057 |   0.074774 |         5.11286e+10 |              180670 |
| national_illness | PatchTST          | S          |        1 |       1 |         24 |        5 |    1.06799 |   0.063156 |   0.837043 |   0.025346 |         5.22194e+10 |              185089 |
| national_illness | Mamba1AutoCon     | S          |        1 |       1 |         24 |        5 |    1.6457  |   0.239258 |   0.994117 |   0.064803 |         8.04666e+10 |              219822 |
| national_illness | Mamba1            | S          |        1 |       1 |         24 |        5 |    1.64575 |   0.238523 |   0.994103 |   0.064519 |         8.0469e+10  |              219819 |
| national_illness | AutoConNet        | S          |        1 |       1 |         24 |        5 |    1.64923 |   0.078268 |   1.00169  |   0.034126 |         8.06396e+10 |              221495 |
| national_illness | AutoConNetAutoCon | S          |        1 |       1 |         24 |        5 |    1.64924 |   0.077975 |   1.00165  |   0.033845 |         8.06398e+10 |              221489 |
| national_illness | Mamba2            | S          |        1 |       1 |         24 |        5 |    1.79947 |   0.225344 |   1.04114  |   0.071825 |         8.79853e+10 |              230220 |
| national_illness | Mamba2AutoCon     | S          |        1 |       1 |         24 |        5 |    1.80033 |   0.224584 |   1.04144  |   0.071399 |         8.80272e+10 |              230286 |
| national_illness | DLinear           | S          |        1 |       1 |         24 |        5 |    1.84092 |   0.078256 |   1.18596  |   0.030574 |         9.00119e+10 |              262242 |
| national_illness | PatchTST          | S          |        1 |       1 |         36 |        5 |    1.16976 |   0.055545 |   0.885935 |   0.025841 |         5.71954e+10 |              195900 |
| national_illness | TimesNet          | S          |        1 |       1 |         36 |        5 |    1.30183 |   0.251524 |   0.924027 |   0.121261 |         6.36532e+10 |              204323 |
| national_illness | DLinear           | S          |        1 |       1 |         36 |        5 |    1.94276 |   0.0353   |   1.22071  |   0.012239 |         9.49915e+10 |              269926 |
| national_illness | Mamba1AutoCon     | S          |        1 |       1 |         36 |        5 |    1.97941 |   0.215488 |   1.11213  |   0.064686 |         9.67834e+10 |              245916 |
| national_illness | Mamba1            | S          |        1 |       1 |         36 |        5 |    1.97968 |   0.215099 |   1.11224  |   0.064492 |         9.67966e+10 |              245942 |
| national_illness | AutoConNet        | S          |        1 |       1 |         36 |        5 |    1.99863 |   0.190656 |   1.1101   |   0.059144 |         9.77235e+10 |              245468 |
| national_illness | AutoConNetAutoCon | S          |        1 |       1 |         36 |        5 |    1.99898 |   0.191361 |   1.1101   |   0.059332 |         9.77404e+10 |              245467 |
| national_illness | Mamba2            | S          |        1 |       1 |         36 |        5 |    2.03994 |   0.260705 |   1.12705  |   0.074017 |         9.97432e+10 |              249217 |
| national_illness | Mamba2AutoCon     | S          |        1 |       1 |         36 |        5 |    2.04045 |   0.261407 |   1.12716  |   0.074327 |         9.97683e+10 |              249241 |
| national_illness | PatchTST          | S          |        1 |       1 |         48 |        5 |    1.28596 |   0.073125 |   0.936386 |   0.02791  |         6.28771e+10 |              207056 |
| national_illness | TimesNet          | S          |        1 |       1 |         48 |        5 |    1.63113 |   0.234191 |   1.06366  |   0.112561 |         7.97545e+10 |              235199 |
| national_illness | AutoConNet        | S          |        1 |       1 |         48 |        5 |    1.91128 |   0.185214 |   1.10468  |   0.060921 |         9.34525e+10 |              244269 |
| national_illness | AutoConNetAutoCon | S          |        1 |       1 |         48 |        5 |    1.91155 |   0.185513 |   1.10475  |   0.060987 |         9.34658e+10 |              244284 |
| national_illness | Mamba1            | S          |        1 |       1 |         48 |        5 |    1.92    |   0.080424 |   1.10456  |   0.028164 |         9.38787e+10 |              244243 |
| national_illness | Mamba1AutoCon     | S          |        1 |       1 |         48 |        5 |    1.92004 |   0.080702 |   1.10458  |   0.028163 |         9.38807e+10 |              244246 |
| national_illness | Mamba2            | S          |        1 |       1 |         48 |        5 |    1.98601 |   0.159553 |   1.12256  |   0.041858 |         9.71064e+10 |              248224 |
| national_illness | Mamba2AutoCon     | S          |        1 |       1 |         48 |        5 |    1.98709 |   0.159425 |   1.12293  |   0.041778 |         9.71591e+10 |              248305 |
| national_illness | DLinear           | S          |        1 |       1 |         48 |        5 |    2.19293 |   0.029873 |   1.30083  |   0.009429 |         1.07223e+11 |              287642 |
| national_illness | PatchTST          | S          |        1 |       1 |         60 |        5 |    1.4403  |   0.084196 |   1.0051   |   0.032503 |         7.0424e+10  |              222251 |
| national_illness | Mamba1            | S          |        1 |       1 |         60 |        5 |    1.68881 |   0.11754  |   1.06582  |   0.042452 |         8.25746e+10 |              235677 |
| national_illness | Mamba1AutoCon     | S          |        1 |       1 |         60 |        5 |    1.68886 |   0.117768 |   1.06581  |   0.042503 |         8.25771e+10 |              235674 |
| national_illness | AutoConNetAutoCon | S          |        1 |       1 |         60 |        5 |    1.69535 |   0.119775 |   1.06715  |   0.047318 |         8.28944e+10 |              235971 |
| national_illness | AutoConNet        | S          |        1 |       1 |         60 |        5 |    1.69537 |   0.119654 |   1.06717  |   0.047276 |         8.28952e+10 |              235975 |
| national_illness | Mamba2            | S          |        1 |       1 |         60 |        5 |    1.71428 |   0.171642 |   1.07492  |   0.053922 |         8.382e+10   |              237690 |
| national_illness | Mamba2AutoCon     | S          |        1 |       1 |         60 |        5 |    1.71505 |   0.173062 |   1.07516  |   0.054384 |         8.38574e+10 |              237741 |
| national_illness | TimesNet          | S          |        1 |       1 |         60 |        5 |    1.95372 |   0.211219 |   1.18825  |   0.079445 |         9.55276e+10 |              262749 |
| national_illness | DLinear           | S          |        1 |       1 |         60 |        5 |    2.58039 |   0.045904 |   1.41741  |   0.016328 |         1.26169e+11 |              313422 |

## Seed-Paired Deltas

| dataset          | features   |   enc_in |   dec_in |   c_out |   seq_len |   pred_len | plain_model   | autocon_model     |   autocon_lambda |   n_pairs |   mean_plain_mse |   mean_autocon_mse |   mean_delta_mse |   relative_change_pct | seed_deltas                                                                                       |
|:-----------------|:-----------|---------:|---------:|--------:|----------:|-----------:|:--------------|:------------------|-----------------:|----------:|-----------------:|-------------------:|-----------------:|----------------------:|:--------------------------------------------------------------------------------------------------|
| national_illness | S          |        1 |        1 |       1 |        96 |         24 | AutoConNet    | AutoConNetAutoCon |             0.01 |         5 |          1.64923 |            1.64924 |      5.4121e-06  |           0.000328158 | 2021:-0.0005210638;2022:-2.6226044e-06;2023:0.0011305809;2024:-0.00067043304;2025:9.059906e-05    |
| national_illness | S          |        1 |        1 |       1 |        96 |         24 | Mamba1        | Mamba1AutoCon     |             0.3  |         5 |          1.64575 |            1.6457  |     -4.90189e-05 |          -0.00297852  | 2021:-0.00042903423;2022:0.00063335896;2023:0.00091612339;2024:-0.001429677;2025:6.4134598e-05    |
| national_illness | S          |        1 |        1 |       1 |        96 |         24 | Mamba2        | Mamba2AutoCon     |             0.3  |         5 |          1.79947 |            1.80033 |      0.000858784 |           0.0477243   | 2021:0.002476573;2022:0.00012207031;2023:2.0623207e-05;2024:0.00047361851;2025:0.0012010336       |
| national_illness | S          |        1 |        1 |       1 |        96 |         36 | AutoConNet    | AutoConNetAutoCon |             0.01 |         5 |          1.99863 |            1.99898 |      0.000344443 |           0.0172339   | 2021:-7.3671341e-05;2022:0.0024011135;2023:-0.00040745735;2024:0.00022864342;2025:-0.00042641163  |
| national_illness | S          |        1 |        1 |       1 |        96 |         36 | Mamba1        | Mamba1AutoCon     |             0.3  |         5 |          1.97968 |            1.97941 |     -0.000268221 |          -0.0135487   | 2021:-0.00011336803;2022:0.00011658669;2023:-7.1525574e-05;2024:-0.0012948513;2025:2.2053719e-05  |
| national_illness | S          |        1 |        1 |       1 |        96 |         36 | Mamba2        | Mamba2AutoCon     |             0.3  |         5 |          2.03994 |            2.04045 |      0.000514317 |           0.0252123   | 2021:-0.00075113773;2022:0.00039172173;2023:0.00017237663;2024:0.0013204813;2025:0.0014381409     |
| national_illness | S          |        1 |        1 |       1 |        96 |         48 | AutoConNet    | AutoConNetAutoCon |             0.01 |         5 |          1.91128 |            1.91155 |      0.000272441 |           0.0142544   | 2021:0.00040304661;2022:0.00084137917;2023:-0.00027859211;2024:0.00050568581;2025:-0.00010931492  |
| national_illness | S          |        1 |        1 |       1 |        96 |         48 | Mamba1        | Mamba1AutoCon     |             0.3  |         5 |          1.92    |            1.92004 |      3.93391e-05 |           0.00204891  | 2021:0.00032603741;2022:-0.000436306;2023:-0.00014138222;2024:-0.00037550926;2025:0.0008238554    |
| national_illness | S          |        1 |        1 |       1 |        96 |         48 | Mamba2        | Mamba2AutoCon     |             0.3  |         5 |          1.98601 |            1.98709 |      0.00107706  |           0.0542321   | 2021:0.0023260117;2022:0.00056993961;2023:0.00031614304;2024:0.0010054111;2025:0.0011677742       |
| national_illness | S          |        1 |        1 |       1 |        96 |         60 | AutoConNet    | AutoConNetAutoCon |             0.01 |         5 |          1.69537 |            1.69535 |     -1.6427e-05  |          -0.000968937 | 2021:-0.00013637543;2022:-7.9870224e-06;2023:-0.00029277802;2024:0.00022292137;2025:0.00013208389 |
| national_illness | S          |        1 |        1 |       1 |        96 |         60 | Mamba1        | Mamba1AutoCon     |             0.3  |         5 |          1.68881 |            1.68886 |      5.12838e-05 |           0.00303669  | 2021:-0.00043165684;2022:2.1934509e-05;2023:0.00045955181;2024:-0.00049221516;2025:0.00069880486  |
| national_illness | S          |        1 |        1 |       1 |        96 |         60 | Mamba2        | Mamba2AutoCon     |             0.3  |         5 |          1.71428 |            1.71505 |      0.000765443 |           0.044651    | 2021:-0.00090110302;2022:0.0010402203;2023:0.00043308735;2024:0.00045275688;2025:0.0028022528     |

## Non-overlapping Origin Sensitivity

Each seed uses every horizon-th forecast origin; this is a sensitivity analysis, not an additional independent-sample test.

| dataset          | features   |   enc_in |   dec_in |   c_out |   seq_len |   pred_len | plain_model   | autocon_model     |   autocon_lambda |   n_pairs |   mean_origin_delta_mse |   relative_change_pct | seed_deltas                                                                                       |
|:-----------------|:-----------|---------:|---------:|--------:|----------:|-----------:|:--------------|:------------------|-----------------:|----------:|------------------------:|----------------------:|:--------------------------------------------------------------------------------------------------|
| national_illness | S          |        1 |        1 |       1 |        96 |         24 | AutoConNet    | AutoConNetAutoCon |             0.01 |         5 |            -0.000402621 |           -0.0213247  | 2021:0.00072850788;2022:-0.00080777374;2023:0.0001721261;2024:-0.0019526879;2025:-0.00015327594   |
| national_illness | S          |        1 |        1 |       1 |        96 |         24 | Mamba1        | Mamba1AutoCon     |             0.3  |         5 |            -0.000708337 |           -0.0437007  | 2021:-0.0025198864;2022:-3.2521813e-05;2023:0.0012983237;2024:-0.0025497247;2025:0.00026212522    |
| national_illness | S          |        1 |        1 |       1 |        96 |         24 | Mamba2        | Mamba2AutoCon     |             0.3  |         5 |             0.00127911  |            0.0663403  | 2021:0.0014654023;2022:0.0010554681;2023:0.0011217934;2024:0.00021331534;2025:0.0025395554        |
| national_illness | S          |        1 |        1 |       1 |        96 |         36 | AutoConNet    | AutoConNetAutoCon |             0.01 |         5 |             0.00151514  |            0.0561526  | 2021:-0.00029426542;2022:0.0072559883;2023:0.00015762074;2024:0.0006516618;2025:-0.00019531564    |
| national_illness | S          |        1 |        1 |       1 |        96 |         36 | Mamba1        | Mamba1AutoCon     |             0.3  |         5 |            -0.00067409  |           -0.0249601  | 2021:-0.0018791349;2022:-0.00034447983;2023:-2.6470087e-05;2024:-0.0013671248;2025:0.00024675955  |
| national_illness | S          |        1 |        1 |       1 |        96 |         36 | Mamba2        | Mamba2AutoCon     |             0.3  |         5 |             0.00093601  |            0.0320257  | 2021:-4.9913096e-05;2022:7.1851012e-05;2023:3.1210575e-05;2024:0.0035998913;2025:0.0010270125     |
| national_illness | S          |        1 |        1 |       1 |        96 |         48 | AutoConNet    | AutoConNetAutoCon |             0.01 |         5 |             0.00108928  |            0.0309875  | 2021:0.001258825;2022:0.00069113426;2023:0.00040683069;2024:0.0028620894;2025:0.00022752358       |
| national_illness | S          |        1 |        1 |       1 |        96 |         48 | Mamba1        | Mamba1AutoCon     |             0.3  |         5 |            -0.000464622 |           -0.0128886  | 2021:8.0733408e-05;2022:-0.00012198588;2023:-0.00012594831;2024:-0.0019068024;2025:-0.00024910795 |
| national_illness | S          |        1 |        1 |       1 |        96 |         48 | Mamba2        | Mamba2AutoCon     |             0.3  |         5 |             0.00298888  |            0.0782529  | 2021:0.0057667173;2022:0.00085500108;2023:0.0015853374;2024:0.00136511;2025:0.0053722485          |
| national_illness | S          |        1 |        1 |       1 |        96 |         60 | AutoConNet    | AutoConNetAutoCon |             0.01 |         5 |            -3.67807e-05 |           -0.00180337 | 2021:-5.8539984e-05;2022:-0.00010716631;2023:-0.00037333299;2024:0.00019893695;2025:0.00015619878 |
| national_illness | S          |        1 |        1 |       1 |        96 |         60 | Mamba1        | Mamba1AutoCon     |             0.3  |         5 |             0.00012422  |            0.00597559 | 2021:-0.00062470848;2022:-0.00019218764;2023:0.00097644102;2024:-0.0010621018;2025:0.0015236559   |
| national_illness | S          |        1 |        1 |       1 |        96 |         60 | Mamba2        | Mamba2AutoCon     |             0.3  |         5 |             0.001811    |            0.0884134  | 2021:0.0047356528;2022:-0.00038594422;2023:0.0015161355;2024:0.00027722336;2025:0.0029119329      |

## Resource Summary

| model             | dataset          | features   |   enc_in |   dec_in |   c_out |   pred_len |   runs |   elapsed_sec_mean |   elapsed_sec_std |   parameters |
|:------------------|:-----------------|:-----------|---------:|---------:|--------:|-----------:|-------:|-------------------:|------------------:|-------------:|
| AutoConNet        | national_illness | S          |        1 |        1 |       1 |         24 |      5 |             2.4756 |        0.0665567  |        19409 |
| AutoConNetAutoCon | national_illness | S          |        1 |        1 |       1 |         24 |      5 |             2.632  |        0.0158588  |        19409 |
| DLinear           | national_illness | S          |        1 |        1 |       1 |         24 |      5 |             2.3212 |        0.0132174  |         4656 |
| Mamba1            | national_illness | S          |        1 |        1 |       1 |         24 |      5 |             2.5874 |        0.0583078  |        15921 |
| Mamba1AutoCon     | national_illness | S          |        1 |        1 |       1 |         24 |      5 |             2.743  |        0.0115974  |        15921 |
| Mamba2            | national_illness | S          |        1 |        1 |       1 |         24 |      5 |            12.9894 |        0.011149   |        13853 |
| Mamba2AutoCon     | national_illness | S          |        1 |        1 |       1 |         24 |      5 |            13.0778 |        0.0133304  |        13853 |
| PatchTST          | national_illness | S          |        1 |        1 |       1 |         24 |      5 |             2.4858 |        0.0468156  |        16280 |
| TimesNet          | national_illness | S          |        1 |        1 |       1 |         24 |      5 |             3.4702 |        0.240518   |       598041 |
| AutoConNet        | national_illness | S          |        1 |        1 |       1 |         36 |      5 |             2.4722 |        0.0533451  |        21737 |
| AutoConNetAutoCon | national_illness | S          |        1 |        1 |       1 |         36 |      5 |             2.5842 |        0.050633   |        21737 |
| DLinear           | national_illness | S          |        1 |        1 |       1 |         36 |      5 |             2.3202 |        0.0102323  |         6984 |
| Mamba1            | national_illness | S          |        1 |        1 |       1 |         36 |      5 |             2.6206 |        0.0119708  |        18249 |
| Mamba1AutoCon     | national_illness | S          |        1 |        1 |       1 |         36 |      5 |             2.7418 |        0.0151228  |        18249 |
| Mamba2            | national_illness | S          |        1 |        1 |       1 |         36 |      5 |            12.9724 |        0.019578   |        16181 |
| Mamba2AutoCon     | national_illness | S          |        1 |        1 |       1 |         36 |      5 |            13.0616 |        0.0621916  |        16181 |
| PatchTST          | national_illness | S          |        1 |        1 |       1 |         36 |      5 |             2.5102 |        0.0219704  |        20900 |
| TimesNet          | national_illness | S          |        1 |        1 |       1 |         36 |      5 |             3.4032 |        0.268722   |       599205 |
| AutoConNet        | national_illness | S          |        1 |        1 |       1 |         48 |      5 |             2.499  |        0.0241247  |        24065 |
| AutoConNetAutoCon | national_illness | S          |        1 |        1 |       1 |         48 |      5 |             2.6046 |        0.0368619  |        24065 |
| DLinear           | national_illness | S          |        1 |        1 |       1 |         48 |      5 |             2.3272 |        0.0116919  |         9312 |
| Mamba1            | national_illness | S          |        1 |        1 |       1 |         48 |      5 |             2.6216 |        0.0111937  |        20577 |
| Mamba1AutoCon     | national_illness | S          |        1 |        1 |       1 |         48 |      5 |             2.7222 |        0.0161462  |        20577 |
| Mamba2            | national_illness | S          |        1 |        1 |       1 |         48 |      5 |            12.9822 |        0.0148054  |        18509 |
| Mamba2AutoCon     | national_illness | S          |        1 |        1 |       1 |         48 |      5 |            13.0832 |        0.0120083  |        18509 |
| PatchTST          | national_illness | S          |        1 |        1 |       1 |         48 |      5 |             2.5104 |        0.011327   |        25520 |
| TimesNet          | national_illness | S          |        1 |        1 |       1 |         48 |      5 |             3.149  |        0.244307   |       600369 |
| AutoConNet        | national_illness | S          |        1 |        1 |       1 |         60 |      5 |             2.4756 |        0.0515393  |        26393 |
| AutoConNetAutoCon | national_illness | S          |        1 |        1 |       1 |         60 |      5 |             2.6284 |        0.0135204  |        26393 |
| DLinear           | national_illness | S          |        1 |        1 |       1 |         60 |      5 |             2.3184 |        0.0126016  |        11640 |
| Mamba1            | national_illness | S          |        1 |        1 |       1 |         60 |      5 |             2.6346 |        0.00658027 |        22905 |
| Mamba1AutoCon     | national_illness | S          |        1 |        1 |       1 |         60 |      5 |             2.737  |        0.0130576  |        22905 |
| Mamba2            | national_illness | S          |        1 |        1 |       1 |         60 |      5 |            12.9918 |        0.0195371  |        20837 |
| Mamba2AutoCon     | national_illness | S          |        1 |        1 |       1 |         60 |      5 |            13.0896 |        0.0205134  |        20837 |
| PatchTST          | national_illness | S          |        1 |        1 |       1 |         60 |      5 |             2.4906 |        0.0656643  |        30140 |
| TimesNet          | national_illness | S          |        1 |        1 |       1 |         60 |      5 |             3.0476 |        0.0276098  |       601533 |

## Caveats

- Metrics are in the repository's train-standardized space; original-scale columns are descriptive.
- Primary test MSE averages all chronological windows; origin sensitivity uses every horizon-th origin.
- National Illness remains exploratory because its records are weekly.
- Parameter counts and environment details are retained in the phase manifest and environment snapshot.

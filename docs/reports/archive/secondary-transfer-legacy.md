# Secondary Transfer Report (Legacy)

> **Provenance status:** legacy/unknown code provenance. This historical
> transfer report depends on a legacy selection run and is retained only for
> diagnostic context. It is not a strict or submission-ready result.

The primary rows use lambdas selected from the validation-only phase; no test metric was used for selection.
The inferential replication unit is the random seed. Sliding windows are not treated as independent samples.

## Frozen Lambdas

```json
{
  "AutoConNetAutoCon": 0.3,
  "Mamba1AutoCon": 0.01,
  "Mamba2AutoCon": 0.1
}
```

## Primary Metrics

| dataset       | model             |   pred_len |   n_runs |   mse_mean |   mse_ci95 |   mae_mean |   mae_ci95 |
|:--------------|:------------------|-----------:|---------:|-----------:|-----------:|-----------:|-----------:|
| exchange_rate | DLinear           |         96 |        5 |   0.1054   |   0.000375 |   0.250235 |   0.000893 |
| exchange_rate | Mamba1            |         96 |        5 |   0.106704 |   0.003189 |   0.242767 |   0.003653 |
| exchange_rate | Mamba1AutoCon     |         96 |        5 |   0.106712 |   0.00317  |   0.242828 |   0.003597 |
| exchange_rate | AutoConNetAutoCon |         96 |        5 |   0.107241 |   0.002113 |   0.243136 |   0.00303  |
| exchange_rate | AutoConNet        |         96 |        5 |   0.107657 |   0.002385 |   0.245372 |   0.001979 |
| exchange_rate | Mamba2AutoCon     |         96 |        5 |   0.108154 |   0.00209  |   0.245332 |   0.003503 |
| exchange_rate | Mamba2            |         96 |        5 |   0.109621 |   0.002599 |   0.246767 |   0.003738 |
| exchange_rate | DLinear           |        336 |        5 |   0.399369 |   0.00209  |   0.502843 |   0.001256 |
| exchange_rate | Mamba1            |        336 |        5 |   0.449808 |   0.012327 |   0.518403 |   0.009465 |
| exchange_rate | Mamba1AutoCon     |        336 |        5 |   0.450428 |   0.011854 |   0.518758 |   0.009062 |
| exchange_rate | AutoConNetAutoCon |        336 |        5 |   0.454304 |   0.008427 |   0.52032  |   0.005071 |
| exchange_rate | AutoConNet        |        336 |        5 |   0.457349 |   0.010657 |   0.519786 |   0.007064 |
| exchange_rate | Mamba2AutoCon     |        336 |        5 |   0.46261  |   0.016466 |   0.525611 |   0.009184 |
| exchange_rate | Mamba2            |        336 |        5 |   0.467515 |   0.014998 |   0.528201 |   0.008397 |
| weather       | DLinear           |         96 |        5 |   0.648513 |   0.013573 |   0.558647 |   0.006108 |
| weather       | Mamba1            |         96 |        5 |   0.698796 |   0.072336 |   0.614169 |   0.044957 |
| weather       | AutoConNetAutoCon |         96 |        5 |   0.759859 |   0.124936 |   0.646164 |   0.055866 |
| weather       | Mamba1AutoCon     |         96 |        5 |   0.784119 |   0.126292 |   0.661373 |   0.065936 |
| weather       | AutoConNet        |         96 |        5 |   0.806687 |   0.113069 |   0.665807 |   0.057285 |
| weather       | Mamba2            |         96 |        5 |   0.82798  |   0.080903 |   0.683383 |   0.046318 |
| weather       | Mamba2AutoCon     |         96 |        5 |   0.855171 |   0.059705 |   0.696207 |   0.030807 |
| weather       | DLinear           |        336 |        5 |   0.906317 |   0.007518 |   0.668503 |   0.002886 |
| weather       | Mamba1AutoCon     |        336 |        5 |   1.13253  |   0.220704 |   0.819552 |   0.094028 |
| weather       | Mamba1            |        336 |        5 |   1.14199  |   0.260232 |   0.823257 |   0.113771 |
| weather       | AutoConNetAutoCon |        336 |        5 |   1.21359  |   0.047116 |   0.858116 |   0.026205 |
| weather       | Mamba2AutoCon     |        336 |        5 |   1.23445  |   0.150995 |   0.876969 |   0.054705 |
| weather       | Mamba2            |        336 |        5 |   1.24964  |   0.203458 |   0.878531 |   0.085627 |
| weather       | AutoConNet        |        336 |        5 |   1.27202  |   0.049893 |   0.878151 |   0.03204  |

## Seed-Paired Deltas

| dataset       | features   |   seq_len |   pred_len | plain_model   | autocon_model     |   autocon_lambda |   n_pairs |   mean_plain_mse |   mean_autocon_mse |   mean_delta_mse |   relative_change_pct | seed_deltas                                                                                     |
|:--------------|:-----------|----------:|-----------:|:--------------|:------------------|-----------------:|----------:|-----------------:|-------------------:|-----------------:|----------------------:|:------------------------------------------------------------------------------------------------|
| exchange_rate | S          |        96 |         96 | AutoConNet    | AutoConNetAutoCon |             0.3  |         5 |         0.107657 |           0.107241 |     -0.000416043 |           -0.386454   | 2021:-0.0014709532;2022:-0.0041324422;2023:-0.00028409064;2024:0.0038956255;2025:-8.8356435e-05 |
| exchange_rate | S          |        96 |         96 | Mamba1        | Mamba1AutoCon     |             0.01 |         5 |         0.106704 |           0.106712 |      7.67261e-06 |            0.00719054 | 2021:4.9270689e-05;2022:0.0001391992;2023:-0.00019954145;2024:0.0001104176;2025:-6.0983002e-05  |
| exchange_rate | S          |        96 |         96 | Mamba2        | Mamba2AutoCon     |             0.1  |         5 |         0.109621 |           0.108154 |     -0.0014671   |           -1.33833    | 2021:-0.0022224337;2022:0.00078381598;2023:-0.0033108965;2024:0.0014381334;2025:-0.0040241256   |
| exchange_rate | S          |        96 |        336 | AutoConNet    | AutoConNetAutoCon |             0.3  |         5 |         0.457349 |           0.454304 |     -0.00304525  |           -0.66585    | 2021:0.00011432171;2022:-0.0031967163;2023:-0.011601746;2024:0.0074610114;2025:-0.0080031455    |
| exchange_rate | S          |        96 |        336 | Mamba1        | Mamba1AutoCon     |             0.01 |         5 |         0.449808 |           0.450428 |      0.000620085 |            0.137855   | 2021:0.00066334009;2022:0.0023879707;2023:2.2858381e-05;2024:-0.0012096167;2025:0.0012358725    |
| exchange_rate | S          |        96 |        336 | Mamba2        | Mamba2AutoCon     |             0.1  |         5 |         0.467515 |           0.46261  |     -0.00490487  |           -1.04914    | 2021:-0.0027820468;2022:-0.0019159019;2023:-0.0055540502;2024:-0.0072008967;2025:-0.0070714355  |
| weather       | S          |        96 |         96 | AutoConNet    | AutoConNetAutoCon |             0.3  |         5 |         0.806687 |           0.759859 |     -0.0468282   |           -5.80501    | 2021:-0.045586228;2022:-0.0065038204;2023:-0.054759264;2024:-0.106381;2025:-0.020910919         |
| weather       | S          |        96 |         96 | Mamba1        | Mamba1AutoCon     |             0.01 |         5 |         0.698796 |           0.784119 |      0.0853229   |           12.21       | 2021:0.021564841;2022:0.029069781;2023:0.33082652;2024:-0.02926898;2025:0.074422121             |
| weather       | S          |        96 |         96 | Mamba2        | Mamba2AutoCon     |             0.1  |         5 |         0.82798  |           0.855171 |      0.0271914   |            3.28407    | 2021:0.052846193;2022:0.046795368;2023:0.051349819;2024:-0.022606313;2025:0.0075720549          |
| weather       | S          |        96 |        336 | AutoConNet    | AutoConNetAutoCon |             0.3  |         5 |         1.27202  |           1.21359  |     -0.0584341   |           -4.59381    | 2021:-0.097685337;2022:-0.084585428;2023:0.017854452;2024:-0.071275353;2025:-0.056478977        |
| weather       | S          |        96 |        336 | Mamba1        | Mamba1AutoCon     |             0.01 |         5 |         1.14199  |           1.13253  |     -0.00946249  |           -0.828597   | 2021:0.052461922;2022:-0.0065777302;2023:-0.070027828;2024:-0.011119962;2025:-0.012048841       |
| weather       | S          |        96 |        336 | Mamba2        | Mamba2AutoCon     |             0.1  |         5 |         1.24964  |           1.23445  |     -0.0151909   |           -1.21562    | 2021:-0.053835392;2022:-0.086813688;2023:-0.046126008;2024:0.016198397;2025:0.094622314         |

## Non-overlapping Origin Sensitivity

Each seed uses every horizon-th forecast origin; this is a sensitivity analysis, not an additional independent-sample test.

| dataset       | features   |   seq_len |   pred_len | plain_model   | autocon_model     |   autocon_lambda |   n_pairs |   mean_origin_delta_mse |   relative_change_pct | seed_deltas                                                                                      |
|:--------------|:-----------|----------:|-----------:|:--------------|:------------------|-----------------:|----------:|------------------------:|----------------------:|:-------------------------------------------------------------------------------------------------|
| exchange_rate | S          |        96 |         96 | AutoConNet    | AutoConNetAutoCon |             0.3  |         5 |             0.000119933 |              0.113984 | 2021:-0.0046745642;2022:-0.0042718172;2023:-0.00046392076;2024:0.0066106919;2025:0.0033992731    |
| exchange_rate | S          |        96 |         96 | Mamba1        | Mamba1AutoCon     |             0.01 |         5 |             8.39474e-05 |              0.081274 | 2021:0.00055531916;2022:0.00020661905;2023:-0.00020444225;2024:0.00018518444;2025:-0.00032294324 |
| exchange_rate | S          |        96 |         96 | Mamba2        | Mamba2AutoCon     |             0.1  |         5 |            -0.00434898  |             -3.93932  | 2021:-0.0042887502;2022:-0.0049233358;2023:0.0039336061;2024:-0.0092803232;2025:-0.007186104     |
| exchange_rate | S          |        96 |        336 | AutoConNet    | AutoConNetAutoCon |             0.3  |         5 |             0.0248041   |              8.59799  | 2021:0.013106302;2022:0.04029899;2023:0.021613945;2024:0.030947202;2025:0.018054151              |
| exchange_rate | S          |        96 |        336 | Mamba1        | Mamba1AutoCon     |             0.01 |         5 |             0.00157841  |              0.505382 | 2021:-0.00098568955;2022:0.0063926116;2023:-0.00019890601;2024:-0.001275169;2025:0.0039592088    |
| exchange_rate | S          |        96 |        336 | Mamba2        | Mamba2AutoCon     |             0.1  |         5 |            -0.000546627 |             -0.17469  | 2021:0.017672986;2022:-0.0097927071;2023:-0.0022317182;2024:0.0047841576;2025:-0.013165852       |
| weather       | S          |        96 |         96 | AutoConNet    | AutoConNetAutoCon |             0.3  |         5 |            -0.0449876   |             -5.99802  | 2021:-0.052512808;2022:-0.015866016;2023:-0.059465709;2024:-0.092659548;2025:-0.00443395         |
| weather       | S          |        96 |         96 | Mamba1        | Mamba1AutoCon     |             0.01 |         5 |             0.0609499   |              9.33609  | 2021:0.017101289;2022:0.02082475;2023:0.23284933;2024:-0.014161111;2025:0.048135389              |
| weather       | S          |        96 |         96 | Mamba2        | Mamba2AutoCon     |             0.1  |         5 |             0.0277123   |              3.52494  | 2021:0.047477808;2022:0.056993448;2023:0.063946635;2024:-0.032040926;2025:0.0021843958           |
| weather       | S          |        96 |        336 | AutoConNet    | AutoConNetAutoCon |             0.3  |         5 |            -0.0781688   |             -5.73059  | 2021:-0.10467399;2022:-0.039152398;2023:-0.033404835;2024:-0.138487;2025:-0.075125755            |
| weather       | S          |        96 |        336 | Mamba1        | Mamba1AutoCon     |             0.01 |         5 |            -0.0217658   |             -1.8089   | 2021:0.045701259;2022:0.014798659;2023:-0.13168983;2024:-0.022638531;2025:-0.01500054            |
| weather       | S          |        96 |        336 | Mamba2        | Mamba2AutoCon     |             0.1  |         5 |            -0.0315154   |             -2.25671  | 2021:-0.022118011;2022:-0.202719;2023:-0.027566651;2024:0.020457586;2025:0.074369318             |

## Resource Summary

| model             | dataset       |   pred_len |   runs |   elapsed_sec_mean |   elapsed_sec_std |   parameters |
|:------------------|:--------------|-----------:|-------:|-------------------:|------------------:|-------------:|
| AutoConNet        | exchange_rate |         96 |      5 |             2.8272 |         0.0353511 |        33377 |
| AutoConNetAutoCon | exchange_rate |         96 |      5 |             3.2398 |         0.135354  |        33377 |
| DLinear           | exchange_rate |         96 |      5 |             3.036  |         0.164738  |        18624 |
| Mamba1            | exchange_rate |         96 |      5 |             3.0978 |         0.0474099 |        29889 |
| Mamba1AutoCon     | exchange_rate |         96 |      5 |             3.3256 |         0.04055   |        29889 |
| Mamba2            | exchange_rate |         96 |      5 |            13.7024 |         0.0958243 |        27821 |
| Mamba2AutoCon     | exchange_rate |         96 |      5 |            13.8656 |         0.0887542 |        27821 |
| AutoConNet        | exchange_rate |        336 |      5 |             3.4876 |         0.245988  |        79937 |
| AutoConNetAutoCon | exchange_rate |        336 |      5 |             3.605  |         0.0251694 |        79937 |
| DLinear           | exchange_rate |        336 |      5 |             2.707  |         0.0408534 |        65184 |
| Mamba1            | exchange_rate |        336 |      5 |             3.9814 |         0.0426063 |        76449 |
| Mamba1AutoCon     | exchange_rate |        336 |      5 |             4.3272 |         0.411045  |        76449 |
| Mamba2            | exchange_rate |        336 |      5 |            14.5242 |         0.482904  |        74381 |
| Mamba2AutoCon     | exchange_rate |        336 |      5 |            14.8574 |         0.678894  |        74381 |
| AutoConNet        | weather       |         96 |      5 |             6.8122 |         0.562662  |        33441 |
| AutoConNetAutoCon | weather       |         96 |      5 |             8.1692 |         0.380038  |        33441 |
| DLinear           | weather       |         96 |      5 |             5.8002 |         0.227563  |        18624 |
| Mamba1            | weather       |         96 |      5 |             6.872  |         0.893039  |        29953 |
| Mamba1AutoCon     | weather       |         96 |      5 |             8.1404 |         0.375375  |        29953 |
| Mamba2            | weather       |         96 |      5 |            20.7038 |         1.67605   |        27885 |
| Mamba2AutoCon     | weather       |         96 |      5 |            22.2698 |         2.65672   |        27885 |
| AutoConNet        | weather       |        336 |      5 |             9.0898 |         2.08381   |        80001 |
| AutoConNetAutoCon | weather       |        336 |      5 |            11.1566 |         0.842045  |        80001 |
| DLinear           | weather       |        336 |      5 |             6.5466 |         0.288294  |        65184 |
| Mamba1            | weather       |        336 |      5 |             7.462  |         1.0221    |        76513 |
| Mamba1AutoCon     | weather       |        336 |      5 |             9.6406 |         2.33713   |        76513 |
| Mamba2            | weather       |        336 |      5 |            22.9668 |         3.27212   |        74445 |
| Mamba2AutoCon     | weather       |        336 |      5 |            25.7942 |         3.55268   |        74445 |

## Caveats

- Metrics are in the repository's train-standardized space.
- Primary test MSE averages all chronological windows; origin sensitivity uses every horizon-th origin.
- Weather has duplicate timestamps and irregular gaps; this transfer phase is secondary, not a primary benchmark claim.
- National Illness remains exploratory because its records are weekly.
- Parameter counts and environment details are retained in the phase manifest and environment snapshot.

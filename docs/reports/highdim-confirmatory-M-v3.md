# Highdim M Results (Complete Posthoc Replay)

> Evidence label: the `_v3` artifact covers 270/270 validated coordinates, but its lambda selection is `posthoc_verified` (`publication_ready=false`). This report is a reproducible posterior replay, not an originally launch-time preregistered confirmatory result.

The validator records `valid=true`, `expected_jobs=270`, `manifest_unique_jobs=270`, and `status_counts.ok=270`; the final deduplicated records are the inferential input. The raw manifest contains ten failed path-length attempts followed by successful retries for the same job IDs.

The primary rows use lambdas selected from the validation-only phase; no test metric was used for selection.
The inferential replication unit is the random seed. Sliding windows are not treated as independent samples.

## Frozen Lambdas

```json
{
  "AutoConNetAutoCon": 0.001,
  "Mamba1AutoCon": 0.1,
  "Mamba2AutoCon": 0.003
}
```

## Primary Metrics

    dataset             model features  enc_in  c_out  pred_len  n_runs mse_mean mse_ci95 mae_mean mae_ci95 original_mse_mean original_mae_mean
electricity          TimesNet        M     321    321        96       5 0.177796 0.003088 0.278292 0.002789    7324595.077756        263.677195
electricity        AutoConNet        M     321    321        96       5 0.178883 0.002080 0.273215 0.001259    5723711.913670        239.585886
electricity AutoConNetAutoCon        M     321    321        96       5 0.179053 0.002431 0.272914 0.001597    5605692.887373        238.810340
electricity     Mamba1AutoCon        M     321    321        96       5 0.180798 0.003732 0.275830 0.004944    5675597.140045        242.311617
electricity     Mamba2AutoCon        M     321    321        96       5 0.182321 0.001211 0.276656 0.001689    5719427.409281        242.520473
electricity            Mamba2        M     321    321        96       5 0.183130 0.001307 0.277704 0.002448    5796740.954536        244.076804
electricity          PatchTST        M     321    321        96       5 0.184448 0.000401 0.272094 0.000741    5415923.952523        240.712945
electricity            Mamba1        M     321    321        96       5 0.184586 0.002848 0.278200 0.002613    5841370.634127        244.278102
electricity           DLinear        M     321    321        96       5 0.201577 0.000211 0.280269 0.001481    5660702.674728        238.927865
electricity     Mamba1AutoCon        M     321    321       336       5 0.204395 0.002471 0.295260 0.001791    8234915.801549        273.339640
electricity        AutoConNet        M     321    321       336       5 0.205658 0.004892 0.296900 0.005399    8433534.234744        276.333059
electricity AutoConNetAutoCon        M     321    321       336       5 0.206717 0.004444 0.297648 0.004266    8478969.263618        277.285644
electricity          PatchTST        M     321    321       336       5 0.207953 0.000723 0.294341 0.000694    9067515.054308        280.778476
electricity     Mamba2AutoCon        M     321    321       336       5 0.210460 0.006020 0.301432 0.005905    8796149.002004        281.261165
electricity            Mamba1        M     321    321       336       5 0.211008 0.005129 0.301063 0.004259    8798446.018732        282.460672
electricity            Mamba2        M     321    321       336       5 0.211477 0.003991 0.302049 0.003844    8690568.755563        281.184637
electricity           DLinear        M     321    321       336       5 0.214592 0.000293 0.299620 0.000857    9061448.765054        279.118431
electricity          TimesNet        M     321    321       336       5 0.216673 0.008889 0.311343 0.008086    9529954.025401        300.246038
electricity     Mamba2AutoCon        M     321    321       720       5 0.241146 0.001847 0.327242 0.005658   10913221.993525        316.853631
electricity            Mamba2        M     321    321       720       5 0.241182 0.002882 0.327371 0.006199   10908740.969421        317.726862
electricity AutoConNetAutoCon        M     321    321       720       5 0.242713 0.003094 0.330341 0.008203   10696730.332245        319.531428
electricity            Mamba1        M     321    321       720       5 0.244689 0.004575 0.329720 0.004010   11470757.742727        320.703611
electricity     Mamba1AutoCon        M     321    321       720       5 0.246006 0.008802 0.333422 0.008920   10656137.060425        320.679307
electricity        AutoConNet        M     321    321       720       5 0.248294 0.007905 0.336296 0.012283   11046275.473363        326.694190
electricity          TimesNet        M     321    321       720       5 0.249996 0.009937 0.332655 0.007181   11338339.234656        325.136033
electricity          PatchTST        M     321    321       720       5 0.252428 0.000366 0.328038 0.000190   12668013.225294        324.888447
electricity           DLinear        M     321    321       720       5 0.252694 0.000742 0.333598 0.001800   12742796.877381        327.891554
    traffic          PatchTST        M     862    862        96       5 0.501767 0.000953 0.314587 0.000525          0.000923          0.013992
    traffic        AutoConNet        M     862    862        96       5 0.638734 0.022160 0.408661 0.002765          0.001199          0.018242
    traffic     Mamba1AutoCon        M     862    862        96       5 0.640048 0.013455 0.405314 0.002927          0.001204          0.018093
    traffic AutoConNetAutoCon        M     862    862        96       5 0.641551 0.016821 0.411026 0.005985          0.001206          0.018347
    traffic            Mamba2        M     862    862        96       5 0.648306 0.002245 0.407675 0.003687          0.001223          0.018189
    traffic     Mamba2AutoCon        M     862    862        96       5 0.649773 0.004160 0.407310 0.003099          0.001227          0.018173
    traffic            Mamba1        M     862    862        96       5 0.650336 0.002382 0.409580 0.003292          0.001229          0.018259
    traffic          TimesNet        M     862    862        96       5 0.651192 0.015612 0.342777 0.003382          0.000896          0.014502
    traffic           DLinear        M     862    862        96       5 0.678933 0.000233 0.404774 0.000315          0.001270          0.017869
    traffic          PatchTST        M     862    862       336       5 0.516942 0.000886 0.318793 0.000541          0.000954          0.014141
    traffic        AutoConNet        M     862    862       336       5 0.619331 0.005843 0.388973 0.003503          0.001164          0.017289
    traffic     Mamba1AutoCon        M     862    862       336       5 0.619543 0.002380 0.386820 0.001383          0.001166          0.017206
    traffic AutoConNetAutoCon        M     862    862       336       5 0.622180 0.007001 0.390540 0.005747          0.001171          0.017362
    traffic     Mamba2AutoCon        M     862    862       336       5 0.623169 0.000398 0.388253 0.002292          0.001174          0.017252
    traffic            Mamba1        M     862    862       336       5 0.623198 0.005007 0.388860 0.002482          0.001173          0.017277
    traffic            Mamba2        M     862    862       336       5 0.623489 0.000801 0.387165 0.002490          0.001174          0.017197
    traffic           DLinear        M     862    862       336       5 0.635166 0.000953 0.384481 0.001435          0.001178          0.016925
    traffic          TimesNet        M     862    862       336       5 0.676624 0.012731 0.354250 0.004338          0.000942          0.014990
    traffic          PatchTST        M     862    862       720       5 0.550696 0.001863 0.335578 0.000695          0.001001          0.014794
    traffic     Mamba1AutoCon        M     862    862       720       5 0.655154 0.002630 0.405642 0.004033          0.001213          0.017947
    traffic AutoConNetAutoCon        M     862    862       720       5 0.657979 0.001805 0.409511 0.002523          0.001219          0.018104
    traffic        AutoConNet        M     862    862       720       5 0.658296 0.002772 0.408053 0.002590          0.001219          0.018027
    traffic     Mamba2AutoCon        M     862    862       720       5 0.658317 0.002371 0.405605 0.002073          0.001219          0.017921
    traffic            Mamba2        M     862    862       720       5 0.659914 0.001340 0.405538 0.000620          0.001225          0.017915
    traffic            Mamba1        M     862    862       720       5 0.661425 0.001793 0.407250 0.001178          0.001228          0.018008
    traffic           DLinear        M     862    862       720       5 0.676952 0.001480 0.405248 0.002096          0.001235          0.017747
    traffic          TimesNet        M     862    862       720       5 0.705246 0.014827 0.368918 0.006050          0.000984          0.015577

## Seed-Paired Deltas

    dataset features  enc_in  dec_in  c_out  seq_len  pred_len plain_model     autocon_model  autocon_lambda  n_pairs  mean_plain_mse  mean_autocon_mse  mean_delta_mse  relative_change_pct                                                                                      seed_deltas
electricity        M     321     321    321       96        96  AutoConNet AutoConNetAutoCon           0.001        5        0.178883          0.179053        0.000170             0.095094   2021:0.00032956752;2022:0.00052960828;2023:-0.00021446462;2024:0.0018580272;2025:-0.0016522023
electricity        M     321     321    321       96        96      Mamba1     Mamba1AutoCon           0.100        5        0.184586          0.180798       -0.003787            -2.051850   2021:-0.0088897992;2022:-0.0048123776;2023:0.00026203191;2024:-0.0018787918;2025:-0.0036181614
electricity        M     321     321    321       96        96      Mamba2     Mamba2AutoCon           0.003        5        0.183130          0.182321       -0.000809            -0.441716  2021:-0.0015165355;2022:0.00022552543;2023:-0.002205207;2024:-0.00042893397;2025:-0.00011942084
electricity        M     321     321    321       96       336  AutoConNet AutoConNetAutoCon           0.001        5        0.205658          0.206717        0.001060             0.515337   2021:-0.00043926749;2022:0.00057964112;2023:0.00034588175;2024:0.0039305519;2025:0.00088233748
electricity        M     321     321    321       96       336      Mamba1     Mamba1AutoCon           0.100        5        0.211008          0.204395       -0.006612            -3.133714     2021:-0.012589108;2022:-0.0037400568;2023:-0.011177632;2024:0.00053311847;2025:-0.0060882276
electricity        M     321     321    321       96       336      Mamba2     Mamba2AutoCon           0.003        5        0.211477          0.210460       -0.001017            -0.480953  2021:-0.0039489417;2022:-0.00052398689;2023:-0.0034388834;2024:-0.00081670823;2025:0.0036430029
electricity        M     321     321    321       96       720  AutoConNet AutoConNetAutoCon           0.001        5        0.248294          0.242713       -0.005581            -2.247882    2021:-0.0030371774;2022:-0.00080476214;2023:-0.01219886;2024:-0.0065149922;2025:-0.0053509905
electricity        M     321     321    321       96       720      Mamba1     Mamba1AutoCon           0.100        5        0.244689          0.246006        0.001317             0.538224      2021:-0.0012765857;2022:0.012865545;2023:0.00011920078;2024:0.0028715531;2025:-0.0079948262
electricity        M     321     321    321       96       720      Mamba2     Mamba2AutoCon           0.003        5        0.241182          0.241146       -0.000036            -0.014968   2021:2.4530057e-06;2022:-0.0012920769;2023:0.0018178873;2024:-0.00050103147;2025:-0.0002077339
    traffic        M     862     862    862       96        96  AutoConNet AutoConNetAutoCon           0.001        5        0.638734          0.641551        0.002818             0.441113       2021:0.0017138773;2022:0.031218132;2023:-0.010575108;2024:-0.0071700589;2025:-0.0010991396
    traffic        M     862     862    862       96        96      Mamba1     Mamba1AutoCon           0.100        5        0.650336          0.640048       -0.010288            -1.581934   2021:-0.0090373436;2022:-0.0033658983;2023:-0.0096809797;2024:-0.029184958;2025:-0.00017023193
    traffic        M     862     862    862       96        96      Mamba2     Mamba2AutoCon           0.003        5        0.648306          0.649773        0.001467             0.226350          2021:0.001629116;2022:0.001539073;2023:0.0049907639;2024:-0.0040758038;2025:0.003254068
    traffic        M     862     862    862       96       336  AutoConNet AutoConNetAutoCon           0.001        5        0.619331          0.622180        0.002849             0.459992      2021:-0.00012682684;2022:0.00075731114;2023:0.0049377781;2024:0.006750952;2025:0.0019251504
    traffic        M     862     862    862       96       336      Mamba1     Mamba1AutoCon           0.100        5        0.623198          0.619543       -0.003655            -0.586546     2021:0.0023448282;2022:-0.0018027734;2023:-0.012107311;2024:-0.0015613044;2025:-0.0051501507
    traffic        M     862     862    862       96       336      Mamba2     Mamba2AutoCon           0.003        5        0.623489          0.623169       -0.000320            -0.051334 2021:0.00026226251;2022:-0.00021658761;2023:-0.00049909081;2024:-0.0014839672;2025:0.00033707542
    traffic        M     862     862    862       96       720  AutoConNet AutoConNetAutoCon           0.001        5        0.658296          0.657979       -0.000316            -0.048062    2021:-0.00049918941;2022:0.00065281709;2023:-0.0043261701;2024:0.0013332056;2025:0.0012573883
    traffic        M     862     862    862       96       720      Mamba1     Mamba1AutoCon           0.100        5        0.661425          0.655154       -0.006271            -0.948085   2021:-0.0055007945;2022:-0.0027851638;2023:-0.0068251742;2024:-0.0099951653;2025:-0.0062480617
    traffic        M     862     862    862       96       720      Mamba2     Mamba2AutoCon           0.003        5        0.659914          0.658317       -0.001597            -0.241950   2021:-0.0018188767;2022:0.0012954524;2023:-0.00039811389;2024:-0.0032767904;2025:-0.0037849798

## Non-overlapping Origin Sensitivity

Each seed uses the registered origin_stride; this is a sensitivity analysis, not an additional independent-sample test.

    dataset features  enc_in  dec_in  c_out  seq_len  pred_len  origin_stride plain_model     autocon_model  autocon_lambda  n_pairs  mean_origin_delta_mse  relative_change_pct                                                                                    seed_deltas
electricity        M     321     321    321       96        96             96  AutoConNet AutoConNetAutoCon           0.001        5               0.000413             0.232236   2021:0.0014009223;2022:-0.00010410394;2023:-0.0011824142;2024:0.003743387;2025:-0.0017912101
electricity        M     321     321    321       96        96             96      Mamba1     Mamba1AutoCon           0.100        5              -0.005936            -3.201788   2021:-0.014423332;2022:-0.0054384056;2023:-0.0036116484;2024:0.0006471546;2025:-0.0068553318
electricity        M     321     321    321       96        96             96      Mamba2     Mamba2AutoCon           0.003        5              -0.000768            -0.416315   2021:-0.0060590907;2022:0.0017729364;2023:-0.0010751762;2024:0.00030683016;2025:0.0012139466
electricity        M     321     321    321       96       336            336  AutoConNet AutoConNetAutoCon           0.001        5               0.000460             0.241322     2021:-0.004353115;2022:0.0015013068;2023:0.00017689937;2024:0.004597852;2025:0.00037710051
electricity        M     321     321    321       96       336            336      Mamba1     Mamba1AutoCon           0.100        5              -0.003245            -1.664533    2021:-0.010910349;2022:-0.007422786;2023:0.00077011048;2024:0.0032358738;2025:-0.0018957071
electricity        M     321     321    321       96       336            336      Mamba2     Mamba2AutoCon           0.003        5               0.003290             1.702982    2021:-0.00036178251;2022:0.0066377724;2023:-0.0012896388;2024:0.001928929;2025:0.0095353382
electricity        M     321     321    321       96       720            720  AutoConNet AutoConNetAutoCon           0.001        5              -0.005191            -2.045013 2021:-0.00026703901;2022:-0.0007712291;2023:-0.013692381;2024:-0.0073608193;2025:-0.0038639488
electricity        M     321     321    321       96       720            720      Mamba1     Mamba1AutoCon           0.100        5              -0.004316            -1.710295     2021:0.002211635;2022:0.0031426539;2023:-0.0058300918;2024:0.00078295573;2025:-0.021885692
electricity        M     321     321    321       96       720            720      Mamba2     Mamba2AutoCon           0.003        5               0.000177             0.071202  2021:0.0078086576;2022:-0.0030671137;2023:0.00057514823;2024:-0.0015432122;2025:-0.0028870053
    traffic        M     862     862    862       96        96             96  AutoConNet AutoConNetAutoCon           0.001        5              -0.002140            -0.330147      2021:-0.010090008;2022:0.028446652;2023:-0.010675094;2024:-0.016083865;2025:-0.0022961633
    traffic        M     862     862    862       96        96             96      Mamba1     Mamba1AutoCon           0.100        5              -0.014319            -2.173385    2021:-0.010322019;2022:-0.0052789516;2023:-0.025680948;2024:-0.028197874;2025:-0.0021177061
    traffic        M     862     862    862       96        96             96      Mamba2     Mamba2AutoCon           0.003        5               0.002109             0.322333     2021:0.0028161216;2022:0.0016869747;2023:0.0047671204;2024:-0.0037317544;2025:0.0050054998
    traffic        M     862     862    862       96       336            336  AutoConNet AutoConNetAutoCon           0.001        5               0.005041             0.697990    2021:0.0067043113;2022:-0.0077586526;2023:-0.00068493061;2024:0.018706267;2025:0.0082402238
    traffic        M     862     862    862       96       336            336      Mamba1     Mamba1AutoCon           0.100        5              -0.008640            -1.179989        2021:0.01523665;2022:-0.014750881;2023:-0.030239191;2024:0.0057005431;2025:-0.019145411
    traffic        M     862     862    862       96       336            336      Mamba2     Mamba2AutoCon           0.003        5              -0.004570            -0.629037  2021:-0.00055604014;2022:0.0018204635;2023:-0.0015740116;2024:-0.0032298176;2025:-0.019311605
    traffic        M     862     862    862       96       720            720  AutoConNet AutoConNetAutoCon           0.001        5              -0.006265            -0.866573  2021:-0.0014760927;2022:-0.0064608131;2023:-0.013645959;2024:-0.0083916388;2025:-0.0013484527
    traffic        M     862     862    862       96       720            720      Mamba1     Mamba1AutoCon           0.100        5              -0.006978            -0.968622     2021:-0.0010931307;2022:0.0046314803;2023:-0.006119334;2024:-0.019077314;2025:-0.013230258
    traffic        M     862     862    862       96       720            720      Mamba2     Mamba2AutoCon           0.003        5               0.003669             0.511191      2021:0.0082703603;2022:0.0089986503;2023:0.011747541;2024:-0.0032710795;2025:-0.007400887

## Resource Summary

            model     dataset features  enc_in  dec_in  c_out  pred_len  runs  elapsed_sec_mean  elapsed_sec_std  parameters
       AutoConNet electricity        M     321     321    321        96     5          186.1570        27.094157    159105.0
AutoConNetAutoCon electricity        M     321     321    321        96     5          186.6696        59.958914    159105.0
          DLinear electricity        M     321     321    321        96     5           76.8222        19.797881     18624.0
           Mamba1 electricity        M     321     321    321        96     5          147.8992        71.996238    138305.0
    Mamba1AutoCon electricity        M     321     321    321        96     5          211.7746        56.770435    138305.0
           Mamba2 electricity        M     321     321    321        96     5          186.8624        92.296001    133753.0
    Mamba2AutoCon electricity        M     321     321    321        96     5          197.3352        68.148910    133753.0
         PatchTST electricity        M     321     321    321        96     5          328.0938         1.222707    100192.0
         TimesNet electricity        M     321     321    321        96     5          239.2852        47.895551   2445185.0
       AutoConNet electricity        M     321     321    321       336     5          190.0246        88.978010    205665.0
AutoConNetAutoCon electricity        M     321     321    321       336     5          222.0748        84.065582    205665.0
          DLinear electricity        M     321     321    321       336     5           80.4628        26.601967     65184.0
           Mamba1 electricity        M     321     321    321       336     5           90.2858        15.262095    184865.0
    Mamba1AutoCon electricity        M     321     321    321       336     5          299.7458        45.992945    184865.0
           Mamba2 electricity        M     321     321    321       336     5          215.2948       101.074325    180313.0
    Mamba2AutoCon electricity        M     321     321    321       336     5          391.0312        88.704250    180313.0
         PatchTST electricity        M     321     321    321       336     5          367.4928         3.384827    284752.0
         TimesNet electricity        M     321     321    321       336     5          349.8086        88.773212   2468465.0
       AutoConNet electricity        M     321     321    321       720     5          152.0052        46.451230    280161.0
AutoConNetAutoCon electricity        M     321     321    321       720     5          148.9168        30.863973    280161.0
          DLinear electricity        M     321     321    321       720     5          189.7922        84.162138    139680.0
           Mamba1 electricity        M     321     321    321       720     5          264.4026       125.291704    259361.0
    Mamba1AutoCon electricity        M     321     321    321       720     5          243.4042       135.809164    259361.0
           Mamba2 electricity        M     321     321    321       720     5          196.1420        66.728992    254809.0
    Mamba2AutoCon electricity        M     321     321    321       720     5          229.4906        63.335366    254809.0
         PatchTST electricity        M     321     321    321       720     5          448.8936         9.131603    580048.0
         TimesNet electricity        M     321     321    321       720     5          583.0732       101.621476   2505713.0
       AutoConNet     traffic        M     862     862    862        96     5          118.7760        71.924917    298142.0
AutoConNetAutoCon     traffic        M     862     862    862        96     5          125.1338        69.058054    298142.0
          DLinear     traffic        M     862     862    862        96     5          106.1020        11.060509     18624.0
           Mamba1     traffic        M     862     862    862        96     5           69.0070         6.187509    277342.0
    Mamba1AutoCon     traffic        M     862     862    862        96     5          144.6292        90.366695    277342.0
           Mamba2     traffic        M     862     862    862        96     5           90.5016        12.297735    272790.0
    Mamba2AutoCon     traffic        M     862     862    862        96     5          109.9052        16.294145    272790.0
         PatchTST     traffic        M     862     862    862        96     5          502.1642         3.715576    100192.0
         TimesNet     traffic        M     862     862    862        96     5          208.8588        58.328342   2584222.0
       AutoConNet     traffic        M     862     862    862       336     5          156.5460        28.907850    344702.0
AutoConNetAutoCon     traffic        M     862     862    862       336     5          222.6682        45.143894    344702.0
          DLinear     traffic        M     862     862    862       336     5          149.3910        41.697373     65184.0
           Mamba1     traffic        M     862     862    862       336     5          215.6146       122.660774    323902.0
    Mamba1AutoCon     traffic        M     862     862    862       336     5          333.7244       148.373146    323902.0
           Mamba2     traffic        M     862     862    862       336     5          222.6272        97.740289    319350.0
    Mamba2AutoCon     traffic        M     862     862    862       336     5          229.5842        48.980475    319350.0
         PatchTST     traffic        M     862     862    862       336     5          631.5388         2.698409    284752.0
         TimesNet     traffic        M     862     862    862       336     5          290.3386        59.874687   2607502.0
       AutoConNet     traffic        M     862     862    862       720     5          480.5872       120.626815    419198.0
AutoConNetAutoCon     traffic        M     862     862    862       720     5          377.9400        78.177951    419198.0
          DLinear     traffic        M     862     862    862       720     5          477.6248       237.623676    139680.0
           Mamba1     traffic        M     862     862    862       720     5          490.8024       216.193610    398398.0
    Mamba1AutoCon     traffic        M     862     862    862       720     5          650.4620       276.111352    398398.0
           Mamba2     traffic        M     862     862    862       720     5          607.3254       159.308997    393846.0
    Mamba2AutoCon     traffic        M     862     862    862       720     5          379.0970        99.498521    393846.0
         PatchTST     traffic        M     862     862    862       720     5          880.4588        12.195472    580048.0
         TimesNet     traffic        M     862     862    862       720     5          443.3312       116.741306   2644750.0

## Caveats

- Metrics are in the repository's train-standardized space; original-scale columns are descriptive.
- Primary test MSE averages all chronological windows; origin sensitivity uses every horizon-th origin.
- Parameter counts and environment details are retained in the phase manifest and environment snapshot.

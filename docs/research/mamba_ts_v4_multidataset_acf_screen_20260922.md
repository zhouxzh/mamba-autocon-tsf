# MambaTS v4 ACF memory screen（2026-09-22）

36/36 个 H96 validation-only 作业成功，覆盖 MambaTSv3 plain、MambaTSv4AutoCon ACF memory + representation fusion 的 `lambda=0` null 和 `lambda=0.001`。测试集没有运行。

| 数据集 | v3 plain | v4 null | v4 lambda=0.001 | v4 null 相对 v3 | lambda 增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.709511 | 0.710455 | 0.710505 | `+0.14%` | `+0.01%` |
| ETTm1 | 0.394196 | 0.392661 | 0.392570 | `-0.39%` | `-0.02%` |
| Electricity | 0.159181 | 0.158264 | 0.158271 | `-0.58%` | `+0.00%` |
| Traffic | 0.370582 | 0.372997 | 0.373024 | `+0.64%` | `+0.01%` |

结论：ACF memory 对数据集敏感，在 Traffic 这一关键高维数据上退化；representation fusion 与 `lambda=0.001` 没有带来独立精度收益。它暂不满足主模型晋级门槛。

这批旧工件是在 MambaTS sidecar 接口补齐前产生的，当前只能作为 validation exploratory evidence。现在的 MambaTS v4/v5 已支持 compact `acf_prior.npz` 的保存、哈希绑定、几何校验和 replay 加载；未来若重跑，必须使用新的 source snapshot，不得把本批工件追溯性标成 sidecar-backed confirmation。

计划：[`mamba_ts_v4_multidataset_acf_autocon_screen_20260922.json`](mamba_ts_v4_multidataset_acf_autocon_screen_20260922.json)。

# Weather / ETTh1 H96 Relation-Auxiliary-Only Resource Readout (2026-09-29)

这是一组绑定到跨域 validation matrix manifest 和 immutable source snapshot 的 prediction-only GPU
资源测量。它使用实际训练 `config.json`、RTX 5090 D、FP32、device-resident synthetic inputs、
`B=32,L=96,H=96`、25 warm-up、100 timed forwards、输入 seed `3331`；不读取数据、不训练、不含
AutoCon relation loss。完整配置/源码绑定审计见
[`resource attestation`](../../artifacts/exploratory/mambatsv3_relation_auxiliary_crossdomain_h96_validation_20260929/resources/readout/readout.md)。

| Dataset | Candidate | RLinear | PatchTST | TimesNet | iTransformer | Mamba2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weather H96 | `.2758 ms` | `.0648` | `.2858` | `3.1347` | `.1535` | `.7158` |
| ETTh1 H96 | `.2751 ms` | `.0659` | `.2827` | `2.9577` | `.1560` | `.7047` |

候选在两个 shape 上快于 PatchTST、TimesNet 和 Mamba2，慢于 RLinear 和 iTransformer。Weather 同时通过
本地 accuracy screen，因此可作为“相对指定非线性比较器的 fixed-shape prediction-only latency 优势”的
补充证据；ETTh1 精度未通过，不应作为正面精度/速度 Pareto 主张。该结果不是训练吞吐、端到端部署延迟或
普适速度结论。

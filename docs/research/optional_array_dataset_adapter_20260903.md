# Optional Solar/PEMS Array Dataset Adapter

This repository now has an opt-in `Dataset_Array` loader for dense Solar/PEMS
benchmarks. No raw Solar/PEMS files are bundled or downloaded by the runner.
The adapter is intentionally strict so an array cannot be mistaken for one of
the existing CSV benchmarks.

## Accepted files

| Format | Required representation | CLI fields |
| --- | --- | --- |
| NPZ | archive key `data` (or an explicitly registered key), shape `[time, channel]` or `[time, channel, feature]` | `--data_format npz --data_key data --data_feature_index 0` |
| TXT | header-free, finite numeric rows (whitespace- or comma-delimited), shape `[time, channel]` | `--data_format txt` |

Three-dimensional NPZ inputs select one feature plane with
`--data_feature_index`. There is no timestamp column in either format. The
loader generates regular calendar marks from `--freq` plus the explicit
`--data_interval_minutes` when `freq=t`, anchored at `2000-01-01`, and applies
the same ratio split (`train_ratio`, 60/20/20 by default) and train-only
`StandardScaler` convention as `Dataset_Custom`.

## Registered optional datasets

The plan registry contains metadata for these exact local paths:

| Name | Expected path | Format | Channels | Target index | Frequency |
| --- | --- | --- | ---: | ---: | --- |
| `solar` | `dataset/solar/solar_AL.txt` | TXT | 137 | 136 | `t` (10 min) |
| `pems03` | `dataset/PEMS03/PEMS03.npz` | NPZ (`data`) | 358 | 357 | `t` (5 min) |
| `pems04` | `dataset/PEMS04/PEMS04.npz` | NPZ (`data`) | 307 | 306 | `t` (5 min) |
| `pems07` | `dataset/PEMS07/PEMS07.npz` | NPZ (`data`) | 883 | 882 | `t` (5 min) |
| `pems08` | `dataset/PEMS08/PEMS08.npz` | NPZ (`data`) | 170 | 169 | `t` (5 min) |

These names are not added to any existing phase. A plan that names one must
provide a matching `data_fingerprints` entry (including `format`, frequency,
channel count, target index, and SHA-256) and the file must already exist.
Missing files produce an explicit path and placement hint during plan
validation; no network access is attempted. The registered Solar file is now
present locally: `solar_AL.txt` has SHA-256
`230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2`,
with 52,560 rows and 137 channels. Its upstream compressed source MD5 is
`41ef7fdc958c2ca3fac9cd06d6227073`.

## Example

```bash
python run.py --is_training 1 --model Mamba2 --data pems03 \
  --root_path dataset/PEMS03 --data_path PEMS03.npz \
  --data_format npz --data_key data --data_feature_index 0 \
  --features M --freq t --data_interval_minutes 5 \
  --enc_in 358 --dec_in 358 --c_out 358 \
  --seq_len 96 --label_len 48 --pred_len 96 --model_id pems03_smoke
```

Use `tests/test_array_data_loader.py` for a no-download loader smoke. It uses
temporary arrays only and does not claim benchmark accuracy.

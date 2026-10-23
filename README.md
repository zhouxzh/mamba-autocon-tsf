# Self-Supervised Contrastive Learning for Long-term Forecasting

This is an official repository for the paper, titled [Self-Supervised Contrastive Learning for Long-term Forecasting](https://openreview.net/forum?id=nBCuRzjqK7), accepted at ICLR'24.

Our repo includes Autocorrelation-based Contrastive Loss (AutoCon) and Model Architecture (AutoConNet).

Our code is implemented based on [TSLib](https://github.com/thuml/Time-Series-Library), and it also includes some time series models. If you want to evaluate more models, please check the TSLib. We sincerely thank the creators of TSLib for their contributions to the time series community.

## Project documentation

The maintained experiment protocol, research notes, paper-facing reports, environment snapshot, and source-paper references are indexed in [`docs/README.md`](docs/README.md). For a one-page human review of the current Mamba/AutoCon/ACF code and numbers, start with [`docs/core-review.md`](docs/core-review.md), then use [`docs/manual-review.md`](docs/manual-review.md) for the full artifact audit. The cleaned repository keeps audits and baselines under `artifacts/`; registered pilot/corrected runs write metrics, plots, logs, manifests, and checkpoints under their explicit `artifacts/paper_final/{strict,corrected}/<phase>/` roots. The latest fair-baseline and variable-token relation-null screens are linked from the docs index and are explicitly exploratory, not publication-ready evidence: the variable candidate is better than PatchTST on Traffic/H96 but worse on ETTh1/H96, and the synthetic resource result does not establish a universal end-to-end speed advantage.

The current code exposes `Mamba1`, `Mamba1AutoCon`, `Mamba2`, `Mamba2AutoCon`, `MambaTS`, `MambaTSAutoCon`, `MambaACF`, `MambaACFAutoCon`, `MambaACF1`, and `MambaACF1AutoCon` through `run.py`. The active corrected core revalidation is [`docs/research/corrected_training_plan.json`](docs/research/corrected_training_plan.json); the PEMS baseline/AutoCon matrix is [`docs/research/mamba_pems_baseline_autocon_pilot_20260909.json`](docs/research/mamba_pems_baseline_autocon_pilot_20260909.json) and remains validation-only until human review. Provenance and result boundaries are documented in [`docs/research/provenance-and-reproducibility.md`](docs/research/provenance-and-reproducibility.md).

## Environment setup

The dependency list below is the legacy environment used by the original paper. The maintained Mamba experiments use the separately verified `mamba` conda environment (Python 3.9.25, PyTorch 2.8.0+cu128, `mamba_ssm` 2.3.1, and CUDA).

Legacy environment:
```
python == 3.8
torch == 1.7.1
numpy == 1.23.5
pandas
statsmodels
scikit-learn
einops
sympy
numba
```

Check the current training environment:
```
conda run --no-capture-output -n mamba python -c "import sys,torch,mamba_ssm; print(sys.executable); print(torch.__version__, torch.cuda.is_available()); print(mamba_ssm.__version__)"
```

## Run with Command Line
```
python run.py --AutoCon --AutoCon_multiscales 720 --AutoCon_wnorm LastVal --AutoCon_lambda 0.1 --d_model 16 --d_ff 16 --e_layers 3 --target OT --c_out 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --model_id ICLR24 --model AutoConNet --data ETTh2 --seq_len 336 --pred_len 720 --enc_in 1 --des 'Exp' --itr 1 --batch_size 64 --learning_rate 0.005 --features S
```

## Training and reproducibility

The maintained entry point is the registered plan in [`docs/research/corrected_training_plan.json`](docs/research/corrected_training_plan.json), executed with [`scripts/run_training_plan.py`](scripts/run_training_plan.py). It records an immutable source snapshot, dataset hashes, per-job configuration, logs, checkpoints, and provenance metadata. The Mamba1/Mamba2 smoke checks are documented in [`docs/README.md`](docs/README.md).

## Citations
```
@inproceedings{park2024self,
  title={Self-Supervised Contrastive Learning for Long-term Forecasting},
  author={Junwoo Park and Daehoon Gwak and Jaegul Choo and Edward Choi},
  booktitle={Proc. the International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

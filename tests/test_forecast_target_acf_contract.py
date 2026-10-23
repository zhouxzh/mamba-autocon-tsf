import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path=Path("scripts/pretrain_mambatsv3_core_autocon.py")
    spec=importlib.util.spec_from_file_location("forecast_target_acf_pretrain",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forecast_target_acf_is_train_only_and_shape_aligned():
    module=_module()
    time=np.arange(600,dtype=np.float32)
    values=np.stack((np.sin(time*2*np.pi/24),np.cos(time*2*np.pi/24)),axis=1)
    peaks=np.asarray([[24,48,0],[24,48,0]],dtype=np.int64)
    table,metadata=module._forecast_target_acf_table(values,96,96,peaks,max_pairs=32)
    assert table.shape==(2,600)
    assert np.allclose(table[:,0],1.0)
    assert metadata["candidate_lag_count"]==2
    assert metadata["evaluated_pair_slots"]==4
    assert np.all(np.isfinite(table))
    assert np.all(table[:,24]>.9)


def test_forecast_target_acf_cli_is_limited_to_forecast_next_patch_views():
    text=Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    for token in ("--forecast-target-acf", "future-residual", "forecast_affine", "forecast_contrastive", "_forecast_target_acf_table"):
        assert token in text

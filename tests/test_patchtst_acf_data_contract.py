import numpy as np
import pandas as pd

from scripts.pretrain_patchtst_acf_core import load_values


def test_patchtst_acf_csv_loader_removes_only_time_column(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"date": [1, 2, 3], "a": [1., 2., 3.], "b": [4., 5., 6.]}).to_csv(path, index=False)
    values = load_values(path, "csv")
    assert values.shape == (3, 2)
    np.testing.assert_allclose(values[0], [1., 4.])

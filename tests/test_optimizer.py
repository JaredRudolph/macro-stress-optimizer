import json

import numpy as np
import pandas as pd
import pytest

from macro_stress_optimizer.optimizer import INDICATOR_COLS, optimize_weights, run


@pytest.fixture
def synthetic_parquet(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.bdate_range("2015-01-01", periods=n)
    data = {col: rng.random(n) for col in INDICATOR_COLS}
    data["SPY"] = np.cumprod(1 + rng.normal(0.0003, 0.01, n)) * 300
    data["STRESS_SCORE"] = rng.random(n)
    df = pd.DataFrame(data, index=idx)
    path = tmp_path / "stress_score.parquet"
    df.to_parquet(path)
    return path


def make_X_y(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    X = pd.DataFrame(
        rng.random((n, len(INDICATOR_COLS))), index=idx, columns=INDICATOR_COLS
    )
    score = X.mean(axis=1)
    y = (score > score.quantile(0.75)).astype(int)
    return X, y


def test_optimize_weights_returns_required_keys():
    X, y = make_X_y()
    result = optimize_weights(X, y, alphas=[1.0], n_jobs=1)
    assert set(result.keys()) == {"weights", "auc_equal", "auc_optimized", "best_alpha"}


def test_weights_has_all_indicators():
    X, y = make_X_y()
    result = optimize_weights(X, y, alphas=[1.0], n_jobs=1)
    assert set(result["weights"].keys()) == set(INDICATOR_COLS)


def test_weights_sum_to_one():
    X, y = make_X_y()
    result = optimize_weights(X, y, alphas=[1.0], n_jobs=1)
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-5


def test_weights_within_bounds():
    X, y = make_X_y()
    result = optimize_weights(X, y, alphas=[1.0], n_jobs=1)
    for w in result["weights"].values():
        assert -0.10 - 1e-6 <= w <= 0.25 + 1e-6


def test_auc_values_are_valid():
    X, y = make_X_y()
    result = optimize_weights(X, y, alphas=[1.0], n_jobs=1)
    assert 0.0 <= result["auc_equal"] <= 1.0
    assert 0.0 <= result["auc_optimized"] <= 1.0


def test_run_writes_json(synthetic_parquet, tmp_path):
    output_path = tmp_path / "optimized_weights.json"
    run(parquet_path=synthetic_parquet, output_path=output_path, alphas=[1.0])
    assert output_path.exists()


def test_run_json_content(synthetic_parquet, tmp_path):
    output_path = tmp_path / "optimized_weights.json"
    result = run(parquet_path=synthetic_parquet, output_path=output_path, alphas=[1.0])

    with open(output_path) as f:
        loaded = json.load(f)

    assert loaded["weights"] == result["weights"]
    assert "auc_equal" in loaded
    assert "auc_optimized" in loaded


def test_run_json_weights_sum_to_one(synthetic_parquet, tmp_path):
    output_path = tmp_path / "optimized_weights.json"
    run(parquet_path=synthetic_parquet, output_path=output_path, alphas=[1.0])

    with open(output_path) as f:
        loaded = json.load(f)

    assert abs(sum(loaded["weights"].values()) - 1.0) < 1e-5

import numpy as np
import pandas as pd

from macro_stress.features import (
    FLIP_COLS,
    SCORE_COLS,
    compute_stress_score,
    flip_direction,
    rolling_percentile_rank,
)


def make_score_df(n=800, seed=42):
    idx = pd.bdate_range("2017-01-01", periods=n)
    rng = np.random.default_rng(seed)
    data = {col: rng.random(n) for col in SCORE_COLS}
    data["SPY"] = rng.random(n) * 400 + 200
    return pd.DataFrame(data, index=idx)


def test_rolling_percentile_rank_bounded():
    s = pd.Series(range(50), dtype=float)
    result = rolling_percentile_rank(s, window=10).dropna()
    assert (result >= 0).all() and (result <= 1).all()


def test_rolling_percentile_rank_nan_warmup():
    s = pd.Series(range(20), dtype=float)
    result = rolling_percentile_rank(s, window=10)
    assert result.iloc[:9].isna().all()
    assert result.iloc[9:].notna().all()


def test_rolling_percentile_rank_monotone_trends_up():
    s = pd.Series(range(50), dtype=float)
    result = rolling_percentile_rank(s, window=10).dropna()
    assert result.iloc[-1] >= result.iloc[0]


def test_flip_direction_values():
    s = pd.Series([0.0, 0.3, 0.5, 0.7, 1.0])
    pd.testing.assert_series_equal(
        flip_direction(s), pd.Series([1.0, 0.7, 0.5, 0.3, 0.0])
    )


def test_flip_direction_is_involution():
    s = pd.Series([0.1, 0.4, 0.9])
    pd.testing.assert_series_equal(flip_direction(flip_direction(s)), s)


def test_compute_stress_score_output_columns():
    result = compute_stress_score(make_score_df())
    assert "stress_score" in result.columns
    assert "SPY" in result.columns


def test_compute_stress_score_bounded():
    result = compute_stress_score(make_score_df())
    scores = result["stress_score"].dropna()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_compute_stress_score_spy_passthrough():
    df = make_score_df()
    result = compute_stress_score(df)
    pd.testing.assert_series_equal(result["SPY"], df["SPY"])


def test_flip_cols_inverted_relative_to_unflipped():
    # A flip col and a non-flip col both trending up should produce opposite ranks.
    flip_col = next(iter(FLIP_COLS))
    non_flip_col = next(col for col in SCORE_COLS if col not in FLIP_COLS)

    n = 800
    idx = pd.bdate_range("2017-01-01", periods=n)
    rng = np.random.default_rng(0)
    data = {col: rng.random(n) for col in SCORE_COLS}
    data["SPY"] = 300.0
    data[flip_col] = np.linspace(0, 1, n)
    data[non_flip_col] = np.linspace(0, 1, n)
    df = pd.DataFrame(data, index=idx)

    result = compute_stress_score(df).dropna()
    assert result[flip_col].iloc[-1] < result[non_flip_col].iloc[-1]

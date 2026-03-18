import numpy as np
import pandas as pd

from macro_stress_ml.labels import compute_drawdown_labels


def make_spy(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_no_drawdown_returns_all_zeros():
    spy = make_spy([100, 101, 102, 103, 104])
    assert (compute_drawdown_labels(spy) == 0).all()


def test_at_all_time_high_is_zero():
    spy = make_spy([100, 110, 120])
    assert (compute_drawdown_labels(spy) == 0).all()


def test_drawdown_above_threshold_is_one():
    spy = make_spy([100, 100, 85])
    assert compute_drawdown_labels(spy, threshold=0.10).iloc[-1] == 1


def test_drawdown_below_threshold_is_zero():
    spy = make_spy([100, 100, 95])
    assert compute_drawdown_labels(spy, threshold=0.10).iloc[-1] == 0


def test_drawdown_just_over_threshold_is_one():
    # 100 -> 89 is -11%, clearly over the 10% threshold
    spy = make_spy([100, 100, 89])
    assert compute_drawdown_labels(spy, threshold=0.10).iloc[-1] == 1


def test_drawdown_just_under_threshold_is_zero():
    # 100 -> 91 is -9%, clearly under the 10% threshold
    spy = make_spy([100, 100, 91])
    assert compute_drawdown_labels(spy, threshold=0.10).iloc[-1] == 0


def test_recovery_to_new_high_resets_label():
    spy = make_spy([100, 80, 110])
    labels = compute_drawdown_labels(spy, threshold=0.10)
    assert labels.iloc[1] == 1
    assert labels.iloc[2] == 0


def test_output_is_binary():
    rng = np.random.default_rng(0)
    spy = make_spy(rng.random(100) * 100 + 200)
    assert set(compute_drawdown_labels(spy).unique()).issubset({0, 1})


def test_output_index_matches_input():
    spy = make_spy([100, 95, 90, 85])
    pd.testing.assert_index_equal(compute_drawdown_labels(spy).index, spy.index)


def test_custom_threshold():
    spy = make_spy([100, 100, 94])
    assert compute_drawdown_labels(spy, threshold=0.05).iloc[-1] == 1

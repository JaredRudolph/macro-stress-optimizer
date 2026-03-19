import pandas as pd


def compute_drawdown_labels(spy: pd.Series, threshold: float = 0.10) -> pd.Series:
    """Return binary series: 1 where SPY is at least `threshold` below its running
    high."""
    drawdown = spy / spy.cummax() - 1
    return (drawdown <= -threshold).astype(int)

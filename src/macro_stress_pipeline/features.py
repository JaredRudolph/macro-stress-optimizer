import pandas as pd

FLIP_COLS = {
    "XLK_XLV",
    "T10Y2Y",
    "USALOLITOAASTSAM",
}

MARKET_SCORE_COLS = [
    "^VIX",
    "VIX_VIX3M",
    "^SKEW",
    "GLD_SPY",
    "DX=F",
    "USDCNY=X",
    "XLK_XLV",
]

FRED_SCORE_COLS = [
    "T10Y2Y",
    "ICSA",
    "CPIAUCSL",
    "DRCCLACBS",
    "USALOLITOAASTSAM",
    "BAMLH0A0HYM2",
]

SCORE_COLS = MARKET_SCORE_COLS + FRED_SCORE_COLS

ROLLING_WINDOW_DAYS = 756


def rolling_percentile_rank(
    series: pd.Series, window: int = ROLLING_WINDOW_DAYS
) -> pd.Series:
    return series.rolling(window).apply(lambda x: (x < x[-1]).mean(), raw=True)


def flip_direction(series: pd.Series) -> pd.Series:
    return 1 - series


def compute_stress_score(df: pd.DataFrame) -> pd.DataFrame:
    required = set(SCORE_COLS) | {"SPY"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_stress_score: missing columns: {sorted(missing)}")
    ranked = {}
    for col in SCORE_COLS:
        ranked_col = rolling_percentile_rank(df[col])
        if col in FLIP_COLS:
            ranked_col = flip_direction(ranked_col)
        ranked[col] = ranked_col

    ranked_df = pd.DataFrame(ranked, index=df.index)
    ranked_df["STRESS_SCORE"] = ranked_df.mean(axis=1)
    ranked_df["SPY"] = df["SPY"]
    return ranked_df[["STRESS_SCORE", "SPY"] + MARKET_SCORE_COLS + FRED_SCORE_COLS]


if __name__ == "__main__":
    from dotenv import load_dotenv

    from macro_stress_pipeline.fetch_data import fetch_fred_data, fetch_market_data
    from macro_stress_pipeline.pipeline import (
        FRED_SERIES,
        MARKET_TICKERS,
        START_DATE,
        get_end_date,
    )
    from macro_stress_pipeline.process_data import merge_all

    load_dotenv()

    df_market = fetch_market_data(MARKET_TICKERS, START_DATE, get_end_date())
    df_fred = fetch_fred_data(FRED_SERIES, START_DATE, get_end_date())
    df_merged = merge_all(df_market, df_fred)
    df_scored = compute_stress_score(df_merged)
    score_min = df_scored["STRESS_SCORE"].min()
    score_max = df_scored["STRESS_SCORE"].max()

    print(df_scored[["STRESS_SCORE"]].dropna().tail(10))
    print(f"\nNaN count:\n{df_scored.isna().sum()}")
    print(f"\nScore range: {score_min:.2f} - {score_max:.2f}")

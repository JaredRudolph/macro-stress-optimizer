import pandas as pd

FLIP_COLS = {"HYG_LQD", "XLK_XLV", "T10Y2Y", "USALOLITOAASTSAM"}

SCORE_COLS = [
    "^VIX",
    "VIX_VIX3M",
    "^SKEW",
    "HYG_LQD",
    "GLD_SPY",
    "DX=F",
    "USDCNY=X",
    "XLK_XLV",
    "T10Y2Y",
    "ICSA",
    "CPIAUCSL",
    "DRCCLACBS",
    "USALOLITOAASTSAM",
]


def rolling_percentile_rank(series: pd.Series, window: int = 756) -> pd.Series:
    return series.rolling(window).apply(lambda x: (x < x[-1]).mean(), raw=True)


def flip_direction(series: pd.Series) -> pd.Series:
    return 1 - series


def compute_stress_score(df: pd.DataFrame) -> pd.DataFrame:
    ranked = {}
    for col in SCORE_COLS:
        ranked_col = rolling_percentile_rank(df[col])
        if col in FLIP_COLS:
            ranked_col = flip_direction(ranked_col)
        ranked[col] = ranked_col

    ranked_df = pd.DataFrame(ranked, index=df.index)
    ranked_df["stress_score"] = ranked_df.mean(axis=1)
    ranked_df["SPY"] = df["SPY"]
    return ranked_df


if __name__ == "__main__":
    from dotenv import load_dotenv

    from macro_stress.fetch_data import fetch_fred_data, fetch_market_data
    from macro_stress.pipeline import (
        FRED_SERIES,
        MARKET_TICKERS,
        START_DATE,
        get_end_date,
    )
    from macro_stress.process_data import merge_all

    load_dotenv()

    df_market = fetch_market_data(MARKET_TICKERS, START_DATE, get_end_date())
    df_fred = fetch_fred_data(FRED_SERIES)
    df_merged = merge_all(df_market, df_fred)
    df_scored = compute_stress_score(df_merged)
    score_min = df_scored["stress_score"].min()
    score_max = df_scored["stress_score"].max()

    print(df_scored[["stress_score"]].dropna().tail(10))
    print(f"\nNaN count:\n{df_scored.isna().sum()}")
    print(f"\nScore range: {score_min:.2f} - {score_max:.2f}")

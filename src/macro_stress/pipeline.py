from pathlib import Path
from loguru import logger
import pandas as pd

from macro_stress.fetch_data import fetch_market_data, fetch_fred_data
from macro_stress.process_data import merge_all
from macro_stress.features import compute_stress_score

MARKET_TICKERS = [
    "SPY",  # S&P 500 (overlay only)
    "^VIX",  # VIX
    "^VIX3M",  # VIX 3-month (for ratio)
    "^SKEW",  # SKEW index
    "HYG",  # High yield bonds (for ratio)
    "LQD",  # Investment grade bonds (for ratio)
    "GLD",  # Gold (for ratio)
    "XLK",  # Tech ETF (for ratio)
    "XLV",  # Healthcare ETF (for ratio)
    "DX-Y.NYB",  # DXY dollar index
    "USDCNY=X",  # USD/CNY
]

FRED_SERIES = [
    "T10Y2Y",  # yield curve spread (10Y-2Y)
    "ICSA",  # initial jobless claims
    "CPIAUCSL",  # CPI
    "DRCCLACBS",  # credit card delinquency rate
    "USSLIND",  # leading economic index
]

OUTPUT_PATH = Path("data/processed/stress_score.parquet")


def run(start: str, end: str) -> pd.DataFrame:
    logger.info("Fetching market data")
    df_market = fetch_market_data(MARKET_TICKERS, start, end)

    logger.info("Fetching FRED data")
    df_fred = fetch_fred_data(FRED_SERIES)

    logger.info("Merging and aligning data")
    df_merged = merge_all(df_market, df_fred)

    logger.info("Computing stress score")
    df_scored = compute_stress_score(df_merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_parquet(OUTPUT_PATH)
    logger.info(f"Saved to {OUTPUT_PATH}")

    return df_scored


if __name__ == "__main__":
    run(start="2020-01-01", end="2027-01-01")

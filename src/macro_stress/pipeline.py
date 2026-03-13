from pathlib import Path

import pandas as pd
from loguru import logger

from macro_stress.features import compute_stress_score
from macro_stress.fetch_data import fetch_fred_data, fetch_market_data
from macro_stress.process_data import merge_all

# START_DATE chosen based on availability of Index/Ticker data
START_DATE = "2008-01-01"


def get_end_date() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


MARKET_TICKERS = [
    "SPY",  # S&P 500 (overlay only)
    "^VIX",  # VIX
    "^VIX3M",  # VIX 3-month (for ratio)
    "^SKEW",  # SKEW index
    "GLD",  # Gold (for ratio)
    "XLK",  # Tech ETF (for ratio)
    "XLV",  # Healthcare ETF (for ratio)
    "DX=F",  # DXY dollar index futures, more reliable than DX-Y.NYB
    "USDCNY=X",  # USD/CNY
]

FRED_SERIES = [
    "T10Y2Y",  # yield curve spread (10Y-2Y)
    "ICSA",  # initial jobless claims
    "CPIAUCSL",  # CPI
    "DRCCLACBS",  # credit card delinquency rate
    "USALOLITOAASTSAM",  # Leading Indicators OECD
    "BAMLH0A0HYM2",  # ICE BofA US High Yield Index Option-Adjusted Spread
]

RAW_PATH = Path("data/raw")

OUTPUT_PATH = Path("data/processed/stress_score.parquet")


def run(start: str = START_DATE, end: str = None) -> pd.DataFrame:
    end = end or get_end_date()

    logger.info("Fetching market data")
    df_market = fetch_market_data(MARKET_TICKERS, start, end)

    logger.info("Fetching FRED data")
    df_fred = fetch_fred_data(FRED_SERIES, start, end)

    RAW_PATH.mkdir(parents=True, exist_ok=True)
    df_market.to_csv(RAW_PATH / "market_raw.csv")
    df_fred.to_csv(RAW_PATH / "fred_raw.csv")
    logger.info("Saved raw data to data/raw/")

    logger.info("Merging and aligning data")
    df_merged = merge_all(df_market, df_fred)

    logger.info("Computing stress score")
    df_scored = compute_stress_score(df_merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_parquet(OUTPUT_PATH)
    logger.info(f"Saved to {OUTPUT_PATH}")

    return df_scored


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    run()

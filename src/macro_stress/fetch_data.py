import os

import pandas as pd
import yfinance as yf
from fredapi import Fred


def fetch_market_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    return yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)[
        "Close"
    ]


def fetch_fred_data(series_ids: list[str]) -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set in environment")

    fred = Fred(api_key=api_key)

    return pd.DataFrame({sid: fred.get_series(sid) for sid in series_ids})


if __name__ == "__main__":
    from dotenv import load_dotenv

    from macro_stress.pipeline import (
        FRED_SERIES,
        MARKET_TICKERS,
        START_DATE,
        get_end_date,
    )

    load_dotenv()

    df_market = fetch_market_data(MARKET_TICKERS, START_DATE, get_end_date())
    df_fred = fetch_fred_data(FRED_SERIES)

    print(df_market.head(10))
    print(df_fred.head(10))

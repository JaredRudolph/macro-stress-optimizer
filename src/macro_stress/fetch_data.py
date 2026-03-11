import os
import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv


load_dotenv()


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

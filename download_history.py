#!/usr/bin/env python3
"""
Download 10 years of daily close data for all industry ETF tickers via yfinance.
Saves to data/industry_etf_daily.parquet (date × ticker close prices).
"""

import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE       = Path(__file__).parent
TICKER_CSV = HERE / "Industry_ETF_Tickers.csv"
OUT_FILE   = HERE / "data" / "industry_etf_daily.parquet"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

END_DATE   = date.today().strftime("%Y-%m-%d")
START_DATE = (date.today() - timedelta(days=365 * 10 + 5)).strftime("%Y-%m-%d")


def load_tickers() -> list[str]:
    raw = TICKER_CSV.read_text(encoding="utf-8-sig").strip().splitlines()
    return [t.strip() for t in raw if t.strip()]


def download(tickers: list[str]) -> pd.DataFrame:
    print(f"Downloading {len(tickers)} tickers  {START_DATE} -> {END_DATE} ...")
    raw = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
        threads=True,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = close.index.tz_localize(None)
    close.index = close.index.strftime("%Y-%m-%d")
    close.index.name = "date"
    close.sort_index(inplace=True)
    return close


def main() -> None:
    tickers = load_tickers()
    close   = download(tickers)

    n_tickers = close.notna().any().sum()
    n_days    = len(close)
    print(f"\n  {n_tickers} tickers returned data  |  {n_days:,} trading days")
    print(f"  Date range: {close.index[0]} -> {close.index[-1]}")

    # Coverage report
    coverage = close.notna().sum()
    short    = coverage[coverage < n_days * 0.5]
    if not short.empty:
        print(f"\n  Tickers with <50% coverage ({len(short)}):")
        for t, c in short.items():
            pct = c / n_days * 100
            first = close[t].first_valid_index()
            print(f"    {t:<8} {c:>4}/{n_days} days ({pct:.0f}%)  first={first}")

    close.to_parquet(OUT_FILE)
    print(f"\nSaved -> {OUT_FILE}  ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

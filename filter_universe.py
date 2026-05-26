#!/usr/bin/env python3
"""
Deduplicate the ETF universe: keep only the highest-AUM representative
per correlated cluster.

Algorithm
---------
1. Load trailing 252-day returns from parquet.
2. Fetch AUM (totalAssets) for each ticker via yfinance.
3. Sort tickers by AUM descending (largest = most liquid/representative).
4. Greedy inclusion: add each ticker only if its max absolute pairwise
   correlation with already-selected tickers is below CORR_THRESH.
5. Write result to Industry_ETF_Tickers_filtered.csv.

Run once (or whenever the universe needs refreshing), then commit the output.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE        = Path(__file__).parent
TICKER_CSV  = HERE / "Industry_ETF_Tickers.csv"
OUT_CSV     = HERE / "Industry_ETF_Tickers_filtered.csv"
PARQUET     = HERE / "data" / "industry_etf_daily.parquet"
CORR_WIN    = 252   # trading days used for correlation
CORR_THRESH = 0.85  # drop if max corr with any selected ticker >= this


def fetch_aum(tickers: list[str]) -> dict[str, float]:
    print(f"Fetching AUM for {len(tickers)} tickers ...")
    aum: dict[str, float] = {}
    batch = yf.Tickers(" ".join(tickers))
    for t in tickers:
        try:
            aum[t] = float(batch.tickers[t].info.get("totalAssets") or 0)
        except Exception:
            aum[t] = 0.0
    return aum


def main() -> None:
    tickers = [
        t.strip()
        for t in TICKER_CSV.read_text(encoding="utf-8-sig").strip().splitlines()
        if t.strip()
    ]
    print(f"Full universe: {len(tickers)} tickers")

    # Load prices and compute trailing correlation
    close = pd.read_parquet(PARQUET)
    close = close[[t for t in tickers if t in close.columns]]
    ret   = close.pct_change().iloc[-CORR_WIN:]

    # Only include tickers with >= 80% data coverage in the window
    min_obs = int(CORR_WIN * 0.80)
    valid   = ret.columns[ret.notna().sum() >= min_obs].tolist()
    sparse  = [t for t in tickers if t in close.columns and t not in valid]
    missing = [t for t in tickers if t not in close.columns]

    corr = ret[valid].corr()
    print(f"Correlation matrix: {len(valid)} tickers  "
          f"({len(sparse)} sparse, {len(missing)} not in parquet)")

    # Fetch AUM and rank
    aum = fetch_aum(valid)
    ranked = sorted(valid, key=lambda t: aum.get(t, 0), reverse=True)

    # Greedy selection
    selected: list[str] = []
    dropped:  list[tuple] = []

    for t in ranked:
        if not selected:
            selected.append(t)
            continue
        # max |corr| with any already-selected ticker
        corrs = [abs(corr.at[t, s]) for s in selected if t in corr.index and s in corr.columns]
        max_c = max(corrs) if corrs else 0.0
        if max_c < CORR_THRESH:
            selected.append(t)
        else:
            best_match = max(selected, key=lambda s: abs(corr.at[t, s]) if s in corr.columns else 0.0)
            dropped.append((t, best_match, max_c, aum.get(t, 0)))

    # Print dropped tickers grouped by the representative they duplicate
    print(f"\nDropped {len(dropped)} redundant tickers "
          f"(corr ≥ {CORR_THRESH} with a higher-AUM ETF):")
    by_rep: dict[str, list] = {}
    for t, rep, c, a in sorted(dropped, key=lambda x: x[1]):
        by_rep.setdefault(rep, []).append((t, c, a))
    for rep in sorted(by_rep):
        dupes = by_rep[rep]
        print(f"  {rep:<8} AUM=${aum.get(rep,0)/1e9:.1f}B  keeps out:")
        for t, c, a in sorted(dupes, key=lambda x: -x[1]):
            print(f"    {t:<8} corr={c:.2f}  AUM=${a/1e9:.1f}B")

    # Sparse / missing tickers: keep them (can't compute correlation)
    kept_extra = sorted(sparse) + sorted(missing)
    if kept_extra:
        print(f"\nKept {len(kept_extra)} tickers with insufficient history "
              f"(no correlation computed): {', '.join(kept_extra)}")

    final = sorted(selected) + kept_extra
    print(f"\nFiltered universe: {len(final)} tickers  (was {len(tickers)})")

    # AUM summary
    total_aum = sum(aum.get(t, 0) for t in selected) / 1e9
    print(f"Combined AUM of selected: ${total_aum:.1f}B")

    OUT_CSV.write_text("\n".join(final) + "\n", encoding="utf-8")
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()

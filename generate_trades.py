#!/usr/bin/env python3
"""
Generate an Interactive Brokers TWS order import CSV for the current
ERC Momentum + ERC Reversal (VIX>20) strategy positions.

Usage:
    python generate_trades.py --notional 100000
    python generate_trades.py --notional 100000 --leg mom       # momentum only
    python generate_trades.py --notional 100000 --leg rev       # reversal only
    python generate_trades.py --notional 100000 --order-type MKT

Output:
    trades/ib_orders_YYYY-MM-DD.csv  — upload via TWS File > Import Orders

TWS import format columns:
    Action, Quantity, Symbol, SecType, Exchange, Currency, TimeInForce, OrderType[, LmtPrice]

Notes:
    - Strategy is dollar-neutral: --notional sets the one-side notional
      (e.g. 100000 = $100k long + $100k short for momentum)
    - Reversal is long-only; --notional sets the total long notional
    - Quantities are rounded to whole shares; small residuals (<0.5 shares) are dropped
    - Prices are fetched live from yfinance at run time
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sco
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE     = Path(__file__).parent
MOM_WIN  = 252
REV_LB   = 5
COV_WIN  = 252
VOL_WIN  = 252
VIX_GATE = 20


# ── ERC helpers ───────────────────────────────────────────────────────────────

def _cov(r, t):
    C = r[t].cov().values
    return C + np.eye(len(C)) * 1e-8

def opt_erc(r, t):
    n, C = len(t), _cov(r, t)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: 0.5*w@C@w - (1/n)*np.sum(np.log(np.maximum(w, 1e-12))),
        x0=w0, jac=lambda w: C@w - (1/n)/np.maximum(w, 1e-12),
        method="L-BFGS-B", bounds=[(1e-6, 1.0)]*n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w / w.sum()

def basket_erc(r, t, sign):
    try:    w = opt_erc(r, t)
    except: w = np.ones(len(t)) / len(t)
    return pd.Series(w * sign, index=t)


# ── Strategy runners (return final weights only) ──────────────────────────────

def current_mom_weights(close):
    mom_sig   = close.pct_change(MOM_WIN)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}
    weights   = pd.Series(0.0, index=close.columns)
    for i in range(1, len(dates)):
        prev = dates[i - 1]
        sig  = mom_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns)
            continue
        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index
        shorts = sig[ranked > len(sig) - n_q].index
        ret_win = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1): date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs]  = basket_erc(ret_win, longs,  +1.0)
        weights[shorts] = basket_erc(ret_win, shorts, -1.0)
    return weights


def current_rev_weights(close, vix):
    rev_sig   = close.pct_change(REV_LB)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}
    weights   = pd.Series(0.0, index=close.columns)
    vix_last  = float(vix.iloc[-1])
    rev_active = vix_last > VIX_GATE

    if not rev_active:
        print(f"  Reversal inactive (VIX={vix_last:.1f} ≤ {VIX_GATE}) — would-be basket shown")
        # compute would-be basket from latest signal regardless
    for i in range(1, len(dates)):
        prev   = dates[i - 1]
        vix_ok = vix.get(prev, 0) > VIX_GATE
        sig    = rev_sig.loc[prev].dropna()
        if not vix_ok or len(sig) < 8:
            if i < len(dates) - 1:   # keep weights flat for non-active days mid-history
                weights = pd.Series(0.0, index=close.columns)
            continue
        n_q    = max(1, len(sig) // 4)
        longs  = sig.nsmallest(n_q).index
        ret_win = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1): date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = basket_erc(ret_win, longs, +1.0)

    if not rev_active:
        # override: use the latest signal's would-be basket
        sig = rev_sig.iloc[-1].dropna()
        n_q = max(1, len(sig) // 4)
        longs = sig.nsmallest(n_q).index
        ret_win = daily_ret.iloc[-COV_WIN:]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = basket_erc(ret_win, longs, +1.0)

    return weights, rev_active


# ── Trade file builder ────────────────────────────────────────────────────────

def weights_to_orders(weights, prices, notional, order_type, lmt_buffer=0.002):
    """
    Convert fractional weights to IB order rows.
    weights: +ve = long, -ve = short (sum of abs = 1 per leg after ERC)
    notional: dollar amount per leg
    """
    rows = []
    for ticker, w in weights.items():
        if abs(w) < 1e-4:
            continue
        price = prices.get(ticker)
        if price is None or np.isnan(price) or price <= 0:
            print(f"  WARNING: no price for {ticker}, skipping")
            continue
        dollar_target = abs(w) * notional
        qty = int(round(dollar_target / price))
        if qty == 0:
            continue
        action = "BUY" if w > 0 else "SELL"
        row = {
            "Action":      action,
            "Quantity":    qty,
            "Symbol":      ticker,
            "SecType":     "STK",
            "Exchange":    "SMART",
            "Currency":    "USD",
            "TimeInForce": "DAY",
            "OrderType":   order_type,
        }
        if order_type == "LMT":
            # BUY: bid slightly above last; SELL: bid slightly below
            buf = 1 + lmt_buffer if action == "BUY" else 1 - lmt_buffer
            row["LmtPrice"] = round(price * buf, 2)
        rows.append(row)
    return rows


def fetch_live_prices(tickers):
    print(f"Fetching live prices for {len(tickers)} tickers ...")
    raw = yf.download(tickers, period="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[-1]
    else:
        close = raw["Close"]
    return close.to_dict()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate IB TWS order CSV")
    parser.add_argument("--notional",   type=float, default=10_000,
                        help="One-side notional in USD (default: 10000)")
    parser.add_argument("--leg",        choices=["both", "mom", "rev"], default="both",
                        help="Which legs to include (default: both)")
    parser.add_argument("--order-type", choices=["MKT", "MOC", "LMT"], default="MOC",
                        help="Order type (default: MOC)")
    args = parser.parse_args()

    today = date.today().strftime("%Y-%m-%d")
    print(f"=== IB Trade File Generator  {today} ===")
    print(f"    Notional: ${args.notional:,.0f} per leg")
    print(f"    Legs:     {args.leg}")
    print(f"    Order:    {args.order_type}")

    # Load prices
    tf = HERE / "Industry_ETF_Tickers_liquid.csv"
    if not tf.exists():
        tf = HERE / "Industry_ETF_Tickers_filtered.csv"
    if not tf.exists():
        tf = HERE / "Industry_ETF_Tickers.csv"
    tickers = [t.strip() for t in tf.read_text(encoding="utf-8-sig").strip().splitlines() if t.strip()]
    close = pd.read_parquet(HERE / "data" / "industry_etf_daily.parquet")
    close = close[[t for t in tickers if t in close.columns]].sort_index()
    close = close.where(close.gt(0)).ffill()

    # VIX
    vix_raw = yf.download("^VIX", start="2014-01-01", auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    vix_last = float(vix.iloc[-1])
    print(f"    VIX:      {vix_last:.1f}")

    # Compute weights
    all_orders = []

    if args.leg in ("both", "mom"):
        print("\nComputing momentum weights ...")
        mom_w = current_mom_weights(close)
        longs  = mom_w[mom_w > 0.001].sort_values(ascending=False)
        shorts = mom_w[mom_w < -0.001].sort_values()
        print(f"  Momentum longs  ({len(longs)}): {', '.join(longs.index)}")
        print(f"  Momentum shorts ({len(shorts)}): {', '.join(shorts.index)}")
        live_px = fetch_live_prices(list(mom_w[mom_w.abs() > 0.001].index))
        all_orders += weights_to_orders(mom_w, live_px, args.notional, args.order_type)

    if args.leg in ("both", "rev"):
        print("\nComputing reversal weights ...")
        rev_w, rev_active = current_rev_weights(close, vix)
        longs = rev_w[rev_w > 0.001].sort_values(ascending=False)
        status = "ACTIVE" if rev_active else "INACTIVE (would-be)"
        print(f"  Reversal {status} longs ({len(longs)}): {', '.join(longs.index)}")
        if not rev_active:
            print(f"  *** VIX={vix_last:.1f} ≤ {VIX_GATE} — reversal is currently OFF ***")
            print(f"  *** Orders shown for information only — do NOT trade reversal leg ***")
        if rev_active:
            live_px = fetch_live_prices(list(rev_w[rev_w > 0.001].index))
            all_orders += weights_to_orders(rev_w, live_px, args.notional, args.order_type)

    if not all_orders:
        print("\nNo orders generated.")
        return

    df = pd.DataFrame(all_orders)

    # Net same-ticker orders across legs (signed qty: BUY=+, SELL=-)
    if len(df["Symbol"].unique()) < len(df):
        df["_signed_qty"] = df.apply(lambda r: r["Quantity"] if r["Action"] == "BUY" else -r["Quantity"], axis=1)
        lmt_prices = df.groupby("Symbol")["LmtPrice"].last() if "LmtPrice" in df.columns else None
        netted = (df.groupby("Symbol", sort=False)
                    .agg(net_qty=("_signed_qty", "sum"),
                         SecType=("SecType", "first"),
                         Exchange=("Exchange", "first"),
                         Currency=("Currency", "first"),
                         TimeInForce=("TimeInForce", "first"),
                         OrderType=("OrderType", "first"))
                    .reset_index())
        netted = netted[netted["net_qty"] != 0].copy()
        netted["Action"]   = netted["net_qty"].apply(lambda q: "BUY" if q > 0 else "SELL")
        netted["Quantity"] = netted["net_qty"].abs()
        cols = ["Action", "Quantity", "Symbol", "SecType", "Exchange", "Currency", "TimeInForce", "OrderType"]
        if lmt_prices is not None:
            netted["LmtPrice"] = netted.apply(
                lambda r: lmt_prices.get(r["Symbol"], np.nan), axis=1)
            cols.append("LmtPrice")
        df = netted[cols]

    # Summary
    buys  = df[df["Action"] == "BUY"]
    sells = df[df["Action"] == "SELL"]
    print(f"\n{'─'*55}")
    print(f"  Orders: {len(buys)} BUY  |  {len(sells)} SELL  |  {len(df)} total")
    print(f"\n{df.to_string(index=False)}")

    # Save
    out_dir = HERE / "trades"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"ib_orders_{today}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved -> {out_file}")
    print(f"\nTo import in TWS: File > Import Orders > select the CSV file")


if __name__ == "__main__":
    main()

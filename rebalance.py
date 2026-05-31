#!/usr/bin/env python3
"""
Daily rebalance via IB TWS API.

Connects to TWS, fetches current positions for model tickers, computes
today's target from the strategy, shows the delta order blotter, and
transmits MOC orders on confirmation.

Requires:
    pip install ib_insync

Usage:
    python rebalance.py                    # $10k/side, preview + confirm
    python rebalance.py --dry-run          # preview only, never transmits
    python rebalance.py --notional 20000
    python rebalance.py --leg mom          # momentum leg only
    python rebalance.py --port 7497        # paper trading
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
VIX_GATE = 20
TWS_HOST = "127.0.0.1"


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


# ── Strategy weight computation ───────────────────────────────────────────────

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
    vix_last   = float(vix.iloc[-1])
    rev_active = vix_last > VIX_GATE
    if not rev_active:
        return pd.Series(0.0, index=close.columns), False

    rev_sig   = close.pct_change(REV_LB)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}
    weights   = pd.Series(0.0, index=close.columns)

    for i in range(1, len(dates)):
        prev   = dates[i - 1]
        vix_ok = vix.get(prev, 0) > VIX_GATE
        sig    = rev_sig.loc[prev].dropna()
        if not vix_ok or len(sig) < 8:
            if i < len(dates) - 1:
                weights = pd.Series(0.0, index=close.columns)
            continue
        n_q    = max(1, len(sig) // 4)
        longs  = sig.nsmallest(n_q).index
        ret_win = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1): date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = basket_erc(ret_win, longs, +1.0)

    return weights, True


# ── Position helpers ──────────────────────────────────────────────────────────

def weights_to_signed_shares(weights, prices, notional):
    """Convert fractional weights to signed share counts (+long, -short)."""
    shares = {}
    for ticker, w in weights.items():
        if abs(w) < 1e-4:
            continue
        price = prices.get(ticker)
        if price is None or np.isnan(price) or price <= 0:
            print(f"  WARNING: no price for {ticker}, skipping")
            continue
        qty = int(round(abs(w) * notional / price))
        if qty == 0:
            continue
        shares[ticker] = qty if w > 0 else -qty
    return shares


def fetch_live_prices(tickers):
    print(f"  Fetching live prices for {len(tickers)} tickers ...")
    raw = yf.download(tickers, period="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[-1]
    else:
        close = raw["Close"]
    return close.to_dict()


def fetch_ib_positions(ib, model_tickers):
    """Return {symbol: signed_qty} for model tickers only."""
    positions = {}
    for item in ib.portfolio():
        sym = item.contract.symbol
        if sym in model_tickers:
            qty = int(item.position)
            if qty != 0:
                positions[sym] = qty
    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IB daily rebalance")
    parser.add_argument("--notional", type=float, default=10_000,
                        help="One-side notional in USD (default: 10000)")
    parser.add_argument("--leg",      choices=["both", "mom", "rev"], default="both")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview orders only, do not transmit")
    parser.add_argument("--port",     type=int, default=7496,
                        help="TWS API port (default: 7496 live, 7497 paper)")
    args = parser.parse_args()

    today = date.today().strftime("%Y-%m-%d")
    print(f"=== IB Rebalance  {today} ===")
    print(f"    Notional : ${args.notional:,.0f}/side")
    print(f"    Legs     : {args.leg}")
    print(f"    Port     : {args.port}")
    if args.dry_run:
        print("    Mode     : DRY RUN (no orders will be transmitted)")

    # ── Load universe & history ───────────────────────────────────────────────
    tf = HERE / "Industry_ETF_Tickers_liquid.csv"
    if not tf.exists():
        tf = HERE / "Industry_ETF_Tickers_filtered.csv"
    if not tf.exists():
        tf = HERE / "Industry_ETF_Tickers.csv"
    model_tickers = set(
        t.strip() for t in tf.read_text(encoding="utf-8-sig").strip().splitlines() if t.strip()
    )

    close = pd.read_parquet(HERE / "data" / "industry_etf_daily.parquet")
    close = close[[t for t in model_tickers if t in close.columns]].sort_index()
    close = close.where(close.gt(0)).ffill()

    # ── VIX ──────────────────────────────────────────────────────────────────
    print("\nFetching VIX ...")
    vix_raw = yf.download("^VIX", start="2014-01-01", auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    vix_last = float(vix.iloc[-1])
    print(f"  VIX: {vix_last:.1f}")

    # ── Target shares (per leg, then net) ─────────────────────────────────────
    target_shares: dict[str, int] = {}

    if args.leg in ("both", "mom"):
        print("\nComputing momentum weights ...")
        mom_w  = current_mom_weights(close)
        longs  = mom_w[mom_w > 0.001].sort_values(ascending=False)
        shorts = mom_w[mom_w < -0.001].sort_values()
        print(f"  Longs  ({len(longs)}): {', '.join(longs.index)}")
        print(f"  Shorts ({len(shorts)}): {', '.join(shorts.index)}")
        live_px = fetch_live_prices(list(mom_w[mom_w.abs() > 0.001].index))
        for sym, qty in weights_to_signed_shares(mom_w, live_px, args.notional).items():
            target_shares[sym] = target_shares.get(sym, 0) + qty

    if args.leg in ("both", "rev"):
        print("\nComputing reversal weights ...")
        rev_w, rev_active = current_rev_weights(close, vix)
        if rev_active:
            longs = rev_w[rev_w > 0.001].sort_values(ascending=False)
            print(f"  Reversal ACTIVE — longs ({len(longs)}): {', '.join(longs.index)}")
            live_px = fetch_live_prices(list(rev_w[rev_w > 0.001].index))
            for sym, qty in weights_to_signed_shares(rev_w, live_px, args.notional).items():
                target_shares[sym] = target_shares.get(sym, 0) + qty
        else:
            print(f"  Reversal INACTIVE (VIX={vix_last:.1f} ≤ {VIX_GATE}) — skipped")

    # Drop any positions netted to zero
    target_shares = {s: q for s, q in target_shares.items() if q != 0}

    # ── Connect to IB ─────────────────────────────────────────────────────────
    try:
        from ib_insync import IB, Stock, Order
    except ImportError:
        print("\nERROR: ib_insync not installed. Run: pip install ib_insync")
        sys.exit(1)

    print(f"\nConnecting to TWS at {TWS_HOST}:{args.port} ...")
    ib = IB()
    try:
        ib.connect(TWS_HOST, args.port, clientId=1, readonly=args.dry_run)
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Make sure TWS is running and API connections are enabled.")
        print("  TWS: Edit > Global Configuration > API > Settings > Enable ActiveX and Socket Clients")
        sys.exit(1)
    print("  Connected.")

    # ── Current IB positions (model tickers only) ─────────────────────────────
    current_shares = fetch_ib_positions(ib, model_tickers)
    print(f"\n  Current model positions: {len(current_shares)} tickers")
    for sym, qty in sorted(current_shares.items()):
        print(f"    {sym:<8} {qty:>+6d} shares")

    # ── Delta ─────────────────────────────────────────────────────────────────
    all_syms = sorted(set(target_shares) | set(current_shares))
    delta_orders = []
    for sym in all_syms:
        current = current_shares.get(sym, 0)
        target  = target_shares.get(sym, 0)
        delta   = target - current
        if delta != 0:
            delta_orders.append((sym, current, target, delta))

    if not delta_orders:
        print("\nPortfolio already at target — no orders needed.")
        ib.disconnect()
        return

    # ── Preview blotter ───────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  {'Symbol':<8} {'Current':>8} {'Target':>8} {'Delta':>8}  Order")
    print(f"{'─'*62}")
    for sym, cur, tgt, delta in delta_orders:
        order_str = f"{'BUY' if delta > 0 else 'SELL'} {abs(delta)} MOC"
        print(f"  {sym:<8} {cur:>8} {tgt:>8} {delta:>+8}  {order_str}")
    print(f"{'─'*62}")
    buys  = sum(1 for *_, d in delta_orders if d > 0)
    sells = sum(1 for *_, d in delta_orders if d < 0)
    print(f"  {buys} BUY  |  {sells} SELL  |  {len(delta_orders)} total")

    if args.dry_run:
        print("\n[dry-run] No orders transmitted.")
        ib.disconnect()
        return

    # ── Confirm ───────────────────────────────────────────────────────────────
    print()
    try:
        answer = input("Transmit these orders? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer != "y":
        print("Aborted — no orders sent.")
        ib.disconnect()
        return

    # ── Submit ────────────────────────────────────────────────────────────────
    print("\nSubmitting ...")
    for sym, cur, tgt, delta in delta_orders:
        contract = Stock(sym, "SMART", "USD")
        order = Order(
            action="BUY" if delta > 0 else "SELL",
            totalQuantity=abs(delta),
            orderType="MOC",
            tif="DAY",
        )
        ib.placeOrder(contract, order)
        print(f"  {'BUY' if delta > 0 else 'SELL':4s} {abs(delta):4d} {sym}")

    ib.sleep(1)
    print(f"\n{len(delta_orders)} orders submitted.")
    ib.disconnect()


if __name__ == "__main__":
    main()

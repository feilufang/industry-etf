#!/usr/bin/env python3
"""
Reversal strategy performance breakdown by VIX bucket.
Runs reversal ungated (every day), then groups daily P&L by prev-day VIX level.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE     = Path(__file__).parent
REV_LB   = 5
COV_WIN  = 252

BUCKETS = [0, 15, 20, 25, 30, 40, 999]
LABELS  = ["<15", "15–20", "20–25", "25–30", "30–40", "40+"]


# ── ERC helpers ───────────────────────────────────────────────────────────────

def _cov(ret_win, tickers):
    C = ret_win[tickers].cov().values
    return C + np.eye(len(C)) * 1e-8

def opt_erc(ret_win, tickers):
    n, C = len(tickers), _cov(ret_win, tickers)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: 0.5 * w @ C @ w - (1/n) * np.sum(np.log(np.maximum(w, 1e-12))),
        x0=w0, jac=lambda w: C @ w - (1/n) / np.maximum(w, 1e-12),
        method="L-BFGS-B", bounds=[(1e-6, 1.0)] * n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w / w.sum()


# ── Ungated reversal loop ─────────────────────────────────────────────────────

def run_reversal_ungated(close):
    """Run reversal every day regardless of VIX; return daily P&L series."""
    rev_sig   = close.pct_change(REV_LB)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}

    weights = pd.Series(0.0, index=close.columns)
    pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))

        sig = rev_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns)
            continue
        n_q   = max(1, len(sig) // 4)
        longs = sig.nsmallest(n_q).index
        ret_win = daily_ret.iloc[max(0, date_pos[prev] - COV_WIN + 1): date_pos[prev] + 1]
        try:
            w = opt_erc(ret_win, longs)
        except Exception:
            w = np.ones(len(longs)) / len(longs)
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = w

    return pd.Series(dict(pnl))


# ── Stats helper ──────────────────────────────────────────────────────────────

def bucket_stats(pnl: pd.Series, label: str) -> dict:
    n    = len(pnl)
    mean = pnl.mean() * 252
    vol  = pnl.std() * np.sqrt(252)
    sharpe = mean / vol if vol > 0 else np.nan
    win  = (pnl > 0).mean()
    cum  = (1 + pnl).prod() - 1
    return dict(label=label, n_days=n, ann_ret=mean, ann_vol=vol,
                sharpe=sharpe, win_rate=win, total_ret=cum)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load tickers
    tf = HERE / "Industry_ETF_Tickers_filtered.csv"
    if not tf.exists():
        tf = HERE / "Industry_ETF_Tickers.csv"
    tickers = [t.strip() for t in tf.read_text(encoding="utf-8-sig").strip().splitlines() if t.strip()]
    print(f"Universe: {len(tickers)} tickers ({tf.name})")

    # Load prices
    close = pd.read_parquet(HERE / "data" / "industry_etf_daily.parquet")
    close = close[[t for t in tickers if t in close.columns]].sort_index()
    close = close.where(close.gt(0)).ffill()
    ind_first = close.index[COV_WIN]

    # Fetch VIX
    print("Fetching VIX ...")
    vix_raw = yf.download("^VIX", start="2014-01-01", auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")

    # Run ungated reversal
    print("Running ungated reversal ...")
    pnl = run_reversal_ungated(close)
    pnl = pnl[pnl.index >= ind_first]

    # Align VIX to pnl dates (prev-day VIX = signal used to trade on date d)
    dates = pd.Series(pnl.index)
    # For each date d, the VIX that drove the signal was the previous trading day
    all_dates = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(all_dates)}
    prev_vix  = {}
    for d in pnl.index:
        pos = date_pos.get(d)
        if pos and pos > 0:
            prev_vix[d] = vix.get(all_dates[pos - 1], np.nan)
    vix_aligned = pd.Series(prev_vix).reindex(pnl.index)

    # ── Bucket breakdown ──────────────────────────────────────────────────────
    rows = []
    pnl_by_bucket = {}
    for lo, hi, lbl in zip(BUCKETS[:-1], BUCKETS[1:], LABELS):
        mask = (vix_aligned >= lo) & (vix_aligned < hi)
        sub  = pnl[mask]
        pnl_by_bucket[lbl] = sub
        rows.append(bucket_stats(sub, lbl))

    # Overall
    rows.append(bucket_stats(pnl, "All days"))

    df = pd.DataFrame(rows).set_index("label")

    print(f"\n{'VIX bucket':<10} {'Days':>6} {'%days':>6} "
          f"{'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>7} {'WinRate':>8} {'Total':>8}")
    print("-" * 70)
    total_days = len(pnl)
    for lbl, row in df.iterrows():
        pct_days = row['n_days'] / total_days * 100
        print(f"  {lbl:<10} {int(row['n_days']):>5}  {pct_days:>5.1f}%"
              f"  {row['ann_ret']:>+7.1%}  {row['ann_vol']:>7.1%}"
              f"  {row['sharpe']:>6.3f}  {row['win_rate']:>7.1%}"
              f"  {row['total_ret']:>+7.1%}")

    # ── Chart: mean daily return by VIX bucket ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#ffffff")

    bucket_labels = LABELS
    sharpes  = [df.loc[l, "sharpe"]  for l in bucket_labels]
    ann_rets = [df.loc[l, "ann_ret"] for l in bucket_labels]
    colors   = ["#e74c3c" if s < 0 else "#2ca02c" for s in sharpes]

    ax = axes[0]
    bars = ax.bar(bucket_labels, sharpes, color=colors, edgecolor="#fff", width=0.6)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("Reversal Sharpe by VIX Bucket", fontsize=11)
    ax.set_xlabel("Prev-day VIX")
    ax.set_ylabel("Sharpe ratio (annualised)")
    ax.set_facecolor("#fafafa")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    ret_colors = ["#e74c3c" if r < 0 else "#1f77b4" for r in ann_rets]
    bars2 = ax2.bar(bucket_labels, [r * 100 for r in ann_rets],
                    color=ret_colors, edgecolor="#fff", width=0.6)
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_title("Reversal Ann Return by VIX Bucket", fontsize=11)
    ax2.set_xlabel("Prev-day VIX")
    ax2.set_ylabel("Annualised return (%)")
    ax2.set_facecolor("#fafafa")
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars2, ann_rets):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.3 if v >= 0 else -1.5),
                 f"{v:+.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(pad=1.5)
    out = HERE / "results_combined" / "reversal_vix_buckets.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Long-only reversal: buy bottom N names by 5d return.
N in [1, 3, 5, 10, 15] — equal weight within basket, daily rebalance, T+1 execution.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE       = Path(__file__).parent
DAILY_FILE = HERE / "data" / "industry_etf_daily.parquet"
TICKER_CSV = HERE / "Industry_ETF_Tickers.csv"
OUT_DIR    = HERE / "results_reversal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END  = "2025-12-31"
MOM_WIN = 252
REV_LB  = 5
N_LIST  = [1, 3, 5, 10, 15]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_tickers():
    raw = TICKER_CSV.read_text(encoding="utf-8-sig").strip().splitlines()
    return [t.strip() for t in raw if t.strip()]


def load_prices(tickers):
    close = pd.read_parquet(DAILY_FILE)
    avail = [t for t in tickers if t in close.columns]
    close = close[avail][close.index <= IS_END].sort_index()
    return close.where(close.gt(0)).ffill()


def run_rev_topn(close: pd.DataFrame, n: int) -> pd.Series:
    """Long bottom-n ETFs by REV_LB-day return. Equal weight, daily rebal, T+1."""
    sig_all   = close.pct_change(REV_LB)
    dates     = close.index.tolist()
    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        sig = sig_all.loc[prev].dropna()
        if len(sig) < n:
            weights = pd.Series(0.0, index=close.columns)
            continue

        # Bottom n by return = worst recent performers
        longs = sig.nsmallest(n).index
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = 1.0 / n

    return pd.Series(dict(daily_pnl), name=f"bot{n}")


def compute_stats(pnl: pd.Series, label: str = "") -> dict:
    cum    = (1 + pnl).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    ann    = pnl.mean() * 252
    vol    = pnl.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    return {
        "label":    label,
        "ann_ret":  round(float(ann), 4),
        "ann_vol":  round(float(vol), 4),
        "sharpe":   round(float(sharpe), 3),
        "max_dd":   round(float(dd.min()), 4),
        "calmar":   round(float(calmar), 3),
        "win_rate": round(float((pnl > 0).mean()), 3),
    }


def monthly_returns(pnl: pd.Series) -> pd.DataFrame:
    idx   = pd.to_datetime(pnl.index)
    m_ret = pd.Series((1 + pnl.values), index=idx).resample("ME").prod() - 1
    m_ret.index = m_ret.index.to_period("M")
    tbl   = m_ret.rename("ret").to_frame()
    tbl["year"]  = tbl.index.year
    tbl["month"] = tbl.index.month
    pivot = tbl.pivot(index="year", columns="month", values="ret")
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
    pivot["Ann"] = tbl.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
    return pivot


def plot_cumret(results: dict, first_sig: str) -> None:
    cmap   = plt.cm.plasma
    n_vals = len(results)
    colors = [cmap(i / (n_vals - 1)) for i in range(n_vals)]

    fig, ax = plt.subplots(figsize=(13, 5))
    for (label, pnl), color in zip(results.items(), colors):
        pnl  = pnl[pnl.index >= first_sig]
        cum  = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                linewidth=1.6, label=label, color=color)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        f"Long-only Reversal  |  Bottom N by {REV_LB}d Return  |  Equal Weight  |  IS: 2017–2025\n"
        "Daily rebalance, T+1 execution",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "cumret_rev_topn.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_heatmap(pnl: pd.Series, label: str, fname: str) -> None:
    pivot      = monthly_returns(pnl)
    month_cols = [c for c in pivot.columns if c != "Ann"]
    fig, axes  = plt.subplots(1, 2, figsize=(16, max(3, len(pivot) * 0.55 + 1.5)),
                               gridspec_kw={"width_ratios": [len(month_cols), 1]})

    def _hm(ax, data, title):
        vals   = data.values.astype(float)
        finite = vals[~np.isnan(vals)]
        vmax   = max(abs(finite).max(), 0.01) if len(finite) else 0.01
        norm   = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ax.imshow(vals, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, fontsize=9)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        for (r, c), v in np.ndenumerate(vals):
            if not np.isnan(v):
                ax.text(c, r, f"{v*100:.1f}", ha="center", va="center",
                        fontsize=7.5, color="black")

    _hm(axes[0], pivot[month_cols], "Monthly Returns (%)")
    _hm(axes[1], pivot[["Ann"]],    "Annual (%)")
    fig.suptitle(f"Reversal Long-only — {label}  |  IS: 2017–2025", fontsize=11, y=1.01)
    fig.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def main():
    tickers = load_tickers()
    close   = load_prices(tickers)
    first_sig = close.index[MOM_WIN]

    print(f"Universe : {len(close.columns)} tickers")
    print(f"Window   : {first_sig} -> {close.index[-1]}\n")

    results = {}
    rows    = []

    print(f"{'N':>4}  {'Label':<18} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'WinRate':>7}")
    print("-" * 72)

    for n in N_LIST:
        label = f"Bottom {n}"
        pnl   = run_rev_topn(close, n)
        pnl   = pnl[pnl.index >= first_sig]
        results[label] = pnl
        s = compute_stats(pnl, label)
        rows.append(s)
        print(f"  {n:>2}  {label:<18} {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  "
              f"{s['calmar']:>7.3f}  {s['win_rate']:>7.1%}")

    print()
    for label, pnl in results.items():
        print(f"\n--- {label} Monthly Returns ---")
        print((monthly_returns(pnl) * 100).round(1).to_string())

    pd.DataFrame(rows).to_csv(OUT_DIR / "stats_rev_topn.csv", index=False)

    print("\nSaving charts ...")
    plot_cumret(results, first_sig)
    for label, pnl in results.items():
        fname = "monthly_rev_" + label.replace(" ", "_").lower() + ".png"
        plot_heatmap(pnl, label, fname)

    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

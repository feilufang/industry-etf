#!/usr/bin/env python3
"""
Long-only reversal: bottom quartile by 5d return, gated by VIX > 20.
Compares: always-on vs VIX > 20.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yfinance as yf

HERE       = Path(__file__).parent
DAILY_FILE = HERE / "data" / "industry_etf_daily.parquet"
TICKER_CSV = HERE / "Industry_ETF_Tickers.csv"
OUT_DIR    = HERE / "results_reversal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END    = "2025-12-31"
MOM_WIN   = 252
REV_LB    = 5
VIX_GATE  = 20
QUANTILE  = 2    # 2 = bottom half, 4 = bottom quartile
MIN_STOCKS = 8

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


def load_vix() -> pd.Series:
    raw = yf.download("^VIX", start="2015-01-01", end=IS_END,
                      auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    return vix


def run_rev_quartile(close: pd.DataFrame, vix: pd.Series,
                     vix_threshold=None) -> pd.Series:
    sig_all = close.pct_change(REV_LB)
    dates   = close.index.tolist()

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        if vix_threshold is not None:
            prev_vix = vix.get(prev, np.nan)
            if np.isnan(prev_vix) or prev_vix <= vix_threshold:
                weights = pd.Series(0.0, index=close.columns)
                continue

        sig = sig_all.loc[prev].dropna()
        if len(sig) < MIN_STOCKS:
            weights = pd.Series(0.0, index=close.columns)
            continue

        n_q    = max(1, len(sig) // QUANTILE)
        longs  = sig.nsmallest(n_q).index
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = 1.0 / len(longs)

    return pd.Series(dict(daily_pnl))


def compute_stats(pnl: pd.Series, label: str = "") -> dict:
    cum    = (1 + pnl).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    ann    = pnl.mean() * 252
    vol    = pnl.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    return {
        "label":        label,
        "ann_ret":      round(float(ann), 4),
        "ann_vol":      round(float(vol), 4),
        "sharpe":       round(float(sharpe), 3),
        "max_dd":       round(float(dd.min()), 4),
        "calmar":       round(float(calmar), 3),
        "win_rate":     round(float((pnl > 0).mean()), 3),
        "pct_invested": round(float((pnl != 0).mean()), 3),
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


def plot_cumret(pnl_always, pnl_gated, first_sig):
    fig, ax = plt.subplots(figsize=(13, 5))

    basket = "Bottom half" if QUANTILE == 2 else "Bottom quartile"
    for pnl, label, color, ls, lw in [
        (pnl_always, f"{basket}  —  Always-on",          "#7f7f7f", "--", 1.3),
        (pnl_gated,  f"{basket}  —  VIX > {VIX_GATE}",  "#d62728",  "-",  2.0),
    ]:
        pnl = pnl[pnl.index >= first_sig]
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                color=color, linestyle=ls, linewidth=lw, label=label)

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Cumulative Return (%)")
    basket = "Bottom Half" if QUANTILE == 2 else "Bottom Quartile"
    ax.set_title(
        f"Long-only Reversal  |  {basket} by {REV_LB}d Return  |  IS: 2017–2025\n"
        f"Always-on vs VIX > {VIX_GATE} gate  |  Equal weight, daily rebalance, T+1",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    bname = "half" if QUANTILE == 2 else "quartile"
    path = OUT_DIR / f"cumret_rev_{bname}_vix{VIX_GATE}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_heatmap(pnl: pd.Series, title: str, fname: str) -> None:
    pivot      = monthly_returns(pnl)
    month_cols = [c for c in pivot.columns if c != "Ann"]
    fig, axes  = plt.subplots(1, 2, figsize=(16, max(3, len(pivot) * 0.55 + 1.5)),
                               gridspec_kw={"width_ratios": [len(month_cols), 1]})

    def _hm(ax, data, hdr):
        vals   = data.values.astype(float)
        finite = vals[~np.isnan(vals)]
        vmax   = max(abs(finite).max(), 0.01) if len(finite) else 0.01
        norm   = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ax.imshow(vals, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, fontsize=9)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=9)
        ax.set_title(hdr, fontsize=10, pad=6)
        for (r, c), v in np.ndenumerate(vals):
            if not np.isnan(v):
                ax.text(c, r, f"{v*100:.1f}", ha="center", va="center",
                        fontsize=7.5, color="black")

    _hm(axes[0], pivot[month_cols], "Monthly Returns (%)")
    _hm(axes[1], pivot[["Ann"]],    "Annual (%)")
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def main():
    tickers = load_tickers()
    close   = load_prices(tickers)
    print("Fetching VIX ...")
    vix       = load_vix()
    first_sig = close.index[MOM_WIN]

    vix_w = vix[vix.index >= first_sig]
    pct_active = (vix_w > VIX_GATE).mean()
    print(f"VIX > {VIX_GATE}: {pct_active:.1%} of days  |  window: {first_sig} -> {close.index[-1]}\n")

    print("Running always-on ...")
    pnl_always = run_rev_quartile(close, vix, vix_threshold=None)
    pnl_always = pnl_always[pnl_always.index >= first_sig]

    print(f"Running VIX > {VIX_GATE} gated ...")
    pnl_gated  = run_rev_quartile(close, vix, vix_threshold=VIX_GATE)
    pnl_gated  = pnl_gated[pnl_gated.index >= first_sig]

    print(f"\n{'Strategy':<40} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'WinRate':>7} {'%Active':>8}")
    print("-" * 96)
    rows = []
    basket = "Bottom half" if QUANTILE == 2 else "Bottom quartile"
    for pnl, label in [
        (pnl_always, f"{basket}  Always-on"),
        (pnl_gated,  f"{basket}  VIX > {VIX_GATE}"),
    ]:
        s = compute_stats(pnl, label)
        rows.append(s)
        print(f"  {label:<38} {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  "
              f"{s['calmar']:>7.3f}  {s['win_rate']:>7.1%}  "
              f"{s['pct_invested']:>7.1%}")

    print(f"\n--- Always-on Monthly Returns ---")
    print((monthly_returns(pnl_always) * 100).round(1).to_string())
    print(f"\n--- VIX > {VIX_GATE} Monthly Returns ---")
    print((monthly_returns(pnl_gated) * 100).round(1).to_string())

    pd.DataFrame(rows).to_csv(OUT_DIR / f"stats_rev_quartile_vix{VIX_GATE}.csv", index=False)

    print("\nSaving charts ...")
    plot_cumret(pnl_always, pnl_gated, first_sig)
    bname  = "half" if QUANTILE == 2 else "quartile"
    blabel = "Half" if QUANTILE == 2 else "Quartile"
    plot_heatmap(
        pnl_always,
        f"Reversal Bottom {blabel}  |  Always-on  |  IS: 2017–2025",
        f"monthly_rev_{bname}_always.png",
    )
    plot_heatmap(
        pnl_gated,
        f"Reversal Bottom {blabel}  |  VIX > {VIX_GATE}  |  IS: 2017–2025",
        f"monthly_rev_{bname}_vix{VIX_GATE}.png",
    )
    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

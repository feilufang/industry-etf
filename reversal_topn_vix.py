#!/usr/bin/env python3
"""
Long-only reversal (bottom N by 5d return) gated by prior-day VIX level.
Thresholds: always-on (baseline), VIX > 15, VIX > 20.
N in [1, 3, 5, 10, 15].
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

IS_END  = "2025-12-31"
MOM_WIN = 252
REV_LB  = 5
N_LIST  = [1, 3, 5, 10, 15]
VIX_THRESHOLDS = [None, 15, 20]   # None = always trade

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_tickers():
    raw = TICKER_CSV.read_text(encoding="utf-8-sig").strip().splitlines()
    return [t.strip() for t in raw if t.strip()]


def load_prices(tickers):
    close = pd.read_parquet(DAILY_FILE)
    avail = [t for t in tickers if t in close.columns]
    close = close[avail][close.index <= IS_END].sort_index()
    return close.where(close.gt(0)).ffill()


def load_vix() -> pd.Series:
    print("Fetching VIX (^VIX) ...")
    raw = yf.download("^VIX", start="2015-01-01", end=IS_END,
                      auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    vix.name = "vix"
    return vix


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_rev_topn(close: pd.DataFrame, n: int,
                 vix: pd.Series, vix_threshold) -> pd.Series:
    """
    Long bottom-n ETFs by REV_LB-day return, T+1 execution.
    Position is flat on day T if prev-day VIX <= vix_threshold.
    vix_threshold=None means always trade.
    """
    sig_all = close.pct_change(REV_LB)
    dates   = close.index.tolist()

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        # VIX gate: check prev day's VIX
        if vix_threshold is not None:
            prev_vix = vix.get(prev, np.nan)
            if np.isnan(prev_vix) or prev_vix <= vix_threshold:
                weights = pd.Series(0.0, index=close.columns)
                continue

        sig = sig_all.loc[prev].dropna()
        if len(sig) < n:
            weights = pd.Series(0.0, index=close.columns)
            continue

        longs = sig.nsmallest(n).index
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = 1.0 / n

    return pd.Series(dict(daily_pnl), name=f"bot{n}")


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(pnl: pd.Series, label: str = "") -> dict:
    cum    = (1 + pnl).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    ann    = pnl.mean() * 252
    vol    = pnl.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    active = (pnl != 0).mean()
    return {
        "label":    label,
        "ann_ret":  round(float(ann), 4),
        "ann_vol":  round(float(vol), 4),
        "sharpe":   round(float(sharpe), 3),
        "max_dd":   round(float(dd.min()), 4),
        "calmar":   round(float(calmar), 3),
        "win_rate": round(float((pnl > 0).mean()), 3),
        "pct_invested": round(float(active), 3),
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


# ── Plots ─────────────────────────────────────────────────────────────────────

N_COLORS = {
    1:  "#d62728",
    3:  "#ff7f0e",
    5:  "#2ca02c",
    10: "#1f77b4",
    15: "#9467bd",
}


def plot_grid(all_results: dict, first_sig: str) -> None:
    """
    3-panel plot: one column per VIX threshold.
    Each panel shows all N values.
    """
    thresholds = [None, 15, 20]
    titles = {None: "Always-on (no VIX filter)", 15: "VIX > 15", 20: "VIX > 20"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, thresh in zip(axes, thresholds):
        key = thresh if thresh is not None else "always"
        for n in N_LIST:
            pnl = all_results[key][n]
            pnl = pnl[pnl.index >= first_sig]
            cum = (1 + pnl).cumprod() - 1
            ax.plot(pd.to_datetime(cum.index), cum * 100,
                    color=N_COLORS[n], linewidth=1.5, label=f"Bottom {n}")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(titles[thresh], fontsize=11)
        ax.set_ylabel("Cumulative Return (%)" if ax is axes[0] else "")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Long-only Reversal  |  Bottom N by {REV_LB}d Return  |  VIX Filter Comparison  |  IS: 2017–2025",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = OUT_DIR / "cumret_rev_topn_vix.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_by_n(all_results: dict, first_sig: str) -> None:
    """
    One plot per N: shows always-on vs VIX>15 vs VIX>20.
    """
    thresh_colors  = {None: "#7f7f7f", 15: "#1f77b4", 20: "#d62728"}
    thresh_labels  = {None: "Always-on", 15: "VIX > 15", 20: "VIX > 20"}
    thresh_styles  = {None: "--", 15: "-", 20: "-"}

    fig, axes = plt.subplots(1, len(N_LIST), figsize=(20, 4), sharey=True)

    for ax, n in zip(axes, N_LIST):
        for thresh in [None, 15, 20]:
            key = thresh if thresh is not None else "always"
            pnl = all_results[key][n]
            pnl = pnl[pnl.index >= first_sig]
            cum = (1 + pnl).cumprod() - 1
            ax.plot(pd.to_datetime(cum.index), cum * 100,
                    color=thresh_colors[thresh],
                    linestyle=thresh_styles[thresh],
                    linewidth=1.6 if thresh is not None else 1.1,
                    label=thresh_labels[thresh])
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_title(f"Bottom {n}", fontsize=11)
        ax.set_ylabel("Cumulative Return (%)" if ax is axes[0] else "")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Long-only Reversal  |  Bottom N by {REV_LB}d Return  |  VIX Gate Effect  |  IS: 2017–2025",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = OUT_DIR / "cumret_rev_vix_by_n.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tickers = load_tickers()
    close   = load_prices(tickers)
    vix     = load_vix()
    first_sig = close.index[MOM_WIN]

    print(f"\nUniverse : {len(close.columns)} tickers")
    print(f"Window   : {first_sig} -> {close.index[-1]}")
    print(f"VIX data : {vix.index[0]} -> {vix.index[-1]}\n")

    # Days above each threshold
    vix_in_window = vix[vix.index >= first_sig]
    total = len(vix_in_window)
    for t in [15, 20]:
        pct = (vix_in_window > t).mean()
        print(f"  VIX > {t}: {pct:.1%} of trading days  ({int(pct*total)}/{total})")
    print()

    # ── Run all combinations ──────────────────────────────────────────────────
    all_results = {}   # {key: {n: pnl}}
    all_rows    = []

    for thresh in VIX_THRESHOLDS:
        key   = thresh if thresh is not None else "always"
        label = "Always-on" if thresh is None else f"VIX > {thresh}"
        all_results[key] = {}

        print(f"--- {label} ---")
        print(f"  {'N':>4}  {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
              f"{'MaxDD':>8} {'Calmar':>8} {'WinRate':>7} {'%Active':>8}")
        print("  " + "-" * 68)

        for n in N_LIST:
            pnl = run_rev_topn(close, n, vix, thresh)
            pnl = pnl[pnl.index >= first_sig]
            all_results[key][n] = pnl
            s   = compute_stats(pnl, f"{label}  Bot{n}")
            all_rows.append(s)
            print(f"  {n:>4}  {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
                  f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  "
                  f"{s['calmar']:>7.3f}  {s['win_rate']:>7.1%}  "
                  f"{s['pct_invested']:>7.1%}")
        print()

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "stats_rev_topn_vix.csv", index=False)
    print(f"Saved -> {OUT_DIR}/stats_rev_topn_vix.csv")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nSaving charts ...")
    plot_grid(all_results, first_sig)
    plot_by_n(all_results, first_sig)

    # Heatmaps for the VIX>20 case across all N
    for n in N_LIST:
        for thresh, key in [(15, 15), (20, 20)]:
            pnl   = all_results[key][n]
            title = f"Reversal Bottom {n}  |  VIX > {thresh}  |  IS: 2017–2025"
            fname = f"monthly_rev_bot{n}_vix{thresh}.png"
            plot_heatmap(pnl, title, fname)

    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

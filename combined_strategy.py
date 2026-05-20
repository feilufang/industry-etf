#!/usr/bin/env python3
"""
Combined: ERC Momentum (252d, quartile L/S) + ERC Reversal (5d, long-only)
Strategy allocation sized by rolling 252d equal-volatility weights.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yfinance as yf

HERE       = Path(__file__).parent
DAILY_FILE = HERE / "data" / "industry_etf_daily.parquet"
TICKER_CSV = HERE / "Industry_ETF_Tickers.csv"
OUT_DIR    = HERE / "results_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END     = "2025-12-31"
MOM_WIN    = 252
COV_WIN    = 252
REV_LB     = 5       # reversal lookback in days
VOL_WIN    = 252     # lookback for equal-vol sizing between strategies
MIN_STOCKS = 8

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


# ── ERC optimizer ─────────────────────────────────────────────────────────────

def _sample_cov(ret_win, tickers):
    C = ret_win[tickers].cov().values
    return C + np.eye(len(C)) * 1e-8


def opt_erc(ret_win, tickers):
    n  = len(tickers)
    C  = _sample_cov(ret_win, tickers)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: 0.5 * w @ C @ w - (1/n) * np.sum(np.log(np.maximum(w, 1e-12))),
        x0=w0,
        jac=lambda w: C @ w - (1/n) / np.maximum(w, 1e-12),
        method="L-BFGS-B",
        bounds=[(1e-6, 1.0)] * n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w / w.sum()


# ── Momentum backtest (ERC, quartile, L/S, daily rebal) ───────────────────────

def run_momentum(close):
    mom       = close.pct_change(MOM_WIN)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        sig = mom.loc[prev].dropna()
        if len(sig) < MIN_STOCKS:
            weights = pd.Series(0.0, index=close.columns)
            continue

        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index
        shorts = sig[ranked > len(sig) - n_q].index

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        weights = pd.Series(0.0, index=close.columns)
        if len(longs):
            try:    weights[longs]  =  opt_erc(ret_win, longs)
            except: weights[longs]  =  1.0 / len(longs)
        if len(shorts):
            try:    weights[shorts] = -opt_erc(ret_win, shorts)
            except: weights[shorts] = -1.0 / len(shorts)

    return pd.Series(dict(daily_pnl), name="momentum")


# ── Reversal backtest (ERC, 5d, long-only bottom quartile, daily rebal) ───────

def run_reversal(close):
    sig_all   = close.pct_change(REV_LB)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        sig = sig_all.loc[prev].dropna()
        if len(sig) < MIN_STOCKS:
            weights = pd.Series(0.0, index=close.columns)
            continue

        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked > len(sig) - n_q].index   # bottom quartile = worst 5d performers

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        weights = pd.Series(0.0, index=close.columns)
        if len(longs):
            try:    weights[longs] = opt_erc(ret_win, longs)
            except: weights[longs] = 1.0 / len(longs)

    return pd.Series(dict(daily_pnl), name="reversal")


# ── Equal-vol combination ─────────────────────────────────────────────────────

def combine_equal_vol(pnl_mom, pnl_rev):
    """
    Rolling VOL_WIN-day equal-vol allocation between two strategies.
    w_i = (1/vol_i) / (1/vol_mom + 1/vol_rev)
    Combined daily P&L = w_mom * pnl_mom + w_rev * pnl_rev.
    Weights computed daily, first valid after VOL_WIN days of both series.
    """
    df = pd.DataFrame({"mom": pnl_mom, "rev": pnl_rev}).dropna()

    ann = np.sqrt(252)
    vol_mom = df["mom"].rolling(VOL_WIN).std() * ann
    vol_rev = df["rev"].rolling(VOL_WIN).std() * ann

    inv_mom = 1.0 / vol_mom.replace(0, np.nan)
    inv_rev = 1.0 / vol_rev.replace(0, np.nan)
    total   = inv_mom + inv_rev

    w_mom = inv_mom / total
    w_rev = inv_rev / total

    combined = (w_mom * df["mom"] + w_rev * df["rev"]).dropna()
    return combined, w_mom.reindex(combined.index), w_rev.reindex(combined.index)


# ── Stats & monthly table ─────────────────────────────────────────────────────

def compute_stats(pnl, label=""):
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


def monthly_returns(pnl):
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

def plot_combined(pnl_mom, pnl_rev, pnl_comb, w_mom, w_rev):
    idx = pnl_comb.index

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    for pnl, label, color, lw, ls in [
        (pnl_mom.reindex(idx).fillna(0), "ERC Momentum (L/S)",          "#1f77b4", 1.3, "--"),
        (pnl_rev.reindex(idx).fillna(0), f"ERC Reversal {REV_LB}d (L-only)", "#ff7f0e", 1.3, "--"),
        (pnl_comb,                        "Combined (Equal-Vol)",         "#2ca02c", 2.0, "-"),
    ]:
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                linewidth=lw, label=label, color=color, linestyle=ls)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        f"ERC Momentum (252d quartile L/S)  +  ERC Reversal ({REV_LB}d long-only)  "
        f"|  IS: 2017–2025\nStrategy allocation: rolling {VOL_WIN}d equal-volatility weights",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    dates_dt = pd.to_datetime(idx)
    ax2.stackplot(
        dates_dt,
        w_mom.fillna(0.5).values * 100,
        w_rev.fillna(0.5).values * 100,
        labels=["w Momentum", "w Reversal"],
        colors=["#1f77b4", "#ff7f0e"],
        alpha=0.75,
    )
    ax2.set_ylabel("Allocation (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3)
    ax2.set_xlabel("Date")

    fig.tight_layout()
    path = OUT_DIR / "cumret_combined.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_heatmap(pnl, title, fname):
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
    print(f"Universe: {len(tickers)} tickers")

    print("Loading prices ...")
    close = load_prices(tickers)
    print(f"  {len(close.columns)} tickers  |  {close.index[0]} -> {close.index[-1]}  ({len(close):,} days)")

    first_sig = close.index[MOM_WIN]
    print(f"  Effective window: {first_sig} -> {close.index[-1]}\n")

    print("Running ERC Momentum (252d, quartile L/S, daily rebal) ...")
    pnl_mom = run_momentum(close)
    pnl_mom = pnl_mom[pnl_mom.index >= first_sig]
    s = compute_stats(pnl_mom, "ERC Momentum")
    print(f"  → {s['ann_ret']:+.2%}  Sharpe {s['sharpe']:.3f}  MaxDD {s['max_dd']:+.2%}")

    print(f"\nRunning ERC Reversal ({REV_LB}d, long-only bottom quartile, daily rebal) ...")
    pnl_rev = run_reversal(close)
    pnl_rev = pnl_rev[pnl_rev.index >= first_sig]
    s = compute_stats(pnl_rev, f"ERC Reversal {REV_LB}d")
    print(f"  → {s['ann_ret']:+.2%}  Sharpe {s['sharpe']:.3f}  MaxDD {s['max_dd']:+.2%}")

    print(f"\nCombining with rolling {VOL_WIN}d equal-vol weights ...")
    pnl_comb, w_mom, w_rev = combine_equal_vol(pnl_mom, pnl_rev)

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    print(f"\n{'Strategy':<28} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'WinRate':>7}")
    print("-" * 82)

    for pnl, label in [
        (pnl_mom.reindex(pnl_comb.index).dropna(), "ERC Momentum (L/S)"),
        (pnl_rev.reindex(pnl_comb.index).dropna(), f"ERC Reversal {REV_LB}d (L-only)"),
        (pnl_comb,                                  "Combined Equal-Vol"),
    ]:
        s = compute_stats(pnl, label)
        rows.append(s)
        print(f"  {label:<26} {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  "
              f"{s['calmar']:>7.3f}  {s['win_rate']:>7.1%}")

    print(f"\n  Avg allocation:  Momentum = {w_mom.mean():.1%}   Reversal = {w_rev.mean():.1%}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "stats_combined.csv", index=False)
    print(f"  Saved -> {OUT_DIR}/stats_combined.csv")

    # ── Monthly tables ────────────────────────────────────────────────────────
    for pnl, label in [
        (pnl_comb,                                   "Combined"),
        (pnl_mom.reindex(pnl_comb.index).dropna(),   "Momentum"),
        (pnl_rev.reindex(pnl_comb.index).dropna(),   "Reversal"),
    ]:
        print(f"\n--- {label} Monthly Returns ---")
        print((monthly_returns(pnl) * 100).round(1).to_string())

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nSaving charts ...")
    plot_combined(pnl_mom, pnl_rev, pnl_comb, w_mom, w_rev)
    plot_heatmap(
        pnl_comb,
        "ERC Momentum + ERC Reversal  |  Equal-Vol Combined  |  IS: 2017–2025",
        "monthly_combined.png",
    )
    plot_heatmap(
        pnl_mom.reindex(pnl_comb.index).dropna(),
        "ERC Momentum (252d Quartile L/S)  |  IS: 2017–2025",
        "monthly_momentum.png",
    )
    plot_heatmap(
        pnl_rev.reindex(pnl_comb.index).dropna(),
        f"ERC Reversal {REV_LB}d (Long-only Bottom Quartile)  |  IS: 2017–2025",
        "monthly_reversal.png",
    )

    # ── SPY correlation ───────────────────────────────────────────────────────
    print("\nFetching SPY ...")
    spy_raw  = yf.download("SPY", start="2016-01-01", end=IS_END,
                           auto_adjust=True, progress=False)
    spy_ret  = spy_raw["Close"].squeeze().pct_change().dropna()
    spy_ret.index = spy_ret.index.tz_localize(None).strftime("%Y-%m-%d")

    aligned = pd.DataFrame({
        "combined": pnl_comb,
        "momentum": pnl_mom.reindex(pnl_comb.index),
        "reversal": pnl_rev.reindex(pnl_comb.index),
        "spy":      spy_ret,
    }).dropna()

    print(f"\n  Static correlation with SPY ({aligned.index[0]} → {aligned.index[-1]}):")
    for col in ["combined", "momentum", "reversal"]:
        corr = aligned[col].corr(aligned["spy"])
        print(f"    {col:<12} {corr:+.3f}")

    # Rolling 90d correlation
    roll = 90
    r_corr = pd.DataFrame({
        col: aligned[col].rolling(roll).corr(aligned["spy"])
        for col in ["combined", "momentum", "reversal"]
    }).dropna()

    print(f"\n  Rolling {roll}d correlation with SPY (mean ± std):")
    for col in ["combined", "momentum", "reversal"]:
        print(f"    {col:<12}  mean={r_corr[col].mean():+.3f}  "
              f"min={r_corr[col].min():+.3f}  max={r_corr[col].max():+.3f}")

    # Plot rolling correlation
    colors_corr = {"combined": "#2ca02c", "momentum": "#1f77b4", "reversal": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(13, 4))
    for col in ["combined", "momentum", "reversal"]:
        ax.plot(pd.to_datetime(r_corr.index), r_corr[col],
                label=col.capitalize(), color=colors_corr[col], linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(pd.to_datetime(r_corr.index),
                    r_corr["combined"], 0,
                    where=r_corr["combined"] > 0, alpha=0.12, color="#d62728")
    ax.fill_between(pd.to_datetime(r_corr.index),
                    r_corr["combined"], 0,
                    where=r_corr["combined"] <= 0, alpha=0.12, color="#2ca02c")
    ax.set_ylabel(f"Rolling {roll}d Correlation with SPY")
    ax.set_title(
        f"Rolling {roll}d Correlation with SPY  |  Combined vs Component Strategies  |  IS: 2017–2025",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "spy_corr_rolling.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  Saved -> {path}")

    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

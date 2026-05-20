#!/usr/bin/env python3
"""
Signal-level combination: 252d momentum + 5d short-term reversal.

Per ETF, at each day T:
  z_mom = cross-sectional z-score of 252d return
  z_rev = cross-sectional z-score of (-5d return)   [negative = buy recent losers]
  composite = z_mom + z_rev

Long top quartile of composite, short bottom quartile.
ERC optimization within each basket using trailing 252d covariance.

Benchmarks reported alongside:
  - ERC Momentum only (252d, quartile L/S)
  - ERC Reversal only  (5d, L/S)
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
REV_LB     = 5
COV_WIN    = 252
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


def basket_erc(ret_win, tickers, sign):
    try:    w = opt_erc(ret_win, tickers)
    except: w = np.ones(len(tickers)) / len(tickers)
    return pd.Series(w * sign, index=tickers)


# ── Cross-sectional z-score ───────────────────────────────────────────────────

def xs_zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score; returns NaN-safe version."""
    mu  = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * np.nan
    return (series - mu) / std


# ── Backtests ─────────────────────────────────────────────────────────────────

def run_combined(close):
    """
    Composite signal = z_mom + z_rev, daily rebalance, ERC sizing.
    """
    mom_sig = close.pct_change(MOM_WIN)           # 252d return
    rev_sig = -close.pct_change(REV_LB)           # negative 5d return (buy losers)

    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        # Both signals must be valid at prev
        m = mom_sig.loc[prev].dropna()
        r = rev_sig.loc[prev].dropna()
        common = m.index.intersection(r.index)
        if len(common) < MIN_STOCKS:
            weights = pd.Series(0.0, index=close.columns)
            continue

        composite = xs_zscore(m[common]) + xs_zscore(r[common])

        n_q    = max(1, len(composite) // 4)
        ranked = composite.rank(ascending=False)
        longs  = composite[ranked <= n_q].index
        shorts = composite[ranked > len(composite) - n_q].index

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        weights = pd.Series(0.0, index=close.columns)
        if len(longs):
            weights[longs]  = basket_erc(ret_win, longs,  +1.0)
        if len(shorts):
            weights[shorts] = basket_erc(ret_win, shorts, -1.0)

    return pd.Series(dict(daily_pnl), name="combined_signal")


def run_mom_only(close):
    """ERC momentum: 252d signal, quartile L/S."""
    mom_sig   = close.pct_change(MOM_WIN)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        sig = mom_sig.loc[prev].dropna()
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
            weights[longs]  = basket_erc(ret_win, longs,  +1.0)
        if len(shorts):
            weights[shorts] = basket_erc(ret_win, shorts, -1.0)

    return pd.Series(dict(daily_pnl), name="momentum")


def run_rev_only(close):
    """ERC reversal: 5d signal, quartile L/S (long losers, short winners)."""
    rev_sig   = close.pct_change(REV_LB)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        sig = rev_sig.loc[prev].dropna()
        if len(sig) < MIN_STOCKS:
            weights = pd.Series(0.0, index=close.columns)
            continue

        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked > len(sig) - n_q].index   # bottom quartile = worst 5d
        shorts = sig[ranked <= n_q].index              # top quartile = best 5d

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        weights = pd.Series(0.0, index=close.columns)
        if len(longs):
            weights[longs]  = basket_erc(ret_win, longs,  +1.0)
        if len(shorts):
            weights[shorts] = basket_erc(ret_win, shorts, -1.0)

    return pd.Series(dict(daily_pnl), name="reversal")


# ── Stats & helpers ───────────────────────────────────────────────────────────

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

def plot_cumret(results, title, fname):
    colors = {"combined_signal": "#2ca02c", "momentum": "#1f77b4", "reversal": "#ff7f0e"}
    labels = {
        "combined_signal": f"Combined signal (z_mom + z_rev{REV_LB}d)  ERC",
        "momentum":        "ERC Momentum only (252d)",
        "reversal":        f"ERC Reversal only ({REV_LB}d L/S)",
    }
    styles = {"combined_signal": "-", "momentum": "--", "reversal": "--"}
    widths = {"combined_signal": 2.2, "momentum": 1.3, "reversal": 1.3}

    fig, ax = plt.subplots(figsize=(13, 5))
    for key, pnl in results.items():
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                color=colors[key], linestyle=styles[key], linewidth=widths[key],
                label=labels[key])
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / fname
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

    print(f"Running combined signal (z_mom252 + z_rev{REV_LB}d), ERC, quartile L/S ...")
    pnl_comb = run_combined(close)
    pnl_comb = pnl_comb[pnl_comb.index >= first_sig]

    print("Running ERC Momentum only (252d, quartile L/S) ...")
    pnl_mom  = run_mom_only(close)
    pnl_mom  = pnl_mom[pnl_mom.index >= first_sig]

    print(f"Running ERC Reversal only ({REV_LB}d, quartile L/S) ...")
    pnl_rev  = run_rev_only(close)
    pnl_rev  = pnl_rev[pnl_rev.index >= first_sig]

    # ── Stats ─────────────────────────────────────────────────────────────────
    rows = []
    print(f"\n{'Strategy':<40} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'WinRate':>7}")
    print("-" * 92)

    for pnl, label in [
        (pnl_comb, f"Combined signal ERC (mom252 + rev{REV_LB}d)"),
        (pnl_mom,  "ERC Momentum only (252d)"),
        (pnl_rev,  f"ERC Reversal only ({REV_LB}d L/S)"),
    ]:
        s = compute_stats(pnl, label)
        rows.append(s)
        print(f"  {label:<38} {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  "
              f"{s['calmar']:>7.3f}  {s['win_rate']:>7.1%}")

    # Pairwise correlations
    df_all = pd.DataFrame({
        "combined": pnl_comb, "momentum": pnl_mom, "reversal": pnl_rev
    }).dropna()
    print(f"\n  Pairwise daily P&L correlations:")
    for a, b in [("combined", "momentum"), ("combined", "reversal"), ("momentum", "reversal")]:
        print(f"    {a} vs {b}: {df_all[a].corr(df_all[b]):+.3f}")

    # SPY correlation
    print("\nFetching SPY ...")
    spy_raw = yf.download("SPY", start="2016-01-01", end=IS_END,
                          auto_adjust=True, progress=False)
    spy_ret = spy_raw["Close"].squeeze().pct_change().dropna()
    spy_ret.index = spy_ret.index.tz_localize(None).strftime("%Y-%m-%d")

    aligned = pd.DataFrame({
        "combined": pnl_comb, "momentum": pnl_mom,
        "reversal": pnl_rev,  "spy": spy_ret,
    }).dropna()

    print(f"\n  Static correlation with SPY:")
    for col in ["combined", "momentum", "reversal"]:
        print(f"    {col:<12}  {aligned[col].corr(aligned['spy']):+.3f}")

    # ── Save & plot ───────────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(OUT_DIR / "stats_signal_combined.csv", index=False)
    print(f"\n  Saved -> {OUT_DIR}/stats_signal_combined.csv")

    results = {"combined_signal": pnl_comb, "momentum": pnl_mom, "reversal": pnl_rev}

    print("\nSaving charts ...")
    plot_cumret(
        results,
        f"Signal-level combination: z_mom(252d) + z_rev({REV_LB}d)  →  ERC  |  Quartile L/S  |  IS: 2017–2025\n"
        "vs ERC Momentum-only and ERC Reversal-only benchmarks  |  Dollar-neutral",
        "cumret_signal_combined.png",
    )
    plot_heatmap(
        pnl_comb,
        f"Combined Signal ERC  (z_mom252 + z_rev{REV_LB}d)  |  Quartile L/S  |  IS: 2017–2025",
        "monthly_signal_combined.png",
    )

    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

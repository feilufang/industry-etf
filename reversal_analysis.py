#!/usr/bin/env python3
"""
Short-term reversal signal analysis on Industry ETFs.

Two approaches:
  1. Cross-sectional reversal  — rank ETFs by prior N-day return,
                                 long bottom quartile, short top quartile.
                                 Lookbacks: 1d, 5d, 10d, 21d.

  2. Time-series (AR) analysis — per-ETF AR(1) coefficient on daily returns.
                                 Aggregate to see if universe has reversal structure.

Timing: signal at T close → trade at T+1 close (no look-ahead).
IS    : 2017-05-16 → 2025-12-31
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as scipy_stats

HERE      = Path(__file__).parent
OUT_DIR   = HERE / "results_reversal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DAILY_FILE  = HERE / "data" / "industry_etf_daily.parquet"
TICKER_CSV  = HERE / "Industry_ETF_Tickers.csv"
IS_END      = "2025-12-31"
MIN_STOCKS  = 8
LOOKBACKS   = [1, 5, 10, 21]   # days


# ── Data ───────────────────────────────────────────────────────────────────────

def load_tickers():
    raw = (HERE / "Industry_ETF_Tickers.csv").read_text(encoding="utf-8-sig").strip().splitlines()
    return [t.strip() for t in raw if t.strip()]


def load_prices(tickers):
    close = pd.read_parquet(DAILY_FILE)
    avail = [t for t in tickers if t in close.columns]
    close = close[avail][close.index <= IS_END].sort_index()
    return close.where(close.gt(0)).ffill()


# ── Stats ──────────────────────────────────────────────────────────────────────

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
        "n_days":   int(len(pnl)),
    }


# ── 1. Cross-sectional reversal backtest ───────────────────────────────────────

def run_xs_reversal(close: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Signal at T: prior `lookback`-day return = close[T] / close[T-lookback] - 1
    Long bottom quartile (worst recent), short top quartile (best recent).
    Trade at T+1 close, equal weight.
    """
    sig_all  = close.pct_change(lookback)   # (date × ticker) prior-N-day return
    dates    = close.index.tolist()
    weights  = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date      = dates[i]
        prev_date = dates[i - 1]

        ret = (close.loc[date] / close.loc[prev_date] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        # Signal computed at prev_date (T), weights active from date (T+1)
        sig = sig_all.loc[prev_date].dropna()
        if len(sig) < MIN_STOCKS:
            weights = pd.Series(0.0, index=close.columns)
            continue

        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        # Reversal: long worst (high rank = low return), short best (low rank = high return)
        longs  = sig[ranked > len(sig) - n_q].index   # bottom quartile by return
        shorts = sig[ranked <= n_q].index              # top quartile by return

        weights = pd.Series(0.0, index=close.columns)
        if len(longs):
            weights[longs]  =  1.0 / len(longs)
        if len(shorts):
            weights[shorts] = -1.0 / len(shorts)

    return pd.Series(dict(daily_pnl), name=f"rev_{lookback}d")


# ── 2. Time-series AR(1) analysis ─────────────────────────────────────────────

def ar1_analysis(close: pd.DataFrame, first_sig: str) -> pd.DataFrame:
    """
    Per-ETF OLS AR(1): r_t = alpha + beta * r_{t-1} + eps
    Returns DataFrame of (ticker, beta, t-stat, p-value, n_obs).
    """
    daily_ret = close.pct_change().loc[first_sig:]
    rows = []
    for ticker in daily_ret.columns:
        r = daily_ret[ticker].dropna()
        if len(r) < 100:
            continue
        r_lag = r.shift(1).dropna()
        r_cur = r.loc[r_lag.index]
        slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(r_lag, r_cur)
        t_stat = slope / std_err
        rows.append({
            "ticker": ticker,
            "ar1_beta":  round(slope, 4),
            "t_stat":    round(t_stat, 3),
            "p_value":   round(p_val, 4),
            "n_obs":     len(r_cur),
        })
    return pd.DataFrame(rows).sort_values("ar1_beta")


def run_ar1_reversal(close: pd.DataFrame) -> pd.Series:
    """
    Daily cross-sectional strategy using AR(1) predicted return as signal.
    Signal: - (beta_i * r_{t-1})  where beta_i is a rolling AR(1) estimate.
    Rolling window: 252 days. Long predicted-down, short predicted-up.
    """
    daily_ret = close.pct_change()
    dates     = close.index.tolist()

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    AR_WIN = 252

    for i in range(1, len(dates)):
        date      = dates[i]
        prev_date = dates[i - 1]

        ret = (close.loc[date] / close.loc[prev_date] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        # Need at least AR_WIN days of history
        if i < AR_WIN + 1:
            weights = pd.Series(0.0, index=close.columns)
            continue

        # Rolling AR(1): for each ticker, regress r_t on r_{t-1} over last AR_WIN days
        win = daily_ret.iloc[i - AR_WIN : i]   # AR_WIN rows ending at prev_date
        r_lag = win.shift(1).iloc[1:]
        r_cur = win.iloc[1:]

        # Predicted reversal signal: -beta_i * r_prev  (sign-flipped = reversal)
        r_prev = daily_ret.iloc[i - 1]   # yesterday's return = r_{t-1}
        signals = {}
        for ticker in close.columns:
            rl = r_lag[ticker].dropna()
            rc = r_cur[ticker].loc[rl.index]
            if len(rl) < 30:
                continue
            try:
                beta = np.cov(rl, rc)[0, 1] / np.var(rl)
                pred = -beta * float(r_prev.get(ticker, np.nan))
                if np.isfinite(pred):
                    signals[ticker] = pred
            except Exception:
                continue

        sig = pd.Series(signals).dropna()
        if len(sig) < MIN_STOCKS:
            weights = pd.Series(0.0, index=close.columns)
            continue

        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index       # highest predicted reversal gain
        shorts = sig[ranked > len(sig) - n_q].index

        weights = pd.Series(0.0, index=close.columns)
        if len(longs):
            weights[longs]  =  1.0 / len(longs)
        if len(shorts):
            weights[shorts] = -1.0 / len(shorts)

    return pd.Series(dict(daily_pnl), name="ar1_rev")


# ── Plots ──────────────────────────────────────────────────────────────────────

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_cumret(results: dict, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for (label, pnl), color in zip(results.items(), COLORS):
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                linewidth=1.4, label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_ar1_dist(ar1_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.hist(ar1_df["ar1_beta"], bins=20, color="#1f77b4", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(ar1_df["ar1_beta"].mean(), color="#d62728", linewidth=1.5,
               linestyle="--", label=f'Mean={ar1_df["ar1_beta"].mean():+.4f}')
    ax.set_xlabel("AR(1) Beta")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of AR(1) Coefficients across ETFs", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sig = ar1_df[ar1_df["p_value"] < 0.1].sort_values("ar1_beta")
    colors = ["#d62728" if b < 0 else "#2ca02c" for b in sig["ar1_beta"]]
    ax.barh(sig["ticker"], sig["ar1_beta"], color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AR(1) Beta")
    ax.set_title("Statistically Significant AR(1) (p < 0.10)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    path = OUT_DIR / "ar1_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_monthly_heatmap(pnl: pd.Series, label: str, fname: str) -> None:
    idx   = pd.to_datetime(pnl.index)
    m_ret = (pd.Series((1 + pnl.values), index=idx).resample("ME").prod() - 1)
    m_ret.index = m_ret.index.to_period("M")
    tbl   = m_ret.rename("ret").to_frame()
    tbl["year"]  = tbl.index.year
    tbl["month"] = tbl.index.month
    pivot = tbl.pivot(index="year", columns="month", values="ret")
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
    pivot["Ann"]  = tbl.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)

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
    _hm(axes[1], pivot[["Ann"]], "Annual (%)")
    fig.suptitle(f"Short-term Reversal — {label}  |  IS: 2017–2025", fontsize=11, y=1.01)
    fig.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    tickers   = load_tickers()
    close     = load_prices(tickers)
    first_sig = close.index[252]

    print(f"Universe : {len(close.columns)} tickers")
    print(f"Data     : {close.index[0]} -> {close.index[-1]}  ({len(close):,} days)")
    print(f"IS window: {first_sig} -> {close.index[-1]}\n")

    # ── 1. AR(1) analysis ──────────────────────────────────────────────────────
    print("Computing AR(1) coefficients ...")
    ar1_df = ar1_analysis(close, first_sig)
    neg    = (ar1_df["ar1_beta"] < 0).sum()
    sig10  = (ar1_df["p_value"] < 0.10).sum()
    mean_b = ar1_df["ar1_beta"].mean()

    print(f"  {len(ar1_df)} ETFs  |  {neg} with negative AR(1) ({neg/len(ar1_df):.0%})")
    print(f"  Mean AR(1) beta: {mean_b:+.4f}")
    print(f"  Significant (p<0.10): {sig10}  |  (p<0.05): {(ar1_df['p_value']<0.05).sum()}")
    print(f"\n  Most negative (reversal):")
    print(ar1_df.head(8)[["ticker","ar1_beta","t_stat","p_value"]].to_string(index=False))
    print(f"\n  Most positive (momentum):")
    print(ar1_df.tail(5)[["ticker","ar1_beta","t_stat","p_value"]].to_string(index=False))

    ar1_df.to_csv(OUT_DIR / "ar1_coefficients.csv", index=False)
    print(f"\n  Saved -> {OUT_DIR}/ar1_coefficients.csv")

    # ── 2. Cross-sectional reversal backtests ──────────────────────────────────
    print(f"\n{'Signal':<15} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("-" * 60)

    xs_results = {}
    xs_rows    = []
    for lb in LOOKBACKS:
        label = f"XS Rev {lb}d"
        pnl   = run_xs_reversal(close, lb)
        pnl   = pnl[pnl.index >= first_sig]
        xs_results[label] = pnl
        s = compute_stats(pnl, label)
        xs_rows.append(s)
        print(f"  {label:<13} {s['ann_ret']:>+8.2%} {s['ann_vol']:>8.2%} "
              f"{s['sharpe']:>8.3f} {s['max_dd']:>+8.2%} {s['win_rate']:>8.1%}")

    # ── 3. AR(1) signal backtest ───────────────────────────────────────────────
    print(f"\n  Running AR(1) rolling signal backtest ...", flush=True)
    pnl_ar1 = run_ar1_reversal(close)
    pnl_ar1 = pnl_ar1[pnl_ar1.index >= first_sig]
    s_ar1   = compute_stats(pnl_ar1, "AR(1) Rev")
    xs_rows.append(s_ar1)
    print(f"  {'AR(1) Rev':<13} {s_ar1['ann_ret']:>+8.2%} {s_ar1['ann_vol']:>8.2%} "
          f"{s_ar1['sharpe']:>8.3f} {s_ar1['max_dd']:>+8.2%} {s_ar1['win_rate']:>8.1%}")

    all_results = {**xs_results, "AR(1) Rev": pnl_ar1}

    # Monthly breakdown
    for label, pnl in all_results.items():
        idx   = pd.to_datetime(pnl.index)
        m_ret = (pd.Series((1 + pnl.values), index=idx).resample("ME").prod() - 1)
        m_ret.index = m_ret.index.to_period("M")
        tbl   = m_ret.rename("ret").to_frame()
        tbl["year"]  = tbl.index.year
        tbl["month"] = tbl.index.month
        pivot = tbl.pivot(index="year", columns="month", values="ret")
        pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
        pivot["Ann"] = tbl.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
        print(f"\n--- {label} Monthly Returns ---")
        print((pivot * 100).round(1).to_string())

    # Save stats
    pd.DataFrame(xs_rows).to_csv(OUT_DIR / "stats_reversal.csv", index=False)

    # Plots
    print("\nSaving charts ...")
    plot_ar1_dist(ar1_df)
    plot_cumret(
        xs_results,
        "Cross-sectional Reversal — Industry ETFs  |  Daily Rebalance  |  IS: 2017–2025\n"
        "Long Bottom Quartile / Short Top Quartile  |  Equal Weight",
        "cumret_xs_reversal.png",
    )
    plot_cumret(
        all_results,
        "Short-term Reversal Signals — Industry ETFs  |  IS: 2017–2025\n"
        "XS Reversal (1/5/10/21d) + AR(1) Rolling Signal  |  Equal Weight",
        "cumret_all_reversal.png",
    )
    for label, pnl in all_results.items():
        fname = "monthly_" + label.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        plot_monthly_heatmap(pnl, label, fname)

    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

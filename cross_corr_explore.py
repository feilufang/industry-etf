#!/usr/bin/env python3
"""
Correlation between:
  A) Industry ETF combined (ERC Momentum L/S + ERC Reversal VIX>20)
  B) ETF Explore momentum backtests — all 6 universes, 252d lookback, equal-weight quartile L/S
     Replicates C:\working\Massive\etf\ETF Explore\momentum_backtest.py logic.

Downloads ETF Explore universe prices via yfinance (data file not present locally).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
import yfinance as yf

HERE       = Path(__file__).parent
EXPLORE    = Path(r"C:\working\Massive\etf\ETF Explore")
OUT_DIR    = HERE / "results_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END     = "2025-12-31"
MOM_WIN    = 252
REV_LB     = 5
COV_WIN    = 252
VOL_WIN    = 252
VIX_GATE   = 20
REBAL_DAYS = 5   # weekly, matching ETF Explore default

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Shared ERC helpers (industry strategy) ────────────────────────────────────

def _sample_cov(ret_win, tickers):
    C = ret_win[tickers].cov().values
    return C + np.eye(len(C)) * 1e-8

def opt_erc(ret_win, tickers):
    n, C = len(tickers), _sample_cov(ret_win, tickers)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: 0.5*w@C@w - (1/n)*np.sum(np.log(np.maximum(w, 1e-12))),
        x0=w0, jac=lambda w: C@w - (1/n)/np.maximum(w, 1e-12),
        method="L-BFGS-B", bounds=[(1e-6, 1.0)]*n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w / w.sum()

def basket_erc(ret_win, tickers, sign):
    try:    w = opt_erc(ret_win, tickers)
    except: w = np.ones(len(tickers)) / len(tickers)
    return pd.Series(w * sign, index=tickers)


# ── Industry combined strategy ────────────────────────────────────────────────

def run_industry_momentum(close, vix):
    mom_sig, daily_ret = close.pct_change(MOM_WIN), close.pct_change()
    dates, date_pos = close.index.tolist(), {d: i for i, d in enumerate(close.index)}
    weights, pnl = pd.Series(0.0, index=close.columns), []
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))
        sig = mom_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns); continue
        n_q = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs, shorts = sig[ranked <= n_q].index, sig[ranked > len(sig)-n_q].index
        ret_win = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1) : date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs]  = basket_erc(ret_win, longs,  +1.0)
        weights[shorts] = basket_erc(ret_win, shorts, -1.0)
    return pd.Series(dict(pnl))

def run_industry_reversal(close, vix):
    rev_sig, daily_ret = close.pct_change(REV_LB), close.pct_change()
    dates, date_pos = close.index.tolist(), {d: i for i, d in enumerate(close.index)}
    weights, pnl = pd.Series(0.0, index=close.columns), []
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))
        if vix.get(prev, np.nan) <= VIX_GATE:
            weights = pd.Series(0.0, index=close.columns); continue
        sig = rev_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns); continue
        n_q = max(1, len(sig) // 4)
        longs = sig.nsmallest(n_q).index
        ret_win = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1) : date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = basket_erc(ret_win, longs, +1.0)
    return pd.Series(dict(pnl))

def combine_ev(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    ia = 1.0 / df["a"].rolling(VOL_WIN).std().replace(0, np.nan)
    ib = 1.0 / df["b"].rolling(VOL_WIN).std().replace(0, np.nan)
    t  = ia + ib
    return ((ia/t)*df["a"] + (ib/t)*df["b"]).dropna()


# ── ETF Explore: equal-weight quartile momentum (replicates their script) ─────

def run_explore_momentum(close, rebal_days=5):
    """
    Exact replication of ETF Explore backtest:
      - 252d signal, top/bottom quartile, equal weight, L/S dollar-neutral
      - Rebalance every rebal_days trading days
    """
    mom_sig = close.pct_change(MOM_WIN)
    dates   = close.index.tolist()
    weights = pd.Series(0.0, index=close.columns)
    pnl     = []
    rebal_counter = 0

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))

        if rebal_counter == 0:
            sig = mom_sig.loc[prev].dropna()
            if len(sig) >= 4:
                n_q    = max(1, len(sig) // 4)
                ranked = sig.rank(ascending=False)
                longs  = sig[ranked <= n_q].index
                shorts = sig[ranked > len(sig) - n_q].index
                weights = pd.Series(0.0, index=close.columns)
                if len(longs):  weights[longs]  =  1.0 / len(longs)
                if len(shorts): weights[shorts] = -1.0 / len(shorts)

        rebal_counter = (rebal_counter + 1) % rebal_days

    return pd.Series(dict(pnl))


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(pnl, label=""):
    pnl = pnl.dropna()
    cum = (1 + pnl).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    ann = pnl.mean() * 252
    vol = pnl.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    return dict(label=label, ann_ret=round(float(ann),4), ann_vol=round(float(vol),4),
                sharpe=round(float(sharpe),3), max_dd=round(float(dd.min()),4),
                calmar=round(float(calmar),3))


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(series_dict, roll_corrs, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    colors = {"Industry combined": "#2ca02c", "SPY": "#7f7f7f"}
    explore_colors = plt.cm.tab10(np.linspace(0, 0.8, 6))

    ax = axes[0]
    ei = 0
    for label, pnl in series_dict.items():
        if label in colors:
            c, lw, ls = colors[label], (2.0 if label != "SPY" else 1.2), ("-" if label != "SPY" else ":")
        else:
            c, lw, ls = explore_colors[ei], 1.3, "--"; ei += 1
        cum = (1 + pnl.dropna()).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum*100, color=c, lw=lw, ls=ls, label=label)

    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        "Industry ETF Combined  vs  ETF Explore Momentum Universes  |  Common window\n"
        "ETF Explore: 252d, equal-weight quartile L/S, weekly rebal",
        fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    for label, rc in roll_corrs.items():
        ax2.plot(pd.to_datetime(rc.index), rc, lw=1.3, label=label)
    ax2.axhline(0, color="black", lw=0.7, ls="--")
    ax2.set_ylabel("Rolling 90d Corr vs Industry")
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Fetch VIX + SPY ───────────────────────────────────────────────────────
    print("Fetching VIX and SPY ...")
    vix_raw = yf.download("^VIX", start="2014-01-01", end=IS_END, auto_adjust=True, progress=False)
    spy_raw = yf.download("SPY",  start="2014-01-01", end=IS_END, auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    spy = spy_raw["Close"].squeeze().pct_change().dropna()
    spy.index = spy.index.tz_localize(None).strftime("%Y-%m-%d")

    # ── Industry combined ─────────────────────────────────────────────────────
    ind_tickers = [t.strip() for t in
                   (HERE / "Industry_ETF_Tickers.csv").read_text(encoding="utf-8-sig")
                   .strip().splitlines() if t.strip()]
    ind_close = pd.read_parquet(HERE / "data" / "industry_etf_daily.parquet")
    ind_close = ind_close[[t for t in ind_tickers if t in ind_close.columns]]
    ind_close = ind_close[ind_close.index <= IS_END].sort_index()
    ind_close = ind_close.where(ind_close.gt(0)).ffill()
    ind_first = ind_close.index[MOM_WIN]

    print(f"\nRunning Industry combined ({len(ind_tickers)} tickers) ...")
    pnl_ind_mom = run_industry_momentum(ind_close, vix)
    pnl_ind_rev = run_industry_reversal(ind_close, vix)
    pnl_industry = combine_ev(
        pnl_ind_mom[pnl_ind_mom.index >= ind_first],
        pnl_ind_rev[pnl_ind_rev.index >= ind_first],
    )
    print(f"  Industry combined: {pnl_industry.index[0]} -> {pnl_industry.index[-1]}")

    # ── ETF Explore universes ─────────────────────────────────────────────────
    universe_files = sorted((EXPLORE / "results_corr").glob("selected_K*_*.csv"))
    universes = {f.stem.replace("selected_", ""): pd.read_csv(f)["ticker"].tolist()
                 for f in universe_files}

    all_exp_tickers = sorted({t for tickers in universes.values() for t in tickers})
    print(f"\nDownloading {len(all_exp_tickers)} ETF Explore tickers ...")
    raw = yf.download(all_exp_tickers, start="2014-01-01", end=IS_END,
                      auto_adjust=True, progress=True, threads=True)
    exp_close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    exp_close.index = exp_close.index.tz_localize(None).strftime("%Y-%m-%d")
    exp_close = exp_close[exp_close.index <= IS_END].sort_index()
    exp_close = exp_close.where(exp_close.gt(0)).ffill()

    # ── Run each universe ─────────────────────────────────────────────────────
    print(f"\nRunning ETF Explore backtests (252d, equal-wt quartile L/S, weekly rebal) ...")
    explore_pnls = {}
    for label, tickers in universes.items():
        avail = [t for t in tickers if t in exp_close.columns]
        sub   = exp_close[avail].copy()
        first = sub.index[MOM_WIN] if len(sub) > MOM_WIN else None
        if first is None: continue
        pnl = run_explore_momentum(sub, rebal_days=REBAL_DAYS)
        pnl = pnl[pnl.index >= first]
        explore_pnls[label] = pnl
        s = compute_stats(pnl, label)
        print(f"  {label:<25} {s['ann_ret']:>+7.2%}  Sharpe {s['sharpe']:.3f}  "
              f"MaxDD {s['max_dd']:>+.2%}  ({len(pnl)} days)")

    # ── Align everything on common window ─────────────────────────────────────
    all_series = {"Industry combined": pnl_industry, "SPY": spy, **explore_pnls}
    df = pd.DataFrame(all_series).dropna()
    print(f"\nCommon window: {df.index[0]} -> {df.index[-1]}  ({len(df)} days)")

    # ── Stats on common window ────────────────────────────────────────────────
    rows = []
    print(f"\n{'Strategy':<28} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 72)
    for col in df.columns:
        s = compute_stats(df[col], col)
        rows.append(s)
        print(f"  {col:<26} {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  {s['calmar']:>7.3f}")

    # ── Correlations ──────────────────────────────────────────────────────────
    print(f"\n  Static correlations with Industry combined:")
    for col in df.columns:
        if col == "Industry combined": continue
        print(f"    {col:<25}  {df['Industry combined'].corr(df[col]):>+.3f}")

    # Rolling 90d
    roll = 90
    roll_corrs = {}
    for col in [c for c in df.columns if c not in ("Industry combined", "SPY")]:
        rc = df["Industry combined"].rolling(roll).corr(df[col]).dropna()
        roll_corrs[col] = rc
        print(f"    {col:<25}  90d mean={rc.mean():>+.3f}  min={rc.min():>+.3f}  max={rc.max():>+.3f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "stats_cross_corr_explore.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nSaving chart ...")
    plot_results(
        {col: df[col] for col in df.columns},
        roll_corrs,
        OUT_DIR / "corr_industry_vs_explore.png",
    )
    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

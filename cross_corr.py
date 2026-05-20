#!/usr/bin/env python3
"""
Correlation between industry ETF combined strategy and sector ETF combined strategy.
Re-runs both strategies, then computes static + rolling correlations vs each other and SPY.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
import yfinance as yf

HERE       = Path(__file__).parent
OUT_DIR    = HERE / "results_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END     = "2025-12-31"
MOM_WIN    = 252
REV_LB     = 5
COV_WIN    = 252
VOL_WIN    = 252
VIX_GATE   = 20

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def run_momentum(close, n_pick, min_stocks):
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
        if len(sig) < min_stocks:
            weights = pd.Series(0.0, index=close.columns); continue
        longs  = sig.nlargest(n_pick).index
        shorts = sig.nsmallest(n_pick).index
        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs]  = basket_erc(ret_win, longs,  +1.0)
        weights[shorts] = basket_erc(ret_win, shorts, -1.0)
    return pd.Series(dict(daily_pnl))


def run_reversal(close, vix, n_pick, min_stocks):
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
        if vix.get(prev, np.nan) <= VIX_GATE:
            weights = pd.Series(0.0, index=close.columns); continue
        sig = rev_sig.loc[prev].dropna()
        if len(sig) < min_stocks:
            weights = pd.Series(0.0, index=close.columns); continue
        longs = sig.nsmallest(n_pick).index
        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = basket_erc(ret_win, longs, +1.0)
    return pd.Series(dict(daily_pnl))


def combine_equal_vol(pnl_a, pnl_b):
    df      = pd.DataFrame({"a": pnl_a, "b": pnl_b}).dropna()
    inv_a   = 1.0 / df["a"].rolling(VOL_WIN).std().replace(0, np.nan)
    inv_b   = 1.0 / df["b"].rolling(VOL_WIN).std().replace(0, np.nan)
    total   = inv_a + inv_b
    combined = (inv_a / total * df["a"] + inv_b / total * df["b"]).dropna()
    return combined


def load_prices(parquet_path, tickers):
    close = pd.read_parquet(parquet_path)
    avail = [t for t in tickers if t in close.columns]
    close = close[avail][close.index <= IS_END].sort_index()
    return close.where(close.gt(0)).ffill()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # VIX and SPY
    print("Fetching VIX and SPY ...")
    vix_raw = yf.download("^VIX", start="2014-01-01", end=IS_END,
                          auto_adjust=True, progress=False)
    spy_raw = yf.download("SPY",  start="2014-01-01", end=IS_END,
                          auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    spy = spy_raw["Close"].squeeze().pct_change().dropna()
    spy.index = spy.index.tz_localize(None).strftime("%Y-%m-%d")

    # ── Industry ETF combined ─────────────────────────────────────────────────
    ind_tickers_raw = (HERE / "Industry_ETF_Tickers.csv").read_text(encoding="utf-8-sig").strip().splitlines()
    ind_tickers = [t.strip() for t in ind_tickers_raw if t.strip()]
    ind_close   = load_prices(HERE / "data" / "industry_etf_daily.parquet", ind_tickers)
    ind_first   = ind_close.index[MOM_WIN]

    print(f"\nIndustry ETF ({len(ind_tickers)} tickers, quartile = ~28 names/leg) ...")
    pnl_ind_mom = run_momentum(ind_close, n_pick=max(1, len(ind_tickers)//4), min_stocks=8)
    pnl_ind_mom = pnl_ind_mom[pnl_ind_mom.index >= ind_first]
    print(f"  Momentum done")

    pnl_ind_rev = run_reversal(ind_close, vix, n_pick=max(1, len(ind_tickers)//4), min_stocks=8)
    pnl_ind_rev = pnl_ind_rev[pnl_ind_rev.index >= ind_first]
    print(f"  Reversal done")

    pnl_industry = combine_equal_vol(pnl_ind_mom, pnl_ind_rev)
    print(f"  Combined: {pnl_industry.index[0]} -> {pnl_industry.index[-1]}  ({len(pnl_industry)} days)")

    # ── Sector ETF combined ───────────────────────────────────────────────────
    sec_tickers_raw = (HERE / "Sector_ETF_Tickers.csv").read_text(encoding="utf-8-sig").strip().splitlines()
    sec_tickers = [t.strip() for t in sec_tickers_raw if t.strip()]
    sec_close   = load_prices(HERE / "data" / "sector_etf_daily.parquet", sec_tickers)
    sec_first   = sec_close.index[MOM_WIN]

    print(f"\nSector ETF ({len(sec_tickers)} tickers, top/bottom 2 names) ...")
    pnl_sec_mom = run_momentum(sec_close, n_pick=2, min_stocks=6)
    pnl_sec_mom = pnl_sec_mom[pnl_sec_mom.index >= sec_first]
    print(f"  Momentum done")

    pnl_sec_rev = run_reversal(sec_close, vix, n_pick=2, min_stocks=6)
    pnl_sec_rev = pnl_sec_rev[pnl_sec_rev.index >= sec_first]
    print(f"  Reversal done")

    pnl_sector = combine_equal_vol(pnl_sec_mom, pnl_sec_rev)
    print(f"  Combined: {pnl_sector.index[0]} -> {pnl_sector.index[-1]}  ({len(pnl_sector)} days)")

    # ── Align all series ──────────────────────────────────────────────────────
    df = pd.DataFrame({
        "industry": pnl_industry,
        "sector":   pnl_sector,
        "spy":      spy,
    }).dropna()

    print(f"\nCommon window: {df.index[0]} -> {df.index[-1]}  ({len(df)} days)\n")

    # ── Stats ─────────────────────────────────────────────────────────────────
    def stats(pnl, label):
        cum    = (1 + pnl).cumprod()
        dd     = (cum - cum.cummax()) / cum.cummax()
        ann    = pnl.mean() * 252
        vol    = pnl.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else np.nan
        calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
        print(f"  {label:<28} {ann:>+8.2%}  {vol:>7.2%}  {sharpe:>7.3f}  "
              f"{dd.min():>+7.2%}  {calmar:>7.3f}")

    print(f"{'Strategy':<30} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 72)
    stats(df["industry"], "Industry combined")
    stats(df["sector"],   "Sector combined")
    stats(df["spy"],      "SPY")

    # ── Correlations ──────────────────────────────────────────────────────────
    print(f"\n  Static correlations (full common window):")
    print(f"    Industry vs Sector : {df['industry'].corr(df['sector']):+.3f}")
    print(f"    Industry vs SPY    : {df['industry'].corr(df['spy']):+.3f}")
    print(f"    Sector   vs SPY    : {df['sector'].corr(df['spy']):+.3f}")

    # Rolling 90d
    roll = 90
    rc_ind_sec = df["industry"].rolling(roll).corr(df["sector"]).dropna()
    rc_ind_spy = df["industry"].rolling(roll).corr(df["spy"]).dropna()
    rc_sec_spy = df["sector"].rolling(roll).corr(df["spy"]).dropna()

    print(f"\n  Rolling {roll}d correlation (mean / min / max):")
    for label, s in [
        ("Industry vs Sector", rc_ind_sec),
        ("Industry vs SPY",    rc_ind_spy),
        ("Sector   vs SPY",    rc_sec_spy),
    ]:
        print(f"    {label:<22}  mean={s.mean():+.3f}  min={s.min():+.3f}  max={s.max():+.3f}")

    # ── Equal-vol portfolio of both combined strategies ───────────────────────
    pnl_both = combine_equal_vol(df["industry"], df["sector"])
    print(f"\n  Equal-vol of both combined strategies:")
    cum    = (1 + pnl_both).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    ann    = pnl_both.mean() * 252
    vol    = pnl_both.std() * np.sqrt(252)
    sharpe = ann / vol
    print(f"    Ann Ret={ann:+.2%}  Ann Vol={vol:.2%}  Sharpe={sharpe:.3f}  MaxDD={dd.min():+.2%}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    for pnl, label, color, ls, lw in [
        (df["spy"],      "SPY",               "#7f7f7f", ":",  1.2),
        (df["industry"], "Industry combined", "#1f77b4", "--", 1.5),
        (df["sector"],   "Sector combined",   "#ff7f0e", "--", 1.5),
        (pnl_both,       "Both combined (EV)","#2ca02c", "-",  2.0),
    ]:
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                color=color, linestyle=ls, linewidth=lw, label=label)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        "Industry ETF Combined vs Sector ETF Combined  |  Equal-vol of both  |  IS overlap window\n"
        "Each combined = ERC Momentum (L/S) + ERC Reversal (long-only, VIX>20)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(pd.to_datetime(rc_ind_sec.index), rc_ind_sec,
             color="#9467bd", linewidth=1.4, label="Industry vs Sector (90d rolling)")
    ax2.axhline(0,                   color="black",   linewidth=0.7, linestyle="--")
    ax2.axhline(rc_ind_sec.mean(),   color="#9467bd", linewidth=1.0, linestyle=":",
                label=f"Mean {rc_ind_sec.mean():+.3f}")
    ax2.set_ylabel("Rolling 90d Correlation")
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel("Date")

    fig.tight_layout()
    path = OUT_DIR / "corr_industry_vs_sector.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  Saved -> {path}")


if __name__ == "__main__":
    main()

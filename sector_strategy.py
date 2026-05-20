#!/usr/bin/env python3
"""
Sector ETF strategy — 10 SPDR Select Sector ETFs.

Leg 1 — ERC Momentum   : 252d signal, top 2 long / bottom 2 short, daily rebal
Leg 2 — ERC Reversal   : 5d signal, long bottom 2, VIX > 20 gated, daily rebal

Combined via rolling 252d equal-vol weights.
Also reports each leg standalone and SPY benchmark.
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
TICKER_CSV = HERE / "Sector_ETF_Tickers.csv"
DATA_FILE  = HERE / "data" / "sector_etf_daily.parquet"
OUT_DIR    = HERE / "results_sector"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END     = "2025-12-31"
MOM_WIN    = 252
REV_LB     = 5
COV_WIN    = 252
VOL_WIN    = 252
VIX_GATE   = 20
N_PICK     = 2      # top/bottom N names per basket
MIN_STOCKS = 6      # minimum valid tickers to trade

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_tickers():
    raw = TICKER_CSV.read_text(encoding="utf-8-sig").strip().splitlines()
    return [t.strip() for t in raw if t.strip()]


def download_prices(tickers) -> pd.DataFrame:
    print(f"Downloading {len(tickers)} sector ETFs ...")
    raw = yf.download(tickers, start="2014-01-01", end=IS_END,
                      auto_adjust=True, progress=True, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = close.index.tz_localize(None)
    close.index = close.index.strftime("%Y-%m-%d")
    close.index.name = "date"
    close.sort_index(inplace=True)
    close.to_parquet(DATA_FILE)
    print(f"  Saved -> {DATA_FILE}")
    return close


def load_prices(tickers) -> pd.DataFrame:
    if DATA_FILE.exists():
        close = pd.read_parquet(DATA_FILE)
        avail = [t for t in tickers if t in close.columns]
        if len(avail) == len(tickers):
            close = close[avail][close.index <= IS_END].sort_index()
            return close.where(close.gt(0)).ffill()
    return download_prices(tickers)


def load_vix() -> pd.Series:
    raw = yf.download("^VIX", start="2014-01-01", end=IS_END,
                      auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    return vix


def load_spy() -> pd.Series:
    raw = yf.download("SPY", start="2014-01-01", end=IS_END,
                      auto_adjust=True, progress=False)
    spy = raw["Close"].squeeze().pct_change().dropna()
    spy.index = spy.index.tz_localize(None).strftime("%Y-%m-%d")
    return spy


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


# ── Leg 1: ERC Momentum (top 2 / bottom 2, always-on) ────────────────────────

def run_momentum(close):
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

        longs  = sig.nlargest(N_PICK).index
        shorts = sig.nsmallest(N_PICK).index

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        weights = pd.Series(0.0, index=close.columns)
        weights[longs]  = basket_erc(ret_win, longs,  +1.0)
        weights[shorts] = basket_erc(ret_win, shorts, -1.0)

    return pd.Series(dict(daily_pnl), name="momentum")


# ── Leg 2: ERC Reversal (bottom 2 long-only, VIX > 20 gated) ─────────────────

def run_reversal(close, vix):
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

        prev_vix = vix.get(prev, np.nan)
        if np.isnan(prev_vix) or prev_vix <= VIX_GATE:
            weights = pd.Series(0.0, index=close.columns)
            continue

        sig = rev_sig.loc[prev].dropna()
        if len(sig) < MIN_STOCKS:
            weights = pd.Series(0.0, index=close.columns)
            continue

        longs = sig.nsmallest(N_PICK).index

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = basket_erc(ret_win, longs, +1.0)

    return pd.Series(dict(daily_pnl), name="reversal")


# ── Equal-vol combination ─────────────────────────────────────────────────────

def combine_equal_vol(pnl_mom, pnl_rev):
    df = pd.DataFrame({"mom": pnl_mom, "rev": pnl_rev}).dropna()

    vol_mom = df["mom"].rolling(VOL_WIN).std() * np.sqrt(252)
    vol_rev = df["rev"].rolling(VOL_WIN).std() * np.sqrt(252)

    inv_mom = 1.0 / vol_mom.replace(0, np.nan)
    inv_rev = 1.0 / vol_rev.replace(0, np.nan)
    total   = inv_mom + inv_rev

    w_mom = inv_mom / total
    w_rev = inv_rev / total

    combined = (w_mom * df["mom"] + w_rev * df["rev"]).dropna()
    return combined, w_mom.reindex(combined.index), w_rev.reindex(combined.index)


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(pnl, label=""):
    pnl    = pnl.dropna()
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
    pnl   = pnl.dropna()
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

def plot_combined(pnl_mom, pnl_rev, pnl_comb, pnl_spy, w_mom, w_rev):
    idx = pnl_comb.index
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    for pnl, label, color, lw, ls in [
        (pnl_spy.reindex(idx).fillna(0),  "SPY (buy & hold)",                    "#7f7f7f", 1.2, ":"),
        (pnl_mom.reindex(idx).fillna(0),  f"ERC Momentum (top{N_PICK}/bot{N_PICK} L/S)",  "#1f77b4", 1.3, "--"),
        (pnl_rev.reindex(idx).fillna(0),  f"ERC Reversal (bot{N_PICK}, VIX>{VIX_GATE})", "#ff7f0e", 1.3, "--"),
        (pnl_comb,                         "Combined (Equal-Vol)",                 "#2ca02c", 2.0, "-"),
    ]:
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                linewidth=lw, label=label, color=color, linestyle=ls)

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        f"Sector ETF Strategy  |  Top/Bottom {N_PICK} names  |  IS: 2015–2025\n"
        f"ERC Momentum (252d L/S)  +  ERC Reversal (5d long-only, VIX>{VIX_GATE})  |  Equal-vol combined",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.stackplot(
        pd.to_datetime(idx),
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
    path = OUT_DIR / "cumret_sector_combined.png"
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
    print(f"Universe: {len(tickers)} sector ETFs: {tickers}")

    close = load_prices(tickers)
    print(f"  {len(close.columns)} tickers  |  {close.index[0]} -> {close.index[-1]}  ({len(close):,} days)")

    print("Fetching VIX and SPY ...")
    vix = load_vix()
    spy = load_spy()

    first_sig = close.index[MOM_WIN]
    vix_pct   = (vix[vix.index >= first_sig] > VIX_GATE).mean()
    print(f"  Effective IS window: {first_sig} -> {close.index[-1]}")
    print(f"  VIX > {VIX_GATE}: {vix_pct:.1%} of IS days\n")

    # ── Run legs ──────────────────────────────────────────────────────────────
    print(f"Running ERC Momentum (top{N_PICK}/bot{N_PICK} L/S, always-on) ...")
    pnl_mom = run_momentum(close)
    pnl_mom = pnl_mom[pnl_mom.index >= first_sig]

    print(f"Running ERC Reversal (bot{N_PICK} long-only, VIX>{VIX_GATE} gated) ...")
    pnl_rev = run_reversal(close, vix)
    pnl_rev = pnl_rev[pnl_rev.index >= first_sig]

    print(f"Combining with rolling {VOL_WIN}d equal-vol weights ...")
    pnl_comb, w_mom, w_rev = combine_equal_vol(pnl_mom, pnl_rev)

    # ── SPY benchmark over same window ────────────────────────────────────────
    pnl_spy = spy.reindex(pnl_comb.index)

    # ── Stats table ───────────────────────────────────────────────────────────
    rows = []
    idx  = pnl_comb.index
    print(f"\n{'Strategy':<38} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'WinRate':>7}")
    print("-" * 88)
    for pnl, label in [
        (pnl_spy.dropna(),                              "SPY buy & hold"),
        (pnl_mom.reindex(idx).dropna(),                f"ERC Momentum top{N_PICK}/bot{N_PICK} L/S"),
        (pnl_rev.reindex(idx).dropna(),                f"ERC Reversal bot{N_PICK} VIX>{VIX_GATE}"),
        (pnl_comb,                                      "Combined Equal-Vol"),
    ]:
        s = compute_stats(pnl, label)
        rows.append(s)
        print(f"  {label:<36} {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  "
              f"{s['calmar']:>7.3f}  {s['win_rate']:>7.1%}")

    print(f"\n  Avg allocation:  Momentum = {w_mom.mean():.1%}   Reversal = {w_rev.mean():.1%}")

    # SPY correlation
    df_corr = pd.DataFrame({
        "Combined":  pnl_comb,
        "Momentum":  pnl_mom.reindex(idx),
        "Reversal":  pnl_rev.reindex(idx),
        "SPY":       pnl_spy,
    }).dropna()
    print(f"\n  Static correlation with SPY:")
    for col in ["Combined", "Momentum", "Reversal"]:
        print(f"    {col:<12}  {df_corr[col].corr(df_corr['SPY']):+.3f}")

    # Pairwise
    print(f"  Momentum vs Reversal:  {df_corr['Momentum'].corr(df_corr['Reversal']):+.3f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "stats_sector.csv", index=False)

    # ── Monthly tables ────────────────────────────────────────────────────────
    for pnl, label in [
        (pnl_comb,                         "Combined"),
        (pnl_mom.reindex(idx).dropna(),    "Momentum"),
        (pnl_rev.reindex(idx).dropna(),    "Reversal"),
        (pnl_spy.dropna(),                 "SPY"),
    ]:
        print(f"\n--- {label} Monthly Returns ---")
        print((monthly_returns(pnl) * 100).round(1).to_string())

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nSaving charts ...")
    plot_combined(pnl_mom, pnl_rev, pnl_comb, pnl_spy, w_mom, w_rev)
    for pnl, title, fname in [
        (pnl_comb,
         f"Sector ETF  |  Combined Equal-Vol  |  IS: 2015–2025",
         "monthly_sector_combined.png"),
        (pnl_mom.reindex(idx).dropna(),
         f"Sector ETF  |  ERC Momentum top{N_PICK}/bot{N_PICK}  |  IS: 2015–2025",
         "monthly_sector_momentum.png"),
        (pnl_rev.reindex(idx).dropna(),
         f"Sector ETF  |  ERC Reversal bot{N_PICK} VIX>{VIX_GATE}  |  IS: 2015–2025",
         "monthly_sector_reversal.png"),
    ]:
        plot_heatmap(pnl, title, fname)

    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cluster-ERC momentum backtest vs standard ERC.

Sizing logic:
  1. Hierarchical clustering on rolling 252d correlation matrix of the long/short basket
  2. Each cluster gets equal notional: 1/n_clusters of the leg
  3. Within each cluster, ERC weights (normalized to sum to cluster allocation)
  4. Long book sums to +1, short book sums to -1 (dollar-neutral)

Compares N_LONG_CLUSTERS x N_SHORT_CLUSTERS grid vs standard ERC quartile L/S.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sco
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE      = Path(__file__).parent
OUT_DIR   = HERE / "results_cluster"
OUT_DIR.mkdir(exist_ok=True)

IS_END    = "2025-12-31"
MOM_WIN   = 252
COV_WIN   = 252
VOL_WIN   = 252
VIX_GATE  = 20
REV_LB    = 5

N_LONG_CLUSTERS  = 3
N_SHORT_CLUSTERS = 4


# ── ERC helpers ───────────────────────────────────────────────────────────────

def _sample_cov(ret_win, tickers):
    C = ret_win[tickers].cov().values
    return C + np.eye(len(C)) * 1e-8

def opt_erc(ret_win, tickers):
    n, C = len(tickers), _sample_cov(ret_win, tickers)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: 0.5 * w @ C @ w - (1/n) * np.sum(np.log(np.maximum(w, 1e-12))),
        x0=w0, jac=lambda w: C @ w - (1/n) / np.maximum(w, 1e-12),
        method="L-BFGS-B", bounds=[(1e-6, 1.0)] * n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w / w.sum()

def basket_erc(ret_win, tickers, alloc):
    """ERC weights for tickers, scaled so they sum to alloc (signed)."""
    sign = np.sign(alloc)
    if len(tickers) == 1:
        return pd.Series([alloc], index=tickers)
    try:
        w = opt_erc(ret_win, tickers)
    except Exception:
        w = np.ones(len(tickers)) / len(tickers)
    return pd.Series(w * alloc, index=tickers)


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_tickers(ret_win, tickers, n_clusters):
    """
    Hierarchical clustering (Ward, correlation distance) → cluster labels.
    Returns dict: cluster_id → [ticker list].
    """
    if len(tickers) <= n_clusters:
        return {i: [t] for i, t in enumerate(tickers)}

    corr = ret_win[tickers].corr().fillna(0).clip(-1, 1)
    # correlation distance: sqrt(0.5 * (1 - rho)), ensures triangle inequality
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)

    condensed = squareform(dist, checks=False)
    Z = sch.linkage(condensed, method="ward")
    labels = sch.fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters = {}
    for ticker, label in zip(tickers, labels):
        clusters.setdefault(int(label), []).append(ticker)
    return clusters


def cluster_erc_weights(ret_win, tickers, sign, n_clusters):
    """
    Equal-cluster-notional ERC weights.
    sign = +1 for longs, -1 for shorts.
    Returns pd.Series of weights summing to sign * 1.0.
    """
    clusters = cluster_tickers(ret_win, tickers, n_clusters)
    n_cl = len(clusters)
    alloc_per_cluster = sign * 1.0 / n_cl

    weights = pd.Series(0.0, index=tickers)
    for _, members in clusters.items():
        w = basket_erc(ret_win, members, alloc_per_cluster)
        weights[w.index] = w.values
    return weights


# ── Standard ERC basket (original approach) ───────────────────────────────────

def basket_erc_standard(ret_win, tickers, sign):
    try:
        w = opt_erc(ret_win, tickers)
    except Exception:
        w = np.ones(len(tickers)) / len(tickers)
    return pd.Series(w * sign, index=tickers)


# ── Backtest runners ──────────────────────────────────────────────────────────

def run_momentum(close, use_cluster=False,
                 n_long=N_LONG_CLUSTERS, n_short=N_SHORT_CLUSTERS):
    mom_sig   = close.pct_change(MOM_WIN)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}
    weights   = pd.Series(0.0, index=close.columns)
    pnl       = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))

        sig = mom_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns)
            continue

        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index.tolist()
        shorts = sig[ranked > len(sig) - n_q].index.tolist()

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        weights = pd.Series(0.0, index=close.columns)
        if use_cluster:
            weights[longs]  = cluster_erc_weights(ret_win, longs,  +1.0, n_long)
            weights[shorts] = cluster_erc_weights(ret_win, shorts, -1.0, n_short)
        else:
            weights[longs]  = basket_erc_standard(ret_win, longs,  +1.0)
            weights[shorts] = basket_erc_standard(ret_win, shorts, -1.0)

    return pd.Series(dict(pnl))


def run_reversal(close, vix, use_cluster=False, n_long=N_LONG_CLUSTERS):
    rev_sig   = close.pct_change(REV_LB)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}
    weights   = pd.Series(0.0, index=close.columns)
    pnl       = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))

        if vix.get(prev, np.nan) <= VIX_GATE:
            weights = pd.Series(0.0, index=close.columns)
            continue
        sig = rev_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns)
            continue

        n_q   = max(1, len(sig) // 4)
        longs = sig.nsmallest(n_q).index.tolist()
        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        weights = pd.Series(0.0, index=close.columns)
        if use_cluster:
            weights[longs] = cluster_erc_weights(ret_win, longs, +1.0, n_long)
        else:
            weights[longs] = basket_erc_standard(ret_win, longs, +1.0)

    return pd.Series(dict(pnl))


def combine_ev(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    ia = 1.0 / df["a"].rolling(VOL_WIN).std().replace(0, np.nan)
    ib = 1.0 / df["b"].rolling(VOL_WIN).std().replace(0, np.nan)
    t  = ia + ib
    return ((ia / t) * df["a"] + (ib / t) * df["b"]).dropna()


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(pnl, label=""):
    pnl = pnl.dropna()
    cum    = (1 + pnl).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    ann    = pnl.mean() * 252
    vol    = pnl.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    return dict(label=label, ann_ret=ann, ann_vol=vol, sharpe=sharpe,
                max_dd=dd.min(), calmar=calmar)


# ── Cluster diagnostics ───────────────────────────────────────────────────────

def show_clusters(ret_win, tickers, n_clusters, side_label):
    clusters = cluster_tickers(ret_win, tickers, n_clusters)
    print(f"\n  {side_label} clusters ({n_clusters}):")
    for cid, members in sorted(clusters.items()):
        print(f"    [{cid}] {', '.join(sorted(members))}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(series_dict, title, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    colors = {
        "ERC (original)":    ("#1f77b4", "--", 1.5),
        "Cluster-ERC":       ("#2ca02c", "-",  2.0),
        "SPY":               ("#7f7f7f", ":",  1.2),
    }

    ax = axes[0]
    for label, pnl in series_dict.items():
        c, ls, lw = colors.get(label, ("#ff7f0e", "-", 1.4))
        cum = (1 + pnl.dropna()).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                color=c, ls=ls, lw=lw, label=label)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Drawdown panel
    ax2 = axes[1]
    for label, pnl in series_dict.items():
        if label == "SPY":
            continue
        c, ls, lw = colors.get(label, ("#ff7f0e", "-", 1.4))
        cum = (1 + pnl.dropna()).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax2.plot(pd.to_datetime(dd.index), dd * 100,
                 color=c, ls=ls, lw=lw, label=label)
    ax2.axhline(0, color="black", lw=0.6, ls=":")
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ──────────────────────────────────────────────────────────
    print("Fetching VIX and SPY ...")
    vix_raw = yf.download("^VIX", start="2014-01-01", end=IS_END, auto_adjust=True, progress=False)
    spy_raw = yf.download("SPY",  start="2014-01-01", end=IS_END, auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    spy = spy_raw["Close"].squeeze().pct_change().dropna()
    spy.index = spy.index.tz_localize(None).strftime("%Y-%m-%d")

    _tf = HERE / "Industry_ETF_Tickers_filtered.csv"
    if not _tf.exists():
        _tf = HERE / "Industry_ETF_Tickers.csv"
    ind_tickers = [t.strip() for t in _tf.read_text(encoding="utf-8-sig").strip().splitlines() if t.strip()]
    close = pd.read_parquet(HERE / "data" / "industry_etf_daily.parquet")
    close = close[[t for t in ind_tickers if t in close.columns]].sort_index()
    close = close[close.index <= IS_END].sort_index()
    close = close.where(close.gt(0)).ffill()
    ind_first = close.index[MOM_WIN]

    # ── Run standard ERC ───────────────────────────────────────────────────
    print("\nRunning standard ERC (original) ...")
    pnl_erc_mom = run_momentum(close, use_cluster=False)
    pnl_erc_rev = run_reversal(close, vix, use_cluster=False)
    pnl_erc_mom = pnl_erc_mom[pnl_erc_mom.index >= ind_first]
    pnl_erc_rev = pnl_erc_rev[pnl_erc_rev.index >= ind_first]
    pnl_erc = combine_ev(pnl_erc_mom, pnl_erc_rev)

    # ── Run Cluster-ERC ────────────────────────────────────────────────────
    print(f"\nRunning Cluster-ERC "
          f"({N_LONG_CLUSTERS} long clusters / {N_SHORT_CLUSTERS} short clusters) ...")
    pnl_cl_mom = run_momentum(close, use_cluster=True,
                              n_long=N_LONG_CLUSTERS, n_short=N_SHORT_CLUSTERS)
    pnl_cl_rev = run_reversal(close, vix, use_cluster=True, n_long=N_LONG_CLUSTERS)
    pnl_cl_mom = pnl_cl_mom[pnl_cl_mom.index >= ind_first]
    pnl_cl_rev = pnl_cl_rev[pnl_cl_rev.index >= ind_first]
    pnl_cl = combine_ev(pnl_cl_mom, pnl_cl_rev)

    # ── Align on common window ─────────────────────────────────────────────
    df = pd.DataFrame({
        "ERC (original)": pnl_erc,
        "Cluster-ERC":    pnl_cl,
        "SPY":            spy,
    }).dropna()
    print(f"\nCommon window: {df.index[0]} -> {df.index[-1]}  ({len(df)} days)")

    # ── Stats ──────────────────────────────────────────────────────────────
    print(f"\n{'Strategy':<28} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Calmar':>8}")
    print("-" * 72)
    for col in df.columns:
        s = compute_stats(df[col], col)
        print(f"  {col:<26} {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  {s['calmar']:>7.3f}")

    # ── SPY correlation ────────────────────────────────────────────────────
    print(f"\n  SPY correlations:")
    for col in ["ERC (original)", "Cluster-ERC"]:
        print(f"    {col:<26}  {df[col].corr(df['SPY']):+.3f}")

    # ── Momentum-only comparison ───────────────────────────────────────────
    df_mom = pd.DataFrame({
        "ERC Mom (original)": pnl_erc_mom,
        "Cluster-ERC Mom":    pnl_cl_mom,
    }).dropna()
    print(f"\n  Momentum leg only:")
    print(f"{'Strategy':<28} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 60)
    for col in df_mom.columns:
        s = compute_stats(df_mom[col], col)
        print(f"  {col:<26} {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}")

    # ── Annual breakdown ───────────────────────────────────────────────────
    print(f"\n  Annual returns — combined:")
    print(f"  {'Year':<6} {'ERC':>10} {'Cluster-ERC':>13}")
    print("  " + "-" * 32)
    for yr in sorted(df.index.str[:4].unique()):
        mask = df.index.str.startswith(yr)
        r_erc = (1 + df.loc[mask, "ERC (original)"]).prod() - 1
        r_cl  = (1 + df.loc[mask, "Cluster-ERC"]).prod() - 1
        print(f"  {yr}   {r_erc:>+9.1%}   {r_cl:>+9.1%}")

    # ── Show latest clusters ───────────────────────────────────────────────
    print(f"\n── Latest cluster snapshot (based on last 252d returns) ─────────")
    ret_win_last = close.pct_change().iloc[-COV_WIN:]
    mom_sig_last = close.pct_change(MOM_WIN).iloc[-1].dropna()
    n_q = max(1, len(mom_sig_last) // 4)
    ranked = mom_sig_last.rank(ascending=False)
    longs_last  = mom_sig_last[ranked <= n_q].index.tolist()
    shorts_last = mom_sig_last[ranked > len(mom_sig_last) - n_q].index.tolist()
    show_clusters(ret_win_last, longs_last,  N_LONG_CLUSTERS,  "Long")
    show_clusters(ret_win_last, shorts_last, N_SHORT_CLUSTERS, "Short")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nSaving plots ...")
    plot_comparison(
        {col: df[col] for col in df.columns},
        f"ERC vs Cluster-ERC Combined  |  {N_LONG_CLUSTERS}L / {N_SHORT_CLUSTERS}S clusters",
        OUT_DIR / f"cluster_erc_combined_L{N_LONG_CLUSTERS}_S{N_SHORT_CLUSTERS}.png",
    )
    plot_comparison(
        {col: df_mom[col] for col in df_mom.columns},
        f"ERC vs Cluster-ERC Momentum only  |  {N_LONG_CLUSTERS}L / {N_SHORT_CLUSTERS}S clusters",
        OUT_DIR / f"cluster_erc_momentum_L{N_LONG_CLUSTERS}_S{N_SHORT_CLUSTERS}.png",
    )

    # ── Save stats ─────────────────────────────────────────────────────────
    rows = [compute_stats(df[c], c) for c in df.columns]
    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"stats_cluster_L{N_LONG_CLUSTERS}_S{N_SHORT_CLUSTERS}.csv", index=False)


if __name__ == "__main__":
    main()

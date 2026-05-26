#!/usr/bin/env python3
"""
Cluster vol-target momentum backtest.

Sizing rules:
  1. Hierarchically cluster the long and short baskets separately.
  2. Size each cluster to $1000 1-day 1-sigma using rolling 252d covariance.
  3. Active clusters per side = min(n_long_clusters, n_short_clusters).
     If one side has more clusters, drop the weakest (lowest mean momentum
     signal on long side, highest mean signal on short side).
  4. Total portfolio 1d vol target ≈ n_active × $1000 long + n_active × $1000 short.

P&L is reported in dollars. Sharpe is scale-invariant.
Compared against ERC (original) rescaled to same ex-ante vol for fair comparison.
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

HERE    = Path(__file__).parent
OUT_DIR = HERE / "results_cluster"
OUT_DIR.mkdir(exist_ok=True)

IS_END         = "2025-12-31"
MOM_WIN        = 252
COV_WIN        = 252
VOL_WIN        = 252
VIX_GATE       = 20
REV_LB         = 5
CLUSTER_VOL    = 1_000.0   # $ 1-day 1-sigma per cluster
N_LONG_CLUST   = 3
N_SHORT_CLUST  = 4


# ── ERC ───────────────────────────────────────────────────────────────────────

def _cov(ret_win, tickers):
    C = ret_win[tickers].cov().values
    return C + np.eye(len(C)) * 1e-8

def opt_erc(ret_win, tickers):
    n, C = len(tickers), _cov(ret_win, tickers)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: 0.5 * w @ C @ w - (1/n) * np.sum(np.log(np.maximum(w, 1e-12))),
        x0=w0, jac=lambda w: C @ w - (1/n) / np.maximum(w, 1e-12),
        method="L-BFGS-B", bounds=[(1e-6, 1.0)] * n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w / w.sum()


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_tickers(ret_win, tickers, n_clusters):
    """Ward hierarchical clustering on correlation distance."""
    if len(tickers) <= n_clusters:
        return {i: [t] for i, t in enumerate(tickers)}
    corr = ret_win[tickers].corr().fillna(0).clip(-1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr.values), 0, None))
    np.fill_diagonal(dist, 0)
    Z      = sch.linkage(squareform(dist, checks=False), method="ward")
    labels = sch.fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = {}
    for ticker, label in zip(tickers, labels):
        clusters.setdefault(int(label), []).append(ticker)
    return clusters


# ── Vol-target sizing ─────────────────────────────────────────────────────────

def size_cluster(ret_win, members, target_vol, sign):
    """
    Dollar weights for a cluster sized to target_vol 1-day 1-sigma.
    sign = +1 (long) or -1 (short).
    Returns pd.Series of dollar weights.
    """
    if len(members) == 1:
        s  = ret_win[members[0]].dropna()
        sv = float(s.std()) if len(s) > 1 else 0.01
        return pd.Series([sign * target_vol / max(sv, 1e-6)], index=members)
    try:
        w = opt_erc(ret_win, members)
    except Exception:
        w = np.ones(len(members)) / len(members)
    C      = _cov(ret_win, members)
    sig_c  = float(np.sqrt(np.maximum(w @ C @ w, 1e-12)))
    notional = target_vol / sig_c          # dollar notional so cluster vol = target_vol
    return pd.Series(sign * notional * w, index=members)


def build_book(ret_win, longs, shorts, sig, n_long_cl, n_short_cl):
    """
    Build the full dollar-weight book.
    Returns pd.Series of dollar weights (positive = long, negative = short).
    """
    # cluster each side
    long_clusters  = cluster_tickers(ret_win, longs,  n_long_cl)
    short_clusters = cluster_tickers(ret_win, shorts, n_short_cl)

    n_l = len(long_clusters)
    n_s = len(short_clusters)
    n_active = min(n_l, n_s)

    # if one side has more clusters, drop the weakest
    def trim_clusters(clusters, tickers_sig, keep_n, ascending):
        """Drop clusters with the weakest signal until keep_n remain."""
        if len(clusters) <= keep_n:
            return clusters
        # score each cluster by mean signal of its members
        scores = {cid: tickers_sig.reindex(members).mean()
                  for cid, members in clusters.items()}
        # ascending=True → drop lowest (for longs keep strongest momentum)
        # ascending=False → drop highest (for shorts keep most negative)
        ordered = sorted(scores, key=scores.__getitem__, reverse=not ascending)
        keep = set(ordered[:keep_n])
        return {cid: clusters[cid] for cid in keep}

    long_clusters  = trim_clusters(long_clusters,  sig, n_active, ascending=False)
    short_clusters = trim_clusters(short_clusters, sig, n_active, ascending=True)

    weights = {}
    for members in long_clusters.values():
        w = size_cluster(ret_win, members, CLUSTER_VOL, +1)
        weights.update(w.to_dict())
    for members in short_clusters.values():
        w = size_cluster(ret_win, members, CLUSTER_VOL, -1)
        weights.update(w.to_dict())

    return pd.Series(weights), n_active


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_cluster_vt_momentum(close, n_long_cl=N_LONG_CLUST, n_short_cl=N_SHORT_CLUST):
    mom_sig   = close.pct_change(MOM_WIN)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}

    # dollar weights (per $1 move in the instrument)
    dol_weights  = pd.Series(0.0, index=close.columns)
    pnl          = []          # daily $ P&L
    n_active_log = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        # dollar P&L = sum of (dollar_weight × return)
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((dol_weights * ret).sum())))

        sig = mom_sig.loc[prev].dropna()
        if len(sig) < 8:
            dol_weights = pd.Series(0.0, index=close.columns)
            n_active_log.append((date, 0))
            continue

        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index.tolist()
        shorts = sig[ranked > len(sig) - n_q].index.tolist()

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        try:
            book, n_act = build_book(ret_win, longs, shorts, sig, n_long_cl, n_short_cl)
            dol_weights = pd.Series(0.0, index=close.columns)
            dol_weights.update(book)
        except Exception:
            dol_weights = pd.Series(0.0, index=close.columns)
            n_act = 0

        n_active_log.append((date, n_act))

    pnl_s      = pd.Series(dict(pnl))
    n_active_s = pd.Series(dict(n_active_log))
    return pnl_s, n_active_s


def run_cluster_vt_reversal(close, vix, n_long_cl=N_LONG_CLUST):
    rev_sig   = close.pct_change(REV_LB)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}

    dol_weights = pd.Series(0.0, index=close.columns)
    pnl         = []

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((dol_weights * ret).sum())))

        if vix.get(prev, np.nan) <= VIX_GATE:
            dol_weights = pd.Series(0.0, index=close.columns)
            continue

        sig = rev_sig.loc[prev].dropna()
        if len(sig) < 8:
            dol_weights = pd.Series(0.0, index=close.columns)
            continue

        n_q   = max(1, len(sig) // 4)
        longs = sig.nsmallest(n_q).index.tolist()

        t_idx   = date_pos[prev]
        ret_win = daily_ret.iloc[max(0, t_idx - COV_WIN + 1) : t_idx + 1]

        try:
            long_clusters = cluster_tickers(ret_win, longs, n_long_cl)
            dol_weights   = pd.Series(0.0, index=close.columns)
            for members in long_clusters.values():
                w = size_cluster(ret_win, members, CLUSTER_VOL, +1)
                dol_weights.update(w)
        except Exception:
            dol_weights = pd.Series(0.0, index=close.columns)

    return pd.Series(dict(pnl))


def combine_ev_dollar(a, b):
    """Equal-vol combination keeping dollar P&L units."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    ia = 1.0 / df["a"].rolling(VOL_WIN).std().replace(0, np.nan)
    ib = 1.0 / df["b"].rolling(VOL_WIN).std().replace(0, np.nan)
    t  = ia + ib
    return ((ia / t) * df["a"] + (ib / t) * df["b"]).dropna()


# ── Stats (dollar P&L) ────────────────────────────────────────────────────────

def compute_stats(pnl, label=""):
    pnl  = pnl.dropna()
    cum  = pnl.cumsum()
    roll_max = cum.cummax()
    dd   = cum - roll_max
    ann  = pnl.mean() * 252
    vol  = pnl.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    return dict(label=label, ann_pnl=ann, ann_vol=vol, sharpe=sharpe,
                max_dd=dd.min(), calmar=calmar)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(series_dict, title, path, dollar=True):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    styles = {
        "ERC (rescaled)":           ("#1f77b4", "--", 1.5),
        "Cluster Vol-Target":       ("#2ca02c", "-",  2.0),
        "Cluster Vol-Target Mom":   ("#2ca02c", "-",  1.6),
        "ERC Mom (rescaled)":       ("#1f77b4", "--", 1.5),
    }

    ax = axes[0]
    for label, s in series_dict.items():
        c, ls, lw = styles.get(label, ("#ff7f0e", "-", 1.4))
        cum = s.dropna().cumsum() if dollar else (1 + s.dropna()).cumprod() - 1
        scale = 1 if dollar else 100
        ax.plot(pd.to_datetime(s.index), cum * scale,
                color=c, ls=ls, lw=lw, label=label)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("Cumulative P&L ($)" if dollar else "Cumulative Return (%)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if dollar:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax2 = axes[1]
    for label, s in series_dict.items():
        c, ls, lw = styles.get(label, ("#ff7f0e", "-", 1.4))
        cum = s.dropna().cumsum()
        dd  = cum - cum.cummax()
        ax2.plot(pd.to_datetime(s.index), dd,
                 color=c, ls=ls, lw=lw, label=label)
    ax2.axhline(0, color="black", lw=0.6, ls=":")
    ax2.set_ylabel("Drawdown ($)")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Standard ERC runners (% return, for comparison baseline) ─────────────────

def _run_erc_momentum(close, vix):
    mom_sig   = close.pct_change(MOM_WIN)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}
    weights   = pd.Series(0.0, index=close.columns)
    pnl = []
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))
        sig = mom_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns); continue
        n_q    = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index
        shorts = sig[ranked > len(sig) - n_q].index
        ret_win = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1):date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        try:
            weights[longs]  = pd.Series(opt_erc(ret_win, longs)  * +1.0, index=longs)
            weights[shorts] = pd.Series(opt_erc(ret_win, shorts) * -1.0, index=shorts)
        except Exception:
            pass
    return pd.Series(dict(pnl))

def _run_erc_reversal(close, vix):
    rev_sig   = close.pct_change(REV_LB)
    daily_ret = close.pct_change()
    dates     = close.index.tolist()
    date_pos  = {d: i for i, d in enumerate(dates)}
    weights   = pd.Series(0.0, index=close.columns)
    pnl = []
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))
        if vix.get(prev, np.nan) <= VIX_GATE:
            weights = pd.Series(0.0, index=close.columns); continue
        sig = rev_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns); continue
        n_q   = max(1, len(sig) // 4)
        longs = sig.nsmallest(n_q).index
        ret_win = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1):date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        try:
            weights[longs] = pd.Series(opt_erc(ret_win, longs) * +1.0, index=longs)
        except Exception:
            pass
    return pd.Series(dict(pnl))


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

    # ── Cluster vol-target ─────────────────────────────────────────────────
    print(f"\nRunning Cluster Vol-Target momentum "
          f"({N_LONG_CLUST}L / {N_SHORT_CLUST}S, ${CLUSTER_VOL:,.0f}/cluster) ...")
    pnl_vt_mom, n_active = run_cluster_vt_momentum(close, N_LONG_CLUST, N_SHORT_CLUST)
    pnl_vt_mom = pnl_vt_mom[pnl_vt_mom.index >= ind_first]
    n_active   = n_active[n_active.index >= ind_first]

    print(f"Running Cluster Vol-Target reversal ...")
    pnl_vt_rev = run_cluster_vt_reversal(close, vix, N_LONG_CLUST)
    pnl_vt_rev = pnl_vt_rev[pnl_vt_rev.index >= ind_first]

    pnl_vt = combine_ev_dollar(pnl_vt_mom, pnl_vt_rev)

    # ── Original ERC (rescaled to same ex-ante vol for comparison) ─────────
    # Reload standard ERC from combined_vix_gated output (re-run inline)
    print(f"\nRunning standard ERC for comparison ...")
    _pnl_erc_mom = _run_erc_momentum(close, vix)
    _pnl_erc_rev = _run_erc_reversal(close, vix)
    _pnl_erc     = combine_ev_dollar(_pnl_erc_mom, _pnl_erc_rev)

    # Rescale ERC so it has the same annualized vol as the vol-target strategy
    # (so Sharpe comparisons are fair, and dollar chart is on same scale)
    common = pd.DataFrame({"vt": pnl_vt, "erc": _pnl_erc}).dropna()
    scale  = common["vt"].std() / common["erc"].std()
    pnl_erc_scaled     = common["erc"] * scale
    pnl_erc_mom_scaled = _pnl_erc_mom.reindex(common.index) * scale
    pnl_vt_c           = common["vt"]

    # ── Stats ──────────────────────────────────────────────────────────────
    daily_vol_target = n_active.mean() * CLUSTER_VOL
    print(f"\n  Average n_active clusters per day: {n_active.mean():.2f}")
    print(f"  Avg ex-ante 1d vol per leg: ${daily_vol_target:,.0f} × 2 sides")
    print(f"  ERC rescale factor: {scale:.3f}x\n")

    print(f"{'Strategy':<28} {'Ann P&L':>10} {'Ann Vol':>10} {'Sharpe':>8} "
          f"{'Max DD':>12} {'Calmar':>8}")
    print("-" * 80)
    for pnl_s, label in [
        (pnl_vt_c,           "Cluster Vol-Target"),
        (pnl_erc_scaled,     "ERC (rescaled)"),
    ]:
        s = compute_stats(pnl_s, label)
        print(f"  {label:<26} ${s['ann_pnl']:>9,.0f}  ${s['ann_vol']:>9,.0f}  "
              f"{s['sharpe']:>7.3f}  ${s['max_dd']:>11,.0f}  {s['calmar']:>7.3f}")

    # Momentum leg
    print(f"\n  Momentum leg only:")
    print(f"{'Strategy':<28} {'Ann P&L':>10} {'Ann Vol':>10} {'Sharpe':>8} {'Max DD':>12}")
    print("-" * 72)
    for pnl_s, label in [
        (pnl_vt_mom.reindex(common.index),   "Cluster Vol-Target Mom"),
        (pnl_erc_mom_scaled,                  "ERC Mom (rescaled)"),
    ]:
        s = compute_stats(pnl_s, label)
        print(f"  {label:<26} ${s['ann_pnl']:>9,.0f}  ${s['ann_vol']:>9,.0f}  "
              f"{s['sharpe']:>7.3f}  ${s['max_dd']:>11,.0f}")

    # Annual breakdown
    print(f"\n  Annual P&L:")
    print(f"  {'Year':<6} {'Cluster VT':>12} {'ERC (rescaled)':>16} {'Ratio':>8}")
    print("  " + "-" * 46)
    for yr in sorted(common.index.str[:4].unique()):
        mask = common.index.str.startswith(yr)
        r_vt  = pnl_vt_c[mask].sum()
        r_erc = pnl_erc_scaled[mask].sum()
        ratio = r_vt / r_erc if r_erc != 0 else np.nan
        print(f"  {yr}   ${r_vt:>10,.0f}   ${r_erc:>13,.0f}   {ratio:>7.2f}x")

    # SPY corr
    spy_c = spy.reindex(common.index)
    print(f"\n  SPY correlation:")
    print(f"    Cluster Vol-Target : {pnl_vt_c.corr(spy_c):+.3f}")
    print(f"    ERC (rescaled)     : {pnl_erc_scaled.corr(spy_c):+.3f}")

    # Cluster diagnostics
    print(f"\n── Latest cluster snapshot ──────────────────────────────────────────")
    ret_win_last = close.pct_change().iloc[-COV_WIN:]
    sig_last     = close.pct_change(MOM_WIN).iloc[-1].dropna()
    n_q          = max(1, len(sig_last) // 4)
    ranked       = sig_last.rank(ascending=False)
    longs_l      = sig_last[ranked <= n_q].index.tolist()
    shorts_l     = sig_last[ranked > len(sig_last) - n_q].index.tolist()

    lc = cluster_tickers(ret_win_last, longs_l,  N_LONG_CLUST)
    sc = cluster_tickers(ret_win_last, shorts_l, N_SHORT_CLUST)
    n_act_now = min(len(lc), len(sc))

    print(f"  Long clusters ({len(lc)}) → active: {n_act_now}")
    for cid, members in sorted(lc.items()):
        w = size_cluster(ret_win_last, members, CLUSTER_VOL, +1)
        cl_vol = sum(abs(v) for v in w.values) * ret_win_last[members].std().mean()
        print(f"    [{cid}] {', '.join(sorted(members))}")

    print(f"\n  Short clusters ({len(sc)}) → active: {n_act_now}")
    for cid, members in sorted(sc.items()):
        print(f"    [{cid}] {', '.join(sorted(members))}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nSaving plots ...")
    plot_comparison(
        {"Cluster Vol-Target": pnl_vt_c, "ERC (rescaled)": pnl_erc_scaled},
        f"Cluster Vol-Target vs ERC (rescaled)  |  "
        f"${CLUSTER_VOL:,.0f}/cluster  ·  {N_LONG_CLUST}L/{N_SHORT_CLUST}S clusters",
        OUT_DIR / "cluster_vt_combined.png",
        dollar=True,
    )
    plot_comparison(
        {"Cluster Vol-Target Mom": pnl_vt_mom.reindex(common.index),
         "ERC Mom (rescaled)":     pnl_erc_mom_scaled},
        f"Momentum leg only  |  ${CLUSTER_VOL:,.0f}/cluster",
        OUT_DIR / "cluster_vt_momentum.png",
        dollar=True,
    )

    pd.DataFrame([
        compute_stats(pnl_vt_c,       "Cluster Vol-Target"),
        compute_stats(pnl_erc_scaled,  "ERC (rescaled)"),
    ]).to_csv(OUT_DIR / "stats_cluster_vt.csv", index=False)


if __name__ == "__main__":
    main()

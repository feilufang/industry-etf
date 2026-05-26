#!/usr/bin/env python3
"""
Hybrid strategy: Cluster Vol-Target Momentum + ERC Reversal (VIX>20).

Three-way comparison:
  A) ERC combined (original)             -- % return, equal-vol combined
  B) Cluster VT combined                 -- $ P&L, equal-vol combined
  C) Hybrid: Cluster VT Mom + ERC Rev   -- both legs vol-scaled to $1k/cluster, equal-vol combined

For fair comparison all strategies are rescaled to the same annualised vol.
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

IS_END       = "2025-12-31"
MOM_WIN      = 252
COV_WIN      = 252
VOL_WIN      = 252
VIX_GATE     = 20
REV_LB       = 5
CLUSTER_VOL  = 1_000.0    # $ 1-day 1-sigma target per cluster
N_LONG_CL    = 3
N_SHORT_CL   = 4


# ── ERC core ──────────────────────────────────────────────────────────────────

def _cov(ret_win, tickers):
    C = ret_win[tickers].cov().values
    return C + np.eye(len(C)) * 1e-8

def opt_erc(ret_win, tickers):
    n, C = len(tickers), _cov(ret_win, tickers)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: 0.5*w@C@w - (1/n)*np.sum(np.log(np.maximum(w, 1e-12))),
        x0=w0, jac=lambda w: C@w - (1/n)/np.maximum(w, 1e-12),
        method="L-BFGS-B", bounds=[(1e-6, 1.0)]*n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w / w.sum()


# ── Clustering + vol-target sizing ───────────────────────────────────────────

def cluster_tickers(ret_win, tickers, n_clusters):
    if len(tickers) <= n_clusters:
        return {i: [t] for i, t in enumerate(tickers)}
    corr = ret_win[tickers].corr().fillna(0).clip(-1, 1)
    dist = np.sqrt(np.clip(0.5*(1 - corr.values), 0, None))
    np.fill_diagonal(dist, 0)
    Z      = sch.linkage(squareform(dist, checks=False), method="ward")
    labels = sch.fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = {}
    for t, lb in zip(tickers, labels):
        clusters.setdefault(int(lb), []).append(t)
    return clusters

def size_cluster_vt(ret_win, members, sign, target_vol=CLUSTER_VOL):
    """Dollar weights so cluster 1d vol = target_vol."""
    if len(members) == 1:
        sv = max(float(ret_win[members[0]].dropna().std()), 1e-6)
        return pd.Series([sign * target_vol / sv], index=members)
    try:    w = opt_erc(ret_win, members)
    except: w = np.ones(len(members)) / len(members)
    C   = _cov(ret_win, members)
    sig = float(np.sqrt(max(w @ C @ w, 1e-12)))
    return pd.Series(sign * (target_vol / sig) * w, index=members)

def trim_clusters(clusters, sig, keep_n, drop_weakest=True):
    """Keep keep_n clusters; drop those with weakest mean momentum signal."""
    if len(clusters) <= keep_n:
        return clusters
    scores  = {cid: sig.reindex(members).mean() for cid, members in clusters.items()}
    ordered = sorted(scores, key=scores.__getitem__, reverse=drop_weakest)
    keep    = set(ordered[:keep_n])
    return {cid: clusters[cid] for cid in keep}


# ── Strategy A: standard ERC (% return) ──────────────────────────────────────

def run_erc_momentum(close, vix):
    mom_sig, daily_ret = close.pct_change(MOM_WIN), close.pct_change()
    dates    = close.index.tolist()
    date_pos = {d: i for i, d in enumerate(dates)}
    weights  = pd.Series(0.0, index=close.columns)
    pnl      = []
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date]/close.loc[prev]-1).fillna(0.0)
        pnl.append((date, float((weights*ret).sum())))
        sig = mom_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns); continue
        n_q    = max(1, len(sig)//4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index
        shorts = sig[ranked > len(sig)-n_q].index
        rw     = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1):date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        try:
            weights[longs]  = pd.Series(opt_erc(rw, longs)  * +1.0, index=longs)
            weights[shorts] = pd.Series(opt_erc(rw, shorts) * -1.0, index=shorts)
        except: pass
    return pd.Series(dict(pnl))

def run_erc_reversal(close, vix):
    rev_sig, daily_ret = close.pct_change(REV_LB), close.pct_change()
    dates    = close.index.tolist()
    date_pos = {d: i for i, d in enumerate(dates)}
    weights  = pd.Series(0.0, index=close.columns)
    pnl      = []
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date]/close.loc[prev]-1).fillna(0.0)
        pnl.append((date, float((weights*ret).sum())))
        if vix.get(prev, np.nan) <= VIX_GATE:
            weights = pd.Series(0.0, index=close.columns); continue
        sig = rev_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns); continue
        n_q   = max(1, len(sig)//4)
        longs = sig.nsmallest(n_q).index
        rw    = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1):date_pos[prev]+1]
        weights = pd.Series(0.0, index=close.columns)
        try:
            weights[longs] = pd.Series(opt_erc(rw, longs) * +1.0, index=longs)
        except: pass
    return pd.Series(dict(pnl))


# ── Strategy B: Cluster VT momentum ($ P&L) ──────────────────────────────────

def run_clustervt_momentum(close, vix, n_long=N_LONG_CL, n_short=N_SHORT_CL):
    mom_sig, daily_ret = close.pct_change(MOM_WIN), close.pct_change()
    dates    = close.index.tolist()
    date_pos = {d: i for i, d in enumerate(dates)}
    dweights = pd.Series(0.0, index=close.columns)
    pnl      = []
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date]/close.loc[prev]-1).fillna(0.0)
        pnl.append((date, float((dweights*ret).sum())))
        sig = mom_sig.loc[prev].dropna()
        if len(sig) < 8:
            dweights = pd.Series(0.0, index=close.columns); continue
        n_q    = max(1, len(sig)//4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index.tolist()
        shorts = sig[ranked > len(sig)-n_q].index.tolist()
        rw     = daily_ret.iloc[max(0, date_pos[prev]-COV_WIN+1):date_pos[prev]+1]
        try:
            lc = cluster_tickers(rw, longs,  n_long)
            sc = cluster_tickers(rw, shorts, n_short)
            n_act = min(len(lc), len(sc))
            lc = trim_clusters(lc, sig, n_act, drop_weakest=False)  # keep highest mom
            sc = trim_clusters(sc, sig, n_act, drop_weakest=True)   # keep lowest mom
            dweights = pd.Series(0.0, index=close.columns)
            for members in lc.values():
                dweights.update(size_cluster_vt(rw, members, +1))
            for members in sc.values():
                dweights.update(size_cluster_vt(rw, members, -1))
        except:
            dweights = pd.Series(0.0, index=close.columns)
    return pd.Series(dict(pnl))


# ── Combination ───────────────────────────────────────────────────────────────

def combine_ev(a, b):
    """Equal-vol combination. Works regardless of whether units are $ or %."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    ia = 1.0 / df["a"].rolling(VOL_WIN).std().replace(0, np.nan)
    ib = 1.0 / df["b"].rolling(VOL_WIN).std().replace(0, np.nan)
    t  = ia + ib
    return ((ia/t)*df["a"] + (ib/t)*df["b"]).dropna()


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(pnl, label="", dollar=False):
    pnl = pnl.dropna()
    cum = pnl.cumsum() if dollar else (1+pnl).cumprod()
    if dollar:
        dd  = cum - cum.cummax()
        ann = pnl.mean() * 252
        vol = pnl.std()  * np.sqrt(252)
    else:
        dd  = (cum - cum.cummax()) / cum.cummax()
        ann = pnl.mean() * 252
        vol = pnl.std()  * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    return dict(label=label, ann=ann, vol=vol, sharpe=sharpe,
                max_dd=dd.min(), calmar=calmar)


# ── Annual breakdown ──────────────────────────────────────────────────────────

def annual_table(series_dict, dollar=False):
    all_idx = sorted(set(idx for s in series_dict.values() for idx in s.index))
    years   = sorted(set(d[:4] for d in all_idx))
    rows    = []
    for yr in years:
        row = {"Year": yr}
        for label, s in series_dict.items():
            mask = s.index.str.startswith(yr)
            if dollar:
                row[label] = s[mask].sum()
            else:
                row[label] = (1 + s[mask]).prod() - 1
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_all(series_dict, title, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    palette = {
        "ERC combined":     ("#1f77b4", "--", 1.4),
        "Cluster VT":       ("#ff7f0e", "-.", 1.4),
        "Hybrid":           ("#2ca02c", "-",  2.2),
    }
    ax = axes[0]
    for label, s in series_dict.items():
        c, ls, lw = palette.get(label, ("#9467bd", "-", 1.4))
        cum = s.dropna().cumsum()
        ax.plot(pd.to_datetime(s.index), cum, color=c, ls=ls, lw=lw, label=label)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("Cumulative P&L (normalised $)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2 = axes[1]
    for label, s in series_dict.items():
        c, ls, lw = palette.get(label, ("#9467bd", "-", 1.4))
        cum = s.dropna().cumsum()
        dd  = cum - cum.cummax()
        ax2.plot(pd.to_datetime(s.index), dd, color=c, ls=ls, lw=lw, label=label)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load ───────────────────────────────────────────────────────────────
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
    close = pd.read_parquet(HERE/"data"/"industry_etf_daily.parquet")
    close = close[[t for t in ind_tickers if t in close.columns]].sort_index()
    close = close[close.index <= IS_END].sort_index()
    close = close.where(close.gt(0)).ffill()
    ind_first = close.index[MOM_WIN]

    # ── Run all legs ────────────────────────────────────────────────────────
    print("\nRunning ERC momentum ...")
    pnl_erc_mom = run_erc_momentum(close, vix)
    pnl_erc_mom = pnl_erc_mom[pnl_erc_mom.index >= ind_first]

    print("Running ERC reversal ...")
    pnl_erc_rev = run_erc_reversal(close, vix)
    pnl_erc_rev = pnl_erc_rev[pnl_erc_rev.index >= ind_first]

    print(f"Running Cluster VT momentum ({N_LONG_CL}L/{N_SHORT_CL}S) ...")
    pnl_cvt_mom = run_clustervt_momentum(close, vix, N_LONG_CL, N_SHORT_CL)
    pnl_cvt_mom = pnl_cvt_mom[pnl_cvt_mom.index >= ind_first]

    # ── Combine ─────────────────────────────────────────────────────────────
    # A: ERC combined (% return)
    pnl_erc_comb = combine_ev(pnl_erc_mom, pnl_erc_rev)

    # B: Cluster VT combined (cluster VT mom + cluster VT rev)
    #    reuse pnl_cvt_rev from cluster_voltarget module logic — run inline
    print("Running Cluster VT reversal ...")
    rev_sig_full, daily_ret_full = close.pct_change(REV_LB), close.pct_change()
    dates_full    = close.index.tolist()
    date_pos_full = {d: i for i, d in enumerate(dates_full)}
    dw_rev = pd.Series(0.0, index=close.columns)
    pnl_cvt_rev_list = []
    for i in range(1, len(dates_full)):
        date, prev = dates_full[i], dates_full[i-1]
        ret = (close.loc[date]/close.loc[prev]-1).fillna(0.0)
        pnl_cvt_rev_list.append((date, float((dw_rev*ret).sum())))
        if vix.get(prev, np.nan) <= VIX_GATE:
            dw_rev = pd.Series(0.0, index=close.columns); continue
        sig = rev_sig_full.loc[prev].dropna()
        if len(sig) < 8:
            dw_rev = pd.Series(0.0, index=close.columns); continue
        n_q   = max(1, len(sig)//4)
        longs = sig.nsmallest(n_q).index.tolist()
        rw    = daily_ret_full.iloc[max(0, date_pos_full[prev]-COV_WIN+1):date_pos_full[prev]+1]
        try:
            lc = cluster_tickers(rw, longs, N_LONG_CL)
            dw_rev = pd.Series(0.0, index=close.columns)
            for members in lc.values():
                dw_rev.update(size_cluster_vt(rw, members, +1))
        except:
            dw_rev = pd.Series(0.0, index=close.columns)
    pnl_cvt_rev = pd.Series(dict(pnl_cvt_rev_list))
    pnl_cvt_rev = pnl_cvt_rev[pnl_cvt_rev.index >= ind_first]
    pnl_cvt_comb = combine_ev(pnl_cvt_mom, pnl_cvt_rev)

    # C: Hybrid = Cluster VT mom + ERC rev (equal-vol combine)
    pnl_hybrid = combine_ev(pnl_cvt_mom, pnl_erc_rev)

    # ── Normalise all to same vol for fair comparison ───────────────────────
    target_vol = 10_000.0   # $10k annualised vol for display
    def rescale(s, tv=target_vol):
        ann_vol = s.dropna().std() * np.sqrt(252)
        return s * (tv / ann_vol) if ann_vol > 0 else s

    common = pd.DataFrame({
        "ERC combined":  pnl_erc_comb,
        "Cluster VT":    pnl_cvt_comb,
        "Hybrid":        pnl_hybrid,
    }).dropna()

    scaled = common.apply(rescale)
    spy_scaled = rescale(spy.reindex(common.index))

    print(f"\nCommon window: {common.index[0]} -> {common.index[-1]}  ({len(common)} days)")
    print(f"All strategies normalised to ${target_vol:,.0f} annualised vol.\n")

    # ── Stats table ─────────────────────────────────────────────────────────
    print(f"{'Strategy':<26} {'Ann P&L':>10} {'Ann Vol':>10} {'Sharpe':>8} "
          f"{'Max DD':>12} {'Calmar':>8} {'SPY corr':>9}")
    print("-" * 90)
    for col in scaled.columns:
        s  = stats(scaled[col], col, dollar=True)
        cr = scaled[col].corr(spy_scaled)
        print(f"  {col:<24} ${s['ann']:>9,.0f}  ${s['vol']:>9,.0f}  "
              f"{s['sharpe']:>7.3f}  ${s['max_dd']:>11,.0f}  {s['calmar']:>7.3f}  {cr:>+8.3f}")

    # ── Leg-level stats ──────────────────────────────────────────────────────
    print(f"\n  Momentum legs (normalised):")
    print(f"{'Leg':<26} {'Ann P&L':>10} {'Ann Vol':>10} {'Sharpe':>8} {'Max DD':>12}")
    print("-" * 70)
    for s_raw, label in [(pnl_erc_mom, "ERC Mom"), (pnl_cvt_mom, "Cluster VT Mom")]:
        s_sc = rescale(s_raw.reindex(common.index))
        st   = stats(s_sc, label, dollar=True)
        print(f"  {label:<24} ${st['ann']:>9,.0f}  ${st['vol']:>9,.0f}  "
              f"{st['sharpe']:>7.3f}  ${st['max_dd']:>11,.0f}")

    print(f"\n  Reversal leg (normalised):")
    s_sc = rescale(pnl_erc_rev.reindex(common.index))
    st   = stats(s_sc, "ERC Rev", dollar=True)
    print(f"  {'ERC Rev':<24} ${st['ann']:>9,.0f}  ${st['vol']:>9,.0f}  "
          f"{st['sharpe']:>7.3f}  ${st['max_dd']:>11,.0f}")

    # ── Annual table ─────────────────────────────────────────────────────────
    print(f"\n  Annual P&L (normalised $):")
    ann = annual_table({"ERC combined": scaled["ERC combined"],
                        "Cluster VT":   scaled["Cluster VT"],
                        "Hybrid":       scaled["Hybrid"]}, dollar=True)
    print(f"  {'Year':<6}", end="")
    for c in ann.columns:
        print(f"  {c:>14}", end="")
    print()
    print("  " + "-" * (6 + 16*3))
    for yr, row in ann.iterrows():
        print(f"  {yr}  ", end="")
        for c in ann.columns:
            print(f"  ${row[c]:>13,.0f}", end="")
        print()

    # ── Plot ────────────────────────────────────────────────────────────────
    print("\nSaving plots ...")
    plot_all(
        {c: scaled[c] for c in scaled.columns},
        f"ERC vs Cluster VT vs Hybrid  |  All at ${target_vol:,.0f} ann vol  "
        f"|  {N_LONG_CL}L/{N_SHORT_CL}S clusters  ·  $1k/cluster",
        OUT_DIR / "hybrid_strategy.png",
    )

    pd.DataFrame([stats(scaled[c], c, dollar=True) for c in scaled.columns]
                 ).to_csv(OUT_DIR / "stats_hybrid.csv", index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
252-day cross-sectional momentum backtest on industry ETFs.

Signal   : 252-trading-day total return  (price[t] / price[t-252] - 1)
Universe : Long top quartile, short bottom quartile (by momentum rank).
Sizing   : Five weighting methods compared:
             equal_vol  — 1/vol weights, normalized per leg
             min_var    — minimum variance QP
             erc        — equal risk contribution
             mdp        — maximum diversification portfolio
             hrp        — hierarchical risk parity
Timing   : Signal at T close → trade at T+1 close (no look-ahead).
Rebal    : Daily (this run). Weekly and Monthly kept for reference.
IS period: 2024-01-04 through 2025-12-31
Cov win  : Trailing 252 trading days (same as signal lookback).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sco
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
DAILY_FILE = HERE / "data" / "industry_etf_daily.parquet"
TICKER_CSV = HERE / "Industry_ETF_Tickers.csv"
OUT_DIR    = HERE / "results_momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END          = "2025-12-31"
MOMENTUM_WINDOW = 252
COV_WINDOW      = 252
MIN_STOCKS      = 8

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_tickers() -> list[str]:
    raw = TICKER_CSV.read_text(encoding="utf-8-sig").strip().splitlines()
    return [t.strip() for t in raw if t.strip()]


def load_prices(tickers: list[str]) -> pd.DataFrame:
    # Wide format: index=date (str), columns=tickers
    close = pd.read_parquet(DAILY_FILE)
    avail = [t for t in tickers if t in close.columns]
    close = close[avail]
    close = close[close.index <= IS_END]
    close.sort_index(inplace=True)
    close = close[close.gt(0) | close.isna()]   # zero prices → NaN
    return close.ffill()


# ── Signal ─────────────────────────────────────────────────────────────────────

def compute_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """252-day total return as the cross-sectional momentum signal."""
    return close.pct_change(MOMENTUM_WINDOW)


# ── Rebalance schedule ─────────────────────────────────────────────────────────

def get_rebal_dates(index: pd.Index, freq: str) -> set:
    dts = pd.to_datetime(index)
    s   = pd.Series(index.tolist(), index=dts)
    if freq == "D":
        return set(index.tolist())
    elif freq == "W":
        week = dts.isocalendar().week.values
        return set(s.groupby([dts.year, week]).last())
    elif freq == "M":
        return set(s.groupby([dts.year, dts.month]).last())
    raise ValueError(freq)


# ── Portfolio optimizers ───────────────────────────────────────────────────────

def _regularize_cov(C: np.ndarray) -> np.ndarray:
    """Add small diagonal to ensure positive definiteness."""
    return C + np.eye(len(C)) * 1e-8


def _sample_cov(ret_window: pd.DataFrame, tickers: pd.Index) -> np.ndarray:
    return _regularize_cov(ret_window[tickers].cov().values)


def opt_equal_vol(ret_window: pd.DataFrame, tickers: pd.Index) -> np.ndarray:
    C    = _sample_cov(ret_window, tickers)
    vols = np.sqrt(np.diag(C))
    vols = np.where(vols <= 0, vols[vols > 0].mean(), vols)
    w    = 1.0 / vols
    return w / w.sum()


def opt_min_var(ret_window: pd.DataFrame, tickers: pd.Index) -> np.ndarray:
    n  = len(tickers)
    C  = _sample_cov(ret_window, tickers)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: w @ C @ w,
        x0=w0,
        jac=lambda w: 2 * C @ w,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = res.x if res.success else w0
    return w / w.sum()


def opt_erc(ret_window: pd.DataFrame, tickers: pd.Index) -> np.ndarray:
    """
    Equal risk contribution via convex log-barrier formulation:
      minimize  (1/2) w'Cw - (1/n) sum log(w_i)
    Unique solution has RC_i = constant. Normalize after.
    """
    n  = len(tickers)
    C  = _sample_cov(ret_window, tickers)
    w0 = np.ones(n) / n
    res = sco.minimize(
        fun=lambda w: 0.5 * w @ C @ w - (1 / n) * np.sum(np.log(np.maximum(w, 1e-12))),
        x0=w0,
        jac=lambda w: C @ w - (1 / n) / np.maximum(w, 1e-12),
        method="L-BFGS-B",
        bounds=[(1e-6, 1.0)] * n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w / w.sum()


def opt_mdp(ret_window: pd.DataFrame, tickers: pd.Index) -> np.ndarray:
    """
    Maximum Diversification Portfolio:
      maximize  w'σ / sqrt(w'Cw)
    where σ_i = individual EWM volatility.
    """
    n    = len(tickers)
    C    = _sample_cov(ret_window, tickers)
    vols = np.sqrt(np.diag(C))
    vols = np.where(vols <= 0, vols[vols > 0].mean(), vols)
    w0   = np.ones(n) / n

    def neg_dr(w):
        port_vol = np.sqrt(w @ C @ w)
        return -(w @ vols) / port_vol

    def grad_neg_dr(w):
        Cw       = C @ w
        port_vol = np.sqrt(w @ Cw)
        return -(vols / port_vol - (w @ vols) * Cw / port_vol**3)

    res = sco.minimize(
        fun=neg_dr, x0=w0, jac=grad_neg_dr,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = res.x if res.success else w0
    return w / w.sum()


def opt_hrp(ret_window: pd.DataFrame, tickers: pd.Index) -> np.ndarray:
    """
    Hierarchical Risk Parity (Lopez de Prado 2016).
    Uses EWM covariance for clustering and risk allocation.
    """
    C    = _sample_cov(ret_window, tickers)
    vols = np.sqrt(np.diag(C))
    vols = np.where(vols <= 0, 1.0, vols)
    corr = C / np.outer(vols, vols)
    np.fill_diagonal(corr, 1.0)

    dist  = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    link  = linkage(squareform(dist), method="ward")
    order = leaves_list(link)   # leaf ordering from dendrogram

    names   = list(tickers)
    ordered = [names[i] for i in order]

    # Map name → index in cov
    idx = {t: i for i, t in enumerate(names)}

    def cluster_var(cluster):
        ii  = [idx[t] for t in cluster]
        Cs  = C[np.ix_(ii, ii)]
        ivp = 1.0 / np.diag(Cs)
        ivp /= ivp.sum()
        return float(ivp @ Cs @ ivp)

    def bisect(cluster: list) -> dict:
        if len(cluster) == 1:
            return {cluster[0]: 1.0}
        mid  = len(cluster) // 2
        L, R = cluster[:mid], cluster[mid:]
        vL, vR = cluster_var(L), cluster_var(R)
        alpha = 1 - vL / (vL + vR)   # fraction to left sub-cluster
        wL = bisect(L)
        wR = bisect(R)
        return {k: v * alpha for k, v in wL.items()} | \
               {k: v * (1 - alpha) for k, v in wR.items()}

    w_dict = bisect(ordered)
    w = np.array([w_dict[t] for t in names])
    return w / w.sum()


OPTIMIZERS = {
    "equal_vol": opt_equal_vol,
    "min_var":   opt_min_var,
    "erc":       opt_erc,
    "mdp":       opt_mdp,
    "hrp":       opt_hrp,
}


def basket_weights(ret_window: pd.DataFrame, tickers: pd.Index,
                   method: str, sign: float) -> pd.Series:
    try:
        w = OPTIMIZERS[method](ret_window, tickers)
    except Exception:
        w = np.ones(len(tickers)) / len(tickers)
    return pd.Series(w * sign, index=tickers)


# ── Backtest ───────────────────────────────────────────────────────────────────

def run_backtest(close: pd.DataFrame, mom: pd.DataFrame,
                 freq: str = "D", method: str = "equal_vol",
                 quantile: int = 4) -> pd.Series:
    """
    Timing: signal at T close → trade at T+1 close → P&L from T+2.
    Covariance: rolling COV_WINDOW-day window.
    quantile: 4 = quartile (top/bottom 25%), 3 = tercile (top/bottom 33%).
    """
    daily_ret   = close.pct_change()
    dates       = close.index.tolist()
    date_pos    = {d: i for i, d in enumerate(dates)}
    rebal_dates = get_rebal_dates(close.index, freq)

    weights   = pd.Series(0.0, index=close.columns)
    daily_pnl = []

    for i in range(1, len(dates)):
        date      = dates[i]
        prev_date = dates[i - 1]

        ret = (close.loc[date] / close.loc[prev_date] - 1).fillna(0.0)
        daily_pnl.append((date, float((weights * ret).sum())))

        if prev_date in rebal_dates:
            sig = mom.loc[prev_date].dropna()
            if len(sig) < MIN_STOCKS:
                weights = pd.Series(0.0, index=close.columns)
                continue

            n_valid = len(sig)
            n_q     = max(1, n_valid // quantile)
            ranked  = sig.rank(ascending=False)
            longs   = sig[ranked <= n_q].index
            shorts  = sig[ranked > n_valid - n_q].index

            t_idx   = date_pos[prev_date]
            start   = max(0, t_idx - COV_WINDOW + 1)
            ret_win = daily_ret.iloc[start : t_idx + 1]

            weights = pd.Series(0.0, index=close.columns)
            if len(longs):
                weights[longs]  = basket_weights(ret_win, longs,  method, +1.0)
            if len(shorts):
                weights[shorts] = basket_weights(ret_win, shorts, method, -1.0)

    return pd.Series(dict(daily_pnl), name="pnl")


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(pnl: pd.Series, label: str = "") -> dict:
    cum    = (1 + pnl).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    ann    = pnl.mean() * 252
    vol    = pnl.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    return {
        "method":   label,
        "ann_ret":  round(float(ann), 4),
        "ann_vol":  round(float(vol), 4),
        "sharpe":   round(float(sharpe), 3),
        "max_dd":   round(float(dd.min()), 4),
        "calmar":   round(float(calmar), 3),
        "win_rate": round(float((pnl > 0).mean()), 3),
        "n_days":   int(len(pnl)),
        "start":    str(pnl.index[0]),
        "end":      str(pnl.index[-1]),
    }


# ── Monthly returns table ──────────────────────────────────────────────────────

def monthly_returns(pnl: pd.Series) -> pd.DataFrame:
    idx   = pd.to_datetime(pnl.index)
    m_ret = (pd.Series((1 + pnl.values), index=idx)
             .resample("ME").prod() - 1)
    m_ret.index = m_ret.index.to_period("M")
    tbl = m_ret.rename("ret").to_frame()
    tbl["year"]  = tbl.index.year
    tbl["month"] = tbl.index.month
    pivot = tbl.pivot(index="year", columns="month", values="ret")
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
    pivot["Ann"] = tbl.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
    return pivot


# ── Plots ──────────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "equal_vol": "Equal Vol",
    "min_var":   "Min Var",
    "erc":       "ERC",
    "mdp":       "MDP",
    "hrp":       "HRP",
}
METHOD_COLORS = {
    "equal_vol": "#7f7f7f",
    "min_var":   "#1f77b4",
    "erc":       "#ff7f0e",
    "mdp":       "#2ca02c",
    "hrp":       "#d62728",
}
METHOD_STYLES = {
    "equal_vol": "--",
    "min_var":   "-",
    "erc":       "-",
    "mdp":       "-",
    "hrp":       "-",
}


def _plot_cumret_labeled(results: dict[str, pd.Series], cov_label: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for method, pnl in results.items():
        cum = (1 + pnl).cumprod() - 1
        ax.plot(
            pd.to_datetime(cum.index), cum * 100,
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=1.5 if method != "equal_vol" else 1.1,
            label=METHOD_LABELS[method],
        )
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        "252-day Momentum — Industry ETFs  |  Daily Rebalance  |  IS: 2017–2025\n"
        "Portfolio Optimization Comparison  |  Rolling 252-day Cov  |  Dollar-Neutral",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / f"cumret_is_{cov_label}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_cumret_methods(results: dict[str, pd.Series]) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for method, pnl in results.items():
        cum = (1 + pnl).cumprod() - 1
        ax.plot(
            pd.to_datetime(cum.index), cum * 100,
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=1.5 if method != "equal_vol" else 1.1,
            label=METHOD_LABELS[method],
        )
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        "252-day Momentum — Industry ETFs  |  Daily Rebalance  |  IS: 2024–2025\n"
        "Portfolio Optimization: Max Diversification Comparison  |  Dollar-Neutral",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "cumret_is_opt_methods.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_monthly_heatmap(pnl: pd.Series, method: str) -> None:
    pivot      = monthly_returns(pnl)
    month_cols = [c for c in pivot.columns if c != "Ann"]
    data_m     = pivot[month_cols]
    ann_col    = pivot[["Ann"]]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(3, len(pivot) * 0.55 + 1.5)),
        gridspec_kw={"width_ratios": [len(month_cols), 1]},
    )

    def _heatmap(ax, data, title):
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

    _heatmap(axes[0], data_m, "Monthly Returns (%)")
    _heatmap(axes[1], ann_col, "Annual (%)")
    fig.suptitle(
        f"252-day Momentum — Industry ETFs  |  {METHOD_LABELS[method]}  |  Daily Rebalance (IS)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = OUT_DIR / f"monthly_returns_is_{method}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def plot_tercile_vs_quartile(results: dict[str, pd.Series]) -> None:
    colors = {"erc_q4": "#1f77b4", "erc_q3": "#ff7f0e"}
    labels = {"erc_q4": "ERC  Quartile (top/bot 25%)", "erc_q3": "ERC  Tercile (top/bot 33%)"}
    fig, ax = plt.subplots(figsize=(13, 5))
    for key, pnl in results.items():
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(cum.index), cum * 100,
                color=colors[key], linewidth=1.8, label=labels[key])
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        "252-day Momentum — Industry ETFs  |  ERC  |  Daily Rebalance  |  IS: 2017–2025\n"
        "Quartile (25%) vs Tercile (33%) basket comparison  |  Dollar-Neutral",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "cumret_erc_q3_vs_q4.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")


def main() -> None:
    tickers = load_tickers()
    print(f"Universe: {len(tickers)} tickers")

    print("Loading IS price data ...")
    close = load_prices(tickers)
    print(f"  {len(close.columns)} tickers  |  {close.index[0]} -> {close.index[-1]}  ({len(close):,} days)")

    mom       = compute_momentum(close)
    first_sig = close.index[MOMENTUM_WINDOW]
    print(f"\nEffective backtest window: {first_sig} -> {close.index[-1]}\n")

    # ── Run ERC: quartile vs tercile ──────────────────────────────────────────
    runs = [("erc_q4", "erc", 4), ("erc_q3", "erc", 3)]
    results = {}
    rows    = []

    print(f"{'Run':<14} {'Basket':>8} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8} {'WinRate':>7}")
    print("-" * 78)

    for key, method, q in runs:
        n_stocks = 114 // q
        label    = f"ERC  {'Q4 (25%)' if q == 4 else 'Q3 (33%)'}"
        print(f"  {label:<22} ~{n_stocks:>2}/leg ...", end=" ", flush=True)
        pnl = run_backtest(close, mom, freq="D", method=method, quantile=q)
        pnl = pnl[pnl.index >= first_sig]
        results[key] = pnl
        s = compute_stats(pnl, label)
        rows.append(s)
        print(
            f"  {s['ann_ret']:>+8.2%}  {s['ann_vol']:>7.2%}  "
            f"{s['sharpe']:>7.3f}  {s['max_dd']:>+7.2%}  "
            f"{s['calmar']:>7.3f}  {s['win_rate']:>7.1%}"
        )

    print()
    for key, pnl in results.items():
        label = "ERC Q4 (quartile)" if key == "erc_q4" else "ERC Q3 (tercile)"
        print(f"\n--- {label} — Monthly Returns ---")
        print((monthly_returns(pnl) * 100).round(1).to_string())

    stats_path = OUT_DIR / "stats_erc_q3_vs_q4.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"\nSaved -> {stats_path}")

    print("\nSaving charts ...")
    plot_tercile_vs_quartile(results)
    for key, pnl in results.items():
        method = "erc"
        # reuse heatmap with patched label
        pivot      = monthly_returns(pnl)
        month_cols = [c for c in pivot.columns if c != "Ann"]
        data_m     = pivot[month_cols]
        ann_col    = pivot[["Ann"]]

        fig, axes = plt.subplots(
            1, 2, figsize=(16, max(3, len(pivot) * 0.55 + 1.5)),
            gridspec_kw={"width_ratios": [len(month_cols), 1]},
        )

        def _heatmap(ax, data, title):
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

        _heatmap(axes[0], data_m, "Monthly Returns (%)")
        _heatmap(axes[1], ann_col, "Annual (%)")
        basket_label = "Q4 (top/bot 25%)" if key == "erc_q4" else "Q3 (top/bot 33%)"
        fig.suptitle(
            f"252-day Momentum — Industry ETFs  |  ERC  {basket_label}  |  Daily Rebalance (IS)",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        path = OUT_DIR / f"monthly_returns_is_erc_{key}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {path}")

    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

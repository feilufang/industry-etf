#!/usr/bin/env python3
"""
Daily monitor for Industry ETF Combined Strategy.
Refreshes price data, runs combined strategy (ERC Momentum + ERC Reversal VIX>20),
and emails an HTML performance report.

Environment variables required:
  GMAIL_USER        - sender Gmail address (e.g. rogerwugang@gmail.com)
  GMAIL_APP_PASS    - Gmail App Password (not your login password)
  MONITOR_TO        - recipient address (defaults to GMAIL_USER if not set)

To generate a Gmail App Password:
  Google Account → Security → 2-Step Verification → App passwords
"""

import os
import sys
import smtplib
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import date

import base64
import io
from email.mime.image import MIMEImage

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE       = Path(__file__).parent
MOM_WIN    = 252
REV_LB     = 5
COV_WIN    = 252
VOL_WIN    = 252
VIX_GATE   = 20
TRAIL_DAYS = 63
NOTIONAL   = 10_000


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

def basket_erc(ret_win, tickers, sign):
    try:    w = opt_erc(ret_win, tickers)
    except: w = np.ones(len(tickers)) / len(tickers)
    return pd.Series(w * sign, index=tickers)


# ── Strategy runners ──────────────────────────────────────────────────────────

def run_momentum(close, vix):
    mom_sig, daily_ret = close.pct_change(MOM_WIN), close.pct_change()
    dates = close.index.tolist()
    date_pos = {d: i for i, d in enumerate(dates)}
    weights = pd.Series(0.0, index=close.columns)
    pnl = []
    last_weights = pd.Series(0.0, index=close.columns)
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        ret = (close.loc[date] / close.loc[prev] - 1).fillna(0.0)
        pnl.append((date, float((weights * ret).sum())))
        sig = mom_sig.loc[prev].dropna()
        if len(sig) < 8:
            weights = pd.Series(0.0, index=close.columns)
            continue
        n_q = max(1, len(sig) // 4)
        ranked = sig.rank(ascending=False)
        longs  = sig[ranked <= n_q].index
        shorts = sig[ranked > len(sig) - n_q].index
        ret_win = daily_ret.iloc[max(0, date_pos[prev] - COV_WIN + 1) : date_pos[prev] + 1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs]  = basket_erc(ret_win, longs,  +1.0)
        weights[shorts] = basket_erc(ret_win, shorts, -1.0)
        last_weights = weights.copy()
    return pd.Series(dict(pnl)), last_weights

def run_reversal(close, vix):
    rev_sig, daily_ret = close.pct_change(REV_LB), close.pct_change()
    dates = close.index.tolist()
    date_pos = {d: i for i, d in enumerate(dates)}
    weights = pd.Series(0.0, index=close.columns)
    pnl = []
    last_weights = pd.Series(0.0, index=close.columns)
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
        n_q = max(1, len(sig) // 4)
        longs = sig.nsmallest(n_q).index
        ret_win = daily_ret.iloc[max(0, date_pos[prev] - COV_WIN + 1) : date_pos[prev] + 1]
        weights = pd.Series(0.0, index=close.columns)
        weights[longs] = basket_erc(ret_win, longs, +1.0)
        last_weights = weights.copy()
    return pd.Series(dict(pnl)), last_weights

def combine_ev(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    ia = 1.0 / df["a"].rolling(VOL_WIN).std().replace(0, np.nan)
    ib = 1.0 / df["b"].rolling(VOL_WIN).std().replace(0, np.nan)
    t  = ia + ib
    return ((ia / t) * df["a"] + (ib / t) * df["b"]).dropna()


# ── Stats ──────────────────────────────────────────────────────────────────────

def stats(pnl, label=""):
    pnl = pnl.dropna()
    cum = (1 + pnl).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    ann = pnl.mean() * 252
    vol = pnl.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    calmar = ann / abs(dd.min()) if dd.min() != 0 else np.nan
    total  = float(cum.iloc[-1]) - 1
    cur_dd = float(dd.iloc[-1])
    return dict(label=label, ann_ret=ann, ann_vol=vol, sharpe=sharpe,
                max_dd=dd.min(), calmar=calmar, total_ret=total, cur_dd=cur_dd)


# ── Chart builder ─────────────────────────────────────────────────────────────

OS_START = "2026-05-01"

def _make_chart(series_list, title, figsize=(11, 3.5)):
    """Shared chart renderer. series_list: [(pnl, label, color, lw), ...]"""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    for pnl, label, color, lw in series_list:
        pnl = pnl.dropna()
        cum = (1 + pnl).cumprod() - 1
        ax.plot(pd.to_datetime(pnl.index), cum * 100, color=color, lw=lw, label=label)
    ax.axhline(0, color="#aaa", lw=0.7, ls=":")
    ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.set_title(title, fontsize=10, color="#444", pad=6)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_cumret_chart(pnl_mom, pnl_rev, pnl_comb):
    """Full-history chart with IS solid / OS dotted."""
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    for pnl, label, color, lw in [
        (pnl_comb, "Combined",           "#2ca02c", 2.2),
        (pnl_mom,  "Momentum",           "#1f77b4", 1.4),
        (pnl_rev,  "Reversal (VIX>20)",  "#ff7f0e", 1.4),
    ]:
        pnl = pnl.dropna()
        cum = (1 + pnl).cumprod() - 1
        is_mask = pnl.index < OS_START
        os_mask = pnl.index >= OS_START
        if is_mask.any():
            ax.plot(pd.to_datetime(pnl.index[is_mask]), cum[is_mask] * 100,
                    color=color, lw=lw, label=label)
        if os_mask.any():
            stitch = pnl.index[is_mask][-1] if is_mask.any() else pnl.index[0]
            os_ext = pnl.index[pnl.index >= stitch]
            ax.plot(pd.to_datetime(os_ext), cum[os_ext] * 100,
                    color=color, lw=lw, ls=":", alpha=0.9, label="_nolegend_")

    os_start_dt = pd.to_datetime(OS_START)
    ax.axvline(os_start_dt, color="#888", lw=1.0, ls="--", alpha=0.7)
    ax.axvspan(os_start_dt, pd.to_datetime(pnl_comb.dropna().index[-1]),
               alpha=0.04, color="#000")
    ylim = ax.get_ylim()
    ax.text(os_start_dt, ylim[1], " OS", fontsize=8, color="#888", va="top")
    ax.axhline(0, color="#aaa", lw=0.7, ls=":")
    ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.grid(axis="x", alpha=0.15, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_os_chart(pnl_mom, pnl_rev, pnl_comb):
    """Trailing 63-day cumulative return chart, rebased to 0 at start of window."""
    series = [
        (pnl_comb, "Combined",           "#2ca02c", 2.2),
        (pnl_mom,  "Momentum",           "#1f77b4", 1.4),
        (pnl_rev,  "Reversal (VIX>20)",  "#ff7f0e", 1.4),
    ]
    trail_series = [(s[0].iloc[-TRAIL_DAYS:],) + s[1:] for s in series]
    if all(len(s[0]) == 0 for s in trail_series):
        return None
    start_date = trail_series[0][0].index[0]
    return _make_chart(
        trail_series,
        title=f"Trailing {TRAIL_DAYS}-Day Performance (from {start_date})",
        figsize=(11, 3.2),
    )


# ── HTML report builder ────────────────────────────────────────────────────────

def pct(v, signed=True):
    if np.isnan(v): return "—"
    return f"{v:+.2%}" if signed else f"{v:.2%}"

def color_cell(v, good_pos=True):
    if np.isnan(v): return "<td>—</td>"
    pos = v > 0
    if good_pos:
        color = "#27ae60" if pos else "#e74c3c"
    else:
        color = "#e74c3c" if pos else "#27ae60"
    return f'<td style="color:{color};font-weight:600">{v:+.2%}</td>'

def _target_shares_html(target_shares, etf_names, vix_last, rev_active, vix_gate=None, vix_gate_date=None):
    if not target_shares:
        return ""
    longs  = sorted([(t, q) for t, q in target_shares.items() if q > 0], key=lambda x: -x[1])
    shorts = sorted([(t, q) for t, q in target_shares.items() if q < 0], key=lambda x:  x[1])
    rows = ""
    for t, q in longs:
        name = (etf_names or {}).get(t, "")
        rows += (f"<tr><td style='color:#27ae60;font-weight:700'>LONG</td>"
                 f"<td style='font-weight:700'>{t}</td><td style='color:#555'>{name}</td>"
                 f"<td style='text-align:right;font-weight:700'>{q}</td></tr>")
    for t, q in shorts:
        name = (etf_names or {}).get(t, "")
        rows += (f"<tr><td style='color:#e74c3c;font-weight:700'>SHORT</td>"
                 f"<td style='font-weight:700'>{t}</td><td style='color:#555'>{name}</td>"
                 f"<td style='text-align:right;font-weight:700'>{q}</td></tr>")
    gate_v = vix_gate if vix_gate is not None else vix_last
    gate_d = f" ({vix_gate_date})" if vix_gate_date else ""
    rev_note = f"active (signal VIX={gate_v:.1f}{gate_d})" if rev_active else f"inactive (signal VIX={gate_v:.1f}{gate_d} &le; {VIX_GATE})"
    return f"""
    <div class="card">
      <h2>Target Positions &mdash; ${NOTIONAL:,}/side MOC &nbsp;
        <span style="font-size:12px;color:#888;font-weight:400">
          Reversal {rev_note}
        </span>
      </h2>
      <table style="width:100%;border-collapse:collapse">
        <tr style="background:#f5f5f5">
          <th style="text-align:left;padding:5px 8px">Side</th>
          <th style="text-align:left;padding:5px 8px">Ticker</th>
          <th style="text-align:left;padding:5px 8px">Name</th>
          <th style="text-align:right;padding:5px 8px">Shares</th>
        </tr>
        {rows}
      </table>
    </div>"""


def build_html(
    run_date, last5, week_blocks, ytd_stats,
    full_stats, vix_last, spy_ytd,
    mom_longs, mom_shorts, rev_longs, rev_active,
    etf_names=None, ticker_rets=None, has_main_chart=False, has_os_chart=False,
    target_shares=None, vix_gate=None, vix_gate_date=None,
):
    def stat_row(s, bold=False):
        b = "<b>" if bold else ""
        e = "</b>" if bold else ""
        return (
            f"<tr><td>{b}{s['label']}{e}</td>"
            + color_cell(s['ann_ret'])
            + f"<td>{s['ann_vol']:.1%}</td>"
            + f"<td>{b}{s['sharpe']:.3f}{e}</td>"
            + color_cell(s['max_dd'], good_pos=False)
            + color_cell(s['cur_dd'], good_pos=False)
            + f"<td>{s['total_ret']:+.1%}</td></tr>"
        )

    # Last 5 days table rows
    day_rows = ""
    for _, row in last5.iterrows():
        vix_val = row.get("VIX", np.nan)
        gate = "✓" if (not np.isnan(vix_val) and vix_val > VIX_GATE) else "—"
        is_intra = row.get("intraday", False)
        style = ' style="background:#fffde7;font-style:italic"' if is_intra else ""
        vix_str = f"{vix_val:.1f}" if not np.isnan(vix_val) else "—"
        day_rows += (
            f"<tr{style}><td>{row['Date']}</td>"
            + color_cell(row['Combined'])
            + color_cell(row['Momentum'])
            + color_cell(row['Reversal'])
            + color_cell(row['SPY'])
            + f"<td>{vix_str}</td><td>{gate}</td></tr>"
        )

    # Week blocks rows
    week_rows = ""
    for _, row in week_blocks.iterrows():
        week_rows += (
            f"<tr><td>{row['Week']}</td>"
            + color_cell(row['Combined'])
            + color_cell(row['Momentum'])
            + color_cell(row['Reversal'])
            + color_cell(row['SPY'])
            + "</tr>"
        )

    # YTD row
    ytd_row = (
        f"<tr><td>YTD {run_date[:4]}</td>"
        + color_cell(ytd_stats['combined'])
        + color_cell(ytd_stats['momentum'])
        + color_cell(ytd_stats['reversal'])
        + color_cell(ytd_stats['spy'])
        + "</tr>"
    )

    # Position tables
    nm = etf_names or {}
    tr = ticker_rets or {}

    def ret_td(v):
        if np.isnan(v): return '<td style="color:#aaa">—</td>'
        color = "#27ae60" if v > 0 else "#e74c3c"
        return f'<td style="color:{color}">{v:+.1%}</td>'

    def pos_table(tickers, side, side_color, descending=True):
        sorted_tickers = sorted(
            tickers,
            key=lambda t: tr.get(t, {}).get("252d", np.nan),
            reverse=descending,
        )
        header = (
            '<tr>'
            '<th style="text-align:left;width:55px">Ticker</th>'
            '<th style="text-align:left">Name</th>'
            f'<th style="color:{side_color};width:50px">Side</th>'
            '<th style="width:60px">1d</th>'
            '<th style="width:60px">5d</th>'
            '<th style="width:70px">252d</th>'
            '</tr>'
        )
        rows = "".join(
            f'<tr>'
            f'<td style="font-family:monospace;font-weight:600">{t}</td>'
            f'<td style="color:#555;font-size:12px">{nm.get(t, "")}</td>'
            f'<td style="color:{side_color};text-align:center;font-weight:600">{side}</td>'
            + ret_td(tr.get(t, {}).get("1d", np.nan))
            + ret_td(tr.get(t, {}).get("5d", np.nan))
            + ret_td(tr.get(t, {}).get("252d", np.nan))
            + '</tr>'
            for t in sorted_tickers
        )
        avgs = {
            lb: np.nanmean([tr.get(t, {}).get(lb, np.nan) for t in sorted_tickers])
            for lb in ("1d", "5d", "252d")
        }
        avg_row = (
            f'<tr style="background:#f7f8fa;font-weight:700;border-top:2px solid #dde">'
            f'<td colspan="3" style="color:#444">Average</td>'
            + ret_td(avgs["1d"])
            + ret_td(avgs["5d"])
            + ret_td(avgs["252d"])
            + '</tr>'
        )
        return header + rows + avg_row

    rev_status = (
        f'<span style="color:#27ae60">Active (VIX={vix_last:.1f} > {VIX_GATE})</span>'
        if rev_active else
        f'<span style="color:#7f8c8d">Inactive (VIX={vix_last:.1f} ≤ {VIX_GATE})</span>'
    )

    css = """
    body{font-family:'Segoe UI',Arial,sans-serif;background:#f5f6fa;margin:0;padding:20px}
    .card{background:#fff;border-radius:8px;padding:20px 24px;margin-bottom:18px;
          box-shadow:0 1px 4px rgba(0,0,0,.10)}
    h1{color:#2c3e50;font-size:20px;margin:0 0 4px}
    h2{color:#34495e;font-size:14px;font-weight:600;margin:0 0 12px;
       text-transform:uppercase;letter-spacing:.5px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th{background:#f0f3f7;color:#555;padding:7px 10px;text-align:right;border-bottom:2px solid #dde}
    th:first-child{text-align:left}
    td{padding:6px 10px;border-bottom:1px solid #eef;text-align:right}
    td:first-child{text-align:left;color:#333}
    tr:last-child td{border-bottom:none}
    .subtitle{color:#7f8c8d;font-size:12px;margin-top:2px}
    .pos-tag{font-size:11px;padding:2px 8px;border-radius:12px;font-weight:600}
    .long-tag{background:#eafaf1;color:#27ae60}
    .short-tag{background:#fdedec;color:#e74c3c}
    """

    def _img_card(cid, caption):
        return (
            f'<div class="card" style="padding:12px 16px">'
            f'<img src="cid:{cid}" '
            f'style="width:100%;max-width:900px;display:block;margin:0 auto" alt="{caption}"/>'
            f'<div style="text-align:center;font-size:11px;color:#999;margin-top:4px">{caption}</div>'
            f'</div>'
        )

    chart_html    = _img_card("chart_main", f"IS (solid) through {OS_START} · OS (dotted) thereafter") if has_main_chart else ""
    os_chart_html = _img_card("chart_os",   f"Trailing {TRAIL_DAYS}-day cumulative return")             if has_os_chart  else ""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>{css}</style></head><body>
    <div class="card">
      <h1>Industry ETF Combined Strategy</h1>
      <div class="subtitle">Daily Monitor — {run_date} &nbsp;|&nbsp; VIX: {vix_last:.1f}
        &nbsp;|&nbsp; Reversal leg: {rev_status}</div>
    </div>
    {chart_html}
    {os_chart_html}

    <div class="card">
      <h2>Last 5 Trading Days</h2>
      <table>
        <tr><th style="text-align:left">Date</th>
            <th>Combined</th><th>Momentum</th><th>Reversal</th><th>SPY</th>
            <th>VIX</th><th>Rev Active</th></tr>
        {day_rows}
        <tr style="background:#f9f9f9;font-weight:600">
          <td>Week (excl. today)</td>
          {"".join(color_cell(last5[~last5.get('intraday', pd.Series(False, index=last5.index)).astype(bool)][c].add(1).prod()-1) for c in ['Combined','Momentum','Reversal','SPY'])}
          <td colspan="2"></td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Last 4 Weeks</h2>
      <table>
        <tr><th style="text-align:left">Week of</th>
            <th>Combined</th><th>Momentum</th><th>Reversal</th><th>SPY</th></tr>
        {week_rows}
        {ytd_row}
      </table>
    </div>

    <div class="card">
      <h2>Full-Period Stats (IS: 2017–{run_date[:4]})</h2>
      <table>
        <tr><th style="text-align:left">Strategy</th>
            <th>Ann Ret</th><th>Ann Vol</th><th>Sharpe</th>
            <th>Max DD</th><th>Cur DD</th><th>Total Ret</th></tr>
        {stat_row(full_stats['combined'], bold=True)}
        {stat_row(full_stats['momentum'])}
        {stat_row(full_stats['reversal'])}
        {stat_row(full_stats['spy'])}
      </table>
    </div>

    <div class="card">
      <h2>Current Positions</h2>
      <table style="width:100%;margin-bottom:16px">
        <tr><td colspan="6" style="font-weight:700;color:#27ae60;padding:6px 0;border:none">
          Momentum Longs ({len(mom_longs)})</td></tr>
        {pos_table(mom_longs, 'LONG', '#27ae60', descending=True)}
      </table>
      <table style="width:100%;margin-bottom:16px">
        <tr><td colspan="6" style="font-weight:700;color:#e74c3c;padding:6px 0;border:none">
          Momentum Shorts ({len(mom_shorts)})</td></tr>
        {pos_table(mom_shorts, 'SHORT', '#e74c3c', descending=False)}
      </table>
      <table style="width:100%">
        <tr><td colspan="6" style="font-weight:700;color:{'#2980b9' if rev_active else '#aaa'};padding:6px 0;border:none">
          Reversal Longs ({len(rev_longs)}) — {'VIX active ✓' if rev_active else f'Inactive (VIX={vix_last:.1f} ≤ {VIX_GATE})'}</td></tr>
        {pos_table(rev_longs, 'LONG', '#2980b9' if rev_active else '#bbb', descending=True)}
      </table>
    </div>

    {_target_shares_html(target_shares, etf_names, vix_last, rev_active, vix_gate=vix_gate, vix_gate_date=vix_gate_date)}

    <div style="color:#aaa;font-size:11px;text-align:center;margin-top:12px">
      ERC Momentum (252d quartile L/S) + ERC Reversal (5d quartile, VIX&gt;20 gated) | Equal-vol combined
    </div>
    </body></html>"""
    return html


# ── Email sender ───────────────────────────────────────────────────────────────

def send_email(subject, html_body, to_addr, from_addr, app_password, images=None):
    """images: dict of {cid_name: png_bytes} embedded via Content-ID references."""
    msg_root = MIMEMultipart("related")
    msg_root["Subject"] = subject
    msg_root["From"]    = from_addr
    msg_root["To"]      = to_addr

    msg_alt = MIMEMultipart("alternative")
    msg_root.attach(msg_alt)
    msg_alt.attach(MIMEText(html_body, "html"))

    for cid, png_bytes in (images or {}).items():
        img = MIMEImage(png_bytes, "png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg_root.attach(img)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(from_addr, app_password)
        server.sendmail(from_addr, to_addr, msg_root.as_string())
    print(f"  Email sent to {to_addr}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    run_date = date.today().strftime("%Y-%m-%d")
    print(f"=== Industry ETF Daily Monitor  {run_date} ===")

    # ── Credentials ──────────────────────────────────────────────────────────
    gmail_user  = os.environ.get("GMAIL_USER")
    gmail_pass  = os.environ.get("GMAIL_APP_PASS")
    to_addr     = os.environ.get("MONITOR_TO", "feilu.fang@gmail.com")
    send_mail   = bool(gmail_user and gmail_pass)
    if not send_mail:
        print("  GMAIL_USER / GMAIL_APP_PASS not set — will print report only (no email).")

    # ── Refresh prices ────────────────────────────────────────────────────────
    print("Refreshing VIX and SPY ...")
    vix_raw = yf.download("^VIX", start="2014-01-01", auto_adjust=True, progress=False)
    spy_raw = yf.download("SPY",  start="2014-01-01", auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze().dropna()
    vix.index = vix.index.tz_localize(None).strftime("%Y-%m-%d")
    spy_ret = spy_raw["Close"].squeeze().pct_change().dropna()
    spy_ret.index = spy_ret.index.tz_localize(None).strftime("%Y-%m-%d")
    vix_last = float(vix.iloc[-1])          # current intraday VIX (for display)

    # ── Industry prices ───────────────────────────────────────────────────────
    _ticker_file = HERE / "Industry_ETF_Tickers_liquid.csv"
    if not _ticker_file.exists():
        _ticker_file = HERE / "Industry_ETF_Tickers_filtered.csv"
    if not _ticker_file.exists():
        _ticker_file = HERE / "Industry_ETF_Tickers.csv"
    ind_tickers = [t.strip() for t in
                   _ticker_file.read_text(encoding="utf-8-sig")
                   .strip().splitlines() if t.strip()]
    print(f"Universe: {len(ind_tickers)} tickers ({_ticker_file.name})")
    close = pd.read_parquet(HERE / "data" / "industry_etf_daily.parquet")
    close = close[[t for t in ind_tickers if t in close.columns]].sort_index()
    close = close.where(close.gt(0)).ffill()
    ind_first = close.index[MOM_WIN]
    print(f"Data: {close.index[0]} -> {close.index[-1]}  ({len(close)} days)")

    # ── Run strategies ────────────────────────────────────────────────────────
    print("Running momentum ...")
    pnl_mom, mom_weights = run_momentum(close, vix)
    pnl_mom = pnl_mom[pnl_mom.index >= ind_first]

    print("Running reversal ...")
    pnl_rev, rev_weights = run_reversal(close, vix)
    pnl_rev = pnl_rev[pnl_rev.index >= ind_first]

    pnl_comb = combine_ev(pnl_mom, pnl_rev)
    print(f"Combined: {pnl_comb.index[0]} -> {pnl_comb.index[-1]}")

    # ── Current positions (needed before intraday) ────────────────────────────
    mom_longs  = sorted(mom_weights[mom_weights > 0.001].index.tolist())
    mom_shorts = sorted(mom_weights[mom_weights < -0.001].index.tolist())
    # Reversal gate uses previous CLOSE VIX (close.index[-1]), not today's intraday.
    # e.g. script runs at 10am June 8: close[-1]=June 5, vix(June 5)=21.5 → ACTIVE.
    vix_gate_date = close.index[-1]
    vix_gate      = float(vix.get(vix_gate_date, vix_last))
    rev_active    = vix_gate > VIX_GATE
    if rev_active:
        rev_longs = sorted(rev_weights[rev_weights > 0.001].index.tolist())
    else:
        # compute would-be basket from latest 5d signal regardless of VIX gate
        rev_sig_last = close.pct_change(REV_LB).iloc[-1].dropna()
        n_q = max(1, len(rev_sig_last) // 4)
        rev_longs = sorted(rev_sig_last.nsmallest(n_q).index.tolist())

    # ── Intraday prices for today ─────────────────────────────────────────────
    print("Fetching intraday prices ...")
    all_intraday_tickers = list(close.columns) + ["SPY", "^VIX"]
    try:
        intra_raw = yf.download(
            all_intraday_tickers, period="1d", interval="1m",
            auto_adjust=True, progress=False,
        )
        intra_close = intra_raw["Close"] if isinstance(intra_raw.columns, pd.MultiIndex) else intra_raw
        intra_close = intra_close.dropna(how="all")

        # latest intraday price per ticker
        latest_price = intra_close.iloc[-1]

        # prev close = yesterday's close from parquet (for ETFs) or SPY daily
        # build a prev_close series aligned to all intraday tickers
        prev_close_dict = {}
        for t in close.columns:
            s = close[t].dropna()
            if len(s) >= 1:
                prev_close_dict[t] = float(s.iloc[-1])
        # SPY and VIX from downloaded daily
        spy_prev = spy_raw["Close"].squeeze().dropna()
        prev_close_dict["SPY"] = float(spy_prev.iloc[-1])
        vix_prev = vix_raw["Close"].squeeze().dropna()
        prev_close_dict["^VIX"] = float(vix_prev.iloc[-1])

        prev_price = pd.Series(prev_close_dict)
        intra_ret = (latest_price / prev_price - 1).dropna()

        # Intraday time label
        last_bar_time = intra_close.index[-1].tz_convert("America/New_York")
        intra_label = f"Today {last_bar_time.strftime('%H:%M ET')} (intraday)"

        # Approximate strategy P&L using yesterday's weights × today's intraday returns
        def _weighted_pnl(weights):
            tickers_in = [t for t in weights.index if t in intra_ret.index]
            if not tickers_in:
                return np.nan
            return float((weights[tickers_in] * intra_ret[tickers_in]).sum())

        # Equal-vol weights for today: reuse last-row allocation ratio
        # (combine_ev already baked into pnl_comb; approximate as 50/50 by vol)
        df_ev = pd.DataFrame({"mom": pnl_mom, "rev": pnl_rev}).dropna()
        ia = 1.0 / df_ev["mom"].rolling(VOL_WIN).std().replace(0, np.nan)
        ib = 1.0 / df_ev["rev"].rolling(VOL_WIN).std().replace(0, np.nan)
        t_ev = ia + ib
        w_mom_today = float((ia / t_ev).iloc[-1])
        w_rev_today = float((ib / t_ev).iloc[-1])

        intra_mom = _weighted_pnl(mom_weights)
        intra_rev = _weighted_pnl(rev_weights) if rev_active else 0.0
        intra_comb = (w_mom_today * intra_mom + w_rev_today * intra_rev
                      if not (np.isnan(intra_mom) or np.isnan(intra_rev))
                      else np.nan)
        intra_spy = float(intra_ret.get("SPY", np.nan))
        intra_vix = float(latest_price.get("^VIX", np.nan))

        today_row = {
            "Date":     intra_label,
            "Combined": intra_comb,
            "Momentum": intra_mom,
            "Reversal": intra_rev,
            "SPY":      intra_spy,
            "VIX":      intra_vix,
            "intraday": True,
        }
        print(f"  Intraday as of {last_bar_time.strftime('%H:%M ET')}: "
              f"Combined={intra_comb:+.2%}  Momentum={intra_mom:+.2%}  SPY={intra_spy:+.2%}  VIX={intra_vix:.1f}")

        intra_ret_today = intra_ret  # stored for later override

    except Exception as e:
        print(f"  Intraday fetch failed: {e}")
        today_row = None
        intra_ret_today = None

    # ── Target shares for today's orders ─────────────────────────────────────
    price_ref = latest_price if (today_row is not None and latest_price is not None) else close.iloc[-1]
    def _target_shares(weights):
        out = {}
        for t, w in weights.items():
            if not np.isfinite(w) or abs(w) < 1e-4: continue
            p = float(price_ref.get(t, 0) or 0)
            if not np.isfinite(p) or p <= 0: continue
            qty = int(round(abs(w) * NOTIONAL / p))
            if qty: out[t] = qty if w > 0 else -qty
        return out

    target_shares = _target_shares(mom_weights)
    if rev_active:
        for t, q in _target_shares(rev_weights).items():
            target_shares[t] = target_shares.get(t, 0) + q
    target_shares = {t: q for t, q in target_shares.items() if q != 0}

    # ── Last 5 days ───────────────────────────────────────────────────────────
    last5_dates = pnl_comb.index[-5:].tolist()
    rows = []
    for d in last5_dates:
        rows.append({
            "Date":     d,
            "Combined": pnl_comb.get(d, np.nan),
            "Momentum": pnl_mom.get(d, np.nan),
            "Reversal": pnl_rev.get(d, np.nan),
            "SPY":      spy_ret.get(d, np.nan),
            "VIX":      vix.get(d, np.nan),
            "intraday": False,
        })
    if today_row:
        rows.append(today_row)
    last5_df = pd.DataFrame(rows)

    print(f"\n{'Date':<12} {'Combined':>10} {'Momentum':>10} {'Reversal':>10} {'SPY':>9}  VIX")
    print("-" * 65)
    for _, r in last5_df.iterrows():
        v = r["VIX"]
        print(f"  {r['Date']}  {r['Combined']:>+9.2%}  {r['Momentum']:>+9.2%}  "
              f"{r['Reversal']:>+9.2%}  {r['SPY']:>+8.2%}  {v:.1f}")
    week_tot = {c: last5_df[c].add(1).prod() - 1 for c in ["Combined","Momentum","Reversal","SPY"]}
    print("-" * 65)
    print(f"  {'Week':10}  {week_tot['Combined']:>+9.2%}  {week_tot['Momentum']:>+9.2%}  "
          f"{week_tot['Reversal']:>+9.2%}  {week_tot['SPY']:>+8.2%}")

    # ── Last 4 weeks ──────────────────────────────────────────────────────────
    all_dates = pnl_comb.index.tolist()
    last20 = all_dates[-20:]
    week_data = []
    for blk in [last20[i:i+5] for i in range(0, 20, 5)]:
        week_data.append({
            "Week":     blk[0],
            "Combined": (1 + pnl_comb.loc[blk]).prod() - 1,
            "Momentum": (1 + pnl_mom.reindex(blk).fillna(0)).prod() - 1,
            "Reversal": (1 + pnl_rev.reindex(blk).fillna(0)).prod() - 1,
            "SPY":      (1 + spy_ret.reindex(blk).fillna(0)).prod() - 1,
        })
    week_blocks_df = pd.DataFrame(week_data)

    # ── YTD ───────────────────────────────────────────────────────────────────
    yr = run_date[:4]
    ytd_stats = {
        "combined": (1 + pnl_comb[pnl_comb.index >= f"{yr}-01-01"]).prod() - 1,
        "momentum": (1 + pnl_mom[pnl_mom.index >= f"{yr}-01-01"]).prod() - 1,
        "reversal": (1 + pnl_rev[pnl_rev.index >= f"{yr}-01-01"]).prod() - 1,
        "spy":      (1 + spy_ret[spy_ret.index >= f"{yr}-01-01"]).prod() - 1,
    }
    print(f"\nYTD {yr}: Combined={ytd_stats['combined']:+.2%}  "
          f"Momentum={ytd_stats['momentum']:+.2%}  "
          f"Reversal={ytd_stats['reversal']:+.2%}  "
          f"SPY={ytd_stats['spy']:+.2%}")

    # ── Full stats ────────────────────────────────────────────────────────────
    full_stats = {
        "combined": stats(pnl_comb,   "Combined (EV)"),
        "momentum": stats(pnl_mom,    "Momentum"),
        "reversal": stats(pnl_rev,    "Reversal VIX>20"),
        "spy":      stats(spy_ret,    "SPY"),
    }

    print(f"\nMomentum longs  ({len(mom_longs)}): {', '.join(mom_longs)}")
    print(f"Momentum shorts ({len(mom_shorts)}): {', '.join(mom_shorts)}")
    if rev_active:
        print(f"Reversal longs  ({len(rev_longs)}): {', '.join(rev_longs)}")
    else:
        print(f"Reversal: inactive (signal VIX={vix_gate:.1f} on {vix_gate_date} <= {VIX_GATE})")

    # ── Load ETF names ────────────────────────────────────────────────────────
    names_file = HERE / "etf_names.csv"
    etf_names = {}
    if names_file.exists():
        etf_names = pd.read_csv(names_file).set_index("ticker")["name"].to_dict()

    # ── Per-ticker returns for position table ─────────────────────────────────
    all_pos = list(set(mom_longs + mom_shorts + rev_longs))
    ticker_rets = {}
    for t in all_pos:
        if t not in close.columns:
            continue
        s = close[t].dropna()
        def _ret(lb, _s=s):
            if len(_s) < lb + 1:
                return np.nan
            return float(_s.iloc[-1] / _s.iloc[-lb-1] - 1)
        ticker_rets[t] = {"1d": _ret(1), "5d": _ret(5), "252d": _ret(252)}

    # Override 1d with live intraday return where available
    if intra_ret_today is not None:
        for t in all_pos:
            if t in intra_ret_today.index:
                ticker_rets.setdefault(t, {})["1d"] = float(intra_ret_today[t])

    # ── Build charts ──────────────────────────────────────────────────────────
    print("Building charts ...")
    chart_bytes    = build_cumret_chart(pnl_mom, pnl_rev, pnl_comb)
    os_chart_bytes = build_os_chart(pnl_mom, pnl_rev, pnl_comb)

    # ── Build HTML (uses cid: references for email) ───────────────────────────
    html = build_html(
        run_date, last5_df, week_blocks_df, ytd_stats,
        full_stats, vix_last, ytd_stats["spy"],
        mom_longs, mom_shorts, rev_longs, rev_active,
        etf_names=etf_names, ticker_rets=ticker_rets,
        has_main_chart=bool(chart_bytes), has_os_chart=bool(os_chart_bytes),
        target_shares=target_shares, vix_gate=vix_gate, vix_gate_date=vix_gate_date,
    )

    # Save HTML + PNG files locally for inspection (replace cid: with file refs)
    out_dir = HERE / "results_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_html = html
    if chart_bytes:
        (out_dir / "chart_main.png").write_bytes(chart_bytes)
        local_html = local_html.replace('src="cid:chart_main"', 'src="chart_main.png"')
    if os_chart_bytes:
        (out_dir / "chart_os.png").write_bytes(os_chart_bytes)
        local_html = local_html.replace('src="cid:chart_os"', 'src="chart_os.png"')
    out_html = out_dir / "daily_monitor_latest.html"
    out_html.write_text(local_html, encoding="utf-8")
    print(f"\nHTML report saved -> {out_html}")

    if send_mail:
        week_pct = f"{week_tot['Combined']:+.1%}"
        ytd_pct  = f"{ytd_stats['combined']:+.1%}"
        subject  = (f"Industry ETF Monitor {run_date} | "
                    f"Week {week_pct}  YTD {ytd_pct}  VIX {vix_last:.1f}")
        images = {}
        if chart_bytes:    images["chart_main"] = chart_bytes
        if os_chart_bytes: images["chart_os"]   = os_chart_bytes
        print(f"Sending email: {subject}")
        send_email(subject, html, to_addr, gmail_user, gmail_pass, images=images)
    else:
        print("\nTo enable email, set environment variables:")
        print("  $env:GMAIL_USER     = 'feilu.fang@gmail.com'")
        print("  $env:GMAIL_APP_PASS = '<your-app-password>'")
        print("  $env:MONITOR_TO     = 'feilu.fang@gmail.com'   (optional, default)")


if __name__ == "__main__":
    main()

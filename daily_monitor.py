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

import numpy as np
import pandas as pd
import scipy.optimize as sco
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE       = Path(__file__).parent
MOM_WIN    = 252
REV_LB     = 5
COV_WIN    = 252
VOL_WIN    = 252
VIX_GATE   = 20


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

def build_html(
    run_date, last5, week_blocks, ytd_stats,
    full_stats, vix_last, spy_ytd,
    mom_longs, mom_shorts, rev_longs, rev_active
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
        day_rows += (
            f"<tr><td>{row['Date']}</td>"
            + color_cell(row['Combined'])
            + color_cell(row['Momentum'])
            + color_cell(row['Reversal'])
            + color_cell(row['SPY'])
            + f"<td>{vix_val:.1f}</td><td>{gate}</td></tr>"
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
    def pos_table(names, side, color):
        rows = "".join(
            f'<tr><td style="font-family:monospace">{t}</td>'
            f'<td style="color:{color};text-align:center">{side}</td></tr>'
            for t in names
        )
        return rows

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

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>{css}</style></head><body>
    <div class="card">
      <h1>Industry ETF Combined Strategy</h1>
      <div class="subtitle">Daily Monitor — {run_date} &nbsp;|&nbsp; VIX: {vix_last:.1f}
        &nbsp;|&nbsp; Reversal leg: {rev_status}</div>
    </div>

    <div class="card">
      <h2>Last 5 Trading Days</h2>
      <table>
        <tr><th style="text-align:left">Date</th>
            <th>Combined</th><th>Momentum</th><th>Reversal</th><th>SPY</th>
            <th>VIX</th><th>Rev Active</th></tr>
        {day_rows}
        <tr style="background:#f9f9f9;font-weight:600">
          <td>Week total</td>
          {"".join(color_cell(last5[c].add(1).prod()-1) for c in ['Combined','Momentum','Reversal','SPY'])}
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
      <table style="width:48%;display:inline-table;vertical-align:top;margin-right:4%">
        <tr><th style="text-align:left" colspan="2">Momentum Longs ({len(mom_longs)})</th></tr>
        {pos_table(mom_longs, 'LONG', '#27ae60')}
      </table>
      <table style="width:48%;display:inline-table;vertical-align:top">
        <tr><th style="text-align:left" colspan="2">Momentum Shorts ({len(mom_shorts)})</th></tr>
        {pos_table(mom_shorts, 'SHORT', '#e74c3c')}
      </table>
      {"" if not rev_active else f'''
      <br><table style="margin-top:12px">
        <tr><th style="text-align:left" colspan="2">Reversal Longs ({len(rev_longs)}) — VIX active</th></tr>
        {pos_table(rev_longs, 'LONG', '#2980b9')}
      </table>'''}
    </div>

    <div style="color:#aaa;font-size:11px;text-align:center;margin-top:12px">
      ERC Momentum (252d quartile L/S) + ERC Reversal (5d quartile, VIX&gt;20 gated) | Equal-vol combined
    </div>
    </body></html>"""
    return html


# ── Email sender ───────────────────────────────────────────────────────────────

def send_email(subject, html_body, to_addr, from_addr, app_password):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(from_addr, app_password)
        server.sendmail(from_addr, to_addr, msg.as_string())
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
    vix_last = float(vix.iloc[-1])

    # ── Industry prices ───────────────────────────────────────────────────────
    ind_tickers = [t.strip() for t in
                   (HERE / "Industry_ETF_Tickers.csv").read_text(encoding="utf-8-sig")
                   .strip().splitlines() if t.strip()]
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
        })
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

    # ── Current positions ─────────────────────────────────────────────────────
    mom_longs  = sorted(mom_weights[mom_weights > 0.001].index.tolist())
    mom_shorts = sorted(mom_weights[mom_weights < -0.001].index.tolist())
    rev_longs  = sorted(rev_weights[rev_weights > 0.001].index.tolist())
    rev_active = vix_last > VIX_GATE

    print(f"\nMomentum longs  ({len(mom_longs)}): {', '.join(mom_longs)}")
    print(f"Momentum shorts ({len(mom_shorts)}): {', '.join(mom_shorts)}")
    if rev_active:
        print(f"Reversal longs  ({len(rev_longs)}): {', '.join(rev_longs)}")
    else:
        print(f"Reversal: inactive (VIX={vix_last:.1f} <= {VIX_GATE})")

    # ── Build HTML and send ───────────────────────────────────────────────────
    html = build_html(
        run_date, last5_df, week_blocks_df, ytd_stats,
        full_stats, vix_last, ytd_stats["spy"],
        mom_longs, mom_shorts, rev_longs, rev_active,
    )

    # Save HTML locally for inspection
    out_html = HERE / "results_combined" / "daily_monitor_latest.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\nHTML report saved -> {out_html}")

    if send_mail:
        week_pct = f"{week_tot['Combined']:+.1%}"
        ytd_pct  = f"{ytd_stats['combined']:+.1%}"
        subject  = (f"Industry ETF Monitor {run_date} | "
                    f"Week {week_pct}  YTD {ytd_pct}  VIX {vix_last:.1f}")
        print(f"Sending email: {subject}")
        send_email(subject, html, to_addr, gmail_user, gmail_pass)
    else:
        print("\nTo enable email, set environment variables:")
        print("  $env:GMAIL_USER     = 'feilu.fang@gmail.com'")
        print("  $env:GMAIL_APP_PASS = '<your-app-password>'")
        print("  $env:MONITOR_TO     = 'feilu.fang@gmail.com'   (optional, default)")


if __name__ == "__main__":
    main()

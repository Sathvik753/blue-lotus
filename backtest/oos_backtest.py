"""
Out-of-sample event-study backtest for the Blue Lotus stress engine.

For every (asset, crisis) cell we fit the engine *only* on data that ended
before the crisis began, then ask one question: did the model's predicted
drawdown distribution contain what actually happened?

The headline statistic is the percentile rank of the realized max drawdown
inside the simulated drawdown distribution (a probability-integral transform).
A well-calibrated tail model puts realized crisis outcomes in its lower tail
only as often as that tail's mass implies — so across many cells the ranks
should look roughly uniform, with breaches of the 5% bound happening ~5% of
the time on ordinary data and the model still covering most named crises.
"""

import os
import io
import sys
import json
import warnings
import contextlib
from dataclasses import dataclass, asdict
from datetime import date

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.core import BlueLotusEngine, fetch_returns

MIN_TRAIN_OBS = 504      # need at least ~2y of history before a crisis to fit
DEFAULT_NPATHS = 5000
N_BOOTSTRAP = 100
SEED = 42

# Crisis windows: roughly peak-to-trough of each episode. The engine never
# sees data on or after `start` for that cell.
EVENTS = [
    {"key": "flash_2010",   "label": "2010 Flash Crash / Euro I",   "start": "2010-04-23", "end": "2010-07-02"},
    {"key": "euro_2011",    "label": "2011 US Downgrade / Euro II",  "start": "2011-07-22", "end": "2011-10-03"},
    {"key": "china_2015",   "label": "2015-16 China / Oil Selloff",  "start": "2015-08-17", "end": "2016-02-11"},
    {"key": "vol_2018",     "label": "Feb 2018 Volmageddon",         "start": "2018-01-26", "end": "2018-02-09"},
    {"key": "q4_2018",      "label": "Q4 2018 Selloff",              "start": "2018-09-20", "end": "2018-12-24"},
    {"key": "covid_2020",   "label": "March 2020 COVID Crash",       "start": "2020-02-19", "end": "2020-03-23"},
    {"key": "rates_2022",   "label": "2022 Rate-Shock Bear",         "start": "2022-01-03", "end": "2022-10-12"},
    {"key": "svb_2023",     "label": "2023 Regional Bank Crisis",    "start": "2023-03-08", "end": "2023-03-24"},
    {"key": "gfc_2008",     "label": "2008 Global Financial Crisis", "start": "2008-09-02", "end": "2009-03-09"},
]

# (ticker, human label, history start). Start dates sit at or after each
# fund's inception so the fetch doesn't waste time on empty ranges.
ASSETS = [
    ("SPY", "S&P 500",            "1995-01-01"),
    ("QQQ", "Nasdaq 100",         "1999-04-01"),
    ("IWM", "Russell 2000",       "2000-06-01"),
    ("EFA", "Developed ex-US",    "2001-09-01"),
    ("TLT", "20Y+ Treasuries",    "2002-08-01"),
    ("GLD", "Gold",               "2004-12-01"),
    ("HYG", "High-Yield Credit",  "2007-05-01"),
    ("XLF", "Financials",         "1999-01-01"),
    ("XLE", "Energy",             "1999-01-01"),
    ("DIA", "Dow 30",             "1998-02-01"),
]


@dataclass
class CellResult:
    asset: str
    asset_label: str
    event: str
    event_label: str
    train_obs: int
    horizon_days: int
    realized_max_dd: float
    realized_es: float
    realized_total_return: float
    pred_dd_median: float
    pred_dd_p5: float
    pred_dd_p1: float
    pred_es_aggregate: float
    realized_pct_rank: float   # where realized DD sits in the simulated dist (0=worst)
    covered_90: bool           # realized inside [p5, p95]
    breach_p5: bool            # realized worse than 5th-pctile prediction
    breach_p1: bool            # realized worse than 1st-pctile prediction
    coverage_ratio: float      # realized / pred_p5 (both negative)


def max_drawdown(returns):
    cum = np.cumsum(returns)
    dd = cum - np.maximum.accumulate(cum)
    return float(dd.min())


def realized_tail_es(returns, alpha=0.05):
    if len(returns) < 5:
        return float(np.min(returns)) if len(returns) else float("nan")
    q = np.quantile(returns, alpha)
    tail = returns[returns <= q]
    return float(tail.mean()) if len(tail) else float(q)


def run_cell(ticker, asset_label, returns, dates, event):
    start = np.datetime64(event["start"], "D")
    end = np.datetime64(event["end"], "D")
    d = np.asarray(dates, dtype="datetime64[D]")

    window = (d >= start) & (d <= end)
    if window.sum() < 8:
        return None, "window too short / no data"

    train_mask = d < start
    n_train = int(train_mask.sum())
    if n_train < MIN_TRAIN_OBS:
        return None, f"only {n_train} train obs"

    train_r = returns[train_mask]
    window_r = returns[window]
    horizon = int(window.sum())

    eng = BlueLotusEngine(
        strategy_name=ticker, n_paths=DEFAULT_NPATHS, horizon=horizon,
        random_seed=SEED, run_sensitivity=False, run_backtest=False,
        n_bootstrap=N_BOOTSTRAP,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        res = eng.run(train_r, dates=None, verbose=False)

    sm = res["stress"]
    dd_sim = np.asarray(sm.drawdown_dist, dtype=float)

    realized_dd = max_drawdown(window_r)
    realized_es = realized_tail_es(window_r)
    realized_ret = float(np.expm1(np.sum(np.log1p(window_r))))

    pred_p1 = float(np.percentile(dd_sim, 1))
    pred_p5 = float(np.percentile(dd_sim, 5))
    pred_p95 = float(np.percentile(dd_sim, 95))
    pct_rank = float((dd_sim <= realized_dd).mean())

    cell = CellResult(
        asset=ticker, asset_label=asset_label,
        event=event["key"], event_label=event["label"],
        train_obs=n_train, horizon_days=horizon,
        realized_max_dd=realized_dd,
        realized_es=realized_es,
        realized_total_return=realized_ret,
        pred_dd_median=float(sm.dd_median),
        pred_dd_p5=pred_p5,
        pred_dd_p1=pred_p1,
        pred_es_aggregate=float(sm.es_aggregate),
        realized_pct_rank=pct_rank,
        covered_90=(realized_dd >= pred_p5) and (realized_dd <= pred_p95),
        breach_p5=(realized_dd < pred_p5),
        breach_p1=(realized_dd < pred_p1),
        coverage_ratio=float(realized_dd / (pred_p5 - 1e-12)),
    )
    return cell, None


def kupiec_pof(n_breaches, n_total, p):
    """Kupiec proportion-of-failures LR statistic for a VaR exception count."""
    if n_total == 0:
        return float("nan"), float("nan")
    x, n = n_breaches, n_total
    pi = x / n
    if x == 0:
        lr = -2.0 * n * np.log(1 - p)
    elif x == n:
        lr = -2.0 * n * np.log(p)
    else:
        lr = -2.0 * ((n - x) * np.log(1 - p) + x * np.log(p)) \
             + 2.0 * ((n - x) * np.log(1 - pi) + x * np.log(pi))
    # chi-square(1) survival function without scipy dependency at call site
    from scipy.stats import chi2
    p_value = float(chi2.sf(lr, df=1))
    return float(lr), p_value


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    today = date.today().strftime("%Y-%m-%d")
    histories = {}
    print("Fetching price histories...")
    for ticker, label, start in ASSETS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r, d, p = fetch_returns(ticker, start=start, end=today)
            histories[ticker] = (r, d, label)
            print(f"  {ticker:<5} {label:<22} {len(r):>5} obs  {str(d[0])[:10]} -> {str(d[-1])[:10]}")
        except Exception as e:
            print(f"  {ticker:<5} FETCH FAILED: {str(e)[:80]}")

    cells = []
    skipped = []
    print("\nRunning out-of-sample cells (fit pre-crisis, test in-crisis)...")
    for event in EVENTS:
        line = []
        for ticker, (r, d, label) in histories.items():
            try:
                cell, why = run_cell(ticker, label, r, d, event)
            except Exception as e:
                cell, why = None, f"error: {str(e)[:60]}"
            if cell is None:
                skipped.append({"asset": ticker, "event": event["key"], "reason": why})
                continue
            cells.append(cell)
            tag = "OK " if cell.covered_90 else ("br1" if cell.breach_p1 else "br5")
            line.append(f"{ticker}:{cell.realized_max_dd*100:+.0f}/{cell.pred_dd_p5*100:+.0f}[{tag}]")
        print(f"  {event['label']:<34} " + "  ".join(line))

    n = len(cells)
    breaches_p5 = sum(c.breach_p5 for c in cells)
    breaches_p1 = sum(c.breach_p1 for c in cells)
    covered = sum(c.covered_90 for c in cells)
    ranks = np.array([c.realized_pct_rank for c in cells])

    lr5, pv5 = kupiec_pof(breaches_p5, n, 0.05)
    lr1, pv1 = kupiec_pof(breaches_p1, n, 0.01)

    summary = {
        "generated_at": today,
        "n_cells": n,
        "n_events": len(set(c.event for c in cells)),
        "n_assets": len(set(c.asset for c in cells)),
        "p5_breaches": breaches_p5,
        "p5_breach_rate": breaches_p5 / n if n else None,
        "p5_kupiec_lr": lr5,
        "p5_kupiec_pvalue": pv5,
        "p1_breaches": breaches_p1,
        "p1_breach_rate": breaches_p1 / n if n else None,
        "p1_kupiec_lr": lr1,
        "p1_kupiec_pvalue": pv1,
        "covered_90_count": covered,
        "covered_90_rate": covered / n if n else None,
        "mean_pct_rank": float(ranks.mean()) if n else None,
        "median_pct_rank": float(np.median(ranks)) if n else None,
        "mean_coverage_ratio": float(np.mean([c.coverage_ratio for c in cells])) if n else None,
        "n_skipped": len(skipped),
    }

    with open(os.path.join(out_dir, "cells.json"), "w") as f:
        json.dump([asdict(c) for c in cells], f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=2)

    print("\n" + "=" * 64)
    print("BACKTEST SUMMARY")
    print("=" * 64)
    print(f"  Cells evaluated         {n}  ({summary['n_events']} events x {summary['n_assets']} assets)")
    print(f"  p5 tail breaches        {breaches_p5}/{n}  ({summary['p5_breach_rate']:.1%})   Kupiec p={pv5:.3f}")
    print(f"  p1 tail breaches        {breaches_p1}/{n}  ({summary['p1_breach_rate']:.1%})   Kupiec p={pv1:.3f}")
    print(f"  90% band coverage       {covered}/{n}  ({summary['covered_90_rate']:.1%})")
    print(f"  Mean realized PIT rank  {summary['mean_pct_rank']:.3f}  (lower = engine ran hotter than reality)")
    print(f"  Skipped cells           {len(skipped)}")
    print(f"\n  Wrote results to {out_dir}/")


if __name__ == "__main__":
    main()

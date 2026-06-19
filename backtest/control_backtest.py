"""
Calm-period control for the out-of-sample backtest.

Same out-of-sample machinery as oos_backtest, but the test windows are
ordinary six-month stretches that do NOT overlap any named crisis. If the
engine is honestly calibrated, the realized drawdown in these windows should
breach the 5th-percentile prediction roughly 5% of the time and the PIT ranks
should look uniform. This is the control that lets us claim the crisis
breaches are genuine tail events rather than a miscalibrated model.
"""

import os
import io
import json
import contextlib
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np

import oos_backtest
from oos_backtest import (
    ASSETS, EVENTS, fetch_returns, run_cell, kupiec_pof,
)

# The PIT ranks come from the raw simulated drawdown distribution, which is
# unaffected by the bootstrap. Drop the bootstrap count so the larger control
# grid runs quickly without changing any number we actually report.
oos_backtest.N_BOOTSTRAP = 20


def build_calm_windows():
    """Semiannual windows 2006-2024 that don't intersect a crisis window."""
    crises = [(np.datetime64(e["start"], "D"), np.datetime64(e["end"], "D")) for e in EVENTS]
    windows = []
    for year in range(2006, 2025):
        for start_md in ("01-05", "07-05"):
            start = date(year, int(start_md[:2]), int(start_md[3:]))
            end = start + timedelta(days=182)
            s64 = np.datetime64(start.isoformat(), "D")
            e64 = np.datetime64(end.isoformat(), "D")
            # skip if it overlaps any crisis window (with a small buffer)
            overlap = any(s64 <= ce + np.timedelta64(20, "D") and e64 >= cs - np.timedelta64(20, "D")
                          for cs, ce in crises)
            if overlap:
                continue
            windows.append({
                "key": f"calm_{start.isoformat()}",
                "label": f"Calm {start.isoformat()}",
                "start": start.isoformat(),
                "end": end.isoformat(),
            })
    return windows


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")

    windows = build_calm_windows()
    print(f"Built {len(windows)} calm control windows.")

    print("Fetching price histories...")
    histories = {}
    for ticker, label, start in ASSETS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r, d, p = fetch_returns(ticker, start=start, end=today)
            histories[ticker] = (r, d, label)
            print(f"  {ticker:<5} {len(r):>5} obs")
        except Exception as e:
            print(f"  {ticker:<5} FETCH FAILED: {str(e)[:60]}")

    cells = []
    print("\nRunning calm control cells...")
    for w in windows:
        n_ok = 0
        for ticker, (r, d, label) in histories.items():
            try:
                cell, why = run_cell(ticker, label, r, d, w)
            except Exception:
                cell = None
            if cell is not None:
                cells.append(cell)
                n_ok += 1
        print(f"  {w['label']:<18} {n_ok} assets")

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
        "n_windows": len(windows),
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
    }

    with open(os.path.join(out_dir, "control_cells.json"), "w") as f:
        json.dump([asdict(c) for c in cells], f, indent=2)
    with open(os.path.join(out_dir, "control_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print("CALM CONTROL SUMMARY")
    print("=" * 64)
    print(f"  Cells evaluated         {n}")
    print(f"  p5 tail breaches        {breaches_p5}/{n}  ({summary['p5_breach_rate']:.1%})  expected ~5%   Kupiec p={pv5:.3f}")
    print(f"  p1 tail breaches        {breaches_p1}/{n}  ({summary['p1_breach_rate']:.1%})  expected ~1%   Kupiec p={pv1:.3f}")
    print(f"  90% band coverage       {covered}/{n}  ({summary['covered_90_rate']:.1%})  expected ~90%")
    print(f"  Mean realized PIT rank  {summary['mean_pct_rank']:.3f}  (0.5 = perfectly centered)")


if __name__ == "__main__":
    main()

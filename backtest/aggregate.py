"""Roll the per-cell backtest output up into the event- and asset-level
tables that go into the report. Writes aggregates.json and prints the
LaTeX table bodies to stdout."""

import os
import json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def agg(cells, keyfn, labelfn):
    groups = defaultdict(list)
    for c in cells:
        groups[keyfn(c)].append(c)
    rows = []
    for key, cs in groups.items():
        n = len(cs)
        rows.append({
            "key": key,
            "label": labelfn(cs[0]),
            "n": n,
            "mean_realized_dd": float(np.mean([c["realized_max_dd"] for c in cs])),
            "worst_realized_dd": float(np.min([c["realized_max_dd"] for c in cs])),
            "mean_pred_p5": float(np.mean([c["pred_dd_p5"] for c in cs])),
            "p5_breaches": sum(c["breach_p5"] for c in cs),
            "p1_breaches": sum(c["breach_p1"] for c in cs),
            "covered_90": sum(c["covered_90"] for c in cs),
            "covered_rate": sum(c["covered_90"] for c in cs) / n,
            "mean_pct_rank": float(np.mean([c["realized_pct_rank"] for c in cs])),
            "mean_coverage_ratio": float(np.mean([c["coverage_ratio"] for c in cs])),
        })
    return rows


def main():
    crisis = load("cells.json")
    control = load("control_cells.json")

    # preserve crisis chronological order via first appearance
    order = []
    for c in crisis:
        if c["event"] not in order:
            order.append(c["event"])

    by_event = agg(crisis, lambda c: c["event"], lambda c: c["event_label"])
    by_event.sort(key=lambda r: order.index(r["key"]))
    by_asset = agg(crisis, lambda c: c["asset"], lambda c: c["asset_label"])
    by_asset.sort(key=lambda r: r["mean_pct_rank"])

    out = {"by_event": by_event, "by_asset": by_asset}
    with open(os.path.join(RES, "aggregates.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n% --- per-event table body ---")
    for r in by_event:
        print(f"{r['label']} & {r['n']} & {r['mean_realized_dd']*100:.1f} & "
              f"{r['mean_pred_p5']*100:.1f} & {r['p5_breaches']}/{r['n']} & "
              f"{r['p1_breaches']}/{r['n']} & {r['covered_rate']*100:.0f}\\% & "
              f"{r['mean_pct_rank']:.2f} \\\\")

    print("\n% --- per-asset table body ---")
    for r in by_asset:
        print(f"{r['label']} ({r['key']}) & {r['n']} & {r['mean_realized_dd']*100:.1f} & "
              f"{r['mean_pred_p5']*100:.1f} & {r['p5_breaches']}/{r['n']} & "
              f"{r['covered_rate']*100:.0f}\\% & {r['mean_pct_rank']:.2f} \\\\")

    # a few exemplar cells for the narrative
    print("\n% --- exemplar cells ---")
    want = [("SPY", "covid_2020"), ("XLF", "gfc_2008"), ("SPY", "china_2015"),
            ("SPY", "svb_2023"), ("XLE", "covid_2020")]
    for tkr, ev in want:
        for c in crisis:
            if c["asset"] == tkr and c["event"] == ev:
                print(f"  {tkr:<4} {c['event_label']:<32} realized={c['realized_max_dd']*100:6.1f}%  "
                      f"pred_p5={c['pred_dd_p5']*100:6.1f}%  pred_p1={c['pred_dd_p1']*100:6.1f}%  "
                      f"rank={c['realized_pct_rank']:.3f}")


if __name__ == "__main__":
    main()

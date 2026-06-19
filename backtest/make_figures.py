"""Render the report figures from the saved backtest cells."""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(RES, "figs")
os.makedirs(FIGS, exist_ok=True)

DARK = "#0D1B2A"
PANEL = "#111E2D"
BLUE = "#1B4F72"
TEAL = "#148F77"
GOLD = "#D4AC0D"
ROSE = "#C0392B"
LIGHT = "#EAF2FF"
GREY = "#5D6D7E"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#33455A", "axes.labelcolor": "#1B2A3A",
    "xtick.color": "#33455A", "ytick.color": "#33455A", "text.color": "#1B2A3A",
    "axes.grid": True, "grid.color": "#D6DEE8", "grid.linestyle": "-",
    "grid.alpha": 0.7, "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 120,
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def fig_calibration(crisis, control):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = np.linspace(0, 1, 11)

    for ax, cells, title, color in [
        (axes[0], control, "Calm control windows", TEAL),
        (axes[1], crisis, "Named crisis windows", ROSE),
    ]:
        ranks = [c["realized_pct_rank"] for c in cells]
        ax.hist(ranks, bins=bins, color=color, alpha=0.85, edgecolor="white")
        ax.axhline(len(ranks) / 10, color=GREY, ls="--", lw=1.3, label="uniform (calibrated)")
        ax.set_title(f"{title}  (n={len(ranks)})")
        ax.set_xlabel("Realized drawdown percentile in simulated distribution")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("Number of cells")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_calibration.pdf"), bbox_inches="tight")
    plt.close(fig)


EVENT_COLORS = {
    "flash_2010": "#5DADE2", "euro_2011": "#48C9B0", "china_2015": "#F5B041",
    "vol_2018": "#AF7AC5", "q4_2018": "#EC7063", "covid_2020": ROSE,
    "rates_2022": "#E59866", "svb_2023": "#82E0AA", "gfc_2008": "#34495E",
}


def fig_scatter(crisis):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = [-1.1, 0.02]
    ax.fill_between(lim, lim, [lim[0], lim[0]], color=ROSE, alpha=0.06)
    ax.plot(lim, lim, color=GREY, ls="--", lw=1.4, label="perfect (realized = p5)", zorder=1)

    seen = set()
    for c in crisis:
        lab = c["event_label"] if c["event"] not in seen else None
        seen.add(c["event"])
        ax.scatter(c["realized_max_dd"], c["pred_dd_p5"],
                   color=EVENT_COLORS.get(c["event"], BLUE), s=42,
                   edgecolor="white", linewidth=0.6, zorder=3, label=lab)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Realized max drawdown (crisis window)")
    ax.set_ylabel("Predicted 5th-percentile drawdown (out-of-sample)")
    ax.set_title("Predicted tail bound vs realized crisis drawdown")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.text(-0.95, -0.18, "below line:\nrealized worse\nthan prediction\n(tail breach)",
            color=ROSE, fontsize=8.5, ha="left", va="top")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_event_rank(agg):
    rows = agg["by_event"]
    labels = [r["label"] for r in rows]
    ranks = [r["mean_pct_rank"] for r in rows]
    colors = [ROSE if x < 0.05 else (GOLD if x < 0.15 else TEAL) for x in ranks]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = np.arange(len(labels))
    ax.barh(y, ranks, color=colors, edgecolor="white", height=0.62)
    ax.axvline(0.5, color=GREY, ls="--", lw=1.2, label="centered (rank = 0.50)")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean realized-drawdown percentile rank (lower = deeper tail miss)")
    ax.set_title("Where each crisis landed in the engine's predicted distribution")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(0, 0.6)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_event_rank.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    crisis = load("cells.json")
    control = load("control_cells.json")
    agg = load("aggregates.json")
    fig_calibration(crisis, control)
    fig_scatter(crisis)
    fig_event_rank(agg)
    print("Wrote figures to", FIGS)
    for f in sorted(os.listdir(FIGS)):
        print("  ", f)


if __name__ == "__main__":
    main()

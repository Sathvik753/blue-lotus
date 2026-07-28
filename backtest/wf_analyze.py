"""Aggregate walk-forward cells into calibration statistics and figures."""

import os, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wf_results")
FIGS = os.path.join(HERE, "results", "figs")
os.makedirs(FIGS, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.size": 10})


def kupiec_pof(x, n, p):
    """Kupiec proportion-of-failures LR statistic and chi2_1 p-value."""
    if n == 0:
        return float("nan"), float("nan")
    pi = x / n
    eps = 1e-12
    ll_null = (n - x) * np.log(1 - p + eps) + x * np.log(p + eps)
    ll_alt = (n - x) * np.log(1 - pi + eps) + x * np.log(pi + eps)
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def cov(cells):
    n = len(cells)
    if n == 0:
        return {}
    xp5 = sum(c["breach_p5"] for c in cells)
    xp1 = sum(c["breach_p1"] for c in cells)
    cov90 = np.mean([c["covered_90"] for c in cells])
    pit = np.array([c["pit_rank"] for c in cells])
    lr5, p5p = kupiec_pof(xp5, n, 0.05)
    lr1, p1p = kupiec_pof(xp1, n, 0.01)
    ks = stats.kstest(pit, "uniform")
    return {
        "n": n,
        "p5_breaches": xp5, "p5_rate": round(xp5 / n, 4),
        "p5_kupiec_lr": round(lr5, 3), "p5_kupiec_p": round(p5p, 4),
        "p1_breaches": xp1, "p1_rate": round(xp1 / n, 4),
        "p1_kupiec_lr": round(lr1, 3), "p1_kupiec_p": round(p1p, 4),
        "coverage_90": round(float(cov90), 4),
        "mean_pit": round(float(pit.mean()), 4),
        "median_pit": round(float(np.median(pit)), 4),
        "ks_stat": round(float(ks.statistic), 4), "ks_p": round(float(ks.pvalue), 4),
    }


def main():
    cells = json.load(open(os.path.join(OUT, "cells.json")))
    print(f"loaded {len(cells)} cells")

    summary = {"overall": cov(cells)}

    # Non-crisis vs crisis years
    calm = [c for c in cells if not c["crisis_year"]]
    crisis = [c for c in cells if c["crisis_year"]]
    summary["calm_years"] = cov(calm)
    summary["crisis_years"] = cov(crisis)

    # By year
    summary["by_year"] = {}
    for Y in sorted({c["year"] for c in cells}):
        summary["by_year"][str(Y)] = cov([c for c in cells if c["year"] == Y])

    # By asset class
    summary["by_class"] = {}
    for k in sorted({c["asset_class"] for c in cells}):
        summary["by_class"][k] = cov([c for c in cells if c["asset_class"] == k])

    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(summary["overall"], indent=2))
    print("calm p5:", summary["calm_years"].get("p5_rate"), "crisis p5:", summary["crisis_years"].get("p5_rate"))
    # KS uniformity test, reported explicitly per stratum (see paper Section 4.3).
    for k in ("overall", "calm_years", "crisis_years"):
        d = summary[k]
        print(f"KS[{k}] D={d['ks_stat']} p={d['ks_p']} meanPIT={d['mean_pit']} medianPIT={d['median_pit']}")

    pit = np.array([c["pit_rank"] for c in cells])

    # Fig 1: PIT histogram (calm vs crisis)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, subset, title in [
        (axes[0], calm, f"Non-crisis years (n={len(calm)})"),
        (axes[1], crisis, f"Crisis years (n={len(crisis)})"),
    ]:
        p = np.array([c["pit_rank"] for c in subset])
        ax.hist(p, bins=20, range=(0, 1), color="#33506e", edgecolor="white", density=True)
        ax.axhline(1.0, color="#c0392b", ls="--", lw=1, label="Uniform")
        ax.set_title(title); ax.set_xlabel("Realized-drawdown PIT rank"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "wf_pit.pdf")); plt.close()

    # Fig 2: reliability / calibration curve (empirical vs nominal quantile coverage)
    levels = np.linspace(0.02, 0.5, 25)
    emp = [np.mean(pit <= q) for q in levels]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 0.5], [0, 0.5], color="#888", ls="--", lw=1, label="Perfect calibration")
    ax.plot(levels, emp, color="#148f77", lw=2, marker="o", ms=3, label="Engine (all cells)")
    ax.set_xlabel("Nominal lower-tail probability")
    ax.set_ylabel("Empirical exceedance frequency")
    ax.set_title("Calibration of predicted drawdown quantiles")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "wf_reliability.pdf")); plt.close()

    # Fig 3: predicted p5 vs realized scatter, colored by crisis
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for subset, col, lab in [(calm, "#33506e", "Non-crisis"), (crisis, "#c0392b", "Crisis")]:
        xr = [c["pred_p5"] for c in subset]
        yr = [c["realized_dd"] for c in subset]
        ax.scatter(xr, yr, s=14, alpha=0.55, color=col, label=lab, edgecolor="none")
    lim = [min(c["realized_dd"] for c in cells) - 0.02, 0.02]
    ax.plot(lim, lim, color="#444", ls="--", lw=1, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Predicted 5th-percentile drawdown")
    ax.set_ylabel("Realized annual max drawdown")
    ax.set_title("Predicted tail vs realized outcome")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "wf_scatter.pdf")); plt.close()

    # Fig 4: p5 breach rate by year
    yrs = sorted({c["year"] for c in cells})
    rates = [summary["by_year"][str(Y)]["p5_rate"] for Y in yrs]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = ["#c0392b" if Y in {2015, 2018, 2020, 2022} else "#33506e" for Y in yrs]
    ax.bar([str(y) for y in yrs], rates, color=colors)
    ax.axhline(0.05, color="#148f77", ls="--", lw=1.2, label="Nominal 5%")
    ax.set_ylabel("p5 breach rate"); ax.set_title("Out-of-sample tail-breach rate by test year")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "wf_byyear.pdf")); plt.close()

    print("figures written to", FIGS)


if __name__ == "__main__":
    main()

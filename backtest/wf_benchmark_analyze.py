"""Aggregate the benchmark: CRPS, skill scores, calibration, and paired tests."""

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

MODELS = ["engine", "gauss", "studt", "histiid", "histblk"]
LABEL = {"engine": "Blue Lotus", "gauss": "Gaussian", "studt": "Student-t",
         "histiid": "Hist. bootstrap", "histblk": "Block bootstrap"}


def kupiec_p(x, n, p):
    if n == 0:
        return float("nan")
    pi = x / n; eps = 1e-12
    lr = -2 * ((n - x) * np.log(1 - p + eps) + x * np.log(p + eps)
               - (n - x) * np.log(1 - pi + eps) - x * np.log(pi + eps))
    return float(1 - stats.chi2.cdf(lr, 1))


def block(cells):
    out = {}
    n = len(cells)
    for m in MODELS:
        crps = np.array([c[f"{m}_crps"] for c in cells])
        b5 = sum(c[f"{m}_breach_p5"] for c in cells)
        b1 = sum(c[f"{m}_breach_p1"] for c in cells)
        pit = np.array([c[f"{m}_pit"] for c in cells])
        out[m] = {
            "n": n,
            "mean_crps": round(float(crps.mean()), 5),
            "p5_rate": round(b5 / n, 4), "p5_kupiec_p": round(kupiec_p(b5, n, 0.05), 4),
            "p1_rate": round(b1 / n, 4), "p1_kupiec_p": round(kupiec_p(b1, n, 0.01), 4),
            "mean_pit": round(float(pit.mean()), 4),
            "ks_p": round(float(stats.kstest(pit, "uniform").pvalue), 4),
        }
    # engine skill vs each baseline + paired Wilcoxon on CRPS differences
    eng = np.array([c["engine_crps"] for c in cells])
    for m in MODELS:
        if m == "engine":
            continue
        base = np.array([c[f"{m}_crps"] for c in cells])
        skill = 1 - eng.mean() / base.mean()            # >0 => engine better
        d = eng - base                                   # <0 => engine better on a cell
        win = float(np.mean(d < 0))
        try:
            w_p = float(stats.wilcoxon(eng, base).pvalue)
        except Exception:
            w_p = float("nan")
        out[m]["crps_skill_vs"] = round(float(skill), 4)
        out[m]["engine_win_rate"] = round(win, 4)
        out[m]["wilcoxon_p"] = round(w_p, 6)
    return out


def main():
    cells = json.load(open(os.path.join(OUT, "benchmark_cells.json")))
    calm = [c for c in cells if not c["crisis_year"]]
    crisis = [c for c in cells if c["crisis_year"]]

    summary = {"overall": block(cells), "calm": block(calm), "crisis": block(crisis)}
    json.dump(summary, open(os.path.join(OUT, "benchmark_summary.json"), "w"), indent=2)

    def show(name, s):
        print(f"\n=== {name} (n={s['engine']['n']}) ===")
        print(f"{'model':16}{'CRPS':>9}{'p5rate':>8}{'p5kup':>7}{'meanPIT':>8}{'ksP':>7}{'skill':>8}{'win%':>7}")
        for m in MODELS:
            d = s[m]
            skill = d.get("crps_skill_vs", "")
            win = d.get("engine_win_rate", "")
            sk = f"{skill:+.3f}" if skill != "" else "   -"
            wn = f"{win*100:.0f}" if win != "" else "  -"
            print(f"{LABEL[m]:16}{d['mean_crps']:>9.4f}{d['p5_rate']:>8.3f}{d['p5_kupiec_p']:>7.2f}"
                  f"{d['mean_pit']:>8.3f}{d['ks_p']:>7.2f}{sk:>8}{wn:>7}")
    for name, key in [("OVERALL", "overall"), ("CALM YEARS", "calm"), ("CRISIS YEARS", "crisis")]:
        show(name, summary[key])

    # Figure: mean CRPS by model, grouped by regime (lower = better)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(MODELS)); w = 0.26
    for i, (key, col, lab) in enumerate([("calm", "#33506e", "Non-crisis"),
                                         ("crisis", "#c0392b", "Crisis"),
                                         ("overall", "#148f77", "All")]):
        vals = [summary[key][m]["mean_crps"] for m in MODELS]
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=15)
    ax.set_ylabel("Mean CRPS  (lower is better)")
    ax.set_title("Forecast skill by model and regime")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "wf_benchmark_crps.pdf")); plt.close()

    # Figure: calm-year p5 breach rate by model (calibration) with nominal line
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vals = [summary["calm"][m]["p5_rate"] for m in MODELS]
    cols = ["#148f77" if m == "engine" else "#33506e" for m in MODELS]
    ax.bar([LABEL[m] for m in MODELS], vals, color=cols)
    ax.axhline(0.05, color="#c0392b", ls="--", lw=1.2, label="Nominal 5%")
    ax.set_ylabel("Non-crisis p5 breach rate"); ax.set_title("Calm-market tail calibration by model")
    ax.legend(fontsize=8); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "wf_benchmark_calib.pdf")); plt.close()

    print("\nfigures written")


if __name__ == "__main__":
    main()

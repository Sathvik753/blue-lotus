"""
╔══════════════════════════════════════════════════════════╗
║   BLUE LOTUS LABS — Stress-Testing Engine                ║
║   Single-file version for Google Colab                   ║
║                                                          ║
║   HOW TO USE:                                            ║
║   1. Open a new Google Colab notebook                    ║
║   2. Create a code cell and paste this entire file       ║
║   3. Run it — the demo will execute automatically        ║
║   4. Swap in your own returns at the bottom              ║
╚══════════════════════════════════════════════════════════╝
"""

# ── Dependencies ─────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import warnings

# ═══════════════════════════════════════════════════════════════
# MODULE 1 — INPUT PROCESSING
# ═══════════════════════════════════════════════════════════════

@dataclass
class InputMetadata:
    n_observations: int
    raw_mean: float
    raw_std: float
    raw_skewness: float
    raw_kurtosis: float
    min_return: float
    max_return: float
    winsorized: bool
    normalization: str


class InputProcessor:
    def __init__(self, winsorize=True, winsorize_limits=(0.01, 0.01),
                 normalization="zscore", target_vol=0.01):
        self.winsorize = winsorize
        self.winsorize_limits = winsorize_limits
        self.normalization = normalization
        self.target_vol = target_vol
        self.mean_ = None
        self.std_ = None
        self.metadata_ = None

    def fit_transform(self, returns):
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 1:
            raise ValueError("returns must be 1-D")
        if len(returns) < 30:
            warnings.warn(f"Only {len(returns)} observations — high uncertainty.", UserWarning)
        returns = returns[~np.isnan(returns)]

        raw_mean = float(np.mean(returns))
        raw_std  = float(np.std(returns, ddof=1))
        mu, sigma = np.mean(returns), np.std(returns, ddof=1)
        raw_skew = float(np.mean(((returns - mu) / (sigma + 1e-12)) ** 3))
        raw_kurt = float(np.mean(((returns - mu) / (sigma + 1e-12)) ** 4)) - 3.0

        if self.winsorize:
            lo, hi = self.winsorize_limits
            returns = np.clip(returns, np.quantile(returns, lo), np.quantile(returns, 1 - hi))

        if self.normalization == "zscore":
            mu, sigma = np.mean(returns), np.std(returns, ddof=1)
            returns = (returns - mu) / (sigma + 1e-12)
            norm_label = "zscore"
        elif self.normalization == "vol_scale":
            returns = returns * (self.target_vol / (np.std(returns, ddof=1) + 1e-12))
            norm_label = f"vol_scale({self.target_vol})"
        else:
            norm_label = "none"

        self.mean_ = float(np.mean(returns))
        self.std_  = float(np.std(returns, ddof=1))
        self.metadata_ = InputMetadata(
            n_observations=len(returns), raw_mean=raw_mean, raw_std=raw_std,
            raw_skewness=raw_skew, raw_kurtosis=raw_kurt,
            min_return=float(np.min(returns)), max_return=float(np.max(returns)),
            winsorized=self.winsorize, normalization=norm_label,
        )
        return returns, self.metadata_


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — STRUCTURAL CONSTRAINT LAYER
# ═══════════════════════════════════════════════════════════════

@dataclass
class RegimeModelOutput:
    transition_matrix: np.ndarray
    regime_means: np.ndarray
    regime_stds: np.ndarray
    regime_labels: np.ndarray
    stationary_dist: np.ndarray


class RegimeModel:
    def fit(self, returns):
        T = len(returns)
        window = max(5, T // 10)
        roll_vol = np.array([np.std(returns[max(0, i-window):i+1], ddof=0) for i in range(T)])
        cumulative = np.cumsum(returns)
        drawdown = cumulative - np.maximum.accumulate(cumulative)
        v33, v66 = np.percentile(roll_vol, 33), np.percentile(roll_vol, 66)
        labels = np.where(roll_vol <= v33, 0, np.where(roll_vol <= v66, 1, 2))
        labels[drawdown < np.percentile(drawdown, 10)] = 2

        P = np.ones((3, 3))
        for t in range(T - 1):
            P[labels[t], labels[t+1]] += 1
        P = P / P.sum(axis=1, keepdims=True)

        means = np.array([np.mean(returns[labels==k]) if (labels==k).sum()>1 else np.mean(returns) for k in range(3)])
        stds  = np.array([np.std(returns[labels==k], ddof=1) if (labels==k).sum()>1 else np.std(returns, ddof=1) for k in range(3)])

        eigvals, eigvecs = np.linalg.eig(P.T)
        pi = np.abs(np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))]))
        pi /= pi.sum()

        return RegimeModelOutput(transition_matrix=P, regime_means=means,
                                 regime_stds=stds, regime_labels=labels, stationary_dist=pi)


@dataclass
class TailConstraints:
    alpha: float
    lower_quantile_bound: float
    upper_quantile_bound: float
    es_target: float
    method: str


class TailConstraintLayer:
    def __init__(self, alpha=0.05, method="student_t"):
        self.alpha = alpha
        self.method = method

    def fit(self, returns):
        tail = returns[returns < np.quantile(returns, self.alpha)]
        q_fitted = float(np.quantile(returns, self.alpha))
        if self.method == "student_t" and len(tail) >= 4:
            df, loc, scale = stats.t.fit(tail, floc=np.mean(tail))
            q_fitted = float(stats.t.ppf(self.alpha, df=df, loc=loc, scale=scale))
        buf = abs(q_fitted) * 0.10
        es_emp = float(np.mean(tail)) if len(tail) > 0 else q_fitted
        return TailConstraints(alpha=self.alpha, lower_quantile_bound=q_fitted - buf,
                               upper_quantile_bound=q_fitted + buf,
                               es_target=es_emp * 1.10, method=self.method)


@dataclass
class BayesianPriors:
    regime_means: np.ndarray
    regime_vars: np.ndarray


class BayesianShrinkageLayer:
    def __init__(self, tau=0.5, alpha_ig=3.0, beta_ig=1.0, shrinkage_strength=0.7):
        self.tau = tau
        self.alpha_ig = alpha_ig
        self.beta_ig = beta_ig
        self.shrinkage_strength = shrinkage_strength

    def fit(self, regime_output, returns):
        n_per = np.array([(regime_output.regime_labels==k).sum() for k in range(3)])
        means, variances = np.zeros(3), np.zeros(3)
        for k in range(3):
            n_k = n_per[k]
            w = self.shrinkage_strength / (1 + n_k / 10)
            means[k] = (1 - w) * regime_output.regime_means[k]
            prior_var = self.beta_ig / (self.alpha_ig + 1)
            post_var  = (self.beta_ig + n_k * regime_output.regime_stds[k]**2 / 2) / (self.alpha_ig + n_k/2 + 1)
            variances[k] = max(post_var, prior_var)
        return BayesianPriors(regime_means=means, regime_vars=variances)


@dataclass
class DrawdownConditioningOutput:
    states: np.ndarray
    conditional_probs: dict
    thresholds: tuple


class DrawdownConditioningLayer:
    def __init__(self, moderate_threshold=-0.05, severe_threshold=-0.15):
        self.moderate_threshold = moderate_threshold
        self.severe_threshold   = severe_threshold

    def fit(self, returns):
        dd = np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns))
        states = np.zeros(len(returns), dtype=int)
        states[dd < self.moderate_threshold] = 1
        states[dd < self.severe_threshold]   = 2
        cp = {}
        for s in range(3):
            mask = states == s
            cp[s] = {"mean": float(np.mean(returns[mask])) if mask.sum()>1 else float(np.mean(returns)),
                     "std":  float(np.std(returns[mask], ddof=1)) if mask.sum()>1 else float(np.std(returns, ddof=1)),
                     "n":    int(mask.sum())}
        return DrawdownConditioningOutput(states=states, conditional_probs=cp,
                                          thresholds=(self.moderate_threshold, self.severe_threshold))


class DistributionalOperatorLayer:
    def __init__(self, mu_prior=0.0, sigma_max=None, es_target=None, alpha=0.05):
        self.mu_prior  = mu_prior
        self.sigma_max = sigma_max
        self.es_target = es_target
        self.alpha     = alpha

    def apply(self, paths):
        # Mean-preserving
        paths = paths - paths.mean(axis=1, keepdims=True) + self.mu_prior
        # Variance-capping
        if self.sigma_max is not None:
            path_vars = paths.var(axis=1, ddof=1)
            scale = np.where(path_vars > self.sigma_max**2,
                             self.sigma_max / np.sqrt(np.maximum(path_vars, 1e-12)), 1.0)
            centered = paths - paths.mean(axis=1, keepdims=True)
            paths = centered * scale[:, np.newaxis] + self.mu_prior
        # Tail-integral (ES ceiling)
        if self.es_target is not None:
            q = np.quantile(paths, self.alpha, axis=1)
            for i in range(len(paths)):
                mask = paths[i] < q[i]
                if mask.sum() > 0:
                    path_es = float(np.mean(paths[i][mask]))
                    if path_es < self.es_target:
                        paths[i][mask] *= self.es_target / (path_es + 1e-12)
        return paths


@dataclass
class ConstraintLayerOutput:
    regime:   RegimeModelOutput
    tail:     TailConstraints
    bayes:    BayesianPriors
    drawdown: DrawdownConditioningOutput
    operator: DistributionalOperatorLayer
    implied_vol: Optional[float]
    known_risk_limit: Optional[float]


class StructuralConstraintLayer:
    def __init__(self, tail_alpha=0.05, tail_method="student_t", tau=0.5,
                 alpha_ig=3.0, beta_ig=1.0, shrinkage_strength=0.7,
                 moderate_dd=-0.05, severe_dd=-0.15,
                 implied_vol=None, known_risk_limit=None,
                 regulatory_scenarios=None, apply_smoothing=False):
        self.regime_model = RegimeModel()
        self.tail_layer   = TailConstraintLayer(alpha=tail_alpha, method=tail_method)
        self.bayes_layer  = BayesianShrinkageLayer(tau=tau, alpha_ig=alpha_ig,
                                                    beta_ig=beta_ig, shrinkage_strength=shrinkage_strength)
        self.dd_layer     = DrawdownConditioningLayer(moderate_threshold=moderate_dd,
                                                      severe_threshold=severe_dd)
        self.implied_vol  = implied_vol
        self.known_risk_limit = known_risk_limit

    def fit(self, returns):
        regime   = self.regime_model.fit(returns)
        tail     = self.tail_layer.fit(returns)
        bayes    = self.bayes_layer.fit(regime, returns)
        drawdown = self.dd_layer.fit(returns)
        op = DistributionalOperatorLayer(
            mu_prior=float(np.mean(bayes.regime_means)),
            sigma_max=self.implied_vol,
            es_target=tail.es_target,
            alpha=self.tail_layer.alpha,
        )
        return ConstraintLayerOutput(regime=regime, tail=tail, bayes=bayes,
                                     drawdown=drawdown, operator=op,
                                     implied_vol=self.implied_vol,
                                     known_risk_limit=self.known_risk_limit)


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — CONSTRAINED MONTE CARLO GENERATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class MonteCarloOutput:
    paths: np.ndarray
    regime_paths: np.ndarray
    scenario_labels: np.ndarray
    n_paths: int
    horizon: int
    rejection_rate: float


class ConstrainedMonteCarloGenerator:
    def __init__(self, n_paths=10000, horizon=252, random_seed=42, stress_fraction=0.20, **kwargs):
        self.n_paths        = n_paths
        self.horizon        = horizon
        self.stress_fraction = stress_fraction
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate(self, constraints):
        P      = constraints.regime.transition_matrix
        b_mean = constraints.bayes.regime_means
        b_std  = np.sqrt(constraints.bayes.regime_vars)
        dd_cp  = constraints.drawdown.conditional_probs

        paths        = np.zeros((self.n_paths, self.horizon))
        regime_paths = np.zeros((self.n_paths, self.horizon), dtype=int)

        # Initial regimes
        n_crisis = int(self.n_paths * self.stress_fraction)
        current_regimes = np.zeros(self.n_paths, dtype=int)
        crisis_idx = np.random.choice(self.n_paths, n_crisis, replace=False)
        current_regimes[crisis_idx] = 2

        cumulative        = np.zeros(self.n_paths)
        current_dd_states = np.zeros(self.n_paths, dtype=int)
        mod_thr, sev_thr  = constraints.drawdown.thresholds

        for t in range(self.horizon):
            new_regimes = np.zeros(self.n_paths, dtype=int)
            for k in range(3):
                mask = current_regimes == k
                if mask.sum() > 0:
                    new_regimes[mask] = np.random.choice(3, size=mask.sum(), p=P[k])
            current_regimes = new_regimes
            regime_paths[:, t] = current_regimes

            returns_t = np.zeros(self.n_paths)
            for k in range(3):
                mask = current_regimes == k
                if mask.sum() > 0:
                    returns_t[mask] = np.random.normal(b_mean[k], b_std[k], size=mask.sum())

            # Drawdown conditioning blend
            blend = {0: 0.10, 1: 0.30, 2: 0.50}
            for s in range(3):
                mask = current_dd_states == s
                if mask.sum() > 0 and dd_cp[s]["n"] > 0:
                    w = blend[s]
                    returns_t[mask] = ((1-w)*returns_t[mask] +
                                       w*np.random.normal(dd_cp[s]["mean"], dd_cp[s]["std"], mask.sum()))

            paths[:, t] = returns_t
            cumulative += returns_t
            running_max = np.maximum.accumulate(np.vstack([cumulative]))[0]
            dd = cumulative - running_max
            current_dd_states = np.where(dd < sev_thr, 2, np.where(dd < mod_thr, 1, 0))

        # Apply distributional operators
        paths = constraints.operator.apply(paths)

        # Hard constraint: reject only the most extreme outlier paths (bottom 1%)
        # Uses relative ranking rather than absolute bound to be scale-invariant
        alpha = constraints.tail.alpha
        path_q = np.quantile(paths, alpha, axis=1)
        cutoff = np.percentile(path_q, 1)   # reject only bottom 1% of paths
        mask   = path_q >= cutoff
        if constraints.known_risk_limit is not None:
            cum = np.cumsum(paths, axis=1)
            max_dd = np.min(cum - np.maximum.accumulate(cum, axis=1), axis=1)
            mask &= max_dd >= constraints.known_risk_limit

        paths        = paths[mask]
        regime_paths = regime_paths[mask]
        rejection_rate = 1.0 - mask.sum() / self.n_paths

        # Scenario labels
        cum = np.cumsum(paths, axis=1)
        max_dd = np.min(cum - np.maximum.accumulate(cum, axis=1), axis=1)
        # Scale scenario thresholds to actual path volatility
        path_vol = paths.std(axis=1).mean()
        thr_normal = -path_vol * 10    # less than 10 daily-vol drawdown = normal
        thr_stress = -path_vol * 25    # 10-25 = stress, beyond = crisis
        labels = np.where(max_dd > thr_normal, "normal",
                          np.where(max_dd > thr_stress, "stress", "crisis"))

        return MonteCarloOutput(paths=paths, regime_paths=regime_paths,
                                scenario_labels=labels, n_paths=len(paths),
                                horizon=self.horizon, rejection_rate=rejection_rate)


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — STRESS METRICS
# ═══════════════════════════════════════════════════════════════

@dataclass
class StressMetricsOutput:
    drawdown_dist: np.ndarray
    dd_mean: float
    dd_median: float
    dd_p5: float
    dd_ci90: tuple
    dd_by_scenario: dict
    es_alpha: float
    es_dist: np.ndarray
    es_mean: float
    es_aggregate: float
    es_ci90: tuple
    worst_returns: np.ndarray
    worst_paths: np.ndarray
    recovery_dist: np.ndarray
    recovery_mean: float
    recovery_median: float
    pct_never_recover: float
    regime_means: dict
    regime_es: dict
    regime_fracs: dict


class StressMetricsEngine:
    def __init__(self, es_alpha=0.05, k_worst=10, ci_level=0.90):
        self.es_alpha = es_alpha
        self.k_worst  = k_worst
        self.ci_lo    = (1 - ci_level) / 2
        self.ci_hi    = 1 - self.ci_lo

    def compute(self, mc):
        paths, labels, regimes = mc.paths, mc.scenario_labels, mc.regime_paths

        # Drawdown
        cum = np.cumsum(paths, axis=1)
        dd  = cum - np.maximum.accumulate(cum, axis=1)
        max_dd = dd.min(axis=1)
        dd_by_sc = {s: float(max_dd[labels==s].mean()) if (labels==s).sum()>0 else float("nan")
                    for s in ["normal", "stress", "crisis"]}

        # ES
        alpha  = self.es_alpha
        q_path = np.quantile(paths, alpha, axis=1)
        es_per = np.array([float(np.mean(paths[i][paths[i]<=q_path[i]]))
                           if (paths[i]<=q_path[i]).sum()>0 else float(q_path[i])
                           for i in range(len(paths))])
        all_r  = paths.flatten()
        agg_q  = np.quantile(all_r, alpha)
        agg_es = float(np.mean(all_r[all_r <= agg_q]))

        # Worst-k
        total_r   = paths.sum(axis=1)
        worst_idx = np.argsort(total_r)[:self.k_worst]

        # Recovery
        recovery = np.full(len(paths), np.nan)
        for i in range(len(paths)):
            t_dd = int(np.argmin(dd[i]))
            if dd[i, t_dd] >= 0:
                recovery[i] = 0
                continue
            peak = np.maximum.accumulate(cum[i])[t_dd]
            post = cum[i, t_dd:]
            rec  = np.where(post >= peak)[0]
            if len(rec) > 0:
                recovery[i] = float(rec[0])
        valid = recovery[~np.isnan(recovery)]

        # Regime losses
        rm, re, rf = {}, {}, {}
        for k in range(3):
            mask = regimes == k
            r    = paths[mask]
            rf[k] = float(mask.sum() / paths.size)
            if len(r) > 0:
                rm[k] = float(r.mean())
                q_k = np.quantile(r, alpha)
                t_k = r[r <= q_k]
                re[k] = float(t_k.mean()) if len(t_k) > 0 else float(q_k)
            else:
                rm[k] = re[k] = float("nan")

        return StressMetricsOutput(
            drawdown_dist=max_dd, dd_mean=float(max_dd.mean()),
            dd_median=float(np.median(max_dd)), dd_p5=float(np.percentile(max_dd, 5)),
            dd_ci90=(float(np.percentile(max_dd, self.ci_lo*100)), float(np.percentile(max_dd, self.ci_hi*100))),
            dd_by_scenario=dd_by_sc,
            es_alpha=alpha, es_dist=es_per, es_mean=float(es_per.mean()),
            es_aggregate=agg_es,
            es_ci90=(float(np.percentile(es_per, self.ci_lo*100)), float(np.percentile(es_per, self.ci_hi*100))),
            worst_returns=total_r[worst_idx], worst_paths=paths[worst_idx],
            recovery_dist=recovery,
            recovery_mean=float(valid.mean()) if len(valid)>0 else float("nan"),
            recovery_median=float(np.median(valid)) if len(valid)>0 else float("nan"),
            pct_never_recover=float(np.isnan(recovery).mean()),
            regime_means=rm, regime_es=re, regime_fracs=rf,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — SENSITIVITY (lightweight inline version)
# ═══════════════════════════════════════════════════════════════

def compute_fragility_index(returns, constraint_kwargs, mc_kwargs, n_trials=10, n_paths=1000):
    """Runs quick perturbation trials and returns a fragility score."""
    base_es_list = []
    for seed in range(n_trials):
        try:
            ip = InputProcessor()
            cleaned, _ = ip.fit_transform(returns)
            cl = StructuralConstraintLayer(**constraint_kwargs)
            c  = cl.fit(cleaned)
            mc = ConstrainedMonteCarloGenerator(**{**mc_kwargs, "n_paths": n_paths, "random_seed": seed})
            out = mc.generate(c)
            sm = StressMetricsEngine()
            m  = sm.compute(out)
            base_es_list.append(m.es_aggregate)
        except Exception:
            pass
    if len(base_es_list) < 2:
        return 0.5, "Unknown"
    arr = np.array(base_es_list)
    cv  = float(np.std(arr, ddof=1) / (abs(np.mean(arr)) + 1e-12))
    fi  = float(np.clip(cv, 0, 1))
    grade = "Robust" if fi < 0.25 else ("Moderate" if fi < 0.55 else "Fragile")
    return fi, grade


# ═══════════════════════════════════════════════════════════════
# MODULE 7 — REPORTING & VISUALIZATION
# ═══════════════════════════════════════════════════════════════

BL_DARK  = "#0D1B2A"
BL_BLUE  = "#1B4F72"
BL_TEAL  = "#148F77"
BL_GOLD  = "#D4AC0D"
BL_ROSE  = "#C0392B"
BL_LIGHT = "#EAF2FF"
BL_GREY  = "#5D6D7E"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": BL_DARK, "axes.facecolor": "#111E2D",
        "axes.edgecolor": "#2E4057", "axes.labelcolor": "#CBD5E0",
        "xtick.color": "#CBD5E0", "ytick.color": "#CBD5E0",
        "text.color": BL_LIGHT, "grid.color": "#1E3A52",
        "grid.linestyle": "--", "grid.alpha": 0.4,
        "font.family": "monospace", "axes.titlesize": 10, "axes.labelsize": 8,
    })


def plot_dashboard(mc, sm, strategy_name="Strategy", fi=None, fi_grade=None):
    apply_style()
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BL_DARK)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    paths  = mc.paths
    labels = mc.scenario_labels
    cum    = np.cumsum(paths, axis=1)
    dd_ser = cum - np.maximum.accumulate(cum, axis=1)
    T      = paths.shape[1]
    x      = np.arange(T)

    # Panel 1 — Drawdown curves
    ax = axes[0]
    colors = {"normal": BL_TEAL, "stress": BL_GOLD, "crisis": BL_ROSE}
    for sc, col in colors.items():
        idx = np.where(labels == sc)[0][:40]
        for i, pi in enumerate(idx):
            ax.plot(x, dd_ser[pi], color=col, alpha=0.07, linewidth=0.5)
    p5  = np.percentile(dd_ser, 5, axis=0)
    p50 = np.percentile(dd_ser, 50, axis=0)
    p95 = np.percentile(dd_ser, 95, axis=0)
    ax.fill_between(x, p5, p95, alpha=0.15, color=BL_BLUE)
    ax.plot(x, p50, color=BL_GOLD, lw=1.5, label="Median")
    ax.plot(x, p5,  color=BL_ROSE, lw=1.0, linestyle="--", label="5th pct")
    legend_els = [Line2D([0],[0], color=BL_TEAL, lw=1.5, label="Normal"),
                  Line2D([0],[0], color=BL_GOLD, lw=1.5, label="Stress"),
                  Line2D([0],[0], color=BL_ROSE, lw=1.5, label="Crisis")]
    ax.legend(handles=legend_els, fontsize=7, framealpha=0.2, loc="lower left")
    ax.set_title("Stress Drawdown Curves", color=BL_LIGHT)
    ax.set_xlabel("Step"); ax.set_ylabel("Drawdown (z-score units)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True)

    # Panel 2 — Max DD histogram
    ax = axes[1]
    ax.hist(sm.drawdown_dist, bins=50, color=BL_BLUE, edgecolor="none", alpha=0.85, density=True)
    ax.axvline(sm.dd_mean,   color=BL_GOLD, lw=1.5, label=f"Mean: {sm.dd_mean:.3f}")
    ax.axvline(sm.dd_p5,     color=BL_ROSE, lw=1.5, linestyle="--", label=f"5th: {sm.dd_p5:.3f}")
    ax.axvline(sm.dd_median, color=BL_TEAL, lw=1.5, linestyle=":",  label=f"Med: {sm.dd_median:.3f}")
    ax.set_title("Max Drawdown Distribution", color=BL_LIGHT)
    ax.set_xlabel("Max Drawdown (z-score units)"); ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True)

    # Panel 3 — Recovery distribution
    ax = axes[2]
    valid = sm.recovery_dist[~np.isnan(sm.recovery_dist)]
    if len(valid) > 0:
        ax.hist(valid, bins=40, color=BL_TEAL, edgecolor="none", alpha=0.85, density=True)
        ax.axvline(sm.recovery_mean,   color=BL_GOLD, lw=1.5, label=f"Mean: {sm.recovery_mean:.1f}")
        ax.axvline(sm.recovery_median, color=BL_ROSE, lw=1.5, linestyle="--", label=f"Med: {sm.recovery_median:.1f}")
        ax.legend(fontsize=7, framealpha=0.2)
    ax.set_title(f"Time-to-Recovery  ({sm.pct_never_recover:.1%} never recover)", color=BL_LIGHT)
    ax.set_xlabel("Steps"); ax.set_ylabel("Density"); ax.grid(True)

    # Panel 4 — Regime heatmap
    ax = axes[3]
    reg = mc.regime_paths
    fracs = np.array([(reg == k).mean(axis=0) for k in range(3)])
    cmap = LinearSegmentedColormap.from_list("bl", [BL_DARK, BL_BLUE, BL_GOLD, BL_ROSE])
    im = ax.imshow(fracs, aspect="auto", cmap=cmap, vmin=0, vmax=1, extent=[0, T, -0.5, 2.5])
    plt.colorbar(im, ax=ax, label="Fraction", fraction=0.046, pad=0.04)
    ax.set_title("Regime Transition Heatmap", color=BL_LIGHT)
    ax.set_xlabel("Step"); ax.set_yticks([0,1,2])
    ax.set_yticklabels(["Calm","Volatile","Crisis"], fontsize=8)

    # Panel 5 — ES distribution
    ax = axes[4]
    ax.hist(sm.es_dist, bins=50, color=BL_ROSE, edgecolor="none", alpha=0.85, density=True)
    ax.axvline(sm.es_mean,      color=BL_GOLD, lw=1.5, label=f"Mean: {sm.es_mean:.4f}")
    ax.axvline(sm.es_aggregate, color=BL_TEAL, lw=1.5, linestyle="--", label=f"Agg: {sm.es_aggregate:.4f}")
    ax.set_title(f"Expected Shortfall (α={sm.es_alpha})", color=BL_LIGHT)
    ax.set_xlabel("ES / CVaR (z-score units)"); ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True)

    # Panel 6 — Worst paths
    ax = axes[5]
    k = len(sm.worst_paths)
    cols = plt.cm.Reds(np.linspace(0.4, 0.9, k))
    for i, path in enumerate(sm.worst_paths):
        ax.plot(np.cumsum(path), color=cols[i], alpha=0.85, lw=1.0)
    ax.set_title(f"Worst-{k} Paths (Cumulative Return)", color=BL_LIGHT)
    ax.set_xlabel("Step"); ax.set_ylabel("Cumulative Return (z-score units)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True)

    fi_str = f"  |  Fragility Index: {fi:.3f} ({fi_grade})" if fi is not None else ""
    fig.suptitle(f"BLUE LOTUS LABS  |  {strategy_name}{fi_str}",
                 fontsize=13, color=BL_GOLD, weight="bold")
    plt.tight_layout()
    return fig


def print_executive_summary(mc, sm, strategy_name, fi=None, fi_grade=None):
    counts = {s: int(np.sum(mc.scenario_labels == s)) for s in ["normal","stress","crisis"]}
    print()
    print("╔" + "═"*58 + "╗")
    print(f"║   BLUE LOTUS LABS  |  EXECUTIVE RISK SUMMARY           ║")
    print(f"║   Strategy: {strategy_name:<45}║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Paths: {mc.n_paths:,}  |  Rejection: {mc.rejection_rate:.1%}  |  Horizon: {mc.horizon}{'':>12}║")
    print(f"║  Normal / Stress / Crisis: {counts['normal']:,} / {counts['stress']:,} / {counts['crisis']:,}{'':>18}║")
    print("╠" + "═"*58 + "╣")
    print(f"║  DRAWDOWN                                              ║")
    print(f"║    Mean: {sm.dd_mean:+.4f}  |  Median: {sm.dd_median:+.4f}  |  5th pct: {sm.dd_p5:+.4f}   ║")
    print(f"║    90% CI: [{sm.dd_ci90[0]:+.4f}, {sm.dd_ci90[1]:+.4f}]{'':>28}║")
    print("╠" + "═"*58 + "╣")
    print(f"║  EXPECTED SHORTFALL (α={sm.es_alpha}){'':>30}║")
    print(f"║    Aggregate ES: {sm.es_aggregate:+.4f}  |  Mean ES: {sm.es_mean:+.4f}{'':>20}║")
    print("╠" + "═"*58 + "╣")
    print(f"║  RECOVERY: Mean={sm.recovery_mean:.1f} steps  |  Never={sm.pct_never_recover:.1%}{'':>20}║")
    if fi is not None:
        print("╠" + "═"*58 + "╣")
        print(f"║  MODEL FRAGILITY INDEX: {fi:.4f}  ({fi_grade}){'':>26}║")
    print("╠" + "═"*58 + "╣")
    print("║  ⚠  Risk distributions only. No return predictions.    ║")
    print("╚" + "═"*58 + "╝")


# ═══════════════════════════════════════════════════════════════
# MAIN ENGINE — wires all modules together
# ═══════════════════════════════════════════════════════════════

class BlueLotusEngine:
    def __init__(self, strategy_name="Strategy", winsorize=True, normalization="zscore",
                 tail_alpha=0.05, tail_method="student_t", tau=0.5, alpha_ig=3.0, beta_ig=1.0,
                 shrinkage_strength=0.7, moderate_dd=-0.05, severe_dd=-0.15,
                 implied_vol=None, known_risk_limit=None,
                 n_paths=10000, horizon=252, random_seed=42, stress_fraction=0.20,
                 k_worst=10, run_sensitivity=True, figsize_scale=1.0):

        self.strategy_name   = strategy_name
        self.run_sensitivity = run_sensitivity

        self._ck = dict(tail_alpha=tail_alpha, tail_method=tail_method, tau=tau,
                        alpha_ig=alpha_ig, beta_ig=beta_ig, shrinkage_strength=shrinkage_strength,
                        moderate_dd=moderate_dd, severe_dd=severe_dd,
                        implied_vol=implied_vol, known_risk_limit=known_risk_limit)
        self._mk = dict(n_paths=n_paths, horizon=horizon, random_seed=random_seed,
                        stress_fraction=stress_fraction)

        self.ip = InputProcessor(winsorize=winsorize, normalization=normalization)
        self.cl = StructuralConstraintLayer(**self._ck)
        self.mc = ConstrainedMonteCarloGenerator(**self._mk)
        self.sm = StressMetricsEngine(es_alpha=tail_alpha, k_worst=k_worst)

    def run(self, returns, verbose=True):
        print(f"\n{'='*55}")
        print(f"  BLUE LOTUS LABS — {self.strategy_name}")
        print(f"{'='*55}")

        print("▶ Module 1: Input Processing...")
        cleaned, meta = self.ip.fit_transform(np.asarray(returns, dtype=float))
        print(f"   n={meta.n_observations}, mean={meta.raw_mean:.4f}, std={meta.raw_std:.4f}")

        print("▶ Module 2: Structural Constraints...")
        constraints = self.cl.fit(cleaned)
        pi = constraints.regime.stationary_dist
        print(f"   Regime dist — calm={pi[0]:.2f}, volatile={pi[1]:.2f}, crisis={pi[2]:.2f}")

        print(f"▶ Module 3: Monte Carlo ({self._mk['n_paths']:,} paths)...")
        mc_out = self.mc.generate(constraints)
        counts = {s: int(np.sum(mc_out.scenario_labels==s)) for s in ["normal","stress","crisis"]}
        print(f"   Accepted={mc_out.n_paths:,}, rejection={mc_out.rejection_rate:.1%}")
        print(f"   Normal={counts['normal']:,} / Stress={counts['stress']:,} / Crisis={counts['crisis']:,}")

        print("▶ Module 4: Stress Metrics...")
        stress = self.sm.compute(mc_out)
        print(f"   Mean max DD={stress.dd_mean:.4f}, Agg ES={stress.es_aggregate:.4f}")

        fi, fi_grade = None, None
        if self.run_sensitivity:
            print("▶ Module 5: Fragility Index (quick)...")
            fi, fi_grade = compute_fragility_index(cleaned, self._ck, self._mk)
            print(f"   Fragility Index: {fi:.4f} ({fi_grade})")

        print("▶ Module 7: Executive Summary...")
        print_executive_summary(mc_out, stress, self.strategy_name, fi, fi_grade)

        self._last_mc     = mc_out
        self._last_stress = stress
        self._last_fi     = fi
        self._last_grade  = fi_grade
        return {"mc": mc_out, "stress": stress, "fi": fi, "fi_grade": fi_grade,
                "constraints": constraints, "metadata": meta}

    def plot(self, results=None):
        mc  = results["mc"]     if results else self._last_mc
        sm  = results["stress"] if results else self._last_stress
        fi  = results["fi"]     if results else self._last_fi
        fig = plot_dashboard(mc, sm, self.strategy_name, fi,
                             results["fi_grade"] if results else self._last_grade)
        plt.show()
        return fig


# ═══════════════════════════════════════════════════════════════
# REAL DATA LOADER — Yahoo Finance
# ═══════════════════════════════════════════════════════════════

def fetch_returns(ticker, start="2010-01-01", end=None, price_col="Close"):
    """Pull daily pct returns for any ticker from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run:  !pip install yfinance  then restart.")
    import datetime
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")
    print(f"   Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}. Check the symbol.")
    prices  = df[price_col].dropna()
    # Newer yfinance returns MultiIndex columns — squeeze to 1-D Series
    if hasattr(prices, 'columns'):
        prices = prices.iloc[:, 0]
    prices = prices.squeeze()
    returns = prices.pct_change().dropna().to_numpy(dtype=float).flatten()
    dates   = prices.index[1:].to_numpy()
    n = len(returns)
    ann_vol = float(returns.std()) * (252 ** 0.5)
    p0 = float(prices.dropna().iloc[0])
    p1 = float(prices.dropna().iloc[-1])
    tot_ret = p1 / p0 - 1
    print(f"   Got {n} daily returns")
    print(f"   Ann vol approx {ann_vol:.2%}  |  Total return approx {tot_ret:.2%}")
    return returns, dates, prices


def run_on_ticker(ticker, start="2010-01-01", n_paths=10_000, horizon=252, run_sensitivity=True):
    """
    One-liner: fetch real data + run full Blue Lotus engine + plot dashboard.

    Examples
    --------
    results = run_on_ticker("SPY")
    results = run_on_ticker("QQQ", start="2015-01-01")
    results = run_on_ticker("BTC-USD", start="2018-01-01")
    """
    print("\n" + "="*55)
    print(f"  BLUE LOTUS LABS  |  {ticker}")
    print("="*55)
    returns, dates, prices = fetch_returns(ticker, start=start)

    # Auto-scale drawdown thresholds to the actual return series scale.
    # moderate_dd = cumulative loss of ~1 month bad returns
    # severe_dd   = cumulative loss of ~3 months bad returns
    daily_std    = float(returns.std())
    moderate_dd  = -daily_std * 15    # ~1 month of 1-sigma down days
    severe_dd    = -daily_std * 45    # ~3 months of 1-sigma down days

    engine = BlueLotusEngine(
        strategy_name   = f"{ticker} daily returns",
        normalization   = "none",
        n_paths         = n_paths,
        horizon         = horizon,
        run_sensitivity = run_sensitivity,
        random_seed     = 42,
        moderate_dd     = moderate_dd,
        severe_dd       = severe_dd,
    )
    results = engine.run(returns, verbose=True)
    results["ticker"] = ticker
    results["dates"]  = dates
    results["prices"] = prices
    engine.plot(results)
    return results


def run_comparison(tickers, start="2010-01-01", n_paths=5_000):
    """
    Run engine on multiple tickers and print a side-by-side risk table.

    Example
    -------
    run_comparison(["SPY", "QQQ", "TLT", "GLD"])
    """
    import warnings
    warnings.filterwarnings("ignore")
    print("\n  BLUE LOTUS LABS  |  Multi-Ticker Comparison")
    print(f"  Tickers: {tickers}\n")
    rows, all_results = [], {}
    for ticker in tickers:
        try:
            returns, dates, prices = fetch_returns(ticker, start=start)
            engine = BlueLotusEngine(
                strategy_name   = ticker,
                normalization   = "none",
                n_paths         = n_paths,
                horizon         = 252,
                run_sensitivity = False,
                random_seed     = 42,
            )
            r  = engine.run(returns, verbose=False)
            sm = r["stress"]
            rows.append({
                "Ticker":      ticker,
                "N obs":       str(len(returns)),
                "Ann Vol":     f"{returns.std()*252**0.5:.2%}",
                "Mean DD":     f"{sm.dd_mean:.4f}",
                "ES (5%)":     f"{sm.es_aggregate:.4f}",
                "No Recovery": f"{sm.pct_never_recover:.1%}",
                "Med Recov":   f"{sm.recovery_median:.0f}d",
            })
            all_results[ticker] = r
            print(f"   OK  {ticker}")
        except Exception as e:
            print(f"   FAIL {ticker}: {e}")
    if rows:
        cols = list(rows[0].keys())
        w = {c: max(len(c), max(len(row[c]) for row in rows)) + 2 for c in cols}
        sep = "  ".join("-" * w[c] for c in cols)
        hdr = "  ".join(c.ljust(w[c]) for c in cols)
        print("\n" + "="*65)
        print("  COMPARISON TABLE")
        print("="*65)
        print(hdr)
        print(sep)
        for row in rows:
            print("  ".join(row[c].ljust(w[c]) for c in cols))
        print("="*65)
        print("  All metrics in raw daily return units.")
        print("="*65)
    return all_results


# ═══════════════════════════════════════════════════════════════
# RUN — edit ticker or switch to comparison mode below
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__" or True:

    import subprocess, sys
    print("Installing yfinance...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("yfinance ready.\n")

    # ── MODE 1: Single ticker — full engine + dashboard ─────────
    # Change the ticker below — try "QQQ", "TLT", "GLD", "BTC-USD"
    results = run_on_ticker(
        ticker  = "SPY",
        start   = "2010-01-01",
        n_paths = 10_000,
        horizon = 252,
    )

    # ── MODE 2: Multi-ticker comparison (uncomment to use) ──────
    # comparison = run_comparison(
    #     tickers = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"],
    #     start   = "2015-01-01",
    # )

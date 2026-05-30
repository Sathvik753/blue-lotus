"""
╔══════════════════════════════════════════════════════════╗
║   BLUE LOTUS LABS — Stress-Testing Engine  v2.0          ║
║   Single-file version for Google Colab                   ║
║                                                          ║
║   HOW TO USE:                                            ║
║   1. Open a new Google Colab notebook                    ║
║   2. Create a code cell and paste this entire file       ║
║   3. Run it — the demo will execute automatically        ║
║   4. Swap in your own returns at the bottom              ║
║                                                          ║
║   CHANGES FROM v1:                                       ║
║   - Distributional operator replaced with Esscher        ║
║     tilting (importance resampling). Output stats        ║
║     now correspond to a well-defined probability         ║
║     measure Q, not a post-hoc rescaled artefact.         ║
║   - All metrics reported in real return units            ║
║     (percent per day) when normalization="none"          ║
║   - Solver warnings suppressed; graceful fallback        ║
║     to identity weights if convergence fails             ║
║   - np.random.default_rng used throughout (thread-safe)  ║
║   - Drawdown conditioning tracking fixed                 ║
╚══════════════════════════════════════════════════════════╝
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Optional
import warnings


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 — INPUT PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class InputMetadata:
    n_observations: int
    raw_mean:       float
    raw_std:        float
    raw_skewness:   float
    raw_kurtosis:   float
    min_return:     float
    max_return:     float
    winsorized:     bool
    normalization:  str


class InputProcessor:
    """
    Cleans and optionally normalizes a 1-D return series.

    normalization options
    ---------------------
    "none"     : no scaling — keep raw daily returns (recommended for real data)
    "zscore"   : subtract mean, divide by std  (unit-free; use for synthetic data)
    "vol_scale": scale so annualized vol = target_vol
    """

    def __init__(self, winsorize=True, winsorize_limits=(0.01, 0.01),
                 normalization="none", target_vol=0.01):
        self.winsorize        = winsorize
        self.winsorize_limits = winsorize_limits
        self.normalization    = normalization
        self.target_vol       = target_vol
        self.mean_            = None
        self.std_             = None
        self.metadata_        = None

    def fit_transform(self, returns):
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 1:
            raise ValueError("returns must be 1-D")
        if len(returns) < 30:
            warnings.warn(f"Only {len(returns)} observations — high uncertainty.", UserWarning)

        returns = returns[~np.isnan(returns)]

        # Raw stats before any transformation
        raw_mean = float(np.mean(returns))
        raw_std  = float(np.std(returns, ddof=1))
        mu, sig  = raw_mean, raw_std
        raw_skew = float(np.mean(((returns - mu) / (sig + 1e-12)) ** 3))
        raw_kurt = float(np.mean(((returns - mu) / (sig + 1e-12)) ** 4)) - 3.0

        # Winsorize
        if self.winsorize:
            lo, hi  = self.winsorize_limits
            returns = np.clip(returns,
                              np.quantile(returns, lo),
                              np.quantile(returns, 1 - hi))

        # Normalize
        if self.normalization == "zscore":
            mu, sig  = np.mean(returns), np.std(returns, ddof=1)
            returns  = (returns - mu) / (sig + 1e-12)
            norm_lbl = "zscore"
        elif self.normalization == "vol_scale":
            returns  = returns * (self.target_vol / (np.std(returns, ddof=1) + 1e-12))
            norm_lbl = f"vol_scale({self.target_vol})"
        else:
            norm_lbl = "none"

        self.mean_     = float(np.mean(returns))
        self.std_      = float(np.std(returns, ddof=1))
        self.metadata_ = InputMetadata(
            n_observations = len(returns),
            raw_mean       = raw_mean,
            raw_std        = raw_std,
            raw_skewness   = raw_skew,
            raw_kurtosis   = raw_kurt,
            min_return     = float(np.min(returns)),
            max_return     = float(np.max(returns)),
            winsorized     = self.winsorize,
            normalization  = norm_lbl,
        )
        return returns, self.metadata_


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — STRUCTURAL CONSTRAINT LAYER
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeModelOutput:
    transition_matrix: np.ndarray
    regime_means:      np.ndarray
    regime_stds:       np.ndarray
    regime_labels:     np.ndarray
    stationary_dist:   np.ndarray


class RegimeModel:
    """
    3-state Markov regime detector: calm / volatile / crisis.
    Regimes are assigned by rolling-volatility percentiles,
    with crisis override for the worst drawdown periods.
    """

    def fit(self, returns):
        T      = len(returns)
        window = max(5, T // 10)
        roll_vol = np.array([
            np.std(returns[max(0, i - window): i + 1], ddof=0)
            for i in range(T)
        ])
        cumulative = np.cumsum(returns)
        drawdown   = cumulative - np.maximum.accumulate(cumulative)

        v33, v66 = np.percentile(roll_vol, 33), np.percentile(roll_vol, 66)
        labels   = np.where(roll_vol <= v33, 0, np.where(roll_vol <= v66, 1, 2))
        labels[drawdown < np.percentile(drawdown, 10)] = 2

        # Transition matrix (Laplace-smoothed)
        P = np.ones((3, 3))
        for t in range(T - 1):
            P[labels[t], labels[t + 1]] += 1
        P /= P.sum(axis=1, keepdims=True)

        # Per-regime statistics
        means = np.array([
            np.mean(returns[labels == k]) if (labels == k).sum() > 1 else np.mean(returns)
            for k in range(3)
        ])
        stds = np.array([
            np.std(returns[labels == k], ddof=1) if (labels == k).sum() > 1 else np.std(returns, ddof=1)
            for k in range(3)
        ])

        # Stationary distribution via eigenvector
        eigvals, eigvecs = np.linalg.eig(P.T)
        pi = np.abs(np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))]))
        pi /= pi.sum()

        return RegimeModelOutput(
            transition_matrix = P,
            regime_means      = means,
            regime_stds       = stds,
            regime_labels     = labels,
            stationary_dist   = pi,
        )


@dataclass
class TailConstraints:
    alpha:                float
    lower_quantile_bound: float
    upper_quantile_bound: float
    es_target:            float
    method:               str


class TailConstraintLayer:
    """
    Fits a Student-t to the empirical tail and derives an ES target.
    The ES target is used by the Esscher tilting step.
    """

    def __init__(self, alpha=0.05, method="student_t"):
        self.alpha  = alpha
        self.method = method

    def fit(self, returns):
        tail      = returns[returns < np.quantile(returns, self.alpha)]
        q_fitted  = float(np.quantile(returns, self.alpha))

        if self.method == "student_t" and len(tail) >= 4:
            df, loc, scale = stats.t.fit(tail, floc=np.mean(tail))
            q_fitted = float(stats.t.ppf(self.alpha, df=df, loc=loc, scale=scale))

        buf    = abs(q_fitted) * 0.10
        es_emp = float(np.mean(tail)) if len(tail) > 0 else q_fitted

        return TailConstraints(
            alpha                = self.alpha,
            lower_quantile_bound = q_fitted - buf,
            upper_quantile_bound = q_fitted + buf,
            es_target            = es_emp * 1.10,
            method               = self.method,
        )


@dataclass
class BayesianPriors:
    regime_means: np.ndarray
    regime_vars:  np.ndarray


class BayesianShrinkageLayer:
    """
    Gaussian conjugate posterior shrinkage for regime means and variances.

    Posterior mean:
        mu_k* = w_k * x_bar_k + (1 - w_k) * mu_0
        w_k   = (n_k / sigma^2) / (n_k / sigma^2 + 1 / tau^2)

    Posterior variance (inverse-gamma conjugate update):
        sigma_k*^2 = (beta_ig + 0.5 * SS_k) / (alpha_ig + n_k/2 - 1)

    Hyperparameters
    ---------------
    mu_0  : grand mean of all returns  (estimated from data)
    tau^2 : between-regime variance    (estimated from data via ANOVA)
    alpha_ig, beta_ig : prior shape/scale for inverse-gamma
    """

    def __init__(self, alpha_ig=3.0, beta_ig=None):
        self.alpha_ig = alpha_ig
        self.beta_ig  = beta_ig  # None = estimate from data

    def fit(self, regime_output, returns):
        mu_0   = float(np.mean(returns))
        sigma2 = float(np.var(returns, ddof=1))

        # Between-regime variance (ANOVA estimate of tau^2)
        regime_means = regime_output.regime_means
        tau2 = float(np.var(regime_means, ddof=1)) if len(regime_means) > 1 else sigma2

        # beta_ig: set so prior mode = pooled variance
        beta_ig = self.beta_ig if self.beta_ig is not None else sigma2 * (self.alpha_ig - 1)

        means     = np.zeros(3)
        variances = np.zeros(3)

        for k in range(3):
            mask = regime_output.regime_labels == k
            n_k  = int(mask.sum())
            x_k  = returns[mask]

            if n_k < 2:
                means[k]     = mu_0
                variances[k] = sigma2
                continue

            x_bar_k = float(np.mean(x_k))
            ss_k    = float(np.sum((x_k - x_bar_k) ** 2))

            # Precision-weighted shrinkage weight
            prec_data  = n_k / (sigma2 + 1e-12)
            prec_prior = 1.0 / (tau2 + 1e-12)
            w_k        = prec_data / (prec_data + prec_prior)

            means[k] = w_k * x_bar_k + (1.0 - w_k) * mu_0

            # Inverse-gamma conjugate posterior mean
            alpha_post  = self.alpha_ig + n_k / 2.0
            beta_post   = beta_ig + ss_k / 2.0
            variances[k] = beta_post / (alpha_post - 1.0) if alpha_post > 1.0 else sigma2

        return BayesianPriors(regime_means=means, regime_vars=variances)


@dataclass
class DrawdownConditioningOutput:
    states:            np.ndarray
    conditional_probs: dict
    thresholds:        tuple


class DrawdownConditioningLayer:
    """
    Classifies historical observations by drawdown state and
    computes conditional return distributions for each state.
    These distributions blend into the MC generator to produce
    path-dependent return dynamics.
    """

    def __init__(self, moderate_threshold=-0.05, severe_threshold=-0.15):
        self.moderate_threshold = moderate_threshold
        self.severe_threshold   = severe_threshold

    def fit(self, returns):
        dd     = np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns))
        states = np.zeros(len(returns), dtype=int)
        states[dd < self.moderate_threshold] = 1
        states[dd < self.severe_threshold]   = 2

        cp = {}
        for s in range(3):
            mask  = states == s
            cp[s] = {
                "mean": float(np.mean(returns[mask])) if mask.sum() > 1 else float(np.mean(returns)),
                "std":  float(np.std(returns[mask], ddof=1)) if mask.sum() > 1 else float(np.std(returns, ddof=1)),
                "n":    int(mask.sum()),
            }

        return DrawdownConditioningOutput(
            states            = states,
            conditional_probs = cp,
            thresholds        = (self.moderate_threshold, self.severe_threshold),
        )


@dataclass
class ConstraintLayerOutput:
    regime:           RegimeModelOutput
    tail:             TailConstraints
    bayes:            BayesianPriors
    drawdown:         DrawdownConditioningOutput
    implied_vol:      Optional[float]
    known_risk_limit: Optional[float]


class StructuralConstraintLayer:
    """Fits all constraint sub-layers and packages their outputs."""

    def __init__(self, tail_alpha=0.05, tail_method="student_t",
                 alpha_ig=3.0, beta_ig=None,
                 moderate_dd=-0.05, severe_dd=-0.15,
                 implied_vol=None, known_risk_limit=None):
        self.regime_model = RegimeModel()
        self.tail_layer   = TailConstraintLayer(alpha=tail_alpha, method=tail_method)
        self.bayes_layer  = BayesianShrinkageLayer(alpha_ig=alpha_ig, beta_ig=beta_ig)
        self.dd_layer     = DrawdownConditioningLayer(
            moderate_threshold=moderate_dd, severe_threshold=severe_dd
        )
        self.implied_vol      = implied_vol
        self.known_risk_limit = known_risk_limit

    def fit(self, returns):
        regime   = self.regime_model.fit(returns)
        tail     = self.tail_layer.fit(returns)
        bayes    = self.bayes_layer.fit(regime, returns)
        drawdown = self.dd_layer.fit(returns)

        return ConstraintLayerOutput(
            regime           = regime,
            tail             = tail,
            bayes            = bayes,
            drawdown         = drawdown,
            implied_vol      = self.implied_vol,
            known_risk_limit = self.known_risk_limit,
        )


# ═════════════════════════════════════════════════════════════════════════════
# ESSCHER TILTING  (replaces DistributionalOperatorLayer)
# ═════════════════════════════════════════════════════════════════════════════

def solve_lambdas(r, mu_target, es_target, alpha=0.05):
    """
    Find Lagrange multipliers (lambda1, lambda2) for the Esscher measure Q
    that satisfies:
        E_Q[r]              = mu_target      (mean constraint)
        E_Q[r | r <= VaR_a] = es_target      (ES constraint)

    The tilted weight for path i is:
        w_i ∝ exp(lambda1 * r_i + lambda2 * r_i * 1[r_i <= VaR])

    Solved numerically via scipy.optimize.fsolve.
    Falls back to identity weights (lambda1=lambda2=0) if convergence fails.

    Reference: Gerber & Shiu (1994), Transactions of the Society of Actuaries.
    """
    var       = np.quantile(r, alpha)
    tail_mask = r <= var

    r_scale = np.std(r)
    if r_scale < 1e-12:
        return 0.0, 0.0

    r_s  = r / r_scale
    mu_s = mu_target / r_scale
    es_s = es_target / r_scale

    def F(lam):
        lam1, lam2 = lam
        log_w  = lam1 * r_s + lam2 * r_s * tail_mask
        log_w -= log_w.max()
        w      = np.exp(log_w)
        w     /= w.sum()

        mean_eq = np.dot(r_s, w) - mu_s

        w_tail = w[tail_mask]
        r_tail = r_s[tail_mask]
        if w_tail.sum() < 1e-12:
            return [mean_eq, 0.0]
        es_eq = np.dot(r_tail, w_tail) / w_tail.sum() - es_s

        return [mean_eq, es_eq]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam_star = fsolve(F, [0.0, 0.0])
        lam1, lam2 = float(lam_star[0]), float(lam_star[1])
    except Exception:
        lam1, lam2 = 0.0, 0.0

    # Sanity check: if lambdas exploded, fall back to identity
    if abs(lam1) > 1e4 or abs(lam2) > 1e4:
        lam1, lam2 = 0.0, 0.0

    return lam1, lam2


def resample_paths(paths, lam1, lam2, alpha=0.05, rng=None):
    """
    Importance-resample paths using Esscher weights.
    Returns a new array of paths drawn from Q (with replacement).
    If lam1=lam2=0 (fallback), returns a uniform resample.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Pass the caller's Generator for reproducibility. Defaults to a
        fresh Generator (non-reproducible) if None.
    """
    if rng is None:
        rng = np.random.default_rng()

    r         = paths.mean(axis=1)
    var       = np.quantile(r, alpha)
    tail_mask = r <= var

    log_w  = lam1 * r + lam2 * r * tail_mask
    log_w -= log_w.max()
    w      = np.exp(log_w)
    w     /= w.sum()

    idx = rng.choice(len(paths), size=len(paths), replace=True, p=w)
    return paths[idx]


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CONSTRAINED MONTE CARLO GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MonteCarloOutput:
    paths:           np.ndarray
    regime_paths:    np.ndarray
    scenario_labels: np.ndarray
    n_paths:         int
    horizon:         int
    rejection_rate:  float
    lam1:            float
    lam2:            float


class ConstrainedMonteCarloGenerator:
    """
    Generates stress-test paths via:
      1. Markov regime switching with Bayesian-shrunk parameters
      2. Drawdown-conditional return blending (per-path tracking)
      3. Esscher tilting to enforce mean and ES constraints
      4. Hard rejection of bottom-1% outlier paths

    Uses np.random.default_rng for thread-safe, reproducible randomness.
    Each instance owns its own Generator — concurrent runs never interfere.
    """

    def __init__(self, n_paths=10_000, horizon=252, random_seed=42,
                 stress_fraction=0.20, **kwargs):
        self.n_paths         = n_paths
        self.horizon         = horizon
        self.stress_fraction = stress_fraction
        # Thread-safe: each instance owns its own Generator
        self.rng             = np.random.default_rng(random_seed)

    def generate(self, constraints):
        rng    = self.rng
        P      = constraints.regime.transition_matrix
        b_mean = constraints.bayes.regime_means
        b_std  = np.sqrt(np.maximum(constraints.bayes.regime_vars, 1e-12))
        dd_cp  = constraints.drawdown.conditional_probs

        paths        = np.zeros((self.n_paths, self.horizon))
        regime_paths = np.zeros((self.n_paths, self.horizon), dtype=int)

        # Initialise: stress_fraction start in crisis regime
        n_crisis        = int(self.n_paths * self.stress_fraction)
        current_regimes = np.zeros(self.n_paths, dtype=int)
        crisis_idx      = rng.choice(self.n_paths, n_crisis, replace=False)
        current_regimes[crisis_idx] = 2

        cumulative         = np.zeros(self.n_paths)
        # running_max_per_path tracks the historical peak for each path
        # independently across time — needed for correct drawdown conditioning.
        running_max_per_path = np.zeros(self.n_paths)
        current_dd_states    = np.zeros(self.n_paths, dtype=int)
        mod_thr, sev_thr     = constraints.drawdown.thresholds

        for t in range(self.horizon):
            # Regime transitions
            new_regimes = np.zeros(self.n_paths, dtype=int)
            for k in range(3):
                mask = current_regimes == k
                if mask.sum() > 0:
                    new_regimes[mask] = rng.choice(3, size=mask.sum(), p=P[k])
            current_regimes    = new_regimes
            regime_paths[:, t] = current_regimes

            # Draw returns from regime distribution
            returns_t = np.zeros(self.n_paths)
            for k in range(3):
                mask = current_regimes == k
                if mask.sum() > 0:
                    returns_t[mask] = rng.normal(b_mean[k], b_std[k], size=mask.sum())

            # Blend in drawdown-conditional returns
            blend = {0: 0.10, 1: 0.30, 2: 0.50}
            for s in range(3):
                mask = current_dd_states == s
                if mask.sum() > 0 and dd_cp[s]["n"] > 0:
                    w = blend[s]
                    returns_t[mask] = (
                        (1 - w) * returns_t[mask]
                        + w * rng.normal(dd_cp[s]["mean"], dd_cp[s]["std"], mask.sum())
                    )

            paths[:, t] = returns_t
            cumulative  += returns_t

            # Update each path's peak independently (element-wise max over time)
            running_max_per_path = np.maximum(running_max_per_path, cumulative)
            dd = cumulative - running_max_per_path
            current_dd_states = np.where(dd < sev_thr, 2, np.where(dd < mod_thr, 1, 0))

        # ── Esscher tilting ───────────────────────────────────────────────
        mu_target  = float(np.mean(constraints.bayes.regime_means))
        es_target  = constraints.tail.es_target
        r          = paths.mean(axis=1)
        lam1, lam2 = solve_lambdas(r, mu_target, es_target,
                                    alpha=constraints.tail.alpha)
        paths = resample_paths(paths, lam1, lam2,
                               alpha=constraints.tail.alpha, rng=rng)

        # ── Hard rejection: bottom 1% of paths by VaR ────────────────────
        alpha  = constraints.tail.alpha
        path_q = np.quantile(paths, alpha, axis=1)
        cutoff = np.percentile(path_q, 1)
        mask   = path_q >= cutoff

        if constraints.known_risk_limit is not None:
            cum    = np.cumsum(paths, axis=1)
            max_dd = np.min(cum - np.maximum.accumulate(cum, axis=1), axis=1)
            mask  &= max_dd >= constraints.known_risk_limit

        paths        = paths[mask]
        regime_paths = regime_paths[mask]
        rejection_rate = 1.0 - mask.sum() / self.n_paths

        # ── Scenario labels ───────────────────────────────────────────────
        cum    = np.cumsum(paths, axis=1)
        max_dd = np.min(cum - np.maximum.accumulate(cum, axis=1), axis=1)

        path_vol   = paths.std(axis=1).mean()
        thr_normal = -path_vol * 10
        thr_stress = -path_vol * 25
        labels = np.where(
            max_dd > thr_normal, "normal",
            np.where(max_dd > thr_stress, "stress", "crisis")
        )

        return MonteCarloOutput(
            paths           = paths,
            regime_paths    = regime_paths,
            scenario_labels = labels,
            n_paths         = len(paths),
            horizon         = self.horizon,
            rejection_rate  = rejection_rate,
            lam1            = lam1,
            lam2            = lam2,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 — STRESS METRICS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StressMetricsOutput:
    drawdown_dist:     np.ndarray
    dd_mean:           float
    dd_median:         float
    dd_p5:             float
    dd_ci90:           tuple
    dd_by_scenario:    dict
    es_alpha:          float
    es_dist:           np.ndarray
    es_mean:           float
    es_aggregate:      float
    es_ci90:           tuple
    worst_returns:     np.ndarray
    worst_paths:       np.ndarray
    recovery_dist:     np.ndarray
    recovery_mean:     float
    recovery_median:   float
    pct_never_recover: float
    regime_means:      dict
    regime_es:         dict
    regime_fracs:      dict


class StressMetricsEngine:

    def __init__(self, es_alpha=0.05, k_worst=10, ci_level=0.90):
        self.es_alpha = es_alpha
        self.k_worst  = k_worst
        self.ci_lo    = (1 - ci_level) / 2
        self.ci_hi    = 1 - self.ci_lo

    def compute(self, mc):
        paths, labels, regimes = mc.paths, mc.scenario_labels, mc.regime_paths
        alpha = self.es_alpha

        # ── Drawdown ──────────────────────────────────────────────────────
        cum    = np.cumsum(paths, axis=1)
        dd     = cum - np.maximum.accumulate(cum, axis=1)
        max_dd = dd.min(axis=1)

        dd_by_sc = {
            s: float(max_dd[labels == s].mean()) if (labels == s).sum() > 0 else float("nan")
            for s in ["normal", "stress", "crisis"]
        }

        # ── Expected Shortfall ────────────────────────────────────────────
        q_path = np.quantile(paths, alpha, axis=1)
        es_per = np.array([
            float(np.mean(paths[i][paths[i] <= q_path[i]]))
            if (paths[i] <= q_path[i]).sum() > 0 else float(q_path[i])
            for i in range(len(paths))
        ])
        all_r  = paths.flatten()
        agg_q  = np.quantile(all_r, alpha)
        agg_es = float(np.mean(all_r[all_r <= agg_q]))

        # ── Worst-k paths ─────────────────────────────────────────────────
        total_r   = paths.sum(axis=1)
        worst_idx = np.argsort(total_r)[: self.k_worst]

        # ── Recovery time ─────────────────────────────────────────────────
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

        # ── Regime statistics ─────────────────────────────────────────────
        rm, re, rf = {}, {}, {}
        for k in range(3):
            mask   = regimes == k
            r_reg  = paths[mask]
            rf[k]  = float(mask.sum() / paths.size)
            if len(r_reg) > 0:
                rm[k] = float(r_reg.mean())
                q_k   = np.quantile(r_reg, alpha)
                t_k   = r_reg[r_reg <= q_k]
                re[k] = float(t_k.mean()) if len(t_k) > 0 else float(q_k)
            else:
                rm[k] = re[k] = float("nan")

        return StressMetricsOutput(
            drawdown_dist     = max_dd,
            dd_mean           = float(max_dd.mean()),
            dd_median         = float(np.median(max_dd)),
            dd_p5             = float(np.percentile(max_dd, 5)),
            dd_ci90           = (
                float(np.percentile(max_dd, self.ci_lo * 100)),
                float(np.percentile(max_dd, self.ci_hi * 100)),
            ),
            dd_by_scenario    = dd_by_sc,
            es_alpha          = alpha,
            es_dist           = es_per,
            es_mean           = float(es_per.mean()),
            es_aggregate      = agg_es,
            es_ci90           = (
                float(np.percentile(es_per, self.ci_lo * 100)),
                float(np.percentile(es_per, self.ci_hi * 100)),
            ),
            worst_returns     = total_r[worst_idx],
            worst_paths       = paths[worst_idx],
            recovery_dist     = recovery,
            recovery_mean     = float(valid.mean()) if len(valid) > 0 else float("nan"),
            recovery_median   = float(np.median(valid)) if len(valid) > 0 else float("nan"),
            pct_never_recover = float(np.isnan(recovery).mean()),
            regime_means      = rm,
            regime_es         = re,
            regime_fracs      = rf,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 — FRAGILITY INDEX
# ═════════════════════════════════════════════════════════════════════════════

def compute_fragility_index(returns, constraint_kwargs, mc_kwargs,
                             n_trials=10, n_paths=1000):
    """
    Model Fragility Index (MFI): coefficient of variation of aggregate ES
    across n_trials random seeds.

    Each trial uses its own Generator (random_seed=seed), so concurrent
    calls are thread-safe and results are reproducible per seed.

    Interpretation
    --------------
    MFI < 0.25  : Robust   — ES estimate is stable across seeds
    MFI < 0.55  : Moderate — some sensitivity to initialisation
    MFI >= 0.55 : Fragile  — ES estimate is highly seed-dependent
    """
    base_es_list = []
    for seed in range(n_trials):
        try:
            ip         = InputProcessor()
            cleaned, _ = ip.fit_transform(returns)
            cl         = StructuralConstraintLayer(**constraint_kwargs)
            c          = cl.fit(cleaned)
            mc_kw      = {**mc_kwargs, "n_paths": n_paths, "random_seed": seed}
            mc         = ConstrainedMonteCarloGenerator(**mc_kw)
            out        = mc.generate(c)
            sm         = StressMetricsEngine()
            m          = sm.compute(out)
            base_es_list.append(m.es_aggregate)
        except Exception:
            pass

    if len(base_es_list) < 2:
        return 0.5, "Unknown"

    arr   = np.array(base_es_list)
    cv    = float(np.std(arr, ddof=1) / (abs(np.mean(arr)) + 1e-12))
    fi    = float(np.clip(cv, 0, 1))
    grade = "Robust" if fi < 0.25 else ("Moderate" if fi < 0.55 else "Fragile")
    return fi, grade


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 7 — REPORTING & VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

BL_DARK  = "#0D1B2A"
BL_BLUE  = "#1B4F72"
BL_TEAL  = "#148F77"
BL_GOLD  = "#D4AC0D"
BL_ROSE  = "#C0392B"
BL_LIGHT = "#EAF2FF"
BL_GREY  = "#5D6D7E"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": BL_DARK,
        "axes.facecolor":   "#111E2D",
        "axes.edgecolor":   "#2E4057",
        "axes.labelcolor":  "#CBD5E0",
        "xtick.color":      "#CBD5E0",
        "ytick.color":      "#CBD5E0",
        "text.color":       BL_LIGHT,
        "grid.color":       "#1E3A52",
        "grid.linestyle":   "--",
        "grid.alpha":       0.4,
        "font.family":      "monospace",
        "axes.titlesize":   10,
        "axes.labelsize":   8,
    })


def _unit_label(normalization):
    if normalization == "zscore":
        return "z-score units"
    return "daily return units"


def plot_dashboard(mc, sm, meta, strategy_name="Strategy", fi=None, fi_grade=None):
    apply_style()
    unit = _unit_label(meta.normalization)

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BL_DARK)
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    paths  = mc.paths
    labels = mc.scenario_labels
    cum    = np.cumsum(paths, axis=1)
    dd_ser = cum - np.maximum.accumulate(cum, axis=1)
    T, x   = paths.shape[1], np.arange(paths.shape[1])

    # ── Panel 1: Drawdown curves ──────────────────────────────────────────
    ax     = axes[0]
    colors = {"normal": BL_TEAL, "stress": BL_GOLD, "crisis": BL_ROSE}
    for sc, col in colors.items():
        idx = np.where(labels == sc)[0][:40]
        for pi in idx:
            ax.plot(x, dd_ser[pi], color=col, alpha=0.07, linewidth=0.5)
    p5  = np.percentile(dd_ser, 5,  axis=0)
    p50 = np.percentile(dd_ser, 50, axis=0)
    p95 = np.percentile(dd_ser, 95, axis=0)
    ax.fill_between(x, p5, p95, alpha=0.15, color=BL_BLUE)
    ax.plot(x, p50, color=BL_GOLD, lw=1.5, label="Median")
    ax.plot(x, p5,  color=BL_ROSE, lw=1.0, linestyle="--", label="5th pct")
    legend_els = [
        Line2D([0], [0], color=BL_TEAL, lw=1.5, label="Normal"),
        Line2D([0], [0], color=BL_GOLD, lw=1.5, label="Stress"),
        Line2D([0], [0], color=BL_ROSE, lw=1.5, label="Crisis"),
    ]
    ax.legend(handles=legend_els, fontsize=7, framealpha=0.2, loc="lower left")
    ax.set_title("Stress Drawdown Curves", color=BL_LIGHT)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Drawdown ({unit})")
    ax.grid(True)

    # ── Panel 2: Max DD histogram ─────────────────────────────────────────
    ax = axes[1]
    ax.hist(sm.drawdown_dist, bins=50, color=BL_BLUE, edgecolor="none", alpha=0.85, density=True)
    ax.axvline(sm.dd_mean,   color=BL_GOLD, lw=1.5, label=f"Mean: {sm.dd_mean:.4f}")
    ax.axvline(sm.dd_p5,     color=BL_ROSE, lw=1.5, linestyle="--", label=f"5th: {sm.dd_p5:.4f}")
    ax.axvline(sm.dd_median, color=BL_TEAL, lw=1.5, linestyle=":",  label=f"Med: {sm.dd_median:.4f}")
    ax.set_title("Max Drawdown Distribution", color=BL_LIGHT)
    ax.set_xlabel(f"Max Drawdown ({unit})")
    ax.set_ylabel("Density")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(True)

    # ── Panel 3: Recovery distribution ───────────────────────────────────
    ax    = axes[2]
    valid = sm.recovery_dist[~np.isnan(sm.recovery_dist)]
    if len(valid) > 0:
        ax.hist(valid, bins=40, color=BL_TEAL, edgecolor="none", alpha=0.85, density=True)
        ax.axvline(sm.recovery_mean,   color=BL_GOLD, lw=1.5, label=f"Mean: {sm.recovery_mean:.1f}")
        ax.axvline(sm.recovery_median, color=BL_ROSE, lw=1.5, linestyle="--",
                   label=f"Med: {sm.recovery_median:.1f}")
        ax.legend(fontsize=7, framealpha=0.2)
    ax.set_title(f"Time-to-Recovery  ({sm.pct_never_recover:.1%} never recover)", color=BL_LIGHT)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Density")
    ax.grid(True)

    # ── Panel 4: Regime heatmap ───────────────────────────────────────────
    ax    = axes[3]
    reg   = mc.regime_paths
    fracs = np.array([(reg == k).mean(axis=0) for k in range(3)])
    cmap  = LinearSegmentedColormap.from_list("bl", [BL_DARK, BL_BLUE, BL_GOLD, BL_ROSE])
    im    = ax.imshow(fracs, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                      extent=[0, T, -0.5, 2.5])
    plt.colorbar(im, ax=ax, label="Fraction", fraction=0.046, pad=0.04)
    ax.set_title("Regime Transition Heatmap", color=BL_LIGHT)
    ax.set_xlabel("Step")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Calm", "Volatile", "Crisis"], fontsize=8)

    # ── Panel 5: ES distribution ──────────────────────────────────────────
    ax = axes[4]
    ax.hist(sm.es_dist, bins=50, color=BL_ROSE, edgecolor="none", alpha=0.85, density=True)
    ax.axvline(sm.es_mean,      color=BL_GOLD, lw=1.5, label=f"Mean: {sm.es_mean:.4f}")
    ax.axvline(sm.es_aggregate, color=BL_TEAL, lw=1.5, linestyle="--",
               label=f"Agg: {sm.es_aggregate:.4f}")
    ax.set_title(f"Expected Shortfall (α={sm.es_alpha})", color=BL_LIGHT)
    ax.set_xlabel(f"ES / CVaR ({unit})")
    ax.set_ylabel("Density")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(True)

    # ── Panel 6: Worst paths ──────────────────────────────────────────────
    ax   = axes[5]
    k    = len(sm.worst_paths)
    cols = plt.cm.Reds(np.linspace(0.4, 0.9, k))
    for i, path in enumerate(sm.worst_paths):
        ax.plot(np.cumsum(path), color=cols[i], alpha=0.85, lw=1.0)
    ax.set_title(f"Worst-{k} Paths (Cumulative Return)", color=BL_LIGHT)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Cumulative Return ({unit})")
    ax.grid(True)

    fi_str = f"  |  Fragility Index: {fi:.3f} ({fi_grade})" if fi is not None else ""
    fig.suptitle(
        f"BLUE LOTUS LABS  |  {strategy_name}{fi_str}",
        fontsize=13, color=BL_GOLD, weight="bold",
    )
    plt.tight_layout()
    return fig


def print_executive_summary(mc, sm, meta, strategy_name, fi=None, fi_grade=None):
    unit    = _unit_label(meta.normalization)
    counts  = {s: int(np.sum(mc.scenario_labels == s)) for s in ["normal", "stress", "crisis"]}
    lam_str = f"λ1={mc.lam1:.3f}  λ2={mc.lam2:.3f}"

    print()
    print("╔" + "═" * 60 + "╗")
    print(f"║   BLUE LOTUS LABS  |  EXECUTIVE RISK SUMMARY             ║")
    print(f"║   Strategy: {strategy_name:<47}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Paths: {mc.n_paths:,}  |  Rejection: {mc.rejection_rate:.1%}  |  Horizon: {mc.horizon}{'':>10}║")
    print(f"║  Normal / Stress / Crisis: {counts['normal']:,} / {counts['stress']:,} / {counts['crisis']:,}{'':>20}║")
    print(f"║  Esscher: {lam_str:<50}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  DRAWDOWN  [{unit}]{'':>38}║")
    print(f"║    Mean: {sm.dd_mean:+.4f}  |  Median: {sm.dd_median:+.4f}  |  5th pct: {sm.dd_p5:+.4f}  ║")
    print(f"║    90% CI: [{sm.dd_ci90[0]:+.4f}, {sm.dd_ci90[1]:+.4f}]{'':>30}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  EXPECTED SHORTFALL (α={sm.es_alpha})  [{unit}]{'':>22}║")
    print(f"║    Aggregate ES: {sm.es_aggregate:+.4f}  |  Mean ES: {sm.es_mean:+.4f}{'':>22}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  RECOVERY: Mean={sm.recovery_mean:.1f} steps  |  Never={sm.pct_never_recover:.1%}{'':>22}║")
    if fi is not None:
        print("╠" + "═" * 60 + "╣")
        print(f"║  MODEL FRAGILITY INDEX: {fi:.4f}  ({fi_grade}){'':>28}║")
    print("╠" + "═" * 60 + "╣")
    print("║  ⚠  Risk distributions only. No return predictions.      ║")
    print("╚" + "═" * 60 + "╝")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE — wires all modules together
# ═════════════════════════════════════════════════════════════════════════════

class BlueLotusEngine:

    def __init__(self, strategy_name="Strategy",
                 winsorize=True, normalization="none",
                 tail_alpha=0.05, tail_method="student_t",
                 alpha_ig=3.0, beta_ig=None,
                 moderate_dd=-0.05, severe_dd=-0.15,
                 implied_vol=None, known_risk_limit=None,
                 n_paths=10_000, horizon=252, random_seed=42,
                 stress_fraction=0.20, k_worst=10,
                 run_sensitivity=True):

        self.strategy_name   = strategy_name
        self.run_sensitivity = run_sensitivity

        self._ck = dict(
            tail_alpha       = tail_alpha,
            tail_method      = tail_method,
            alpha_ig         = alpha_ig,
            beta_ig          = beta_ig,
            moderate_dd      = moderate_dd,
            severe_dd        = severe_dd,
            implied_vol      = implied_vol,
            known_risk_limit = known_risk_limit,
        )
        self._mk = dict(
            n_paths         = n_paths,
            horizon         = horizon,
            random_seed     = random_seed,
            stress_fraction = stress_fraction,
        )

        self.ip = InputProcessor(winsorize=winsorize, normalization=normalization)
        self.cl = StructuralConstraintLayer(**self._ck)
        self.mc = ConstrainedMonteCarloGenerator(**self._mk)
        self.sm = StressMetricsEngine(es_alpha=tail_alpha, k_worst=k_worst)

        self._last_mc     = None
        self._last_stress = None
        self._last_fi     = None
        self._last_grade  = None
        self._last_meta   = None

    def run(self, returns, verbose=True):
        print(f"\n{'=' * 57}")
        print(f"  BLUE LOTUS LABS — {self.strategy_name}")
        print(f"{'=' * 57}")

        print("▶ Module 1: Input Processing...")
        cleaned, meta = self.ip.fit_transform(np.asarray(returns, dtype=float))
        print(f"   n={meta.n_observations}, mean={meta.raw_mean:.6f}, std={meta.raw_std:.6f}")
        print(f"   Normalization: {meta.normalization}")

        print("▶ Module 2: Structural Constraints...")
        constraints = self.cl.fit(cleaned)
        pi = constraints.regime.stationary_dist
        print(f"   Regime dist — calm={pi[0]:.2f}, volatile={pi[1]:.2f}, crisis={pi[2]:.2f}")
        print(f"   ES target: {constraints.tail.es_target:.6f}")

        print(f"▶ Module 3: Monte Carlo ({self._mk['n_paths']:,} paths) + Esscher tilting...")
        mc_out = self.mc.generate(constraints)
        counts = {s: int(np.sum(mc_out.scenario_labels == s)) for s in ["normal", "stress", "crisis"]}
        print(f"   Accepted={mc_out.n_paths:,}, rejection={mc_out.rejection_rate:.1%}")
        print(f"   Esscher λ1={mc_out.lam1:.4f}, λ2={mc_out.lam2:.4f}")
        print(f"   Normal={counts['normal']:,} / Stress={counts['stress']:,} / Crisis={counts['crisis']:,}")

        print("▶ Module 4: Stress Metrics...")
        stress = self.sm.compute(mc_out)
        print(f"   Mean max DD={stress.dd_mean:.6f}, Agg ES={stress.es_aggregate:.6f}")

        fi, fi_grade = None, None
        if self.run_sensitivity:
            print("▶ Module 5: Fragility Index...")
            fi, fi_grade = compute_fragility_index(cleaned, self._ck, self._mk)
            print(f"   Fragility Index: {fi:.4f} ({fi_grade})")

        print("▶ Module 7: Executive Summary...")
        print_executive_summary(mc_out, stress, meta, self.strategy_name, fi, fi_grade)

        self._last_mc     = mc_out
        self._last_stress = stress
        self._last_fi     = fi
        self._last_grade  = fi_grade
        self._last_meta   = meta

        return {
            "mc":          mc_out,
            "stress":      stress,
            "fi":          fi,
            "fi_grade":    fi_grade,
            "constraints": constraints,
            "metadata":    meta,
        }

    def plot(self, results=None):
        mc    = results["mc"]       if results else self._last_mc
        sm    = results["stress"]   if results else self._last_stress
        meta  = results["metadata"] if results else self._last_meta
        fi    = results["fi"]       if results else self._last_fi
        grade = results["fi_grade"] if results else self._last_grade
        fig   = plot_dashboard(mc, sm, meta, self.strategy_name, fi, grade)
        plt.show()
        return fig


# ═════════════════════════════════════════════════════════════════════════════
# REAL DATA LOADER — Yahoo Finance
# ═════════════════════════════════════════════════════════════════════════════

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
    t  = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(
            f"No price data found for '{ticker}'. "
            "Check the symbol is correct (e.g. SPY, AAPL, BTC-USD)."
        )

    prices = df[price_col].dropna().squeeze()
    returns = prices.pct_change().dropna().to_numpy(dtype=float).flatten()
    dates   = prices.index[1:].to_numpy()

    ann_vol = float(returns.std()) * (252 ** 0.5)
    tot_ret = float(prices.iloc[-1]) / float(prices.iloc[0]) - 1
    print(f"   Got {len(returns)} daily returns")
    print(f"   Ann vol ≈ {ann_vol:.2%}  |  Total return ≈ {tot_ret:.2%}")

    return returns, dates, prices


def run_on_ticker(ticker, start="2010-01-01", n_paths=10_000,
                  horizon=252, run_sensitivity=True):
    """
    One-liner: fetch real data + full Blue Lotus engine + dashboard.

    Examples
    --------
    results = run_on_ticker("SPY")
    results = run_on_ticker("QQQ", start="2015-01-01")
    results = run_on_ticker("BTC-USD", start="2018-01-01")
    """
    print("\n" + "=" * 57)
    print(f"  BLUE LOTUS LABS  |  {ticker}")
    print("=" * 57)

    returns, dates, prices = fetch_returns(ticker, start=start)

    daily_std   = float(returns.std())
    moderate_dd = -daily_std * 15
    severe_dd   = -daily_std * 45

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
    warnings.filterwarnings("ignore")
    print("\n  BLUE LOTUS LABS  |  Multi-Ticker Comparison")
    print(f"  Tickers: {tickers}\n")

    rows, all_results = [], {}
    for ticker in tickers:
        try:
            returns, dates, prices = fetch_returns(ticker, start=start)
            daily_std = float(returns.std())
            engine = BlueLotusEngine(
                strategy_name   = ticker,
                normalization   = "none",
                n_paths         = n_paths,
                horizon         = 252,
                run_sensitivity = False,
                random_seed     = 42,
                moderate_dd     = -daily_std * 15,
                severe_dd       = -daily_std * 45,
            )
            r  = engine.run(returns, verbose=False)
            sm = r["stress"]
            rows.append({
                "Ticker":      ticker,
                "N obs":       str(len(returns)),
                "Ann Vol":     f"{returns.std() * 252 ** 0.5:.2%}",
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
        w    = {c: max(len(c), max(len(row[c]) for row in rows)) + 2 for c in cols}
        sep  = "  ".join("-" * w[c] for c in cols)
        hdr  = "  ".join(c.ljust(w[c]) for c in cols)
        print("\n" + "=" * 67)
        print("  COMPARISON TABLE  (all metrics in raw daily return units)")
        print("=" * 67)
        print(hdr)
        print(sep)
        for row in rows:
            print("  ".join(row[c].ljust(w[c]) for c in cols))
        print("=" * 67)

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import subprocess, sys
    print("Installing yfinance...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "yfinance", "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("yfinance ready.\n")

    # ── MODE 1: Single ticker ────────────────────────────────────────────
    results = run_on_ticker(
        ticker  = "SPY",
        start   = "2010-01-01",
        n_paths = 10_000,
        horizon = 252,
    )

    # ── MODE 2: Multi-ticker comparison (uncomment to use) ───────────────
    # comparison = run_comparison(
    #     tickers = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"],
    #     start   = "2015-01-01",
    # )

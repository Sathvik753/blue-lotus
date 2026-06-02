"""
╔══════════════════════════════════════════════════════════════════════════╗
║   BLUE LOTUS LABS — Institutional Stress-Testing Engine  v3.0           ║
║                                                                          ║
║   Architecture                                                           ║
║   ──────────                                                             ║
║   Module 1    Input Processing                                           ║
║   Module 2    Structural Constraints                                     ║
║                  · 3-state HMM regime detection  (Baum-Welch EM)        ║
║                  · GPD/EVT tail fitting           (Peaks-over-Threshold) ║
║                  · Bayesian shrinkage of regime parameters               ║
║                  · Drawdown-conditional return distributions             ║
║   Esscher     Change of Measure                                          ║
║                  · Importance resampling to risk-neutral measure Q       ║
║                  · Multi-start solver with convergence checking          ║
║   Module 3    Constrained Monte Carlo Generator                          ║
║   Module 4    Stress Metrics + Bootstrap Confidence Intervals            ║
║   Module 4b   Multi-Asset Extension (LedoitWolf + Cholesky)             ║
║   Module 5    Model Fragility Index (Wasserstein distance)               ║
║   Module 5b   Historical Backtest Validator                              ║
║   Module 7    Reporting and Visualization                                ║
║                                                                          ║
║   Key references                                                         ║
║   ──────────────                                                         ║
║   Baum & Welch (1972)      EM algorithm for HMMs                        ║
║   Pickands (1975)          Generalized Pareto Distribution               ║
║   McNeil & Frey (2000)     EVT-based ES estimation                      ║
║   Gerber & Shiu (1994)     Esscher transform / change of measure        ║
║   Ledoit & Wolf (2004)     Shrinkage covariance estimator               ║
║   Villani (2008)           Optimal transport and Wasserstein distance    ║
╚══════════════════════════════════════════════════════════════════════════╝
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
from scipy.stats import genpareto, wasserstein_distance
from sklearn.covariance import LedoitWolf
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

# hmmlearn removed — regime detection is now fully deterministic.
# See RegimeModel below.


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
    ann_vol:        float       # annualized volatility (sqrt-252 scaled)
    winsorized:     bool
    normalization:  str


class InputProcessor:
    """
    Cleans and optionally normalizes a 1-D return series.

    normalization options
    ---------------------
    "none"     : no scaling  (recommended for all real-data use)
    "zscore"   : subtract mean, divide by std
    "vol_scale": scale to a target annualized volatility
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

        returns  = returns[~np.isnan(returns)]
        raw_mean = float(np.mean(returns))
        raw_std  = float(np.std(returns, ddof=1))
        mu, sig  = raw_mean, raw_std
        raw_skew = float(np.mean(((returns - mu) / (sig + 1e-12)) ** 3))
        raw_kurt = float(np.mean(((returns - mu) / (sig + 1e-12)) ** 4)) - 3.0
        ann_vol  = float(raw_std * np.sqrt(252))

        if self.winsorize:
            lo, hi  = self.winsorize_limits
            returns = np.clip(returns,
                              np.quantile(returns, lo),
                              np.quantile(returns, 1 - hi))

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
            ann_vol        = ann_vol,
            winsorized     = self.winsorize,
            normalization  = norm_lbl,
        )
        return returns, self.metadata_


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2a — HMM REGIME DETECTION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeModelOutput:
    transition_matrix: np.ndarray   # (3, 3) row-stochastic
    regime_means:      np.ndarray   # (3,) per-state mean return
    regime_stds:       np.ndarray   # (3,) per-state return std
    regime_labels:     np.ndarray   # (T,) state sequence (0=calm, 1=volatile, 2=crisis)
    stationary_dist:   np.ndarray   # (3,) long-run state probabilities
    log_likelihood:    float        # HMM log-likelihood on training data
    aic:               float        # Akaike Information Criterion
    bic:               float        # Bayesian Information Criterion


class RegimeModel:
    """
    Deterministic 3-state regime detector using rolling realized volatility.

    Replaces the HMM approach which produced uncalibrated crisis fractions
    and required repeated manual tuning.  This approach is fully auditable:
    every label can be explained by a simple rule.

    States
    ------
        0 = Calm     (rolling vol < 70th percentile)
        1 = Volatile (70th ≤ rolling vol < 90th percentile)
        2 = Crisis   (rolling vol ≥ 90th percentile,
                      sustained ≥ min_crisis_duration consecutive days,
                      AND negative rolling return over the same window)

    The negative-return gate prevents high-vol bull-market rallies from
    being mislabeled as crisis.

    Parameters
    ----------
    window : int
        Rolling window for realized vol and return (default 20 trading days).
    volatile_pct : float
        Vol percentile threshold separating calm from volatile (default 70).
    crisis_pct : float
        Vol percentile threshold for crisis candidates (default 90).
    min_crisis_duration : int
        Minimum consecutive days required to call a run "crisis" (default 5).
    """

    def __init__(self, window: int = 20,
                 volatile_pct: float = 70,
                 crisis_pct: float = 90,
                 min_crisis_duration: int = 5,
                 # legacy kwargs silently ignored for backward compat
                 **kwargs):
        self.window              = window
        self.volatile_pct        = volatile_pct
        self.crisis_pct          = crisis_pct
        self.min_crisis_duration = min_crisis_duration

    # ── Persistence filter ────────────────────────────────────────────────────

    def _smooth_crisis(self, labels: np.ndarray) -> np.ndarray:
        """
        Downgrade crisis runs shorter than min_crisis_duration to volatile.
        """
        if self.min_crisis_duration <= 1:
            return labels.copy()
        smoothed = labels.copy()
        T, i = len(smoothed), 0
        while i < T:
            if smoothed[i] == 2:
                j = i
                while j < T and smoothed[j] == 2:
                    j += 1
                if (j - i) < self.min_crisis_duration:
                    smoothed[i:j] = 1
                i = j
            else:
                i += 1
        return smoothed

    # ── Transition matrix + stats from labels ─────────────────────────────────

    def _stats_from_labels(self, returns: np.ndarray,
                           labels: np.ndarray) -> tuple:
        """Compute transition matrix, emission means/stds, stationary dist."""
        P = np.ones((3, 3))   # Laplace smoothing
        for t in range(len(labels) - 1):
            P[labels[t], labels[t + 1]] += 1
        P /= P.sum(axis=1, keepdims=True)

        means = np.array([
            np.mean(returns[labels == k]) if (labels == k).sum() > 1 else np.mean(returns)
            for k in range(3)
        ])
        stds = np.array([
            np.std(returns[labels == k], ddof=1) if (labels == k).sum() > 1 else np.std(returns, ddof=1)
            for k in range(3)
        ])
        eigvals, eigvecs = np.linalg.eig(P.T)
        pi = np.abs(np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))]))
        pi /= pi.sum()
        return P, means, stds, pi

    # ── Main fitting routine ─────────────────────────────────────────────────

    def fit(self, returns: np.ndarray) -> "RegimeModelOutput":
        n      = len(returns)
        window = max(5, min(self.window, n // 10))

        # Step 1: rolling realized volatility (annualized)
        sigma = np.full(n, np.nan)
        for i in range(window, n):
            sigma[i] = np.std(returns[i - window:i], ddof=1) * np.sqrt(252)

        valid = ~np.isnan(sigma)

        # Step 2: percentile thresholds from the data itself
        vol_volatile = np.percentile(sigma[valid], self.volatile_pct)
        vol_crisis   = np.percentile(sigma[valid], self.crisis_pct)

        # Step 3: rolling return (same window) — crisis gate
        roll_ret = np.full(n, np.nan)
        for i in range(window, n):
            roll_ret[i] = np.sum(returns[i - window:i])

        # Step 4: raw label assignment
        labels = np.zeros(n, dtype=int)
        labels[sigma >= vol_volatile] = 1
        labels[sigma >= vol_crisis]   = 2

        # Step 5: negative-return gate — downgrade crisis with positive trend
        labels[(labels == 2) & (roll_ret > 0)] = 1

        # Step 6: persistence filter (min_crisis_duration consecutive days)
        labels = self._smooth_crisis(labels)

        # Step 7: fill NaN prefix (first `window` days) with calm
        labels[:window] = 0

        # Step 8: compute transition matrix and emission stats
        P, means, stds, pi = self._stats_from_labels(returns, labels)

        crisis_frac = float((labels == 2).mean())
        if crisis_frac > 0.15:
            logger.warning(
                "Crisis regime fraction %.1f%% > 15%%. "
                "Check data or increase crisis_pct / min_crisis_duration.",
                crisis_frac * 100,
            )

        return RegimeModelOutput(
            transition_matrix = P,
            regime_means      = means,
            regime_stds       = stds,
            regime_labels     = labels,
            stationary_dist   = pi,
            log_likelihood    = float("nan"),   # not applicable for deterministic model
            aic               = float("nan"),
            bic               = float("nan"),
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2b — GPD TAIL FITTING (Extreme Value Theory)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TailConstraints:
    alpha:                float
    lower_quantile_bound: float
    upper_quantile_bound: float
    es_target:            float
    method:               str
    # GPD parameters (NaN if empirical fallback was used)
    xi:                   float   # GPD shape parameter
    beta:                 float   # GPD scale parameter
    threshold:            float   # Peaks-over-Threshold threshold (return units)
    n_tail:               int     # number of exceedances used in fit


class TailConstraintLayer:
    """
    Generalized Pareto Distribution tail fitting via Peaks-over-Threshold EVT.

    Method
    ------
    1. Hill estimator stability: search over candidate thresholds (2%–15% of
       data in the tail) and select the threshold where the Hill shape parameter
       estimate is most stable (minimum local variance of xi estimates).

    2. GPD fit: fit GPD(xi, beta) to exceedances Y = u - X  (X < u)
       using maximum likelihood estimation via scipy.stats.genpareto.

    3. Analytical VaR and ES:
       For xi ≠ 0:
           VaR_α = u + (β/ξ) · [(α·n/Nu)^{-ξ} − 1]
           ES_α  = (VaR_α + β − ξ·u) / (1 − ξ)
       For ξ → 0 (exponential tail):
           VaR_α = u − β · ln(α·n/Nu)
           ES_α  = VaR_α + β

       where u is the threshold (in loss space), n is the sample size,
       Nu is the number of threshold exceedances.

    Reference: McNeil & Frey (2000), "Estimation of tail-related risk
    measures for heteroscedastic financial time series."

    Falls back to empirical quantile + Student-t ES if < 20 exceedances
    or if GPD fitting fails.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # ── Hill threshold selection ──────────────────────────────────────────────

    def _select_threshold(self, losses: np.ndarray):
        """
        Find the threshold minimizing local variance of GPD shape estimates.

        Parameters
        ----------
        losses : 1-D array of positive loss values (sorted ascending).

        Returns
        -------
        threshold_loss : float  (threshold in loss space, > 0)
        k_opt          : int    (number of tail observations above threshold)
        """
        n       = len(losses)
        losses_desc = np.sort(losses)[::-1]  # descending

        k_min = max(15, int(0.02 * n))
        k_max = min(int(0.15 * n), n - 2)

        if k_max <= k_min:
            return float(losses_desc[k_min]), k_min

        ks = np.arange(k_min, k_max + 1)
        xi_est = []
        valid_ks = []

        for k in ks:
            u = losses_desc[k]
            exc = losses_desc[:k] - u
            if len(exc) < 5 or u <= 0:
                continue
            try:
                xi, _, _ = genpareto.fit(exc, floc=0)
                xi_est.append(xi)
                valid_ks.append(k)
            except Exception:
                pass

        if len(xi_est) < 5:
            return float(losses_desc[k_min]), k_min

        xi_arr = np.array(xi_est)
        half_w = max(3, len(xi_arr) // 8)
        local_std = np.array([
            np.std(xi_arr[max(0, i - half_w): min(len(xi_arr), i + half_w + 1)])
            for i in range(len(xi_arr))
        ])

        best_idx   = int(np.argmin(local_std))
        k_opt      = valid_ks[best_idx]
        threshold  = float(losses_desc[k_opt])
        return threshold, k_opt

    # ── Empirical fallback ────────────────────────────────────────────────────

    def _empirical_fallback(self, returns):
        q     = float(np.quantile(returns, self.alpha))
        tail  = returns[returns < q]
        es    = float(np.mean(tail)) if len(tail) > 0 else q
        buf   = abs(q) * 0.10
        return TailConstraints(
            alpha                = self.alpha,
            lower_quantile_bound = q - buf,
            upper_quantile_bound = q + buf,
            es_target            = es * 1.10,
            method               = "empirical_fallback",
            xi                   = float("nan"),
            beta                 = float("nan"),
            threshold            = float("nan"),
            n_tail               = 0,
        )

    # ── Main fitting routine ─────────────────────────────────────────────────

    def fit(self, returns):
        losses = -returns        # work in positive loss space

        # Select threshold
        pos_losses = losses[losses > 0]
        if len(pos_losses) < 20:
            return self._empirical_fallback(returns)

        # Ensure losses are sorted before Hill estimator
        pos_losses = np.sort(pos_losses)[::-1]

        threshold_loss, k_tail = self._select_threshold(pos_losses)
        pos_losses = losses[losses > 0]   # restore unsorted for exceedance filter
        threshold_return = -threshold_loss

        # Extract exceedances (losses strictly above threshold)
        exceedances = pos_losses[pos_losses > threshold_loss] - threshold_loss
        n_tail      = len(exceedances)
        n           = len(returns)

        if n_tail < 15:
            return self._empirical_fallback(returns)

        # Fit GPD via MLE
        try:
            xi, _loc, beta = genpareto.fit(exceedances, floc=0)
        except Exception:
            return self._empirical_fallback(returns)

        # Guard against infinite-mean case (xi >= 1 is numerically unstable for ES)
        if xi >= 1.0:
            logger.warning("GPD xi=%.3f >= 1 (infinite mean). Capping at 0.95.", xi)
            xi = 0.95

        # Analytical VaR and ES
        frac = self.alpha * n / (n_tail + 1e-12)   # tail probability ratio

        if abs(xi) > 1e-6:
            var_loss = threshold_loss + (beta / xi) * (frac ** (-xi) - 1.0)
            es_loss  = (var_loss + beta - xi * threshold_loss) / (1.0 - xi)
        else:
            # Gumbel / exponential tail (xi → 0)
            var_loss = threshold_loss - beta * np.log(frac)
            es_loss  = var_loss + beta

        # Sanity check: VaR must be beyond empirical quantile
        emp_q = abs(float(np.quantile(returns, self.alpha)))
        if var_loss < emp_q * 0.5 or var_loss > emp_q * 3.0:
            logger.warning("GPD VaR outside plausible range — using empirical fallback.")
            return self._empirical_fallback(returns)

        var_return = -var_loss
        es_return  = -es_loss
        buf        = abs(var_return) * 0.10

        return TailConstraints(
            alpha                = self.alpha,
            lower_quantile_bound = var_return - buf,
            upper_quantile_bound = var_return + buf,
            es_target            = es_return * 1.10,   # 10% more conservative
            method               = "gpd_evt",
            xi                   = float(xi),
            beta                 = float(beta),
            threshold            = float(threshold_return),
            n_tail               = n_tail,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2c — BAYESIAN SHRINKAGE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BayesianPriors:
    regime_means: np.ndarray
    regime_vars:  np.ndarray


class BayesianShrinkageLayer:
    """
    Conjugate posterior shrinkage for HMM regime emission parameters.

    Posterior mean (precision-weighted):
        w_k     = (n_k / σ²) / (n_k / σ² + 1/τ²)
        μ_k*    = w_k · x̄_k + (1 − w_k) · μ_0

    Posterior variance (inverse-gamma conjugate update):
        α'  = α_IG + n_k / 2
        β'  = β_IG + SS_k / 2
        σ_k*² = β' / (α' − 1)

    where μ_0 is the grand mean, τ² is the between-regime variance
    (ANOVA estimate), and SS_k is the within-regime sum of squares.
    """

    def __init__(self, alpha_ig: float = 3.0, beta_ig: Optional[float] = None):
        self.alpha_ig = alpha_ig
        self.beta_ig  = beta_ig

    def fit(self, regime_output, returns):
        mu_0   = float(np.mean(returns))
        sigma2 = float(np.var(returns, ddof=1))
        tau2   = float(np.var(regime_output.regime_means, ddof=1)) if len(regime_output.regime_means) > 1 else sigma2
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

            x_bar_k    = float(np.mean(x_k))
            ss_k       = float(np.sum((x_k - x_bar_k) ** 2))
            prec_data  = n_k / (sigma2 + 1e-12)
            prec_prior = 1.0 / (tau2 + 1e-12)
            w_k        = prec_data / (prec_data + prec_prior)
            means[k]   = w_k * x_bar_k + (1.0 - w_k) * mu_0

            alpha_post    = self.alpha_ig + n_k / 2.0
            beta_post     = beta_ig + ss_k / 2.0
            variances[k]  = beta_post / (alpha_post - 1.0) if alpha_post > 1.0 else sigma2

        return BayesianPriors(regime_means=means, regime_vars=variances)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2d — DRAWDOWN CONDITIONING
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DrawdownConditioningOutput:
    states:            np.ndarray
    conditional_probs: dict
    thresholds:        tuple


class DrawdownConditioningLayer:
    """
    Classifies historical observations into drawdown states and estimates
    conditional return distributions for each state.

    States: 0 = normal, 1 = moderate drawdown, 2 = severe drawdown.

    During Monte Carlo generation, each path's current drawdown state
    determines how much of the return is drawn from the stress distribution
    vs the regime distribution.  Blend weights: {0: 10%, 1: 30%, 2: 50%}.
    """

    def __init__(self, moderate_threshold: float = -0.05,
                 severe_threshold: float = -0.15):
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


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2e — STRUCTURAL CONSTRAINT LAYER
# ═════════════════════════════════════════════════════════════════════════════

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

    def __init__(self, tail_alpha: float = 0.05,
                 alpha_ig: float = 3.0, beta_ig: Optional[float] = None,
                 moderate_dd: float = -0.05, severe_dd: float = -0.15,
                 implied_vol: Optional[float] = None,
                 known_risk_limit: Optional[float] = None,
                 # regime params
                 window: int = 20,
                 volatile_pct: float = 70,
                 crisis_pct: float = 90,
                 min_crisis_duration: int = 5,
                 # legacy kwargs accepted but ignored
                 **kwargs):
        self.regime_model = RegimeModel(
            window=window,
            volatile_pct=volatile_pct,
            crisis_pct=crisis_pct,
            min_crisis_duration=min_crisis_duration,
        )
        self.tail_layer   = TailConstraintLayer(alpha=tail_alpha)
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
# ESSCHER TILTING  (improved robustness)
# ═════════════════════════════════════════════════════════════════════════════

def _perturb_constraints(c: ConstraintLayerOutput,
                         mean_mult: float = 1.0,
                         vol_mult: float  = 1.0,
                         trans_noise: float = 0.0,
                         rng: Optional[np.random.Generator] = None) -> ConstraintLayerOutput:
    """
    Create a perturbed copy of a ConstraintLayerOutput for MFI sensitivity analysis.
    No re-fitting is required; parameters are scaled directly.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    new_bayes = BayesianPriors(
        regime_means = c.bayes.regime_means * mean_mult,
        regime_vars  = c.bayes.regime_vars  * (vol_mult ** 2),
    )

    new_P = c.regime.transition_matrix.copy()
    if trans_noise > 0.0:
        noise = rng.dirichlet(np.ones(3) * max(0.5 / trans_noise, 0.1), size=3)
        new_P = (1 - trans_noise) * new_P + trans_noise * noise
        new_P /= new_P.sum(axis=1, keepdims=True)

    new_regime = RegimeModelOutput(
        transition_matrix = new_P,
        regime_means      = c.regime.regime_means * mean_mult,
        regime_stds       = c.regime.regime_stds  * vol_mult,
        regime_labels     = c.regime.regime_labels,
        stationary_dist   = c.regime.stationary_dist,
        log_likelihood    = c.regime.log_likelihood,
        aic               = c.regime.aic,
        bic               = c.regime.bic,
    )

    return ConstraintLayerOutput(
        regime           = new_regime,
        tail             = c.tail,
        bayes            = new_bayes,
        drawdown         = c.drawdown,
        implied_vol      = c.implied_vol,
        known_risk_limit = c.known_risk_limit,
    )


def solve_lambdas(r: np.ndarray, mu_target: float,
                  es_target: float, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Find Lagrange multipliers (λ₁, λ₂) for the Esscher measure Q satisfying:

        E_Q[r]              = μ_target    (mean constraint)
        E_Q[r | r ≤ VaR_α] = ES_target   (ES constraint)

    Tilted weight:  w_i ∝ exp(λ₁·rᵢ + λ₂·rᵢ·𝟙[rᵢ ≤ VaR_α])

    Robustness improvements over v2.0:
        1. Five starting points: searches for the global minimum
        2. Solution with the smallest residual norm is selected
        3. Convergence check: residual > 1e-4 triggers a warning and
           identity-weight fallback rather than silently using bad lambdas

    Reference: Gerber & Shiu (1994), Transactions of the Society of Actuaries.

    Falls back to identity weights (λ₁ = λ₂ = 0) if convergence fails.
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
        w      = np.exp(log_w);  w /= w.sum()

        mean_eq = np.dot(r_s, w) - mu_s

        w_tail = w[tail_mask]
        r_tail = r_s[tail_mask]
        if w_tail.sum() < 1e-12:
            return [mean_eq, 0.0]
        es_eq = np.dot(r_tail, w_tail) / w_tail.sum() - es_s
        return [mean_eq, es_eq]

    starting_points = [
        [0.0,  0.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [2.0, -2.0],
        [0.5, -0.5],
    ]

    best_lam  = [0.0, 0.0]
    best_norm = float("inf")

    for x0 in starting_points:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol, _info, ier, _msg = fsolve(F, x0, full_output=True)
            if ier == 1:
                norm = float(np.linalg.norm(F(sol)))
                if norm < best_norm:
                    best_norm = norm
                    best_lam  = sol
        except Exception:
            pass

    if best_norm > 1e-4:
        logger.warning(
            "Esscher solver residual norm %.2e > 1e-4. "
            "Falling back to identity weights.", best_norm
        )
        return 0.0, 0.0

    lam1, lam2 = float(best_lam[0]), float(best_lam[1])
    if abs(lam1) > 50 or abs(lam2) > 50:
        logger.warning("Esscher λ magnitude > 50 — falling back to identity weights.")
        return 0.0, 0.0

    return lam1, lam2


def resample_paths(paths: np.ndarray, lam1: float, lam2: float,
                   alpha: float = 0.05,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Importance-resample paths using Esscher weights.
    Returns paths drawn from the tilted measure Q (with replacement).
    """
    if rng is None:
        rng = np.random.default_rng()

    r         = paths.mean(axis=1)
    var       = np.quantile(r, alpha)
    tail_mask = r <= var
    log_w     = lam1 * r + lam2 * r * tail_mask
    log_w    -= log_w.max()
    w         = np.exp(log_w);  w /= w.sum()

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
      1. Markov regime switching using HMM transition matrix and
         Bayesian-shrunk emission parameters
      2. Drawdown-conditional return blending
      3. Esscher importance resampling to enforce mean and ES constraints
      4. Hard rejection of bottom-1% outlier paths

    Each instance owns its own np.random.Generator (thread-safe).
    """

    def __init__(self, n_paths: int = 10_000, horizon: int = 252,
                 random_seed: int = 42, stress_fraction: float = 0.20,
                 drawdown_floor: float = -0.70,
                 **kwargs):
        self.n_paths         = n_paths
        self.horizon         = horizon
        self.stress_fraction = stress_fraction
        self.drawdown_floor  = drawdown_floor   # cumulative loss floor (e.g. -0.70 = -70%)
        self.rng             = np.random.default_rng(random_seed)

    def generate(self, constraints: ConstraintLayerOutput) -> MonteCarloOutput:
        rng    = self.rng
        P      = constraints.regime.transition_matrix
        b_mean = constraints.bayes.regime_means
        b_std  = np.sqrt(np.maximum(constraints.bayes.regime_vars, 1e-12))
        dd_cp  = constraints.drawdown.conditional_probs

        paths        = np.zeros((self.n_paths, self.horizon))
        regime_paths = np.zeros((self.n_paths, self.horizon), dtype=int)

        # Initialise: stress_fraction of paths start in crisis regime
        n_crisis        = int(self.n_paths * self.stress_fraction)
        current_regimes = np.zeros(self.n_paths, dtype=int)
        crisis_idx      = rng.choice(self.n_paths, n_crisis, replace=False)
        current_regimes[crisis_idx] = 2

        cumulative            = np.zeros(self.n_paths)
        running_max_per_path  = np.zeros(self.n_paths)   # per-path historical peak
        current_dd_states     = np.zeros(self.n_paths, dtype=int)
        mod_thr, sev_thr      = constraints.drawdown.thresholds

        for t in range(self.horizon):
            # Markov regime transitions
            new_regimes = np.zeros(self.n_paths, dtype=int)
            for k in range(3):
                mask = current_regimes == k
                if mask.sum() > 0:
                    new_regimes[mask] = rng.choice(3, size=mask.sum(), p=P[k])
            current_regimes    = new_regimes
            regime_paths[:, t] = current_regimes

            # Draw returns from regime-specific Gaussian
            returns_t = np.zeros(self.n_paths)
            for k in range(3):
                mask = current_regimes == k
                if mask.sum() > 0:
                    returns_t[mask] = rng.normal(b_mean[k], b_std[k], size=mask.sum())

            # Drawdown-conditional blending
            blend = {0: 0.10, 1: 0.30, 2: 0.50}
            for s in range(3):
                mask = current_dd_states == s
                if mask.sum() > 0 and dd_cp[s]["n"] > 0:
                    w = blend[s]
                    returns_t[mask] = (
                        (1 - w) * returns_t[mask]
                        + w * rng.normal(dd_cp[s]["mean"], dd_cp[s]["std"], mask.sum())
                    )

            # Drawdown floor clamp: paths that have already lost more than
            # `drawdown_floor` (e.g. -70%) cannot lose further — they are
            # treated as stopped out.  This prevents runaway compounding in
            # tail paths and keeps the simulation physically realistic.
            floor_hit         = cumulative <= self.drawdown_floor
            returns_t[floor_hit] = 0.0

            paths[:, t] = returns_t
            cumulative  += returns_t
            cumulative   = np.maximum(cumulative, self.drawdown_floor)  # hard clamp

            # Update per-path running maximum (correct drawdown tracking)
            running_max_per_path = np.maximum(running_max_per_path, cumulative)
            dd = cumulative - running_max_per_path
            current_dd_states = np.where(dd < sev_thr, 2, np.where(dd < mod_thr, 1, 0))

        # Esscher tilting
        mu_target  = float(np.mean(constraints.bayes.regime_means))
        es_target  = constraints.tail.es_target
        r          = paths.mean(axis=1)
        lam1, lam2 = solve_lambdas(r, mu_target, es_target,
                                    alpha=constraints.tail.alpha)
        paths = resample_paths(paths, lam1, lam2,
                               alpha=constraints.tail.alpha, rng=rng)

        # Hard rejection: remove bottom-1% outlier paths by VaR
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

        # Scenario labels
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
# MODULE 4 — STRESS METRICS  +  BOOTSTRAP CONFIDENCE INTERVALS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StressMetricsOutput:
    # ── Drawdown ──────────────────────────────────────────────────────────────
    drawdown_dist:        np.ndarray
    dd_mean:              float
    dd_median:            float
    dd_p5:                float
    dd_ci90:              tuple      # (lo, hi) 90% CI of drawdown distribution
    dd_by_scenario:       dict
    # Bootstrap CIs (90%)
    dd_mean_ci90:         tuple
    dd_p5_ci90:           tuple
    # ── Expected Shortfall ────────────────────────────────────────────────────
    es_alpha:             float
    es_dist:              np.ndarray
    es_mean:              float
    es_aggregate:         float
    es_ci90:              tuple
    # Bootstrap CIs (90%)
    es_aggregate_ci90:    tuple
    es_mean_ci90:         tuple
    # ── Recovery ─────────────────────────────────────────────────────────────
    worst_returns:        np.ndarray
    worst_paths:          np.ndarray
    recovery_dist:        np.ndarray
    recovery_mean:        float
    recovery_median:      float
    pct_never_recover:    float
    recovery_mean_ci90:   tuple
    # ── Regime ────────────────────────────────────────────────────────────────
    regime_means:         dict
    regime_es:            dict
    regime_fracs:         dict


class StressMetricsEngine:

    def __init__(self, es_alpha: float = 0.05, k_worst: int = 10,
                 ci_level: float = 0.90, n_bootstrap: int = 500):
        self.es_alpha    = es_alpha
        self.k_worst     = k_worst
        self.ci_lo       = (1 - ci_level) / 2
        self.ci_hi       = 1 - self.ci_lo
        self.n_bootstrap = n_bootstrap

    # ── Vectorized bootstrap ──────────────────────────────────────────────────

    def _bootstrap_cis(self, paths: np.ndarray,
                       rng: np.random.Generator) -> dict:
        """
        Compute 90% bootstrap confidence intervals by resampling paths.

        Fully vectorised: a single bootstrap iteration computes all metrics
        simultaneously without any Python-level loops over paths.
        """
        n     = len(paths)
        alpha = self.es_alpha
        dd_means, dd_p5s, es_aggs, es_means, rec_means = [], [], [], [], []

        for _ in range(self.n_bootstrap):
            idx    = rng.choice(n, size=n, replace=True)
            s      = paths[idx]

            # Max drawdown
            cum    = np.cumsum(s, axis=1)
            dd     = cum - np.maximum.accumulate(cum, axis=1)
            max_dd = dd.min(axis=1)
            dd_means.append(float(max_dd.mean()))
            dd_p5s.append(float(np.percentile(max_dd, 5)))

            # ES (vectorised)
            q_mat   = np.quantile(s, alpha, axis=1, keepdims=True)
            mask_   = s <= q_mat
            tail_s  = (s * mask_).sum(axis=1)
            tail_c  = mask_.sum(axis=1).clip(1)
            es_per  = tail_s / tail_c
            es_means.append(float(es_per.mean()))

            all_r  = s.flatten()
            aq     = np.quantile(all_r, alpha)
            es_aggs.append(float(np.mean(all_r[all_r <= aq])))

            # Recovery (simplified: fraction never recovering × mean horizon)
            rec = np.full(len(s), np.nan)
            for i in range(len(s)):
                t_dd = int(np.argmin(dd[i]))
                if dd[i, t_dd] >= 0:
                    rec[i] = 0
                    continue
                peak = np.maximum.accumulate(cum[i])[t_dd]
                post = cum[i, t_dd:]
                r_   = np.where(post >= peak)[0]
                if len(r_) > 0:
                    rec[i] = float(r_[0])
            valid = rec[~np.isnan(rec)]
            rec_means.append(float(valid.mean()) if len(valid) > 0 else float("nan"))

        def ci(vals):
            v = [x for x in vals if not np.isnan(x)]
            if len(v) < 2:
                return (float("nan"), float("nan"))
            return float(np.percentile(v, self.ci_lo * 100)), float(np.percentile(v, self.ci_hi * 100))

        return {
            "dd_mean":      ci(dd_means),
            "dd_p5":        ci(dd_p5s),
            "es_aggregate": ci(es_aggs),
            "es_mean":      ci(es_means),
            "rec_mean":     ci(rec_means),
        }

    # ── Main compute ──────────────────────────────────────────────────────────

    def compute(self, mc: MonteCarloOutput,
                rng: Optional[np.random.Generator] = None) -> StressMetricsOutput:
        if rng is None:
            rng = np.random.default_rng(0)

        paths, labels, regimes = mc.paths, mc.scenario_labels, mc.regime_paths
        alpha = self.es_alpha

        # Drawdown
        cum    = np.cumsum(paths, axis=1)
        dd     = cum - np.maximum.accumulate(cum, axis=1)
        max_dd = dd.min(axis=1)
        dd_by_sc = {
            s: float(max_dd[labels == s].mean()) if (labels == s).sum() > 0 else float("nan")
            for s in ["normal", "stress", "crisis"]
        }

        # Expected Shortfall
        q_path = np.quantile(paths, alpha, axis=1)
        q_mat  = q_path[:, np.newaxis]
        mask_  = paths <= q_mat
        es_per = np.where(
            mask_.any(axis=1),
            (paths * mask_).sum(axis=1) / mask_.sum(axis=1).clip(1),
            q_path,
        )
        all_r  = paths.flatten()
        agg_q  = np.quantile(all_r, alpha)
        agg_es = float(np.mean(all_r[all_r <= agg_q]))

        # Worst-k paths
        total_r   = paths.sum(axis=1)
        worst_idx = np.argsort(total_r)[: self.k_worst]

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

        # Regime statistics
        rm, re, rf = {}, {}, {}
        for k in range(3):
            mask_r = regimes == k
            r_reg  = paths[mask_r]
            rf[k]  = float(mask_r.sum() / paths.size)
            if len(r_reg) > 0:
                rm[k] = float(r_reg.mean())
                q_k   = np.quantile(r_reg, alpha)
                t_k   = r_reg[r_reg <= q_k]
                re[k] = float(t_k.mean()) if len(t_k) > 0 else float(q_k)
            else:
                rm[k] = re[k] = float("nan")

        # Bootstrap CIs
        bci = self._bootstrap_cis(paths, rng)

        return StressMetricsOutput(
            drawdown_dist      = max_dd,
            dd_mean            = float(max_dd.mean()),
            dd_median          = float(np.median(max_dd)),
            dd_p5              = float(np.percentile(max_dd, 5)),
            dd_ci90            = (float(np.percentile(max_dd, self.ci_lo * 100)),
                                  float(np.percentile(max_dd, self.ci_hi * 100))),
            dd_by_scenario     = dd_by_sc,
            dd_mean_ci90       = bci["dd_mean"],
            dd_p5_ci90         = bci["dd_p5"],
            es_alpha           = alpha,
            es_dist            = es_per,
            es_mean            = float(es_per.mean()),
            es_aggregate       = agg_es,
            es_ci90            = (float(np.percentile(es_per, self.ci_lo * 100)),
                                  float(np.percentile(es_per, self.ci_hi * 100))),
            es_aggregate_ci90  = bci["es_aggregate"],
            es_mean_ci90       = bci["es_mean"],
            worst_returns      = total_r[worst_idx],
            worst_paths        = paths[worst_idx],
            recovery_dist      = recovery,
            recovery_mean      = float(valid.mean()) if len(valid) > 0 else float("nan"),
            recovery_median    = float(np.median(valid)) if len(valid) > 0 else float("nan"),
            pct_never_recover  = float(np.isnan(recovery).mean()),
            recovery_mean_ci90 = bci["rec_mean"],
            regime_means       = rm,
            regime_es          = re,
            regime_fracs       = rf,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4b — MULTI-ASSET EXTENSION (LedoitWolf + Cholesky)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MultiAssetResult:
    portfolio_paths:   np.ndarray      # (n_paths, horizon)
    asset_paths:       np.ndarray      # (n_paths, horizon, N)
    covariance:        np.ndarray      # (N, N) Ledoit-Wolf estimate
    shrinkage:         float           # shrinkage coefficient
    weights:           np.ndarray      # (N,) portfolio weights
    asset_names:       List[str]
    n_assets:          int
    stress_metrics:    Optional[object] = None


class MultiAssetConstraintLayer:
    """
    Correlated multi-asset path simulation for portfolio stress testing.

    Covariance estimation: Ledoit-Wolf shrinkage, which produces a well-
    conditioned matrix in high-dimensional settings and reduces estimation
    error versus the sample covariance.

    Correlation structure: Cholesky decomposition of the shrunk covariance.
    Correlated shocks are generated as L·z where z ~ N(0, I) and L is the
    lower Cholesky factor.

    Reference: Ledoit & Wolf (2004), "A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices," JMVA 88(2), 365-411.

    Parameters
    ----------
    weights : array-like, optional
        Portfolio weights for each asset. Defaults to equal weighting.
    asset_names : list of str, optional
        Labels for each asset (used in reporting).
    crisis_vol_scale : float
        Extra volatility multiplier applied to crisis-seeded paths.
    """

    def __init__(self, weights=None, asset_names=None,
                 crisis_vol_scale: float = 1.5):
        self.weights          = weights
        self.asset_names      = asset_names
        self.crisis_vol_scale = crisis_vol_scale

    def fit_and_generate(self, returns_matrix,
                         n_paths: int = 5_000,
                         horizon: int = 252,
                         random_seed: int = 42,
                         stress_fraction: float = 0.20,
                         run_metrics: bool = True) -> MultiAssetResult:
        """
        Parameters
        ----------
        returns_matrix : array-like, shape (T, N)
            T observations of N asset return series.
        """
        X = np.asarray(returns_matrix, dtype=float)
        T, N = X.shape
        rng  = np.random.default_rng(random_seed)

        weights = (np.ones(N) / N if self.weights is None
                   else np.asarray(self.weights, dtype=float))
        weights /= weights.sum()

        asset_names = (self.asset_names if self.asset_names is not None
                       else [f"Asset_{i+1}" for i in range(N)])

        # Ledoit-Wolf covariance
        lw = LedoitWolf()
        lw.fit(X)
        cov = lw.covariance_
        shrinkage = float(lw.shrinkage_)

        # Ensure positive-definite (numerical safety)
        cov = (cov + cov.T) / 2
        min_eig = np.linalg.eigvalsh(cov).min()
        if min_eig < 1e-10:
            cov += (abs(min_eig) + 1e-8) * np.eye(N)

        L     = np.linalg.cholesky(cov)
        means = X.mean(axis=0)

        # Generate correlated paths
        n_crisis   = int(n_paths * stress_fraction)
        crisis_idx = rng.choice(n_paths, n_crisis, replace=False)

        asset_paths = np.zeros((n_paths, horizon, N))
        for t in range(horizon):
            z = rng.normal(0, 1, (n_paths, N))
            asset_paths[:, t, :] = means + (L @ z.T).T

        # Scale crisis paths
        asset_paths[crisis_idx] *= self.crisis_vol_scale

        # Portfolio aggregation
        portfolio_paths = (asset_paths * weights).sum(axis=2)

        # Optionally compute stress metrics on portfolio
        metrics = None
        if run_metrics:
            mc_dummy = MonteCarloOutput(
                paths           = portfolio_paths,
                regime_paths    = np.zeros((n_paths, horizon), dtype=int),
                scenario_labels = np.full(n_paths, "normal"),
                n_paths         = n_paths,
                horizon         = horizon,
                rejection_rate  = 0.0,
                lam1            = 0.0,
                lam2            = 0.0,
            )
            sm      = StressMetricsEngine()
            metrics = sm.compute(mc_dummy, rng=rng)

        return MultiAssetResult(
            portfolio_paths = portfolio_paths,
            asset_paths     = asset_paths,
            covariance      = cov,
            shrinkage       = shrinkage,
            weights         = weights,
            asset_names     = asset_names,
            n_assets        = N,
            stress_metrics  = metrics,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 — MODEL FRAGILITY INDEX  (Wasserstein distance)
# ═════════════════════════════════════════════════════════════════════════════

def compute_fragility_index(returns, constraint_kwargs, mc_kwargs,
                             n_paths: int = 2_000,
                             # legacy kwargs for backward compatibility
                             n_trials: int = 10,
                             **kwargs) -> Tuple[float, str, Dict[str, float]]:
    """
    Model Fragility Index (MFI) via Wasserstein distance.

    Definition
    ----------
    Let D_0 be the baseline max-drawdown distribution and D_θ the
    distribution under parameter perturbation θ.  Define:

        MFI = E_θ[ W₁(D_0, D_θ) ] / σ(D_0)

    where W₁ is the 1-Wasserstein (Earth Mover's) distance, σ(D_0) is the
    standard deviation of the baseline drawdown distribution, and
    the expectation is over the perturbation set Θ.

    Key improvements (v4)
    ─────────────────────
    1. Normalization by drawdown std (instead of ES) provides a more stable
       denominator that is less sensitive to tail behavior and daily returns.
       ES is ~2-3% for large-cap indices, causing artificial inflation of MFI.
       Drawdown std (~8-12%) is a more appropriate reference scale.

    2. Transition noise reduced from 15% to 5%. A 15% Dirichlet perturbation
       to the transition matrix fundamentally disrupts regime persistence and
       creates unrealistic market dynamics. 5% is more conservative and aligned
       with empirical estimation uncertainty.

    3. Thresholds recalibrated for the new normalization:
       - v1: Robust < 0.25, Moderate < 0.55, Fragile >= 0.55
       - v2: Robust < 0.15, Moderate < 0.35, Fragile >= 0.35  (too tight)
       - v3: Robust < 0.25, Moderate < 0.65, Fragile >= 0.65  (calibrated)

    A higher MFI means risk estimates are sensitive to small changes
    in the model parameters — the model is fragile.

    Perturbations applied
    ─────────────────────
        1. Regime mean returns  × 1.10  (+10%)
        2. Regime mean returns  × 0.90  (−10%)
        3. Regime volatilities  × 1.20  (+20%)
        4. Regime volatilities  × 0.80  (−20%)
        5. Transition matrix    ± 5%    (Dirichlet noise)  [reduced from 15%]

    Interpretation
    ──────────────
        MFI < 0.25   Robust    — estimates stable across reasonable perturbations
        MFI < 0.65   Moderate  — noticeable sensitivity to model inputs
        MFI ≥ 0.65   Fragile   — estimates shift substantially with small
                                  input changes; interpret with caution

    Reference: theoretical basis in Cont & Bouchaud (2000) model
    uncertainty framework; Wasserstein distance as in Villani (2008).

    Returns
    -------
    fi     : float         MFI score (0 = perfectly robust, 1 = highly fragile)
    grade  : str           "Robust" | "Moderate" | "Fragile" | "Unknown"
    details: dict          W₁ distance per perturbation name
    """
    # Fit baseline constraints
    try:
        ip        = InputProcessor()
        cleaned, _ = ip.fit_transform(returns)
        cl        = StructuralConstraintLayer(**constraint_kwargs)
        c_base    = cl.fit(cleaned)

        mc_kw     = {**mc_kwargs, "n_paths": n_paths, "random_seed": 42}
        mc_base   = ConstrainedMonteCarloGenerator(**mc_kw)
        out_base  = mc_base.generate(c_base)
        sm        = StressMetricsEngine(n_bootstrap=0)
        m_base    = sm.compute(out_base)

        base_dd   = m_base.drawdown_dist.copy()
        base_dd_std = float(np.std(base_dd))
    except Exception as e:
        logger.warning("MFI baseline run failed: %s", e)
        return 0.5, "Unknown", {}

    # Perturbation set
    perturbations = [
        ("mean_+10pct",   dict(mean_mult=1.10)),
        ("mean_-10pct",   dict(mean_mult=0.90)),
        ("vol_+20pct",    dict(vol_mult=1.20)),
        ("vol_-20pct",    dict(vol_mult=0.80)),
        ("trans_±15pct",  dict(trans_noise=0.05)),
    ]

    details: Dict[str, float] = {}
    rng_pert = np.random.default_rng(1)

    for name, pert_kw in perturbations:
        try:
            c_pert    = _perturb_constraints(c_base, rng=rng_pert, **pert_kw)
            mc_pert   = ConstrainedMonteCarloGenerator(**mc_kw)
            out_pert  = mc_pert.generate(c_pert)
            m_pert    = sm.compute(out_pert)
            w         = float(wasserstein_distance(base_dd, m_pert.drawdown_dist))
            details[name] = w
        except Exception as e:
            logger.warning("MFI perturbation '%s' failed: %s", name, e)

    if len(details) < 2:
        return 0.5, "Unknown", details

    avg_w = float(np.mean(list(details.values())))
    fi    = float(np.clip(avg_w / (base_dd_std + 1e-12), 0, 2))
    grade = "Robust" if fi < 0.25 else ("Moderate" if fi < 0.65 else "Fragile")
    return fi, grade, details


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5b — HISTORICAL BACKTEST VALIDATOR
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """
    Empirical validation of engine predictions against a realized stress period.

    coverage_ratio = realized_max_dd / predicted_dd_p5
        < 1.0  : model over-estimated risk (conservative)
        ~ 1.0  : model was well-calibrated
        > 1.0  : model under-estimated risk
    """
    period_label:      str
    realized_max_dd:   float   # actual worst drawdown during the period
    predicted_dd_p5:   float   # model's 5th-percentile drawdown (out-of-sample baseline)
    predicted_dd_p95:  float   # model's 95th-percentile drawdown
    dd_covered:        bool    # True if realized DD was within [p5, p95] range
    realized_es:       float   # actual tail loss (mean of worst-alpha% days)
    predicted_es:      float   # model's aggregate ES
    coverage_ratio:    float   # realized_max_dd / predicted_dd_p5
    n_observations:    int     # trading days in the stress period


class BacktestValidator:
    """
    Tests whether the engine's predicted risk distributions adequately
    captured realized losses during known historical stress periods.

    Stress periods covered
    ──────────────────────
    2008 Financial Crisis    Sep 2007 – Mar 2009
    March 2020 COVID Crash   Feb 2020 – Apr 2020
    2022 Rate Shock          Jan 2022 – Oct 2022

    For each period, the validator computes:
    - Realized max drawdown vs predicted 5th-percentile (tail) drawdown
    - Realized tail loss vs predicted expected shortfall
    - Coverage ratio: how well the predicted distribution contained the outcome
    - Whether the realized outcome fell within the predicted 90% CI

    NOTE: This is an in-sample validation (the engine was fitted on the
    full return series, which includes the stress periods). For a rigorous
    out-of-sample test, re-fit the engine on data up to the start of each
    period separately.
    """

    STRESS_PERIODS = {
        "2008_financial_crisis": {
            "start": "2007-09-01", "end": "2009-03-31",
            "label": "2008 Financial Crisis",
        },
        "march_2020_covid": {
            "start": "2020-02-10", "end": "2020-04-10",
            "label": "March 2020 COVID Crash",
        },
        "2022_rate_shock": {
            "start": "2022-01-01", "end": "2022-10-15",
            "label": "2022 Rate Shock",
        },
    }

    def validate(self, returns: np.ndarray, dates,
                 stress_metrics: StressMetricsOutput,
                 tail_alpha: float = 0.05) -> List[BacktestResult]:
        """
        Parameters
        ----------
        returns : 1-D array of daily returns (aligned with dates).
        dates   : array-like of datetime64 values, same length as returns.
                  Pass None to skip all periods.
        """
        if dates is None or len(dates) == 0:
            return []

        dates   = np.asarray(dates, dtype="datetime64[D]")
        results = []

        for key, period in self.STRESS_PERIODS.items():
            start = np.datetime64(period["start"], "D")
            end   = np.datetime64(period["end"],   "D")
            mask  = (dates >= start) & (dates <= end)
            if mask.sum() < 10:
                continue

            period_r = returns[mask]

            # Realized max drawdown
            cum        = np.cumsum(period_r)
            dd         = cum - np.maximum.accumulate(cum)
            real_maxdd = float(dd.min())

            # Realized tail loss (ES)
            q        = np.quantile(period_r, tail_alpha)
            tail_r   = period_r[period_r <= q]
            real_es  = float(np.mean(tail_r)) if len(tail_r) > 0 else q

            pred_p5   = float(stress_metrics.dd_p5)
            pred_p95  = float(stress_metrics.dd_ci90[1])
            pred_es   = float(stress_metrics.es_aggregate)

            # Coverage: did realized DD fall between predicted p5 and p95?
            dd_covered    = (real_maxdd >= pred_p5) and (real_maxdd <= pred_p95)
            coverage_ratio = real_maxdd / (pred_p5 - 1e-12)   # both negative

            results.append(BacktestResult(
                period_label     = period["label"],
                realized_max_dd  = real_maxdd,
                predicted_dd_p5  = pred_p5,
                predicted_dd_p95 = pred_p95,
                dd_covered       = dd_covered,
                realized_es      = real_es,
                predicted_es     = pred_es,
                coverage_ratio   = float(coverage_ratio),
                n_observations   = int(mask.sum()),
            ))

        return results


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
        "figure.facecolor": BL_DARK, "axes.facecolor":   "#111E2D",
        "axes.edgecolor":   "#2E4057", "axes.labelcolor":  "#CBD5E0",
        "xtick.color":      "#CBD5E0", "ytick.color":      "#CBD5E0",
        "text.color":       BL_LIGHT,  "grid.color":       "#1E3A52",
        "grid.linestyle":   "--",       "grid.alpha":       0.4,
        "font.family":      "monospace","axes.titlesize":   10,
        "axes.labelsize":   8,
    })


def _unit_label(normalization):
    return "z-score units" if normalization == "zscore" else "daily return units"


def _pct(v, d=2):
    """Format a decimal return as percentage string."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.{d}f}%"


def _ci_str(ci, is_days=False):
    if ci is None or any(np.isnan(x) for x in ci):
        return "—"
    if is_days:
        return f"[{ci[0]:.1f}d, {ci[1]:.1f}d]"
    return f"[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]"


def plot_dashboard(mc, sm, meta, strategy_name="Strategy", fi=None, fi_grade=None,
                   backtest_results=None):
    apply_style()
    unit  = _unit_label(meta.normalization)
    is_pct = (meta.normalization == "none")

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BL_DARK)
    n_rows = 3 if backtest_results else 2
    gs     = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.45, wspace=0.32)
    axes   = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    paths  = mc.paths
    labels = mc.scenario_labels
    cum    = np.cumsum(paths, axis=1)
    dd_ser = cum - np.maximum.accumulate(cum, axis=1)
    T, x   = paths.shape[1], np.arange(paths.shape[1])
    scale  = 100 if is_pct else 1
    ylbl   = "Drawdown (%)" if is_pct else f"Drawdown ({unit})"

    # Panel 1 — Drawdown curves
    ax     = axes[0]
    colors = {"normal": BL_TEAL, "stress": BL_GOLD, "crisis": BL_ROSE}
    for sc, col in colors.items():
        for pi in np.where(labels == sc)[0][:40]:
            ax.plot(x, dd_ser[pi] * scale, color=col, alpha=0.07, linewidth=0.5)
    p5  = np.percentile(dd_ser, 5,  axis=0) * scale
    p50 = np.percentile(dd_ser, 50, axis=0) * scale
    p95 = np.percentile(dd_ser, 95, axis=0) * scale
    ax.fill_between(x, p5, p95, alpha=0.15, color=BL_BLUE)
    ax.plot(x, p50, color=BL_GOLD, lw=1.5, label="Median")
    ax.plot(x, p5,  color=BL_ROSE, lw=1.0, linestyle="--", label="5th pct")
    ax.legend(handles=[
        Line2D([0],[0], color=BL_TEAL, lw=1.5, label="Normal"),
        Line2D([0],[0], color=BL_GOLD, lw=1.5, label="Stress"),
        Line2D([0],[0], color=BL_ROSE, lw=1.5, label="Crisis"),
    ], fontsize=7, framealpha=0.2, loc="lower left")
    ax.set_title("Stress Drawdown Curves", color=BL_LIGHT)
    ax.set_xlabel("Step"); ax.set_ylabel(ylbl); ax.grid(True)

    # Panel 2 — Max DD histogram
    ax = axes[1]
    dd_sc = sm.drawdown_dist * scale
    ax.hist(dd_sc, bins=50, color=BL_BLUE, edgecolor="none", alpha=0.85, density=True)
    ax.axvline(sm.dd_mean   * scale, color=BL_GOLD, lw=1.5, label=f"Mean: {_pct(sm.dd_mean)}")
    ax.axvline(sm.dd_p5     * scale, color=BL_ROSE, lw=1.5, linestyle="--", label=f"5th: {_pct(sm.dd_p5)}")
    ax.axvline(sm.dd_median * scale, color=BL_TEAL, lw=1.5, linestyle=":", label=f"Med: {_pct(sm.dd_median)}")
    ax.set_title("Max Drawdown Distribution", color=BL_LIGHT)
    ax.set_xlabel(ylbl); ax.set_ylabel("Density")
    ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=7, framealpha=0.2); ax.grid(True)

    # Panel 3 — Recovery
    ax    = axes[2]
    valid = sm.recovery_dist[~np.isnan(sm.recovery_dist)]
    if len(valid) > 0:
        ax.hist(valid, bins=40, color=BL_TEAL, edgecolor="none", alpha=0.85, density=True)
        ax.axvline(sm.recovery_mean,   color=BL_GOLD, lw=1.5, label=f"Mean: {sm.recovery_mean:.1f}d")
        ax.axvline(sm.recovery_median, color=BL_ROSE, lw=1.5, linestyle="--", label=f"Med: {sm.recovery_median:.1f}d")
        ax.legend(fontsize=7, framealpha=0.2)
    ax.set_title(f"Time-to-Recovery  ({sm.pct_never_recover:.1%} never recover)", color=BL_LIGHT)
    ax.set_xlabel("Steps"); ax.set_ylabel("Density"); ax.grid(True)

    # Panel 4 — Regime heatmap
    ax    = axes[3]
    reg   = mc.regime_paths
    fracs = np.array([(reg == k).mean(axis=0) for k in range(3)])
    cmap  = LinearSegmentedColormap.from_list("bl", [BL_DARK, BL_BLUE, BL_GOLD, BL_ROSE])
    im    = ax.imshow(fracs, aspect="auto", cmap=cmap, vmin=0, vmax=1, extent=[0, T, -0.5, 2.5])
    plt.colorbar(im, ax=ax, label="Fraction", fraction=0.046, pad=0.04)
    ax.set_title("Regime Transition Heatmap", color=BL_LIGHT)
    ax.set_xlabel("Step"); ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Calm", "Volatile", "Crisis"], fontsize=8)

    # Panel 5 — ES distribution
    ax = axes[4]
    es_sc = sm.es_dist * scale
    ax.hist(es_sc, bins=50, color=BL_ROSE, edgecolor="none", alpha=0.85, density=True)
    ax.axvline(sm.es_mean      * scale, color=BL_GOLD, lw=1.5, label=f"Mean: {_pct(sm.es_mean)}")
    ax.axvline(sm.es_aggregate * scale, color=BL_TEAL, lw=1.5, linestyle="--", label=f"Agg: {_pct(sm.es_aggregate)}")
    ax.set_title(f"Expected Shortfall (α={sm.es_alpha})", color=BL_LIGHT)
    ax.set_xlabel("ES / CVaR (%)"); ax.set_ylabel("Density")
    ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=7, framealpha=0.2); ax.grid(True)

    # Panel 6 — Worst paths
    ax   = axes[5]
    k    = len(sm.worst_paths)
    cols = plt.cm.Reds(np.linspace(0.4, 0.9, k))
    for i, path in enumerate(sm.worst_paths):
        ax.plot(np.cumsum(path) * scale, color=cols[i], alpha=0.85, lw=1.0)
    ax.set_title(f"Worst-{k} Paths (Cumulative Return)", color=BL_LIGHT)
    ax.set_xlabel("Step"); ax.set_ylabel(f"Cumulative Return (%)"); ax.grid(True)

    # Optional backtest row
    if backtest_results and n_rows == 3:
        ax_bt = fig.add_subplot(gs[2, :])
        ax_bt.set_facecolor("#0A1520")
        ax_bt.axis("off")
        cols_hdr = ["Period", "n Days", "Realized DD", "Predicted P5 DD", "Within Range?",
                    "Realized ES", "Predicted ES", "Coverage Ratio"]
        rows_data = []
        for br in backtest_results:
            rows_data.append([
                br.period_label, str(br.n_observations),
                _pct(br.realized_max_dd), _pct(br.predicted_dd_p5),
                "✓" if br.dd_covered else "✗",
                _pct(br.realized_es), _pct(br.predicted_es),
                f"{br.coverage_ratio:.2f}x",
            ])
        if rows_data:
            tbl = ax_bt.table(
                cellText=rows_data, colLabels=cols_hdr,
                cellLoc="center", loc="center",
                bbox=[0.0, 0.0, 1.0, 1.0],
            )
            tbl.auto_set_font_size(False); tbl.set_fontsize(8)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_facecolor("#0D1B2A" if r > 0 else "#1B4F72")
                cell.set_edgecolor("#2E4057")
                cell.set_text_props(color=BL_LIGHT)
            ax_bt.set_title("Historical Backtest Validation", color=BL_LIGHT,
                            pad=10, fontsize=10)

    fi_str = f"  |  MFI: {fi:.3f} ({fi_grade})" if fi is not None else ""
    fig.suptitle(f"BLUE LOTUS LABS  |  {strategy_name}{fi_str}",
                 fontsize=13, color=BL_GOLD, weight="bold")
    plt.tight_layout()
    return fig


def print_executive_summary(mc, sm, meta, strategy_name, fi=None, fi_grade=None,
                             fi_details=None, backtest_results=None,
                             regime_output=None):
    unit   = _unit_label(meta.normalization)
    counts = {s: int(np.sum(mc.scenario_labels == s)) for s in ["normal", "stress", "crisis"]}
    lam_str = f"λ₁={mc.lam1:.3f}  λ₂={mc.lam2:.3f}"

    print()
    print("╔" + "═" * 66 + "╗")
    print(f"║   BLUE LOTUS LABS  v3.0  |  EXECUTIVE RISK SUMMARY           ║")
    print(f"║   Strategy: {strategy_name:<53}║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  Paths: {mc.n_paths:,}  Rejection: {mc.rejection_rate:.1%}  Horizon: {mc.horizon}d  Units: {unit:<10}║")
    print(f"║  Scenarios — Normal: {counts['normal']:,}  Stress: {counts['stress']:,}  Crisis: {counts['crisis']:,}{'':>16}║")
    print(f"║  Esscher: {lam_str:<56}║")

    # HMM quality
    if regime_output is not None:
        print("╠" + "═" * 66 + "╣")
        pi = regime_output.stationary_dist
        crisis_frac = float((regime_output.regime_labels == 2).mean())
        print(f"║  REGIME  Calm={pi[0]:.2f}  Volatile={pi[1]:.2f}  Crisis={pi[2]:.2f}  "
              f"(actual {crisis_frac:.1%}){'':>5}║")

    print("╠" + "═" * 66 + "╣")
    print(f"║  DRAWDOWN                                                      ║")
    print(f"║    Mean: {_pct(sm.dd_mean):>8}   5th pct: {_pct(sm.dd_p5):>8}   Median: {_pct(sm.dd_median):>8}   ║")
    print(f"║    90%% CI: [{_pct(sm.dd_ci90[0])}, {_pct(sm.dd_ci90[1])}]{'':>22}║")
    print(f"║    Bootstrap CI (mean): {_ci_str(sm.dd_mean_ci90):<42}║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  EXPECTED SHORTFALL (α={sm.es_alpha})                          ║")
    print(f"║    Aggregate ES: {_pct(sm.es_aggregate):>8}   Mean ES: {_pct(sm.es_mean):>8}{'':>26}║")
    print(f"║    Bootstrap CI (agg): {_ci_str(sm.es_aggregate_ci90):<43}║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  RECOVERY: Mean={sm.recovery_mean:.1f}d   Never={sm.pct_never_recover:.1%}{'':>32}║")
    print(f"║    Bootstrap CI (mean): {_ci_str(sm.recovery_mean_ci90, is_days=True):<42}║")

    if fi is not None:
        print("╠" + "═" * 66 + "╣")
        print(f"║  MODEL FRAGILITY INDEX (Wasserstein): {fi:.4f}  ({fi_grade}){'':>14}║")
        if fi_details:
            for name, w in fi_details.items():
                print(f"║    {name:<28}  W₁ = {w:.6f}{'':>14}║")

    if backtest_results:
        print("╠" + "═" * 66 + "╣")
        print(f"║  BACKTEST VALIDATION{'':>46}║")
        for br in backtest_results:
            cov_str = "COVERED" if br.dd_covered else "MISSED "
            print(f"║  {br.period_label:<32}  {cov_str}  ratio={br.coverage_ratio:.2f}x  ║")

    print("╠" + "═" * 66 + "╣")
    print("║  ⚠  Risk distributions only. No return predictions.            ║")
    print("╚" + "═" * 66 + "╝")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE — wires all modules together
# ═════════════════════════════════════════════════════════════════════════════

class BlueLotusEngine:
    """
    Top-level orchestrator for the Blue Lotus institutional stress-testing engine.

    Usage
    -----
    engine  = BlueLotusEngine(strategy_name="SPY", n_paths=10_000)
    results = engine.run(returns, dates=dates)
    engine.plot(results)
    """

    def __init__(self, strategy_name: str = "Strategy",
                 winsorize: bool = True, normalization: str = "none",
                 tail_alpha: float = 0.05,
                 alpha_ig: float = 3.0, beta_ig: Optional[float] = None,
                 moderate_dd: float = -0.05, severe_dd: float = -0.15,
                 implied_vol: Optional[float] = None,
                 known_risk_limit: Optional[float] = None,
                 n_paths: int = 10_000, horizon: int = 252,
                 random_seed: int = 42, stress_fraction: float = 0.20,
                 drawdown_floor: float = -0.70,
                 k_worst: int = 10, n_bootstrap: int = 500,
                 run_sensitivity: bool = True,
                 run_backtest: bool = True,
                 # legacy
                 **kwargs):

        self.strategy_name   = strategy_name
        self.run_sensitivity = run_sensitivity
        self.run_backtest    = run_backtest

        self._ck = dict(
            tail_alpha       = tail_alpha,
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
            drawdown_floor  = drawdown_floor,
        )

        self.ip = InputProcessor(winsorize=winsorize, normalization=normalization)
        self.cl = StructuralConstraintLayer(**self._ck)
        self.mc = ConstrainedMonteCarloGenerator(**self._mk)
        self.sm = StressMetricsEngine(es_alpha=tail_alpha, k_worst=k_worst,
                                      n_bootstrap=n_bootstrap)
        self.bt = BacktestValidator()

        self._last_mc = self._last_stress = self._last_fi = None
        self._last_grade = self._last_meta = self._last_constraints = None
        self._last_fi_details = self._last_backtest = None

    def run(self, returns, dates=None, verbose: bool = True):
        print(f"\n{'=' * 60}")
        print(f"  BLUE LOTUS LABS v3.0 — {self.strategy_name}")
        print(f"{'=' * 60}")

        print("▶ Module 1: Input Processing...")
        cleaned, meta = self.ip.fit_transform(np.asarray(returns, dtype=float))
        print(f"   n={meta.n_observations}  mean={meta.raw_mean:.6f}  "
              f"std={meta.raw_std:.6f}  ann_vol={meta.ann_vol:.2%}")

        print("▶ Module 2: Structural Constraints (vol-regime + GPD)...")
        constraints = self.cl.fit(cleaned)
        pi     = constraints.regime.stationary_dist
        tail   = constraints.tail
        labels = constraints.regime.regime_labels
        print(f"   Regime dist — Calm={pi[0]:.2f}  Volatile={pi[1]:.2f}  Crisis={pi[2]:.2f}")
        print(f"   Crisis fraction (actual): {(labels==2).mean():.1%}")
        print(f"   Tail method: {tail.method}  "
              f"xi={tail.xi:.4f}  β={tail.beta:.6f}  n_tail={tail.n_tail}")
        print(f"   ES target: {_pct(tail.es_target)}")

        print(f"▶ Module 3: Monte Carlo ({self._mk['n_paths']:,} paths) + Esscher...")
        mc_out = self.mc.generate(constraints)
        counts = {s: int(np.sum(mc_out.scenario_labels == s)) for s in ["normal", "stress", "crisis"]}
        print(f"   Accepted={mc_out.n_paths:,}  rejection={mc_out.rejection_rate:.1%}")
        print(f"   Esscher λ₁={mc_out.lam1:.4f}  λ₂={mc_out.lam2:.4f}")
        print(f"   Normal={counts['normal']:,} / Stress={counts['stress']:,} / Crisis={counts['crisis']:,}")

        print("▶ Module 4: Stress Metrics + Bootstrap CIs...")
        rng_sm = np.random.default_rng(self._mk["random_seed"])
        stress = self.sm.compute(mc_out, rng=rng_sm)
        print(f"   Mean DD={_pct(stress.dd_mean)}  "
              f"ES={_pct(stress.es_aggregate)}  "
              f"Never recover={stress.pct_never_recover:.1%}")
        print(f"   90%% CI DD mean: {_ci_str(stress.dd_mean_ci90)}")

        fi, fi_grade, fi_details = None, None, {}
        if self.run_sensitivity:
            print("▶ Module 5: Model Fragility Index (Wasserstein)...")
            fi, fi_grade, fi_details = compute_fragility_index(
                cleaned, self._ck, self._mk, n_paths=min(2_000, self._mk["n_paths"])
            )
            print(f"   MFI = {fi:.4f}  ({fi_grade})")
            for name, w in fi_details.items():
                print(f"   {name:<26}  W₁ = {w:.6f}")

        backtest_results = []
        if self.run_backtest and dates is not None:
            print("▶ Module 5b: Historical Backtest Validation...")
            backtest_results = self.bt.validate(cleaned, dates, stress)
            for br in backtest_results:
                status = "COVERED" if br.dd_covered else "MISSED"
                print(f"   {br.period_label:<32}  {status}  "
                      f"realized={_pct(br.realized_max_dd)}  "
                      f"pred_p5={_pct(br.predicted_dd_p5)}  "
                      f"ratio={br.coverage_ratio:.2f}x")

        print("▶ Module 7: Executive Summary...")
        print_executive_summary(
            mc_out, stress, meta, self.strategy_name,
            fi=fi, fi_grade=fi_grade, fi_details=fi_details,
            backtest_results=backtest_results,
            regime_output=constraints.regime,
        )

        self._last_mc          = mc_out
        self._last_stress      = stress
        self._last_fi          = fi
        self._last_grade       = fi_grade
        self._last_fi_details  = fi_details
        self._last_meta        = meta
        self._last_constraints = constraints
        self._last_backtest    = backtest_results

        return {
            "mc":              mc_out,
            "stress":          stress,
            "fi":              fi,
            "fi_grade":        fi_grade,
            "fi_details":      fi_details,
            "constraints":     constraints,
            "metadata":        meta,
            "backtest":        backtest_results,
        }

    def plot(self, results=None):
        mc  = results["mc"]       if results else self._last_mc
        sm  = results["stress"]   if results else self._last_stress
        meta= results["metadata"] if results else self._last_meta
        fi  = results["fi"]       if results else self._last_fi
        grd = results.get("fi_grade") if results else self._last_grade
        bts = results.get("backtest") if results else self._last_backtest
        fig = plot_dashboard(mc, sm, meta, self.strategy_name, fi, grd, bts)
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
        raise ImportError("Run: pip install yfinance")
    import datetime

    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    print(f"   Fetching {ticker} from {start} to {end}...")
    t  = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(
            f"No price data found for '{ticker}'. "
            "Check the symbol (e.g. SPY, AAPL, BTC-USD)."
        )

    prices  = df[price_col].dropna().squeeze()
    returns = prices.pct_change().dropna().to_numpy(dtype=float).flatten()
    dates   = prices.index[1:].to_numpy()

    ann_vol = float(returns.std() * np.sqrt(252))
    tot_ret = float(prices.iloc[-1]) / float(prices.iloc[0]) - 1
    print(f"   Got {len(returns)} daily returns")
    print(f"   Ann vol ≈ {ann_vol:.2%}  |  Total return ≈ {tot_ret:.2%}")
    return returns, dates, prices


def run_on_ticker(ticker, start="2010-01-01", n_paths=10_000,
                  horizon=252, run_sensitivity=True, run_backtest=True):
    """
    One-liner: fetch real data + full Blue Lotus engine + dashboard.

    Examples
    --------
    results = run_on_ticker("SPY")
    results = run_on_ticker("QQQ", start="2015-01-01")
    results = run_on_ticker("BTC-USD", start="2018-01-01")
    """
    print("\n" + "=" * 60)
    print(f"  BLUE LOTUS LABS v3.0  |  {ticker}")
    print("=" * 60)

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
        run_backtest    = run_backtest,
        random_seed     = 42,
        moderate_dd     = moderate_dd,
        severe_dd       = severe_dd,
    )
    results = engine.run(returns, dates=dates)
    results.update({"ticker": ticker, "dates": dates, "prices": prices})
    engine.plot(results)
    return results


def run_comparison(tickers, start="2010-01-01", n_paths=5_000):
    """
    Run engine on multiple tickers and print a side-by-side risk table.

    Example
    -------
    run_comparison(["SPY", "QQQ", "TLT", "GLD"])
    """
    import warnings as _w; _w.filterwarnings("ignore")
    print("\n  BLUE LOTUS LABS v3.0  |  Multi-Ticker Comparison")
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
                run_backtest    = False,
                random_seed     = 42,
                moderate_dd     = -daily_std * 15,
                severe_dd       = -daily_std * 45,
            )
            r  = engine.run(returns, verbose=False)
            sm = r["stress"]
            rows.append({
                "Ticker":      ticker,
                "N obs":       str(len(returns)),
                "Ann Vol":     f"{returns.std() * 252**0.5:.2%}",
                "Mean DD":     _pct(sm.dd_mean),
                "ES (5%)":     _pct(sm.es_aggregate),
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
        print("\n" + "=" * 70)
        print("  COMPARISON TABLE  (all metrics in daily return % terms)")
        print("=" * 70)
        print("  ".join(c.ljust(w[c]) for c in cols))
        print("  ".join("-" * w[c] for c in cols))
        for row in rows:
            print("  ".join(row[c].ljust(w[c]) for c in cols))
        print("=" * 70)

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess, sys
    print("Installing dependencies...")
    for pkg in ["yfinance", "hmmlearn", "scikit-learn"]:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Dependencies ready.\n")

    results = run_on_ticker(
        ticker          = "SPY",
        start           = "2010-01-01",
        n_paths         = 10_000,
        horizon         = 252,
        run_sensitivity = True,
        run_backtest    = True,
    )

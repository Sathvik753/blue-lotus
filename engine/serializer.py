"""
Engine Serializer — Blue Lotus Labs v3.0
Converts numpy/dataclass outputs to JSON-safe dicts.
Includes new v3.0 fields: HMM quality, GPD parameters, bootstrap CIs, MFI details.
"""

import numpy as np
from typing import Any, Optional, Dict


def to_json(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _safe(v, decimals=6):
    """Round a float, return None for NaN/Inf."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, decimals)
    except Exception:
        return None


def _ci(t, decimals=6):
    """Serialize a (lo, hi) CI tuple."""
    if t is None:
        return {"lo": None, "hi": None}
    return {"lo": _safe(t[0], decimals), "hi": _safe(t[1], decimals)}


def serialize_run_results(mc, sm, constraints, metadata, fi, fi_grade,
                           fi_details: Optional[Dict] = None,
                           ticker=None,
                           backtest_results=None) -> dict:
    """
    Convert a full v3.0 engine run into a clean JSON-safe dict.

    New fields vs v2.0:
    - metadata.ann_vol
    - regime.log_likelihood / aic / bic
    - tail.method / xi / beta / threshold / n_tail
    - drawdown.*_ci90  (bootstrap confidence intervals)
    - expected_shortfall.*_ci90
    - recovery.mean_ci90
    - fragility.details  (Wasserstein per perturbation)
    """
    dd    = sm.drawdown_dist
    es    = sm.es_dist
    rec   = sm.recovery_dist
    valid_rec = rec[~np.isnan(rec)]

    labels = mc.scenario_labels
    scenario_counts = {
        "normal": int((labels == "normal").sum()),
        "stress": int((labels == "stress").sum()),
        "crisis": int((labels == "crisis").sum()),
    }

    pi   = constraints.regime.stationary_dist
    tail = constraints.tail

    return {
        "ticker":   ticker,

        # ── Data & simulation ─────────────────────────────────────────────
        "metadata": {
            "n_observations": int(metadata.n_observations),
            "raw_mean":       _safe(metadata.raw_mean),
            "raw_std":        _safe(metadata.raw_std),
            "raw_skewness":   _safe(metadata.raw_skewness, 4),
            "raw_kurtosis":   _safe(metadata.raw_kurtosis, 4),
            "ann_vol":        _safe(metadata.ann_vol, 4),
            "normalization":  metadata.normalization,
        },

        "simulation": {
            "n_paths":        int(mc.n_paths),
            "horizon":        int(mc.horizon),
            "rejection_rate": _safe(mc.rejection_rate, 4),
            "scenario_counts": scenario_counts,
            "esscher_lam1":   _safe(mc.lam1, 6),
            "esscher_lam2":   _safe(mc.lam2, 6),
        },

        # ── HMM regime ───────────────────────────────────────────────────
        "regime": {
            "stationary_dist": {
                "calm":     _safe(pi[0], 4),
                "volatile": _safe(pi[1], 4),
                "crisis":   _safe(pi[2], 4),
            },
            "transition_matrix": to_json(constraints.regime.transition_matrix.round(4)),
            "regime_means":      to_json(constraints.regime.regime_means.round(6)),
            "regime_stds":       to_json(constraints.regime.regime_stds.round(6)),
            "fit_quality": {
                "log_likelihood": _safe(constraints.regime.log_likelihood, 2),
                "aic":            _safe(constraints.regime.aic, 2),
                "bic":            _safe(constraints.regime.bic, 2),
            },
        },

        # ── GPD tail ─────────────────────────────────────────────────────
        "tail_constraints": {
            "alpha":                float(tail.alpha),
            "method":               tail.method,
            "es_target":            _safe(tail.es_target),
            "lower_quantile_bound": _safe(tail.lower_quantile_bound),
            "upper_quantile_bound": _safe(tail.upper_quantile_bound),
            "gpd": {
                "xi":        _safe(tail.xi, 6),
                "beta":      _safe(tail.beta, 8),
                "threshold": _safe(tail.threshold, 8),
                "n_tail":    int(tail.n_tail),
            },
        },

        # ── Drawdown ─────────────────────────────────────────────────────
        "drawdown": {
            "mean":       _safe(sm.dd_mean),
            "median":     _safe(sm.dd_median),
            "p5":         _safe(sm.dd_p5),
            "p95":        _safe(float(np.percentile(dd, 95))),
            "ci_90_low":  _safe(sm.dd_ci90[0], 6),
            "ci_90_high": _safe(sm.dd_ci90[1], 6),
            "by_scenario": {
                k: _safe(v) for k, v in sm.dd_by_scenario.items()
            },
            "bootstrap_ci": {
                "mean": _ci(sm.dd_mean_ci90),
                "p5":   _ci(sm.dd_p5_ci90),
            },
            "histogram": _histogram(dd, bins=50),
        },

        # ── Expected Shortfall ───────────────────────────────────────────
        "expected_shortfall": {
            "alpha":      float(sm.es_alpha),
            "aggregate":  _safe(sm.es_aggregate),
            "mean":       _safe(sm.es_mean),
            "ci_90_low":  _safe(sm.es_ci90[0]),
            "ci_90_high": _safe(sm.es_ci90[1]),
            "bootstrap_ci": {
                "aggregate": _ci(sm.es_aggregate_ci90),
                "mean":      _ci(sm.es_mean_ci90),
            },
            "histogram": _histogram(es, bins=50),
        },

        # ── Recovery ─────────────────────────────────────────────────────
        "recovery": {
            "mean":          _safe(sm.recovery_mean, 2),
            "median":        _safe(sm.recovery_median, 2),
            "pct_never":     _safe(sm.pct_never_recover, 4),
            "bootstrap_ci":  _ci(sm.recovery_mean_ci90, 2),
            "histogram":     _histogram(valid_rec, bins=40) if len(valid_rec) > 0 else [],
        },

        # ── Regime losses ─────────────────────────────────────────────────
        "regime_losses": {
            str(k): {
                "mean":     _safe(sm.regime_means.get(k)),
                "es":       _safe(sm.regime_es.get(k)),
                "fraction": _safe(sm.regime_fracs.get(k), 4),
            }
            for k in range(3)
        },

        # ── Worst paths ──────────────────────────────────────────────────
        "worst_scenarios": {
            "k":             int(len(sm.worst_returns)),
            "total_returns": to_json(np.round(sm.worst_returns, 6)),
            "paths":         to_json(np.round(sm.worst_paths[:5], 6)),
        },

        # ── Model Fragility Index (Wasserstein) ───────────────────────────
        "fragility": {
            "index":   _safe(fi, 6) if fi is not None else None,
            "grade":   fi_grade,
            "details": {k: _safe(v, 6) for k, v in (fi_details or {}).items()},
        },

        # ── Historical backtest validation ────────────────────────────────
        "backtest": [
            {
                "period_label":      br.period_label,
                "realized_max_dd":   _safe(br.realized_max_dd),
                "predicted_dd_p5":   _safe(br.predicted_dd_p5),
                "predicted_dd_p95":  _safe(br.predicted_dd_p95),
                "dd_covered":        bool(br.dd_covered),
                "realized_es":       _safe(br.realized_es),
                "predicted_es":      _safe(br.predicted_es),
                "coverage_ratio":    _safe(br.coverage_ratio, 3),
                "n_observations":    int(br.n_observations),
            }
            for br in (backtest_results or [])
        ],
    }


def _histogram(arr: np.ndarray, bins: int = 50) -> list:
    """Return histogram as list of {x, y} points for frontend charting."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return []
    counts, edges = np.histogram(arr, bins=bins, density=True)
    midpoints = (edges[:-1] + edges[1:]) / 2
    return [{"x": round(float(x), 6), "y": round(float(y), 6)}
            for x, y in zip(midpoints, counts)]

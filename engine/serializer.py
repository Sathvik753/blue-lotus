"""
Engine Serializer
Blue Lotus Labs — converts numpy/dataclass outputs to JSON-safe dicts
"""

import numpy as np
from typing import Any


def to_json(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def serialize_run_results(mc, sm, constraints, metadata, fi, fi_grade, ticker=None) -> dict:
    """
    Convert a full engine run into a clean JSON-safe dict.
    All drawdown/ES values converted to proper percentage strings where appropriate.
    """
    dd   = sm.drawdown_dist
    es   = sm.es_dist
    rec  = sm.recovery_dist
    valid_rec = rec[~np.isnan(rec)]

    # Scenario counts
    labels = mc.scenario_labels
    scenario_counts = {
        "normal": int((labels == "normal").sum()),
        "stress": int((labels == "stress").sum()),
        "crisis": int((labels == "crisis").sum()),
    }

    # Regime info
    pi = constraints.regime.stationary_dist

    return {
        "ticker": ticker,
        "metadata": {
            "n_observations":  int(metadata.n_observations),
            "raw_mean":        round(float(metadata.raw_mean), 6),
            "raw_std":         round(float(metadata.raw_std), 6),
            "raw_skewness":    round(float(metadata.raw_skewness), 4),
            "raw_kurtosis":    round(float(metadata.raw_kurtosis), 4),
            "normalization":   metadata.normalization,
        },
        "simulation": {
            "n_paths":         int(mc.n_paths),
            "horizon":         int(mc.horizon),
            "rejection_rate":  round(float(mc.rejection_rate), 4),
            "scenario_counts": scenario_counts,
        },
        "regime": {
            "stationary_dist": {
                "calm":     round(float(pi[0]), 4),
                "volatile": round(float(pi[1]), 4),
                "crisis":   round(float(pi[2]), 4),
            },
            "transition_matrix": to_json(constraints.regime.transition_matrix.round(4)),
            "regime_means": to_json(constraints.regime.regime_means.round(6)),
            "regime_stds":  to_json(constraints.regime.regime_stds.round(6)),
        },
        "drawdown": {
            "mean":            round(float(sm.dd_mean), 6),
            "median":          round(float(sm.dd_median), 6),
            "p5":              round(float(sm.dd_p5), 6),
            "p95":             round(float(np.percentile(dd, 95)), 6),
            "ci_90_low":       round(float(sm.dd_ci90[0]), 6),
            "ci_90_high":      round(float(sm.dd_ci90[1]), 6),
            "by_scenario":     {k: round(float(v), 6) if v is not None and not np.isnan(v) else None
                                for k, v in sm.dd_by_scenario.items()},
            # Histogram for charting
            "histogram":       _histogram(dd, bins=50),
        },
        "expected_shortfall": {
            "alpha":           float(sm.es_alpha),
            "aggregate":       round(float(sm.es_aggregate), 6),
            "mean":            round(float(sm.es_mean), 6),
            "ci_90_low":       round(float(sm.es_ci90[0]), 6),
            "ci_90_high":      round(float(sm.es_ci90[1]), 6),
            "histogram":       _histogram(es, bins=50),
        },
        "recovery": {
            "mean":            round(float(sm.recovery_mean), 2) if not np.isnan(sm.recovery_mean) else None,
            "median":          round(float(sm.recovery_median), 2) if not np.isnan(sm.recovery_median) else None,
            "pct_never":       round(float(sm.pct_never_recover), 4),
            "histogram":       _histogram(valid_rec, bins=40) if len(valid_rec) > 0 else [],
        },
        "regime_losses": {
            str(k): {
                "mean":     round(float(sm.regime_means[k]), 6) if not np.isnan(sm.regime_means[k]) else None,
                "es":       round(float(sm.regime_es[k]), 6)    if not np.isnan(sm.regime_es[k]) else None,
                "fraction": round(float(sm.regime_fracs[k]), 4),
            }
            for k in range(3)
        },
        "worst_scenarios": {
            "k":            int(len(sm.worst_returns)),
            "total_returns": to_json(np.round(sm.worst_returns, 6)),
            # Send only 10 worst paths for charting (sampled, not all)
            "paths":        to_json(np.round(sm.worst_paths[:5], 6)),
        },
        "fragility": {
            "index": round(float(fi), 6) if fi is not None else None,
            "grade": fi_grade,
        },
        "tail_constraints": {
            "alpha":               float(constraints.tail.alpha),
            "es_target":           round(float(constraints.tail.es_target), 6),
            "lower_quantile_bound": round(float(constraints.tail.lower_quantile_bound), 6),
            "upper_quantile_bound": round(float(constraints.tail.upper_quantile_bound), 6),
        },
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

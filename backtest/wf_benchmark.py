"""Benchmark the Blue Lotus engine against naive baselines on the same panel.

For every walk-forward cell (identical universe, years, train/test split as
walk_forward.py) we fit five models on the pre-year training window and score
each against the realized one-year max drawdown:

  engine   - Blue Lotus v3.1
  gauss    - i.i.d. Gaussian (textbook Monte Carlo)
  studt    - i.i.d. Student-t (fat-tailed parametric)
  histiid  - historical i.i.d. bootstrap of daily returns
  histblk  - circular block bootstrap, block length 20 (preserves vol clustering)

Scoring uses the Continuous Ranked Probability Score (CRPS, a strictly proper
scoring rule for a probabilistic forecast of a scalar) plus the same p5/p1 breach
and PIT statistics as the main study. Reuses the on-disk price cache.
"""

import os, sys, json, time, warnings
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)

import walk_forward as wf  # reuse universe, cache loader, engine simulator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wf_results")
N = wf.N_PATHS
SEED = wf.SEED
MODELS = ["engine", "gauss", "studt", "histiid", "histblk"]


def maxdd(daily_paths):
    """Compounded max drawdown for each simulated path (N, H) -> (N,)."""
    w = np.cumprod(1.0 + daily_paths, axis=1)
    return np.min(w / np.maximum.accumulate(w, axis=1) - 1.0, axis=1)


def sim_gauss(train, H, rng):
    mu, sd = float(train.mean()), float(train.std(ddof=1))
    return maxdd(rng.normal(mu, sd, size=(N, H)))


def sim_studt(train, H, rng):
    try:
        df, loc, scale = stats.t.fit(train)
        if not np.isfinite(df) or df < 2.0:
            df = 2.0
        X = stats.t.rvs(df, loc=loc, scale=scale, size=(N, H), random_state=rng)
    except Exception:
        mu, sd = float(train.mean()), float(train.std(ddof=1))
        X = rng.normal(mu, sd, size=(N, H))
    return maxdd(X)


def sim_histiid(train, H, rng):
    idx = rng.integers(0, len(train), size=(N, H))
    return maxdd(train[idx])


def sim_histblk(train, H, rng, L=20):
    nblk = int(np.ceil(H / L))
    starts = rng.integers(0, len(train), size=(N, nblk))
    offs = np.arange(L)
    idx = (starts[:, :, None] + offs[None, None, :]) % len(train)
    idx = idx.reshape(N, -1)[:, :H]
    return maxdd(train[idx])


def crps(ens, y):
    """CRPS of an ensemble forecast against scalar y (lower is better).
    Energy form via the sorted estimator, O(n log n)."""
    n = len(ens)
    xs = np.sort(ens)
    term1 = float(np.mean(np.abs(ens - y)))
    i = np.arange(1, n + 1)
    term2 = (2.0 / n**2) * float(np.sum((2 * i - n - 1) * xs))
    return term1 - 0.5 * term2


def scores(ens, real):
    p5 = float(np.percentile(ens, 5)); p1 = float(np.percentile(ens, 1))
    return {
        "crps": round(crps(ens, real), 6),
        "pit": round(float(np.mean(ens <= real)), 4),
        "breach_p5": bool(real < p5),
        "breach_p1": bool(real < p1),
    }


def run():
    cells = []
    tickers = list(wf.UNIVERSE.keys())
    t0 = time.time()

    for ti, ticker in enumerate(tickers, 1):
        data = wf.load_prices(ticker)
        if data is None:
            continue
        rets, dates = data
        years = dates.astype("datetime64[Y]").astype(int) + 1970

        for Y in wf.TEST_YEARS:
            train_mask = dates < np.datetime64(f"{Y}-01-01")
            test_mask = years == Y
            if train_mask.sum() < wf.MIN_TRAIN or test_mask.sum() < wf.MIN_TEST:
                continue
            train = rets[train_mask]
            test = rets[test_mask]
            H = int(test_mask.sum())
            real = wf.realized_dd(test)

            # deterministic per-cell rng for the baselines
            rng = np.random.default_rng(SEED + Y * 1000 + ti)

            rec = {"ticker": ticker, "asset_class": wf.UNIVERSE[ticker], "year": Y,
                   "crisis_year": Y in wf.CRISIS_YEARS, "realized_dd": round(real, 5)}
            try:
                eng, _ = wf.simulate_dd(train, H)          # Blue Lotus v3.1
                ens = {
                    "engine":  eng,
                    "gauss":   sim_gauss(train, H, rng),
                    "studt":   sim_studt(train, H, rng),
                    "histiid": sim_histiid(train, H, rng),
                    "histblk": sim_histblk(train, H, rng),
                }
            except Exception as e:
                print(f"    {ticker} {Y}: FAIL {type(e).__name__}", flush=True)
                continue

            for m in MODELS:
                for k, v in scores(ens[m], real).items():
                    rec[f"{m}_{k}"] = v
            cells.append(rec)

        with open(os.path.join(OUT, "benchmark_cells.json"), "w") as fh:
            json.dump(cells, fh)
        print(f"[{ti}/{len(tickers)}] {ticker}: total {len(cells)} cells ({time.time()-t0:.0f}s)", flush=True)

    print(f"DONE: {len(cells)} cells in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()

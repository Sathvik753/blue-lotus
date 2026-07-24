"""Walk-forward (expanding-window) out-of-sample backtest of Blue Lotus v3.1.

For each (asset, test-year) cell the engine is fit ONLY on data strictly before
the test year, then its predicted one-year maximum-drawdown distribution is
compared against the realized drawdown over that year. This removes the
single-window / look-ahead weakness of a one-shot 2025 hold-out: every year is
an out-of-sample test, and the panel spans many regimes.

Outputs incremental checkpoints to backtest/wf_results/cells.json so the run is
resumable. Downloads are cached to disk so re-runs cost no network.
"""

import os, sys, json, time, warnings, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

from engine.core import (
    InputProcessor, StructuralConstraintLayer, ConstrainedMonteCarloGenerator,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wf_results")
CACHE = os.path.join(OUT, "price_cache")
os.makedirs(CACHE, exist_ok=True)

# --- Configuration --------------------------------------------------------------
SEED = 42
N_PATHS = 3000
MIN_TRAIN = 504          # >= ~2y of pre-year history required to fit a cell
MIN_TEST = 200           # >= ~10 months of realized data required in the test year
TEST_YEARS = list(range(2013, 2025))   # 2013 .. 2024 inclusive
MODERATE_DD, SEVERE_DD = -0.05, -0.15  # documented engine defaults

# --- Universe: liquid, long-history names across seven asset classes ------------
UNIVERSE = {
    # US equity beta
    "SPY": "US Equity", "QQQ": "US Equity", "DIA": "US Equity", "IWM": "US Equity",
    "IVV": "US Equity", "VTI": "US Equity", "RSP": "US Equity", "MDY": "US Equity",
    # Sectors (SPDR)
    "XLF": "Sector", "XLE": "Sector", "XLK": "Sector", "XLV": "Sector",
    "XLI": "Sector", "XLP": "Sector", "XLU": "Sector", "XLB": "Sector", "XLY": "Sector",
    # International equity
    "EFA": "Intl Equity", "EEM": "Intl Equity", "EWJ": "Intl Equity",
    "EWZ": "Intl Equity", "FXI": "Intl Equity", "VGK": "Intl Equity", "EWG": "Intl Equity",
    # Rates / duration
    "TLT": "Rates", "IEF": "Rates", "SHY": "Rates", "AGG": "Rates", "TIP": "Rates",
    # Credit
    "HYG": "Credit", "LQD": "Credit", "JNK": "Credit",
    # Commodities
    "GLD": "Commodity", "SLV": "Commodity", "USO": "Commodity", "DBC": "Commodity",
    # Single-name equity (long history)
    "AAPL": "Single Name", "MSFT": "Single Name", "JPM": "Single Name",
    "XOM": "Single Name", "JNJ": "Single Name", "PG": "Single Name",
    "KO": "Single Name", "WMT": "Single Name", "CVX": "Single Name",
    "PFE": "Single Name", "INTC": "Single Name", "CSCO": "Single Name",
    "BAC": "Single Name", "C": "Single Name", "T": "Single Name",
    "VZ": "Single Name", "IBM": "Single Name", "MCD": "Single Name",
    "HD": "Single Name", "DIS": "Single Name", "BA": "Single Name",
    "CAT": "Single Name", "MMM": "Single Name", "GS": "Single Name",
    "GE": "Single Name", "WFC": "Single Name",
}

# Crisis calendar years (for regime stratification of the panel).
CRISIS_YEARS = {2015, 2018, 2020, 2022}


def load_prices(ticker):
    """Daily adjusted returns + dates, cached to disk."""
    cf = os.path.join(CACHE, f"{ticker}.pkl")
    if os.path.exists(cf):
        with open(cf, "rb") as fh:
            return pickle.load(fh)
    import yfinance as yf
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(start="2004-01-01", end="2025-01-01", auto_adjust=True)
            if df is not None and not df.empty and len(df) > MIN_TRAIN:
                px = df["Close"].dropna()
                rets = px.pct_change().dropna().to_numpy(dtype=float).flatten()
                dates = px.index[1:].to_numpy().astype("datetime64[D]")
                data = (rets, dates)
                with open(cf, "wb") as fh:
                    pickle.dump(data, fh)
                return data
        except Exception:
            time.sleep(2)
    return None


def simulate_dd(train_returns, horizon):
    """Fit v3.1 engine on train_returns, return simulated compounded max-DD array."""
    ip = InputProcessor(winsorize=True, normalization="none")
    cleaned, meta = ip.fit_transform(train_returns)
    raw = ip.raw_returns_
    cl = StructuralConstraintLayer(moderate_dd=MODERATE_DD, severe_dd=SEVERE_DD)
    constraints = cl.fit(cleaned, raw_returns=raw)
    gen = ConstrainedMonteCarloGenerator(n_paths=N_PATHS, horizon=horizon, random_seed=SEED)
    out = gen.generate(constraints)
    w = np.cumprod(1.0 + out.paths, axis=1)
    max_dd = np.min(w / np.maximum.accumulate(w, axis=1) - 1.0, axis=1)
    return max_dd, float(meta.ann_vol)


def realized_dd(test_returns):
    """Compounded realized max drawdown over the test window (raw returns)."""
    w = np.cumprod(1.0 + test_returns)
    return float(np.min(w / np.maximum.accumulate(w) - 1.0))


def run():
    cells = []
    tickers = list(UNIVERSE.keys())
    t0 = time.time()

    for ti, ticker in enumerate(tickers, 1):
        data = load_prices(ticker)
        if data is None:
            print(f"[{ti}/{len(tickers)}] {ticker}: NO DATA", flush=True)
            continue
        rets, dates = data
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        n_cells_ticker = 0

        for Y in TEST_YEARS:
            train_mask = dates < np.datetime64(f"{Y}-01-01")
            test_mask = years == Y
            if train_mask.sum() < MIN_TRAIN or test_mask.sum() < MIN_TEST:
                continue

            train = rets[train_mask]
            test = rets[test_mask]
            horizon = int(test_mask.sum())

            try:
                sim, ann_vol = simulate_dd(train, horizon)
            except Exception as e:
                print(f"    {ticker} {Y}: FIT FAIL {type(e).__name__}", flush=True)
                continue

            real = realized_dd(test)
            u = float(np.mean(sim <= real))                 # PIT rank (fraction as bad or worse)
            p5 = float(np.percentile(sim, 5))
            p1 = float(np.percentile(sim, 1))
            p95 = float(np.percentile(sim, 95))
            cells.append({
                "ticker": ticker, "asset_class": UNIVERSE[ticker], "year": Y,
                "crisis_year": Y in CRISIS_YEARS,
                "n_train": int(train_mask.sum()), "n_test": horizon,
                "ann_vol": round(ann_vol, 4),
                "realized_dd": round(real, 5),
                "pred_p5": round(p5, 5), "pred_p1": round(p1, 5),
                "pred_p50": round(float(np.percentile(sim, 50)), 5),
                "pred_p95": round(p95, 5),
                "pit_rank": round(u, 4),
                "breach_p5": bool(real < p5),
                "breach_p1": bool(real < p1),
                "covered_90": bool(p5 <= real <= p95),
            })
            n_cells_ticker += 1

        # Incremental checkpoint after each ticker.
        with open(os.path.join(OUT, "cells.json"), "w") as fh:
            json.dump(cells, fh)
        print(f"[{ti}/{len(tickers)}] {ticker}: {n_cells_ticker} cells  "
              f"(total {len(cells)}, {time.time()-t0:.0f}s)", flush=True)

    print(f"DONE: {len(cells)} cells in {time.time()-t0:.0f}s", flush=True)
    return cells


if __name__ == "__main__":
    run()

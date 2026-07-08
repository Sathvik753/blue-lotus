"""Async background execution of stress-testing engine runs."""

import sys
import time
import traceback
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import select

sys.path.insert(0, "/app")

from db.database import AsyncSessionLocal
from db.models import Run, Result, RunStatus
from engine.serializer import serialize_run_results

async def execute_run(run_id: str, returns: np.ndarray, config: dict):
    """
    Execute a full Blue Lotus engine run in the background.
    Updates Run.status and writes Result on completion.
    """
    async with AsyncSessionLocal() as db:
        result_q = await db.execute(select(Run).where(Run.id == run_id))
        run = result_q.scalar_one_or_none()
        if run is None:
            return

        run.status = RunStatus.running
        await db.commit()

        t_start = time.perf_counter()

        try:

            import sys, os
            sys.path.insert(0, "/app")
            sys.path.insert(0, "/app/engine")
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from engine.core import (
                InputProcessor, StructuralConstraintLayer,
                ConstrainedMonteCarloGenerator, StressMetricsEngine,
                compute_fragility_index, BacktestValidator,
            )
            import numpy as _np

            ip = InputProcessor(winsorize=True, normalization="none")
            cleaned, meta = ip.fit_transform(returns)
            raw = ip.raw_returns_

            # Documented engine defaults: -5% moderate / -15% severe. The old
            # ±15σ/45σ overrides put "moderate" near -16% on an equity index,
            # which effectively disabled the drawdown-conditional blend in
            # production while the whitepaper described the defaults.
            moderate_dd = -0.05
            severe_dd = -0.15

            cl = StructuralConstraintLayer(
                moderate_dd=moderate_dd,
                severe_dd=severe_dd,
            )
            constraints = cl.fit(cleaned, raw_returns=raw)

            mc_gen = ConstrainedMonteCarloGenerator(
                n_paths=config.get("n_paths", 10_000),
                horizon=config.get("horizon", 252),
                random_seed=42,
            )
            mc_out = mc_gen.generate(constraints)

            sm_engine = StressMetricsEngine()
            stress = sm_engine.compute(mc_out)

            fi, fi_grade, fi_details = None, None, {}
            if config.get("run_sensitivity", True):
                ck = dict(moderate_dd=moderate_dd, severe_dd=severe_dd)
                mk = dict(n_paths=config.get("n_paths", 10_000), horizon=config.get("horizon", 252))
                fi, fi_grade, fi_details = compute_fragility_index(
                    cleaned, ck, mk,
                    n_paths=min(2_000, config.get("n_paths", 10_000)),
                    raw_returns=raw,
                )

            backtest_results = []
            raw_dates = config.get("dates")
            if raw_dates is not None:
                try:
                    bv = BacktestValidator()
                    # Raw returns: realized drawdowns must include the actual
                    # worst days, not winsorized ones.
                    backtest_results = bv.validate(raw, raw_dates, stress)
                except Exception:
                    pass

            payload = serialize_run_results(
                mc=mc_out, sm=stress, constraints=constraints,
                metadata=meta, fi=fi, fi_grade=fi_grade, fi_details=fi_details,
                ticker=config.get("ticker"),
                backtest_results=backtest_results,
            )

            duration = time.perf_counter() - t_start

            db_result = Result(
                run_id=run_id,
                dd_mean=float(stress.dd_mean),
                dd_p5=float(stress.dd_p5),
                es_aggregate=float(stress.es_aggregate),
                recovery_mean=float(stress.recovery_mean) if not np.isnan(stress.recovery_mean) else None,
                pct_never_recover=float(stress.pct_never_recover),
                fragility_index=float(fi) if fi is not None else None,
                fragility_grade=fi_grade,
                ann_vol=float(meta.ann_vol),   # raw (pre-winsorize) volatility
                payload=payload,
            )
            db.add(db_result)

            run.status = RunStatus.completed
            run.completed_at = datetime.now(timezone.utc)
            run.duration_sec = round(duration, 2)
            run.n_observations = int(meta.n_observations)
            await db.commit()

        except Exception as e:
            run.status = RunStatus.failed
            run.error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()

async def fetch_ticker_and_run(run_id: str, ticker: str, start_date: str, config: dict):
    """Fetch Yahoo Finance data then execute run."""
    async with AsyncSessionLocal() as db:
        try:
            import yfinance as yf
            import datetime as dt

            t = yf.Ticker(ticker)
            df = t.history(start=start_date,
                           end=dt.date.today().strftime("%Y-%m-%d"),
                           auto_adjust=True)
            if df.empty:
                raise ValueError(
                    f"No price data found for '{ticker}'. "
                    "Check the symbol is correct (e.g. SPY, AAPL, BTC-USD)."
                )

            prices = df["Close"].dropna().squeeze()
            returns = prices.pct_change().dropna().to_numpy(dtype=float).flatten()
            dates = prices.index[1:].to_numpy()

            result_q = await db.execute(select(Run).where(Run.id == run_id))
            run = result_q.scalar_one_or_none()
            if run:
                run.start_date = start_date
                run.end_date = dt.date.today().strftime("%Y-%m-%d")
                await db.commit()

        except Exception as e:
            result_q = await db.execute(select(Run).where(Run.id == run_id))
            run = result_q.scalar_one_or_none()
            if run:
                run.status = RunStatus.failed
                run.error_msg = str(e)
                await db.commit()
            return

    await execute_run(run_id, returns, {**config, "ticker": ticker, "dates": dates})

"""
Job Runner — Blue Lotus Labs
Async background execution of stress-testing engine runs.
Uses FastAPI BackgroundTasks for lightweight async job handling.
For scale: swap drop-in for Celery + Redis.
"""

import sys
import time
import traceback
import numpy as np
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

sys.path.insert(0, "/app")   # Docker path; adjust for local dev

from db.models import Run, Result, RunStatus
from engine.serializer import serialize_run_results


async def execute_run(run_id: str, returns: np.ndarray, config: dict, db: AsyncSession):
    """
    Execute a full Blue Lotus engine run in the background.
    Updates Run.status and writes Result on completion.

    Parameters
    ----------
    run_id  : DB run UUID
    returns : cleaned 1-D numpy return array
    config  : dict with n_paths, horizon, strategy_name, run_sensitivity, ticker
    db      : async DB session
    """
    # Mark as running
    result_q = await db.execute(select(Run).where(Run.id == run_id))
    run = result_q.scalar_one_or_none()
    if run is None:
        return

    run.status = RunStatus.running
    await db.commit()

    t_start = time.perf_counter()

    try:
        # ── Import engine (lazy to keep startup fast) ──────────
        from engine.core import (
            InputProcessor, StructuralConstraintLayer,
            ConstrainedMonteCarloGenerator, StressMetricsEngine,
            compute_fragility_index,
        )

        # ── Module 1: Input Processing ─────────────────────────
        ip = InputProcessor(winsorize=True, normalization="none")
        cleaned, meta = ip.fit_transform(returns)

        # ── Module 2: Constraints ──────────────────────────────
        daily_std   = float(cleaned.std())
        moderate_dd = -daily_std * 15
        severe_dd   = -daily_std * 45

        cl = StructuralConstraintLayer(
            moderate_dd=moderate_dd,
            severe_dd=severe_dd,
        )
        constraints = cl.fit(cleaned)

        # ── Module 3: Monte Carlo ──────────────────────────────
        mc_gen = ConstrainedMonteCarloGenerator(
            n_paths=config.get("n_paths", 10_000),
            horizon=config.get("horizon", 252),
            random_seed=42,
        )
        mc_out = mc_gen.generate(constraints)

        # ── Module 4: Stress Metrics ───────────────────────────
        sm_engine = StressMetricsEngine()
        stress    = sm_engine.compute(mc_out)

        # ── Module 5: Fragility Index ──────────────────────────
        fi, fi_grade = None, None
        if config.get("run_sensitivity", True):
            ck = dict(moderate_dd=moderate_dd, severe_dd=severe_dd)
            mk = dict(n_paths=config.get("n_paths", 10_000), horizon=config.get("horizon", 252))
            fi, fi_grade = compute_fragility_index(cleaned, ck, mk, n_trials=10, n_paths=1_000)

        # ── Serialize ──────────────────────────────────────────
        payload = serialize_run_results(
            mc=mc_out, sm=stress, constraints=constraints,
            metadata=meta, fi=fi, fi_grade=fi_grade,
            ticker=config.get("ticker"),
        )

        duration = time.perf_counter() - t_start

        # ── Write result to DB ─────────────────────────────────
        db_result = Result(
            run_id            = run_id,
            dd_mean           = float(stress.dd_mean),
            dd_p5             = float(stress.dd_p5),
            es_aggregate      = float(stress.es_aggregate),
            recovery_mean     = float(stress.recovery_mean) if not np.isnan(stress.recovery_mean) else None,
            pct_never_recover = float(stress.pct_never_recover),
            fragility_index   = float(fi) if fi is not None else None,
            fragility_grade   = fi_grade,
            ann_vol           = float(cleaned.std() * (252 ** 0.5)),
            payload           = payload,
        )
        db.add(db_result)

        run.status        = RunStatus.completed
        run.completed_at  = datetime.now(timezone.utc)
        run.duration_sec  = round(duration, 2)
        run.n_observations = int(meta.n_observations)
        await db.commit()

    except Exception as e:
        run.status    = RunStatus.failed
        run.error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-500:]}"
        run.completed_at = datetime.now(timezone.utc)
        await db.commit()
        raise


async def fetch_ticker_and_run(run_id: str, ticker: str, start_date: str,
                                config: dict, db: AsyncSession):
    """Fetch Yahoo Finance data then execute run."""
    try:
        import yfinance as yf
        import datetime as dt

        df = yf.download(ticker, start=start_date,
                         end=dt.date.today().strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        prices = df["Close"].dropna()
        if hasattr(prices, "columns"):
            prices = prices.iloc[:, 0]
        prices  = prices.squeeze()
        returns = prices.pct_change().dropna().to_numpy(dtype=float).flatten()

        # Update run with date range
        result_q = await db.execute(select(Run).where(Run.id == run_id))
        run = result_q.scalar_one_or_none()
        if run:
            run.start_date = start_date
            run.end_date   = dt.date.today().strftime("%Y-%m-%d")
            await db.commit()

        await execute_run(run_id, returns, {**config, "ticker": ticker}, db)

    except Exception as e:
        result_q = await db.execute(select(Run).where(Run.id == run_id))
        run = result_q.scalar_one_or_none()
        if run:
            run.status    = RunStatus.failed
            run.error_msg = str(e)
            await db.commit()
        raise

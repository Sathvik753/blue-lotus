"""
FastAPI Application — Blue Lotus Labs
REST API for the Stress-Testing Engine

Endpoints:
  POST /auth/register
  POST /auth/login
  GET  /auth/me
  POST /auth/api-keys
  GET  /auth/api-keys

  POST /run/ticker          → async job
  POST /run/custom          → async job
  GET  /run/{run_id}        → job status + result
  GET  /runs                → paginated run history

  POST /compare             → multi-ticker comparison

  GET  /health
"""

import os
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from fastapi import (
    FastAPI, Depends, HTTPException, BackgroundTasks,
    status, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from db.database import get_db, init_db
from db.models import User, ApiKey, Run, Result, RunStatus
from api.auth import (
    hash_password, verify_password, create_access_token,
    generate_api_key, hash_api_key, get_current_user,
)
from api.schemas import (
    RegisterRequest, TokenResponse, ApiKeyResponse, UserResponse,
    TickerRunRequest, CustomRunRequest, CompareRequest,
    RunStatusResponse, FullResultResponse, RunSummary,
    PaginatedRuns, CompareResponse, CompareRow,
)
from api.jobs import fetch_ticker_and_run, execute_run


# ── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Blue Lotus Labs — Stress-Testing API",
    description="Constraint-driven Monte Carlo stress-testing for financial strategies.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()


# ── Health ────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "Blue Lotus Labs API", "version": "1.0.0"}


# ── Auth ──────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered.")

    user = User(
        email=req.email,
        name=req.name,
        hashed_pw=hash_password(req.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token, user_id=user.id,
        email=user.email, plan=user.plan,
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == form.username))
    user   = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token, user_id=user.id,
        email=user.email, plan=user.plan,
    )


@app.get("/auth/me", response_model=UserResponse, tags=["Auth"])
async def me(user: User = Depends(get_current_user)):
    return UserResponse(
        id=user.id, email=user.email, name=user.name,
        plan=user.plan, created_at=user.created_at,
    )


@app.post("/auth/api-keys", response_model=ApiKeyResponse, tags=["Auth"])
async def create_api_key(
    name: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    raw, hashed = generate_api_key()
    key_obj = ApiKey(user_id=user.id, key_hash=hashed, name=name)
    db.add(key_obj)
    await db.commit()
    await db.refresh(key_obj)
    return ApiKeyResponse(
        key=raw,               # only time plaintext is returned
        key_id=key_obj.id,
        name=key_obj.name,
        created_at=key_obj.created_at,
    )


@app.get("/auth/api-keys", tags=["Auth"])
async def list_api_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.is_active == True)
    )
    keys = result.scalars().all()
    return [{"key_id": k.id, "name": k.name,
             "last_used": k.last_used, "created_at": k.created_at}
            for k in keys]


# ── Run: Ticker ───────────────────────────────────────────────────

@app.post("/run/ticker", response_model=RunStatusResponse, tags=["Runs"])
async def run_ticker(
    req: TickerRunRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a stress-test run on a ticker (fetched from Yahoo Finance).
    Returns immediately with a run_id. Poll GET /run/{run_id} for results.
    """
    run = Run(
        user_id=user.id,
        ticker=req.ticker,
        strategy_name=req.strategy_name or f"{req.ticker} daily returns",
        n_paths=req.n_paths,
        horizon=req.horizon,
        status=RunStatus.pending,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    config = {
        "n_paths":         req.n_paths,
        "horizon":         req.horizon,
        "run_sensitivity": req.run_sensitivity,
        "ticker":          req.ticker,
    }

    background_tasks.add_task(
        fetch_ticker_and_run,
        run_id=run.id, ticker=req.ticker,
        start_date=req.start_date, config=config, db=db,
    )

    return RunStatusResponse(
        run_id=run.id, status=run.status, created_at=run.created_at,
    )


# ── Run: Custom Returns ───────────────────────────────────────────

@app.post("/run/custom", response_model=RunStatusResponse, tags=["Runs"])
async def run_custom(
    req: CustomRunRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a stress-test run on a user-supplied return series.
    """
    run = Run(
        user_id=user.id,
        ticker=None,
        strategy_name=req.strategy_name,
        n_paths=req.n_paths,
        horizon=req.horizon,
        status=RunStatus.pending,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    returns = np.array(req.returns, dtype=float)
    config  = {
        "n_paths":         req.n_paths,
        "horizon":         req.horizon,
        "run_sensitivity": req.run_sensitivity,
        "strategy_name":   req.strategy_name,
    }

    background_tasks.add_task(execute_run, run_id=run.id, returns=returns,
                               config=config, db=db)

    return RunStatusResponse(
        run_id=run.id, status=run.status, created_at=run.created_at,
    )


# ── Run: Status + Result ──────────────────────────────────────────

@app.get("/run/{run_id}", response_model=FullResultResponse, tags=["Runs"])
async def get_run(
    run_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Poll for run status. When status='completed', result is included."""
    result = await db.execute(
        select(Run)
        .where(Run.id == run_id, Run.user_id == user.id)
        .options(selectinload(Run.result))
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    payload = None
    if run.result:
        payload = run.result.payload

    return FullResultResponse(
        run_id=run.id,
        status=run.status,
        ticker=run.ticker,
        strategy_name=run.strategy_name,
        created_at=run.created_at,
        completed_at=run.completed_at,
        duration_sec=run.duration_sec,
        result=payload,
    )


# ── Runs: History ─────────────────────────────────────────────────

@app.get("/runs", response_model=PaginatedRuns, tags=["Runs"])
async def list_runs(
    page: int      = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: User     = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Paginated list of all runs for the current user."""
    offset = (page - 1) * page_size

    total_q = await db.execute(
        select(func.count(Run.id)).where(Run.user_id == user.id)
    )
    total = total_q.scalar()

    runs_q = await db.execute(
        select(Run)
        .where(Run.user_id == user.id)
        .options(selectinload(Run.result))
        .order_by(Run.created_at.desc())
        .offset(offset).limit(page_size)
    )
    runs = runs_q.scalars().all()

    summaries = []
    for r in runs:
        res = r.result
        summaries.append(RunSummary(
            run_id=r.id, ticker=r.ticker,
            strategy_name=r.strategy_name, status=r.status,
            n_observations=r.n_observations,
            dd_mean=res.dd_mean if res else None,
            es_aggregate=res.es_aggregate if res else None,
            fragility_index=res.fragility_index if res else None,
            fragility_grade=res.fragility_grade if res else None,
            created_at=r.created_at,
        ))

    return PaginatedRuns(runs=summaries, total=total, page=page, page_size=page_size)


# ── Compare ───────────────────────────────────────────────────────

@app.post("/compare", response_model=CompareResponse, tags=["Analysis"])
async def compare(
    req: CompareRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Synchronous multi-ticker comparison.
    Runs engine on each ticker and returns side-by-side risk table.
    For large n_paths consider the async /run/ticker approach instead.
    """
    import sys, warnings
    sys.path.insert(0, "/app/engine")
    warnings.filterwarnings("ignore")

    from api.jobs import fetch_ticker_and_run
    import yfinance as yf
    import datetime as dt
    from engine.core import (
        InputProcessor, StructuralConstraintLayer,
        ConstrainedMonteCarloGenerator, StressMetricsEngine,
    )

    rows = []
    for ticker in req.tickers:
        try:
            df = yf.download(ticker, start=req.start_date,
                             end=dt.date.today().strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if df.empty:
                continue

            prices  = df["Close"].dropna().squeeze()
            returns = prices.pct_change().dropna().to_numpy(dtype=float).flatten()

            ip      = InputProcessor(winsorize=True, normalization="none")
            cleaned, meta = ip.fit_transform(returns)

            daily_std   = float(cleaned.std())
            cl = StructuralConstraintLayer(
                moderate_dd=-daily_std * 15,
                severe_dd=-daily_std * 45,
            )
            constraints = cl.fit(cleaned)

            mc  = ConstrainedMonteCarloGenerator(n_paths=req.n_paths, horizon=req.horizon, random_seed=42)
            mc_out = mc.generate(constraints)
            sm  = StressMetricsEngine()
            stress = sm.compute(mc_out)

            # Save lightweight run record
            run = Run(
                user_id=user.id, ticker=ticker,
                strategy_name=f"{ticker} comparison",
                n_paths=req.n_paths, horizon=req.horizon,
                status=RunStatus.completed,
                n_observations=int(meta.n_observations),
                completed_at=datetime.now(timezone.utc),
            )
            db.add(run)
            await db.commit()
            await db.refresh(run)

            rows.append(CompareRow(
                ticker=ticker,
                n_observations=int(meta.n_observations),
                ann_vol=round(float(cleaned.std() * 252**0.5), 4),
                dd_mean=round(float(stress.dd_mean), 6),
                es_aggregate=round(float(stress.es_aggregate), 6),
                pct_never_recover=round(float(stress.pct_never_recover), 4),
                recovery_median=round(float(stress.recovery_median), 2) if not np.isnan(stress.recovery_median) else None,
                fragility_index=None,
                fragility_grade=None,
                run_id=run.id,
            ))

        except Exception as e:
            continue

    return CompareResponse(
        tickers=req.tickers,
        rows=rows,
        generated_at=datetime.now(timezone.utc),
    )

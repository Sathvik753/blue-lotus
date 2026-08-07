"""FastAPI application for the Blue Lotus Labs stress-testing engine."""

import os
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import (
    FastAPI, Depends, HTTPException, BackgroundTasks,
    status, Query, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload

from db.database import get_db, init_db, AsyncSessionLocal
from db.models import (
    User, ApiKey, Run, Result, Organization,
    RunStatus, PlanTier, Role, SubscriptionStatus,
)
from api.auth import (
    SECRET_KEY, DEVELOPER_EMAILS, DEVELOPER_UNLOCK_CODE,
    hash_password, verify_password, create_access_token,
    generate_api_key, get_current_user, get_current_org,
    require_developer, is_developer, log_action,
)
from api import billing
from api.security import SecurityHeadersMiddleware, RateLimitMiddleware, IS_PRODUCTION
from reports.pdf import generate_pdf
from api.schemas import (
    RegisterRequest, TokenResponse, ApiKeyResponse, MeResponse, OrgInfo,
    TickerRunRequest, CustomRunRequest, CompareRequest,
    DeveloperUnlockRequest,
    RunStatusResponse, FullResultResponse, RunSummary,
    PaginatedRuns, CompareResponse, CompareRow,
    PlanInfo, BillingStatus, CheckoutRequest, CheckoutResponse,
    ComponentStatus, StatusResponse, DevStats,
)
from api.jobs import fetch_ticker_and_run, execute_run

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bluelotus")

APP_VERSION = "1.1.0"

app = FastAPI(
    title="Blue Lotus Labs — Stress-Testing API",
    description="Constraint-driven Monte Carlo stress-testing for financial strategies.",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middleware ----------------------------------------------------------------
_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS or ["*"],
    allow_credentials=bool(_ALLOWED_ORIGINS),   # credentials require explicit origins
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    # Do not leak internals in production.
    detail = "Internal server error" if IS_PRODUCTION else f"{type(exc).__name__}: {exc}"
    return JSONResponse(status_code=500, content={"detail": detail})


@app.on_event("startup")
async def startup():
    if IS_PRODUCTION and SECRET_KEY == "dev-insecure-key-change-in-production":
        raise RuntimeError("SECRET_KEY must be set to a strong value in production.")
    if IS_PRODUCTION and not _ALLOWED_ORIGINS:
        logger.warning("ALLOWED_ORIGINS is unset in production; CORS is permissive.")
    await init_db()
    logger.info("Blue Lotus API %s started (stripe=%s, prod=%s)",
                APP_VERSION, billing.STRIPE_ENABLED, IS_PRODUCTION)


# =============================================================== System / status
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "Blue Lotus Labs API", "version": APP_VERSION}


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def system_status():
    """Public status endpoint powering the status page."""
    components = []

    api_ok = True
    components.append(ComponentStatus(name="API", status="operational"))

    # Database round-trip.
    db_ok = True
    try:
        async with AsyncSessionLocal() as s:
            await s.execute(select(func.count(User.id)))
    except Exception as e:
        db_ok = False
        components.append(ComponentStatus(name="Database", status="down", detail=str(e)[:120]))
    if db_ok:
        components.append(ComponentStatus(name="Database", status="operational"))

    components.append(ComponentStatus(
        name="Billing",
        status="operational" if billing.STRIPE_ENABLED else "degraded",
        detail=None if billing.STRIPE_ENABLED else "Running in mock mode (Stripe not configured).",
    ))
    components.append(ComponentStatus(name="Engine", status="operational"))

    overall = "operational" if (api_ok and db_ok) else "degraded"
    return StatusResponse(
        status=overall, version=APP_VERSION,
        time=datetime.now(timezone.utc), components=components,
    )


# =============================================================== Auth
@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(req: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered.")

    org = Organization(
        name=req.org_name or f"{(req.name or req.email.split('@')[0])}'s workspace",
        plan=PlanTier.free,
        subscription_status=SubscriptionStatus.inactive,
    )
    db.add(org)
    await db.commit()
    await db.refresh(org)

    dev = req.email.strip().lower() in DEVELOPER_EMAILS
    user = User(
        email=req.email,
        name=req.name,
        hashed_pw=hash_password(req.password),
        org_id=org.id,
        role=(Role.developer if dev else Role.owner),
        plan=org.plan,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    await log_action(db, "user.register", user=user, request=request)
    token = create_access_token(user.id, user.email, dev=is_developer(user))
    return TokenResponse(access_token=token, user_id=user.id, email=user.email, plan=user.plan)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_pw):
        await log_action(db, "user.login_failed", detail=form.username, request=request)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    await log_action(db, "user.login", user=user, request=request)
    token = create_access_token(user.id, user.email, dev=is_developer(user))
    return TokenResponse(access_token=token, user_id=user.id, email=user.email, plan=user.plan)


@app.get("/auth/me", response_model=MeResponse, tags=["Auth"])
async def me(
    user: User = Depends(get_current_user),
    org: Organization = Depends(get_current_org),
):
    return MeResponse(
        id=user.id, email=user.email, name=user.name,
        role=str(user.role), plan=org.plan, is_developer=is_developer(user),
        org=OrgInfo(id=org.id, name=org.name, plan=org.plan,
                    subscription_status=str(org.subscription_status)),
    )




@app.post("/auth/developer/unlock", response_model=TokenResponse, tags=["Auth"])
async def developer_unlock(
    req: DeveloperUnlockRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Grant the developer role to the signed-in user in exchange for the
    unlock code. Disabled entirely when DEVELOPER_UNLOCK_CODE is unset. The
    fresh token carries the dev claim, which exempts the account from rate
    limiting and run quotas."""
    import secrets as _secrets
    if not DEVELOPER_UNLOCK_CODE:
        raise HTTPException(status_code=404, detail="Developer unlock is not enabled.")
    if not _secrets.compare_digest(req.code.strip(), DEVELOPER_UNLOCK_CODE):
        await log_action(db, "dev.unlock_failed", user=user, request=request)
        raise HTTPException(status_code=403, detail="Invalid code.")

    user.role = Role.developer
    await db.commit()
    await log_action(db, "dev.unlock", user=user, request=request)

    token = create_access_token(user.id, user.email, dev=True)
    return TokenResponse(access_token=token, user_id=user.id,
                         email=user.email, plan=user.plan)


@app.post("/auth/api-keys", response_model=ApiKeyResponse, tags=["Auth"])
async def create_api_key(
    name: Optional[str] = None,
    user: User = Depends(get_current_user),
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    raw, hashed, prefix = generate_api_key()
    key_obj = ApiKey(user_id=user.id, org_id=org.id, key_hash=hashed, prefix=prefix, name=name)
    db.add(key_obj)
    await db.commit()
    await db.refresh(key_obj)
    await log_action(db, "apikey.create", user=user)
    return ApiKeyResponse(key=raw, key_id=key_obj.id, name=key_obj.name, created_at=key_obj.created_at)


@app.get("/auth/api-keys", tags=["Auth"])
async def list_api_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.is_active == True)
    )
    keys = result.scalars().all()
    return [{"key_id": k.id, "name": k.name, "prefix": k.prefix,
             "last_used": k.last_used, "created_at": k.created_at} for k in keys]


# =============================================================== Billing
@app.get("/billing/plans", response_model=list[PlanInfo], tags=["Billing"])
async def list_plans():
    out = []
    for tier, p in billing.PLANS.items():
        out.append(PlanInfo(
            tier=tier.value, name=p["name"], price_usd=p["price_usd"],
            monthly_runs=p["monthly_runs"], blurb=p["blurb"], features=p["features"],
        ))
    return out


@app.get("/billing/status", response_model=BillingStatus, tags=["Billing"])
async def billing_status(
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    return BillingStatus(**await billing.usage_summary(db, org))


@app.post("/billing/checkout", response_model=CheckoutResponse, tags=["Billing"])
async def billing_checkout(
    req: CheckoutRequest,
    user: User = Depends(get_current_user),
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    if user.role not in (Role.owner, "owner", Role.developer, "developer"):
        raise HTTPException(status_code=403, detail="Only the org owner can change the plan.")
    result = await billing.create_checkout_session(db, org, req.tier)
    await log_action(db, "billing.checkout", user=user, detail=req.tier)
    return CheckoutResponse(**result)


@app.post("/billing/portal", tags=["Billing"])
async def billing_portal(org: Organization = Depends(get_current_org)):
    return await billing.create_portal_session(org)


@app.post("/billing/webhook", tags=["Billing"])
async def billing_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    return await billing.handle_webhook(db, payload, sig)


# =============================================================== Runs
@app.post("/run/ticker", response_model=RunStatusResponse, tags=["Runs"])
async def run_ticker(
    req: TickerRunRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Submit a stress-test run on a ticker (fetched from Yahoo Finance)."""
    await billing.enforce_quota(db, org, user)

    run = Run(
        user_id=user.id, org_id=org.id,
        ticker=req.ticker,
        strategy_name=req.strategy_name or f"{req.ticker} daily returns",
        n_paths=req.n_paths, horizon=req.horizon,
        status=RunStatus.pending,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    await log_action(db, "run.create", user=user, detail=f"ticker:{req.ticker}")

    config = {"n_paths": req.n_paths, "horizon": req.horizon,
              "run_sensitivity": req.run_sensitivity, "ticker": req.ticker}
    background_tasks.add_task(
        fetch_ticker_and_run, run_id=run.id, ticker=req.ticker,
        start_date=req.start_date, config=config,
    )
    return RunStatusResponse(run_id=run.id, status=run.status, created_at=run.created_at)


@app.post("/run/custom", response_model=RunStatusResponse, tags=["Runs"])
async def run_custom(
    req: CustomRunRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Submit a stress-test run on a user-supplied return series."""
    await billing.enforce_quota(db, org, user)

    run = Run(
        user_id=user.id, org_id=org.id, ticker=None,
        strategy_name=req.strategy_name,
        n_paths=req.n_paths, horizon=req.horizon,
        status=RunStatus.pending,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    await log_action(db, "run.create", user=user, detail="custom")

    returns = np.array(req.returns, dtype=float)
    config = {"n_paths": req.n_paths, "horizon": req.horizon,
              "run_sensitivity": req.run_sensitivity, "strategy_name": req.strategy_name}
    background_tasks.add_task(execute_run, run_id=run.id, returns=returns, config=config)
    return RunStatusResponse(run_id=run.id, status=run.status, created_at=run.created_at)


@app.get("/run/{run_id}", response_model=FullResultResponse, tags=["Runs"])
async def get_run(
    run_id: str,
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Run).where(Run.id == run_id, Run.org_id == org.id)
        .options(selectinload(Run.result))
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    payload = run.result.payload if run.result else None
    return FullResultResponse(
        run_id=run.id, status=run.status, ticker=run.ticker,
        strategy_name=run.strategy_name, created_at=run.created_at,
        completed_at=run.completed_at, duration_sec=run.duration_sec,
        error_msg=run.error_msg, result=payload,
    )


async def _load_completed_run(run_id: str, org: Organization, db: AsyncSession) -> Run:
    result = await db.execute(
        select(Run).where(Run.id == run_id, Run.org_id == org.id)
        .options(selectinload(Run.result))
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run.status != RunStatus.completed or run.result is None:
        raise HTTPException(status_code=409, detail="Run is not completed yet.")
    return run


def _slug(run: Run) -> str:
    base = (run.strategy_name or run.ticker or "run").replace(" ", "_")
    return "".join(c for c in base if c.isalnum() or c in "_-")[:40]


@app.get("/run/{run_id}/export", tags=["Runs"])
async def export_run_json(
    run_id: str,
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    run = await _load_completed_run(run_id, org, db)
    body = {
        "run_id": run.id, "strategy_name": run.strategy_name, "ticker": run.ticker,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "result": run.result.payload,
    }
    filename = f"bluelotus_{_slug(run)}_{run.id[:8]}.json"
    return JSONResponse(content=body,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/run/{run_id}/pdf", tags=["Runs"])
async def export_run_pdf(
    run_id: str,
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    run = await _load_completed_run(run_id, org, db)
    pdf_bytes = generate_pdf(
        result=run.result.payload,
        strategy_name=run.strategy_name or run.ticker or "Strategy",
        ticker=run.ticker, run_id=run.id,
    )
    filename = f"bluelotus_{_slug(run)}_{run.id[:8]}.pdf"
    return Response(content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/runs", response_model=PaginatedRuns, tags=["Runs"])
async def list_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    # Multi-ticker comparison runs are one-shot side-by-sides with no stored
    # report, so they don't belong in History. They are created with a
    # strategy name of "<ticker> comparison"; exclude them here.
    not_comparison = or_(Run.strategy_name.is_(None), Run.strategy_name.notlike("% comparison"))

    total_q = await db.execute(
        select(func.count(Run.id)).where(Run.org_id == org.id, not_comparison)
    )
    total = total_q.scalar()

    runs_q = await db.execute(
        select(Run).where(Run.org_id == org.id, not_comparison)
        .options(selectinload(Run.result))
        .order_by(Run.created_at.desc())
        .offset(offset).limit(page_size)
    )
    runs = runs_q.scalars().all()

    summaries = []
    for r in runs:
        res = r.result
        summaries.append(RunSummary(
            run_id=r.id, ticker=r.ticker, strategy_name=r.strategy_name, status=r.status,
            n_observations=r.n_observations,
            dd_mean=res.dd_mean if res else None,
            es_aggregate=res.es_aggregate if res else None,
            fragility_index=res.fragility_index if res else None,
            fragility_grade=res.fragility_grade if res else None,
            created_at=r.created_at,
        ))
    return PaginatedRuns(runs=summaries, total=total, page=page, page_size=page_size)


@app.post("/compare", response_model=CompareResponse, tags=["Analysis"])
async def compare(
    req: CompareRequest,
    user: User = Depends(get_current_user),
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Synchronous multi-ticker comparison."""
    await billing.enforce_quota(db, org, user)

    import sys, warnings
    sys.path.insert(0, "/app/engine")
    warnings.filterwarnings("ignore")

    import yfinance as yf
    import datetime as dt
    from engine.core import (
        InputProcessor, StructuralConstraintLayer,
        ConstrainedMonteCarloGenerator, StressMetricsEngine,
    )

    rows = []
    for ticker in req.tickers:
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=req.start_date,
                           end=dt.date.today().strftime("%Y-%m-%d"),
                           auto_adjust=True)
            if df.empty:
                continue

            prices = df["Close"].dropna().squeeze()
            returns = prices.pct_change().dropna().to_numpy(dtype=float).flatten()

            ip = InputProcessor(winsorize=True, normalization="none")
            cleaned, meta = ip.fit_transform(returns)

            # Documented engine defaults; tail fit on raw returns.
            cl = StructuralConstraintLayer(moderate_dd=-0.05, severe_dd=-0.15)
            constraints = cl.fit(cleaned, raw_returns=ip.raw_returns_)

            mc = ConstrainedMonteCarloGenerator(n_paths=req.n_paths, horizon=req.horizon, random_seed=42)
            mc_out = mc.generate(constraints)
            sm = StressMetricsEngine()
            stress = sm.compute(mc_out)

            run = Run(
                user_id=user.id, org_id=org.id, ticker=ticker,
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
                ticker=ticker, n_observations=int(meta.n_observations),
                ann_vol=round(float(meta.ann_vol), 4),
                dd_mean=round(float(stress.dd_mean), 6),
                es_aggregate=round(float(stress.es_aggregate), 6),
                pct_never_recover=round(float(stress.pct_never_recover), 4),
                recovery_median=round(float(stress.recovery_median), 2) if not np.isnan(stress.recovery_median) else None,
                fragility_index=None, fragility_grade=None, run_id=run.id,
            ))
        except Exception:
            continue

    return CompareResponse(tickers=req.tickers, rows=rows, generated_at=datetime.now(timezone.utc))


# =============================================================== Developer (gated)
@app.get("/dev/stats", response_model=DevStats, tags=["Developer"])
async def dev_stats(
    _dev: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    """System-wide statistics. Restricted to allowlisted developer accounts."""
    orgs = (await db.execute(select(func.count(Organization.id)))).scalar() or 0
    users = (await db.execute(select(func.count(User.id)))).scalar() or 0
    runs_total = (await db.execute(select(func.count(Run.id)))).scalar() or 0

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    runs_24h = (await db.execute(
        select(func.count(Run.id)).where(Run.created_at >= since)
    )).scalar() or 0

    by_status = {}
    for st in RunStatus:
        c = (await db.execute(
            select(func.count(Run.id)).where(Run.status == st.value)
        )).scalar() or 0
        by_status[st.value] = int(c)

    return DevStats(
        organizations=int(orgs), users=int(users),
        runs_total=int(runs_total), runs_24h=int(runs_24h),
        runs_by_status=by_status, stripe_enabled=billing.STRIPE_ENABLED,
        env=os.environ.get("ENV", "development"),
    )


@app.get("/dev/organizations", tags=["Developer"])
async def dev_orgs(
    _dev: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Organization).order_by(Organization.created_at.desc()).limit(200)
    )).scalars().all()
    return [{
        "id": o.id, "name": o.name, "plan": o.plan,
        "subscription_status": str(o.subscription_status),
        "created_at": o.created_at,
    } for o in rows]

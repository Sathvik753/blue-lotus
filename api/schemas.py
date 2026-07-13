"""Pydantic request/response schemas for the Blue Lotus Labs API."""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: Optional[str] = None
    org_name: Optional[str] = None   # defaults to "<name>'s workspace"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    plan: str

class ApiKeyResponse(BaseModel):
    key: str  # only returned once on creation
    key_id: str
    name: Optional[str]
    created_at: datetime

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    plan: str
    created_at: datetime

class OrgInfo(BaseModel):
    id: str
    name: str
    plan: str
    subscription_status: str

class MeResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    role: str
    plan: str
    is_developer: bool
    org: OrgInfo

# --- Billing -------------------------------------------------------------------
class PlanInfo(BaseModel):
    tier: str
    name: str
    price_usd: Optional[int]
    monthly_runs: Optional[int]
    blurb: str
    features: List[str]

class BillingStatus(BaseModel):
    plan: str
    plan_name: str
    subscription_status: str
    period: str
    runs_used: int
    runs_limit: Optional[int]
    runs_remaining: Optional[int]
    stripe_enabled: bool

class CheckoutRequest(BaseModel):
    tier: str = Field(examples=["pro", "enterprise"])

class CheckoutResponse(BaseModel):
    mode: str
    checkout_url: str
    message: Optional[str] = None

class DeveloperUnlockRequest(BaseModel):
    code: str = Field(min_length=1, max_length=64)

# --- System / status -----------------------------------------------------------
class ComponentStatus(BaseModel):
    name: str
    status: str            # "operational" | "degraded" | "down"
    detail: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    version: str
    time: datetime
    components: List[ComponentStatus]

# --- Developer -----------------------------------------------------------------
class DevStats(BaseModel):
    organizations: int
    users: int
    runs_total: int
    runs_24h: int
    runs_by_status: Dict[str, int]
    stripe_enabled: bool
    env: str

class TickerRunRequest(BaseModel):
    """Run stress test on a ticker fetched from Yahoo Finance."""
    ticker: str = Field(examples=["SPY", "QQQ", "BTC-USD"])
    start_date: str = Field(default="2010-01-01", examples=["2010-01-01"])
    n_paths: int = Field(default=10_000, ge=1_000, le=100_000)
    horizon: int = Field(default=252, ge=21, le=1260)
    strategy_name: Optional[str] = None
    run_sensitivity: bool = True

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v):
        return v.strip().upper()

class CustomRunRequest(BaseModel):
    """Run stress test on user-supplied return series."""
    returns: List[float] = Field(min_length=30, max_length=100_000, description="Daily return series (max 100 000 observations)")
    strategy_name: str = Field(default="Custom Strategy")
    n_paths: int = Field(default=10_000, ge=1_000, le=100_000)
    horizon: int = Field(default=252, ge=21, le=1260)
    run_sensitivity: bool = True

    @field_validator("returns")
    @classmethod
    def validate_returns(cls, v):
        if any(abs(r) > 10.0 for r in v):
            raise ValueError(
                "Returns look like percentages (e.g. 5.2). "
                "Please supply decimal returns (e.g. 0.052)."
            )
        return v

class CompareRequest(BaseModel):
    """Compare multiple tickers side by side."""
    tickers: List[str] = Field(min_length=2, max_length=10)
    start_date: str = Field(default="2010-01-01")
    n_paths: int = Field(default=5_000, ge=1_000, le=50_000)
    horizon: int = Field(default=252, ge=21, le=1260)

    @field_validator("tickers")
    @classmethod
    def tickers_upper(cls, v):
        return [t.strip().upper() for t in v]

class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_sec: Optional[float] = None
    error_msg: Optional[str] = None

class RunSummary(BaseModel):
    run_id: str
    ticker: Optional[str]
    strategy_name: Optional[str]
    status: str
    n_observations: Optional[int]
    dd_mean: Optional[float]
    es_aggregate: Optional[float]
    fragility_index: Optional[float]
    fragility_grade: Optional[str]
    created_at: datetime

class FullResultResponse(BaseModel):
    run_id: str
    status: str
    ticker: Optional[str]
    strategy_name: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    duration_sec: Optional[float]
    error_msg: Optional[str] = None
    result: Optional[Dict[str, Any]]

class CompareRow(BaseModel):
    ticker: str
    n_observations: int
    ann_vol: Optional[float]
    dd_mean: Optional[float]
    es_aggregate: Optional[float]
    pct_never_recover: Optional[float]
    recovery_median: Optional[float]
    fragility_index: Optional[float]
    fragility_grade: Optional[str]
    run_id: str

class CompareResponse(BaseModel):
    tickers: List[str]
    rows: List[CompareRow]
    generated_at: datetime

class PaginatedRuns(BaseModel):
    runs: List[RunSummary]
    total: int
    page: int
    page_size: int

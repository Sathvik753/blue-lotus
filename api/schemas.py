"""
API Schemas — Blue Lotus Labs
Pydantic models for request validation and response serialization
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Auth ─────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: Optional[str] = None


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
    key: str          # only returned once on creation
    key_id: str
    name: Optional[str]
    created_at: datetime


class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    plan: str
    created_at: datetime


# ── Run Requests ──────────────────────────────────────────────────

class TickerRunRequest(BaseModel):
    """Run stress test on a ticker fetched from Yahoo Finance."""
    ticker: str             = Field(examples=["SPY", "QQQ", "BTC-USD"])
    start_date: str         = Field(default="2010-01-01", examples=["2010-01-01"])
    n_paths: int            = Field(default=10_000, ge=1_000, le=100_000)
    horizon: int            = Field(default=252,    ge=21,    le=1260)
    strategy_name: Optional[str] = None
    run_sensitivity: bool   = True

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v):
        return v.strip().upper()


class CustomRunRequest(BaseModel):
    """Run stress test on user-supplied return series."""
    returns: List[float]    = Field(min_length=30, description="Daily return series")
    strategy_name: str      = Field(default="Custom Strategy")
    n_paths: int            = Field(default=10_000, ge=1_000, le=100_000)
    horizon: int            = Field(default=252,    ge=21,    le=1260)
    run_sensitivity: bool   = True

    @field_validator("returns")
    @classmethod
    def validate_returns(cls, v):
        if any(abs(r) > 1.0 for r in v):
            raise ValueError(
                "Returns look like percentages (e.g. 5.2). "
                "Please supply decimal returns (e.g. 0.052)."
            )
        return v


class CompareRequest(BaseModel):
    """Compare multiple tickers side by side."""
    tickers: List[str]      = Field(min_length=2, max_length=10)
    start_date: str         = Field(default="2010-01-01")
    n_paths: int            = Field(default=5_000, ge=1_000, le=50_000)
    horizon: int            = Field(default=252, ge=21, le=1260)

    @field_validator("tickers")
    @classmethod
    def tickers_upper(cls, v):
        return [t.strip().upper() for t in v]


# ── Run Responses ─────────────────────────────────────────────────

class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_sec: Optional[float]    = None
    error_msg: Optional[str]         = None


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
    result: Optional[Dict[str, Any]]   # full serialized engine output


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


# ── Pagination ────────────────────────────────────────────────────

class PaginatedRuns(BaseModel):
    runs: List[RunSummary]
    total: int
    page: int
    page_size: int

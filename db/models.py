"""
Database Models — Blue Lotus Labs
PostgreSQL via SQLAlchemy (async)
Tables: users, api_keys, runs, results
"""

from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Text, ForeignKey, JSON, Enum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid
import enum

Base = declarative_base()


def new_uuid():
    return str(uuid.uuid4())


# ── Enums ────────────────────────────────────────────────────────

class RunStatus(str, enum.Enum):
    pending   = "pending"
    running   = "running"
    completed = "completed"
    failed    = "failed"


class PlanTier(str, enum.Enum):
    free       = "free"
    pro        = "pro"
    enterprise = "enterprise"


# ── Tables ───────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id         = Column(String, primary_key=True, default=new_uuid)
    email      = Column(String, unique=True, nullable=False, index=True)
    name       = Column(String, nullable=True)
    hashed_pw  = Column(String, nullable=False)
    plan       = Column(String, default=PlanTier.free)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete")
    runs     = relationship("Run",    back_populates="user", cascade="all, delete")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id         = Column(String, primary_key=True, default=new_uuid)
    user_id    = Column(String, ForeignKey("users.id"), nullable=False)
    key_hash   = Column(String, unique=True, nullable=False)   # store hash, never plaintext
    name       = Column(String, nullable=True)                 # e.g. "production", "test"
    is_active  = Column(Boolean, default=True)
    last_used  = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="api_keys")


class Run(Base):
    __tablename__ = "runs"

    id           = Column(String, primary_key=True, default=new_uuid)
    user_id      = Column(String, ForeignKey("users.id"), nullable=False)
    ticker       = Column(String, nullable=True)               # null if custom returns uploaded
    strategy_name = Column(String, nullable=True)
    status       = Column(String, default=RunStatus.pending)
    error_msg    = Column(Text, nullable=True)

    # Engine config stored so runs are reproducible
    n_paths      = Column(Integer, default=10_000)
    horizon      = Column(Integer, default=252)
    n_observations = Column(Integer, nullable=True)
    start_date   = Column(String, nullable=True)
    end_date     = Column(String, nullable=True)

    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_sec = Column(Float, nullable=True)

    user    = relationship("User",       back_populates="runs")
    result  = relationship("Result",     back_populates="run",  uselist=False, cascade="all, delete")


class Result(Base):
    __tablename__ = "results"

    id      = Column(String, primary_key=True, default=new_uuid)
    run_id  = Column(String, ForeignKey("runs.id"), unique=True, nullable=False)

    # Top-level risk metrics (denormalized for fast queries / dashboards)
    dd_mean            = Column(Float, nullable=True)
    dd_p5              = Column(Float, nullable=True)
    es_aggregate       = Column(Float, nullable=True)
    recovery_mean      = Column(Float, nullable=True)
    pct_never_recover  = Column(Float, nullable=True)
    fragility_index    = Column(Float, nullable=True)
    fragility_grade    = Column(String, nullable=True)
    ann_vol            = Column(Float, nullable=True)

    # Full serialized result blob (everything the frontend needs)
    payload = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("Run", back_populates="result")

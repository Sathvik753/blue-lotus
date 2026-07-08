"""Database models for Blue Lotus Labs (PostgreSQL via SQLAlchemy async)."""

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

class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"

class PlanTier(str, enum.Enum):
    free = "free"
    pro = "pro"
    enterprise = "enterprise"

class Role(str, enum.Enum):
    owner = "owner"
    member = "member"
    developer = "developer"

class SubscriptionStatus(str, enum.Enum):
    inactive = "inactive"      # no paid subscription (free tier)
    trialing = "trialing"
    active = "active"
    past_due = "past_due"
    canceled = "canceled"


class Organization(Base):
    """A billing tenant. Every user, run, and API key belongs to exactly one org,
    which is the unit of data isolation and of subscription/metering."""
    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=new_uuid)
    name = Column(String, nullable=False)
    plan = Column(String, default=PlanTier.free)

    # Stripe linkage (null until a checkout completes).
    stripe_customer_id = Column(String, nullable=True, index=True)
    stripe_subscription_id = Column(String, nullable=True, index=True)
    subscription_status = Column(String, default=SubscriptionStatus.inactive)
    current_period_end = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    users = relationship("User", back_populates="org", cascade="all, delete")
    runs = relationship("Run", back_populates="org", cascade="all, delete")


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    hashed_pw = Column(String, nullable=False)
    role = Column(String, default=Role.owner)
    plan = Column(String, default=PlanTier.free)   # mirror of org plan for convenience
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    org = relationship("Organization", back_populates="users")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete")
    runs = relationship("Run", back_populates="user", cascade="all, delete")

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=new_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=True, index=True)
    key_hash = Column(String, unique=True, nullable=False)
    prefix = Column(String, nullable=True)   # first chars, shown in UI to identify a key
    name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="api_keys")

class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, default=new_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=True, index=True)
    ticker = Column(String, nullable=True)
    strategy_name = Column(String, nullable=True)
    status = Column(String, default=RunStatus.pending)
    error_msg = Column(Text, nullable=True)

    n_paths = Column(Integer, default=10_000)
    horizon = Column(Integer, default=252)
    n_observations = Column(Integer, nullable=True)
    start_date = Column(String, nullable=True)
    end_date = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_sec = Column(Float, nullable=True)

    user = relationship("User", back_populates="runs")
    org = relationship("Organization", back_populates="runs")
    result = relationship("Result", back_populates="run", uselist=False, cascade="all, delete")

class Result(Base):
    __tablename__ = "results"

    id = Column(String, primary_key=True, default=new_uuid)
    run_id = Column(String, ForeignKey("runs.id"), unique=True, nullable=False)

    dd_mean = Column(Float, nullable=True)
    dd_p5 = Column(Float, nullable=True)
    es_aggregate = Column(Float, nullable=True)
    recovery_mean = Column(Float, nullable=True)
    pct_never_recover = Column(Float, nullable=True)
    fragility_index = Column(Float, nullable=True)
    fragility_grade = Column(String, nullable=True)
    ann_vol = Column(Float, nullable=True)

    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("Run", back_populates="result")


class AuditLog(Base):
    """Append-only security/audit trail. One row per security-relevant action."""
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=True, index=True)
    user_id = Column(String, nullable=True, index=True)
    action = Column(String, nullable=False)          # e.g. "user.login", "run.create"
    detail = Column(Text, nullable=True)
    ip = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

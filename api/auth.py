"""JWT auth, API keys, org resolution, developer gating, and audit logging."""

import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Security, status, Request
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.database import get_db
from db.models import User, ApiKey, Organization, AuditLog, Role

# SECRET_KEY must be set in production. A dev fallback keeps local boots working;
# main.py refuses to start in production if the insecure default is still in use.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Comma-separated allowlist of emails granted developer access, e.g.
# DEVELOPER_EMAILS="you@firm.com,partner@firm.com". Case-insensitive.
DEVELOPER_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("DEVELOPER_EMAILS", "").split(",")
    if e.strip()
}

# Secret unlock code: a signed-in user who submits it gains the developer
# role. Deliberately NOT defaulted in source (this repo is public) — the
# unlock endpoint stays disabled unless the env var is set on the service.
DEVELOPER_UNLOCK_CODE = os.environ.get("DEVELOPER_UNLOCK_CODE", "")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password[:72])

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain[:72], hashed)
    except Exception:
        return False

def create_access_token(user_id: str, email: str, dev: bool = False) -> str:
    """The `dev` claim lets the rate-limit middleware exempt developer traffic
    without a database lookup — the claim is trusted because the token is
    signature-verified."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "exp": expire}
    if dev:
        payload["dev"] = True
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def generate_api_key() -> tuple[str, str, str]:
    """Returns (raw_key, hashed_key, prefix). Store only the hash and prefix."""
    raw = "bl_" + secrets.token_urlsafe(32)
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hashed, raw[:10]

def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def is_developer(user: User) -> bool:
    """A user is a developer if their role is developer or their email is on the
    server-side allowlist. The allowlist wins, so access can be granted without a
    database write and revoked by redeploying with a new env var."""
    if user.email and user.email.lower() in DEVELOPER_EMAILS:
        return True
    return user.role == Role.developer or user.role == "developer"


async def log_action(
    db: AsyncSession, action: str, *,
    user: Optional[User] = None, org_id: Optional[str] = None,
    detail: Optional[str] = None, request: Optional[Request] = None,
) -> None:
    """Best-effort append to the audit trail; never raises into the request path."""
    try:
        ip = None
        if request is not None:
            ip = request.headers.get("x-forwarded-for", request.client.host if request.client else None)
        entry = AuditLog(
            action=action,
            user_id=user.id if user else None,
            org_id=org_id or (user.org_id if user else None),
            detail=detail,
            ip=ip,
        )
        db.add(entry)
        await db.commit()
    except Exception:
        await db.rollback()


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Authenticate via a Bearer JWT or an X-API-Key header."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user = None

    if token:
        payload = decode_token(token)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                result = await db.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()

    if user is None and api_key:
        key_hash = hash_api_key(api_key)
        result = await db.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
        )
        key_obj = result.scalar_one_or_none()
        if key_obj:
            result = await db.execute(select(User).where(User.id == key_obj.user_id))
            user = result.scalar_one_or_none()
            if user:
                key_obj.last_used = datetime.now(timezone.utc)
                await db.commit()

    if user is None or not user.is_active:
        raise credentials_exception

    return user


async def get_current_org(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Organization:
    """Resolve the caller's organization, self-healing if a legacy user has none."""
    if user.org_id:
        result = await db.execute(select(Organization).where(Organization.id == user.org_id))
        org = result.scalar_one_or_none()
        if org:
            return org

    # Legacy or orphaned user: attach a personal org so isolation still holds.
    org = Organization(name=(user.name or user.email.split("@")[0]) + "'s workspace")
    db.add(org)
    await db.commit()
    await db.refresh(org)
    user.org_id = org.id
    await db.commit()
    return org


async def require_developer(user: User = Depends(get_current_user)) -> User:
    """Gate for developer-only endpoints."""
    if not is_developer(user):
        raise HTTPException(status_code=403, detail="Developer access required.")
    return user

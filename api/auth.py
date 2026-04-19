"""
Auth Utilities — Blue Lotus Labs
JWT tokens + API key generation + bcrypt password hashing
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.database import get_db
from db.models import User, ApiKey

# ── Config ───────────────────────────────────────────────────────
SECRET_KEY      = os.getenv("SECRET_KEY", "change-me-in-production-use-openssl-rand-hex-32")
ALGORITHM       = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24   # 24 hours

pwd_context     = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme   = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
api_key_header  = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Password ─────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password[:72])


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT ──────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": user_id, "email": email, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ── API Key ──────────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str]:
    """Returns (raw_key, hashed_key). Store only the hash."""
    raw  = "bl_" + secrets.token_urlsafe(32)
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hashed


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ── FastAPI Auth Dependencies ─────────────────────────────────────

async def get_current_user(
    token: Optional[str]    = Depends(oauth2_scheme),
    api_key: Optional[str]  = Security(api_key_header),
    db: AsyncSession        = Depends(get_db),
) -> User:
    """
    Accepts either:
      - Bearer JWT token  (browser / Streamlit frontend)
      - X-API-Key header  (programmatic API access)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user = None

    # Try JWT first
    if token:
        payload = decode_token(token)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                result = await db.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()

    # Try API key
    if user is None and api_key:
        key_hash = hash_api_key(api_key)
        result   = await db.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
        )
        key_obj = result.scalar_one_or_none()
        if key_obj:
            result = await db.execute(select(User).where(User.id == key_obj.user_id))
            user   = result.scalar_one_or_none()
            # Update last_used
            if user:
                key_obj.last_used = datetime.now(timezone.utc)
                await db.commit()

    if user is None or not user.is_active:
        raise credentials_exception

    return user


async def get_current_user_optional(
    token: Optional[str]   = Depends(oauth2_scheme),
    api_key: Optional[str] = Security(api_key_header),
    db: AsyncSession       = Depends(get_db),
) -> Optional[User]:
    """Same as get_current_user but returns None instead of raising."""
    try:
        return await get_current_user(token, api_key, db)
    except HTTPException:
        return None

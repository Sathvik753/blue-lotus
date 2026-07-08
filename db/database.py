"""Async database connection, session factory, and table init."""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from db.models import Base

# Postgres in production (DATABASE_URL set by the host); SQLite locally so the
# app runs end-to-end with no external database.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./bluelotus.db")

# Railway/Render hand out postgres:// URLs; SQLAlchemy's async driver needs the
# explicit +asyncpg dialect.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    poolclass=NullPool,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Columns added after the initial schema shipped. create_all() creates new
# *tables* but never alters existing ones, so we add these defensively on boot.
# Each runs in its own transaction and swallows "already exists" errors, which
# keeps it idempotent on both fresh and previously-deployed databases.
_ADDITIVE_COLUMNS = [
    ("users", "org_id", "VARCHAR"),
    ("users", "role", "VARCHAR"),
    ("runs", "org_id", "VARCHAR"),
    ("api_keys", "org_id", "VARCHAR"),
    ("api_keys", "prefix", "VARCHAR"),
]


async def _ensure_columns():
    from sqlalchemy import text
    for table, column, coltype in _ADDITIVE_COLUMNS:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {column} {coltype}'))
        except Exception:
            # Column already exists (or table is brand-new from create_all) — fine.
            pass


async def init_db():
    """Create all tables and apply additive column migrations. Call once on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _ensure_columns()


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

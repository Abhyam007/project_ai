"""
Async PostgreSQL connection pool using psycopg3.
Provides a context-managed connection with pgvector support.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from core.config import settings

# Module-level connection pool (initialised on startup)
_pool: AsyncConnectionPool | None = None


async def init_db() -> None:
    """Create the connection pool and register the pgvector type."""
    global _pool

    # Ensure SSL is always enabled for Supabase connections
    conninfo = settings.database_url
    if "sslmode" not in conninfo:
        conninfo += "?sslmode=require" if "?" not in conninfo else "&sslmode=require"

    _pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=2,
        max_size=10,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    await _pool.open()
    # Register pgvector type on a sample connection
    async with _pool.connection() as conn:
        await register_vector_async(conn)


async def close_db() -> None:
    """Close the connection pool on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Yield a pooled async database connection."""
    if _pool is None:
        raise RuntimeError("Database pool is not initialised. Call init_db() first.")
    async with _pool.connection() as conn:
        yield conn

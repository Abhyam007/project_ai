"""
FastAPI application entry point.
Mounts all routers and handles startup / shutdown lifecycle.
"""
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import documents, chat
from core.database import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise resources on startup; clean up on shutdown."""
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="Project AI — RAG API",
    description="Upload PDFs and ask questions about them using Retrieval-Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (all origins — development only) ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "RAG API is running"}

"""Pydantic request/response schemas."""
from pydantic import BaseModel, Field


# ── Upload ──────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: int
    filename: str
    chunk_count: int


# ── Chat ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    document_id: int = Field(..., gt=0, description="ID of the uploaded document")
    question: str = Field(..., min_length=1, description="Question to answer")


class Citation(BaseModel):
    label: str          # e.g. "S1"
    chunk_id: int
    page_number: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]

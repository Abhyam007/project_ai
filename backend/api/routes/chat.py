"""
POST /chat — Ask a question about an uploaded document via RAG.
"""
from fastapi import APIRouter, HTTPException, status

from models.schemas import ChatRequest, ChatResponse
from services.rag_service import document_exists, generate_answer

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Given a document_id and a question, retrieve relevant passages via
    pgvector similarity search and return an LLM-generated answer with citations.
    """
    # ── 1. Validate document exists ──────────────────────────────────────────
    if not await document_exists(request.document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {request.document_id} was not found.",
        )

    # ── 2. Run RAG pipeline ──────────────────────────────────────────────────
    try:
        answer, citations = await generate_answer(
            document_id=request.document_id,
            question=request.question,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {exc}",
        ) from exc

    return ChatResponse(answer=answer, citations=citations)

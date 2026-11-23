"""
POST /documents — Upload a PDF, extract, embed, and store in Supabase + Postgres.
"""
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from core.config import settings
from core.database import get_connection
from models.schemas import UploadResponse
from services.pdf_service import read_and_chunk
from services.embedding_service import embed_texts
from services.storage_service import upload_pdf, delete_pdf

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_CONTENT_TYPES = {"application/pdf", "binary/octet-stream"}


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PDF file, extract its text, generate embeddings, and persist
    everything to Supabase Storage and the Postgres database.
    """
    # ── 1. Validate content type ─────────────────────────────────────────────
    if file.content_type not in _ALLOWED_CONTENT_TYPES and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted.",
        )

    # ── 2. Read file bytes and enforce size limit ────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum allowed size of "
                   f"{settings.max_upload_bytes // (1024 * 1024)} MB.",
        )

    # ── 3. Extract text and chunk ────────────────────────────────────────────
    try:
        chunks = read_and_chunk(file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # ── 4. Generate embeddings ───────────────────────────────────────────────
    texts = [chunk.text for chunk in chunks]
    try:
        embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding generation failed: {exc}",
        ) from exc

    # ── 5. Upload PDF to Supabase Storage ────────────────────────────────────
    storage_key = f"uploads/{uuid.uuid4()}.pdf"
    try:
        upload_pdf(storage_key, file_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Storage upload failed: {exc}",
        ) from exc

    # ── 6. Persist to database in a single transaction ───────────────────────
    filename = file.filename or "document.pdf"
    try:
        async with get_connection() as conn:
            async with conn.transaction():
                # Insert document row
                doc_row = await conn.execute(
                    """
                    INSERT INTO documents (filename, storage_key)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (filename, storage_key),
                )
                doc = await doc_row.fetchone()
                document_id: int = doc["id"]

                # Insert all chunk rows
                import numpy as np
                from pgvector.psycopg import register_vector_async
                await register_vector_async(conn)

                chunk_records = [
                    (
                        document_id,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.text,
                        np.array(embedding),
                    )
                    for chunk, embedding in zip(chunks, embeddings)
                ]
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        INSERT INTO document_chunks
                            (document_id, page_number, chunk_index, passage_text, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        chunk_records,
                    )

    except Exception as exc:
        # Roll back storage on DB failure
        delete_pdf(storage_key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database write failed and storage was cleaned up: {exc}",
        ) from exc

    # ── 7. Return result ─────────────────────────────────────────────────────
    return UploadResponse(
        document_id=document_id,
        filename=filename,
        chunk_count=len(chunks),
    )

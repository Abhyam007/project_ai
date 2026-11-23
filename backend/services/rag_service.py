"""
RAG service: vector search + LLM answer generation with citation validation.
"""
import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import settings
from core.database import get_connection
from models.schemas import Citation
from services.embedding_service import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: int
    page_number: int
    passage_text: str
    distance: float


async def search_chunks(document_id: int, question_embedding: list[float]) -> list[RetrievedChunk]:
    """
    Run pgvector cosine similarity search filtered to a specific document.
    Returns the top-k closest chunks.
    """
    from pgvector.psycopg import register_vector_async
    import numpy as np

    query_vec = np.array(question_embedding)

    async with get_connection() as conn:
        await register_vector_async(conn)
        rows = await conn.execute(
            """
            SELECT
                id,
                page_number,
                passage_text,
                (embedding <=> %s::vector) AS distance
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY distance
            LIMIT %s
            """,
            (query_vec, document_id, settings.top_k_passages),
        )
        results = await rows.fetchall()

    return [
        RetrievedChunk(
            chunk_id=row["id"],
            page_number=row["page_number"],
            passage_text=row["passage_text"],
            distance=float(row["distance"]),
        )
        for row in results
    ]


async def document_exists(document_id: int) -> bool:
    """Check whether a document row exists in the database."""
    async with get_connection() as conn:
        row = await conn.execute(
            "SELECT 1 FROM documents WHERE id = %s LIMIT 1",
            (document_id,),
        )
        result = await row.fetchone()
    return result is not None


def _build_prompt_sources(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, RetrievedChunk]]:
    """
    Assign S1, S2, … labels to retrieved chunks.
    Returns the formatted source block string and a label→chunk mapping.
    """
    mapping: dict[str, RetrievedChunk] = {}
    lines: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        label = f"S{i}"
        mapping[label] = chunk
        lines.append(f"[{label}] (Page {chunk.page_number}):\n{chunk.passage_text}")

    return "\n\n".join(lines), mapping


_SYSTEM_PROMPT = (
    "You are a precise, helpful AI assistant. "
    "Answer the user's question using ONLY the provided source passages. "
    "Cite every factual claim with its source label in square brackets, e.g. [S1], [S2]. "
    "If a passage does not support a claim, do not cite it. "
    "If the passages do not contain enough information to answer the question, "
    "say so explicitly and do not make up information."
)

_HUMAN_TEMPLATE = (
    "Question: {question}\n\n"
    "Sources:\n{sources}"
)

_llm = ChatGoogleGenerativeAI(
    model=settings.llm_model,
    google_api_key=settings.google_api_key,
)
_parser = StrOutputParser()
_chain = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_TEMPLATE),
]) | _llm | _parser


def _extract_citations(
    answer: str,
    mapping: dict[str, RetrievedChunk],
) -> list[Citation]:
    """
    Find all S1, S2, … labels in the answer text (e.g. [S1] or [S1, S2]).
    Map each valid label to a Citation object. Ignore invalid labels.
    """
    found_labels = set(re.findall(r"\bS\d+\b", answer))
    citations: list[Citation] = []

    for label in sorted(found_labels, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        if label in mapping:
            chunk = mapping[label]
            citations.append(
                Citation(
                    label=label,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                )
            )

    return citations


EVIDENCE_GAP_ANSWER = (
    "The document does not contain enough information to answer this question."
)


async def generate_answer(
    document_id: int,
    question: str,
) -> tuple[str, list[Citation]]:
    """
    Full RAG pipeline:
    1. Embed the question.
    2. Retrieve top-k chunks via pgvector.
    3. If no chunks, return evidence-gap answer.
    4. Build labelled source block.
    5. Call LLM.
    6. Validate and extract citations.
    7. Return (answer, citations).
    """
    # 1. Embed question
    question_embedding = embed_query(question)

    # 2. Retrieve chunks
    chunks = await search_chunks(document_id, question_embedding)

    # 3. Evidence gap
    if not chunks:
        return EVIDENCE_GAP_ANSWER, []

    # 4. Build labelled sources
    sources_text, mapping = _build_prompt_sources(chunks)

    # 5. Call LLM
    answer = await _chain.ainvoke({
        "question": question,
        "sources": sources_text,
    })

    # 6. Extract and validate citations
    citations = _extract_citations(answer, mapping)

    return answer, citations

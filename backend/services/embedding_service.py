"""
Embedding generation service using Google text-embedding-004.
Batches requests to stay within API limits.
"""
from google import genai
from google.genai import types as genai_types

from core.config import settings

# Initialise the Google GenAI client once
_client = genai.Client(api_key=settings.google_api_key)

# Max passages Google allows per embed_content call
_BATCH_SIZE = 100


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Embed a list of text strings.

    Args:
        texts:     List of strings to embed.
        task_type: "RETRIEVAL_DOCUMENT" for indexing, "RETRIEVAL_QUERY" for queries.

    Returns:
        List of embedding vectors (each a list of floats, length 768).
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = _client.models.embed_content(
            model=settings.embedding_model,
            contents=batch,
            config=genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dimensions,
            ),
        )
        all_embeddings.extend([e.values for e in response.embeddings])

    return all_embeddings


def embed_query(question: str) -> list[float]:
    """Embed a single question for similarity search."""
    embeddings = embed_texts([question], task_type="RETRIEVAL_QUERY")
    return embeddings[0]

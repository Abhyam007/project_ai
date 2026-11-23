"""
Supabase Storage service.
Handles uploading and deleting PDFs in the 'Documents' bucket.
"""
from supabase import create_client, Client

from core.config import settings

_BUCKET = "Documents"

# Initialise the Supabase client once
_supabase: Client = create_client(settings.supabase_url, settings.supabase_service_key)


def upload_pdf(storage_key: str, file_bytes: bytes) -> None:
    """
    Upload a PDF to the Documents bucket.

    Args:
        storage_key: Unique path/key inside the bucket (e.g. "uploads/uuid.pdf").
        file_bytes:  Raw bytes of the PDF file.

    Raises:
        RuntimeError: If the Supabase upload fails.
    """
    response = _supabase.storage.from_(_BUCKET).upload(
        path=storage_key,
        file=file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "false"},
    )
    # supabase-py raises an exception on failure, but double-check
    if hasattr(response, "error") and response.error:
        raise RuntimeError(f"Supabase Storage upload failed: {response.error}")


def delete_pdf(storage_key: str) -> None:
    """
    Delete a PDF from the Documents bucket (used on rollback).

    Args:
        storage_key: The same key used during upload.
    """
    try:
        _supabase.storage.from_(_BUCKET).remove([storage_key])
    except Exception:
        # Best-effort cleanup — do not mask the original error
        pass

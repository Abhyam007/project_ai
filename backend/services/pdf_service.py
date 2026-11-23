"""
PDF reading and text chunking service.
Uses PyMuPDF for extraction and LangChain's text splitter for chunking.
Chunks never cross page boundaries so citations remain precise.
"""
from dataclasses import dataclass

import pymupdf as fitz  # PyMuPDF (new non-deprecated API)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings


@dataclass
class Chunk:
    page_number: int   # 1-based
    chunk_index: int   # 0-based index within the whole document
    text: str


def read_and_chunk(file_bytes: bytes) -> list[Chunk]:
    """
    Open a PDF, extract text page-by-page, split each page into chunks.
    Returns a flat list of Chunk objects.

    Raises:
        ValueError: If the file is not a valid PDF or has no extractable text.
    """
    # Open as PDF — raises fitz.FileDataError on corrupt/non-PDF files
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open file as a PDF: {exc}") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    all_chunks: list[Chunk] = []
    global_index = 0
    total_text = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text().strip()
        total_text += len(page_text)

        if not page_text:
            continue  # Skip blank pages

        page_chunks = splitter.split_text(page_text)
        for chunk_text in page_chunks:
            if chunk_text.strip():
                all_chunks.append(
                    Chunk(
                        page_number=page_num + 1,  # Convert to 1-based
                        chunk_index=global_index,
                        text=chunk_text.strip(),
                    )
                )
                global_index += 1

    doc.close()

    if total_text == 0:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned document that requires OCR before uploading."
        )

    if not all_chunks:
        raise ValueError("Could not split the PDF into any usable text chunks.")

    return all_chunks

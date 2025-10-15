"""
Utility functions for text processing:
- clean_text: normalize whitespace, remove control chars
- chunk_text: split long text into overlapping chunks
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and invisible characters."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for embedding.

    Args:
        text: The full text string.
        chunk_size: Max characters per chunk.
        overlap: Number of characters overlapped between chunks.

    Returns:
        List of text chunks.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap  # step back for overlap

    return chunks

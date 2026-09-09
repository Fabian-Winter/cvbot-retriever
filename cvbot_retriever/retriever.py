"""Retrieval of the chunks that are relevant for a question."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

LOGGER = logging.getLogger(__name__)


def retrieve(store: Any, question: str, top_k: int) -> list[Document]:
    """Looks up the chunks that match a question.

    The store embeds the question with the model it was opened with and returns
    the nearest chunks including the metadata written by cvbot-embedder
    (``source``, ``filename``, ``chunk_index`` and the Markdown headers).

    Args:
        store: Source store exposing a ``similarity_search`` method.
        question: The user question.
        top_k: Number of chunks to retrieve.

    Returns:
        The matching chunks, ordered by decreasing similarity.

    Raises:
        ValueError: If the question is empty or ``top_k`` is not positive.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError(f"top_k must be positive: {top_k}")

    chunks = store.similarity_search(question, k=top_k)
    LOGGER.info("retrieved %d chunk(s) for top_k=%d", len(chunks), top_k)
    return chunks

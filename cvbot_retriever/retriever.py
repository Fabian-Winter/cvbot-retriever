"""Retrieval of the chunks that are relevant for a question."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from cvbot_core.metadata import split_values
from cvbot_core.protocols import VectorStoreReader
from langchain_core.documents import Document
from .config import DEFAULT_FILTER_OVERFETCH_FACTOR

LOGGER = logging.getLogger(__name__)


def retrieve(
    store: VectorStoreReader,
    question: str,
    top_k: int,
    filters: Mapping[str, Sequence[str]] | None = None,
    overfetch_factor: int = DEFAULT_FILTER_OVERFETCH_FACTOR,
) -> list[Document]:
    """Looks up the chunks that match a question.

    The store embeds the question with the model it was opened with and returns
    the nearest chunks including the metadata written by cvbot-embedder
    (``source``, ``filename``, ``chunk_index``, the Markdown headers and the
    section metadata).

    Filters only re-rank: a matching chunk moves up, a chunk that lacks the
    field keeps its place. Nothing is ever excluded, so incomplete metadata and
    filters without a single match both degrade into plain semantic search.

    Args:
        store: Source store the chunks are read from.
        question: The user question.
        top_k: Number of chunks to retrieve.
        filters: Metadata fields mapped onto the values to boost.
        overfetch_factor: How many times ``top_k`` is fetched before re-ranking,
            so that a boosted chunk outside the first ``top_k`` can still win.

    Returns:
        The matching chunks, ordered by boost and then by decreasing
        similarity.

    Raises:
        ValueError: If the question is empty or ``top_k`` is not positive.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError(f"top_k must be positive: {top_k}")

    if not filters:
        chunks = store.similarity_search(question, k=top_k)
        LOGGER.info("retrieved %d chunk(s) for top_k=%d", len(chunks), top_k)
        return chunks

    candidates = store.similarity_search(question, k=top_k * max(overfetch_factor, 1))
    scores = [_boost_score(chunk, filters) for chunk in candidates]
    # Python's sort is stable, so the similarity order survives as tie-breaker.
    ranked = [
        chunk
        for _, chunk in sorted(
            zip(scores, candidates, strict=True),
            key=lambda pair: -pair[0],
        )
    ]
    LOGGER.info(
        "retrieved %d chunk(s) for top_k=%d, %d boosted by %d filter field(s)",
        len(ranked[:top_k]),
        top_k,
        sum(1 for score in scores if score),
        len(filters),
    )
    return ranked[:top_k]


def _boost_score(chunk: Document, filters: Mapping[str, Sequence[str]]) -> int:
    """Counts how many filter fields a chunk satisfies.

    A missing field scores zero rather than negative, which is what keeps
    chunks without metadata eligible.

    Args:
        chunk: The retrieved chunk.
        filters: Metadata fields mapped onto the values to boost.

    Returns:
        The number of matching fields.
    """
    score = 0
    for name, wanted in filters.items():
        value = chunk.metadata.get(name)
        if not isinstance(value, str):
            continue
        if any(candidate in wanted for candidate in split_values(value)):
            score += 1
    return score

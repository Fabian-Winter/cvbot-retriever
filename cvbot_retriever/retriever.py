"""Retrieval of the chunks that are relevant for a question.

This is the retrieval stage of the pipeline: it validates the request, asks
the store for candidates and hands them to the ranking stage. All scoring
knowledge lives in ``ranking``; this module only decides how many candidates
to fetch so that the ranking has room to promote a fresher or matching chunk.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from cvbot_core.protocols import VectorStoreReader
from langchain_core.documents import Document

from .config import RankingConfig
from .ranking import rerank

LOGGER = logging.getLogger(__name__)


def retrieve(
    store: VectorStoreReader,
    question: str,
    top_k: int,
    config: RankingConfig,
    *,
    boost: Mapping[str, Sequence[str]] | None = None,
) -> list[Document]:
    """Looks up the chunks that match a question.

    The store embeds the question with the model it was opened with and returns
    the nearest chunks together with their vector distance, including the
    metadata written by cvbot-embedder. ``top_k`` is over-fetched by
    ``config.overfetch_factor`` before ``rerank`` orders the candidates, so a
    chunk that only wins through its boost or recency is still in the pool.

    Args:
        store: Source store the chunks are read from.
        question: The user question.
        top_k: Number of chunks to return.
        config: Weights, over-fetch factor and window of the ranking.
        boost: Metadata fields mapped onto the values to boost.

    Returns:
        The matching chunks, ordered by decreasing total score. Ties keep the
        order of the store, which is the similarity order.

    Raises:
        ValueError: If the question is empty or ``top_k`` is not positive.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError(f"top_k must be positive: {top_k}")

    candidates = store.similarity_search_with_score(
        question, k=top_k * config.overfetch_factor
    )
    ranked = rerank(candidates, config, boost)[:top_k]
    LOGGER.info(
        "retrieved %d chunk(s) for top_k=%d from %d candidate(s)",
        len(ranked),
        top_k,
        len(candidates),
    )
    return ranked

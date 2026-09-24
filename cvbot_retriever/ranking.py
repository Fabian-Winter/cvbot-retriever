"""Re-ranking of retrieved candidates by relevance, boost and recency.

This is the ranking stage of the pipeline: it receives the raw candidates of
a similarity search and orders them by a total score made of three additive
parts, each bounded to ``(0, 1]`` so that the configured weights stay
comparable:

- ``relevance``: the distance mapped onto ``(0, 1]`` via ``1 / (1 + d)``, so a
  closer chunk always contributes more.
- ``boost factor``: ``filter_weight`` scaled by the share of the extracted
  boost fields the chunk satisfies, so a chunk matching the boost moves up
  without ever outranking relevance through field count alone.
- ``recency factor``: ``recency_weight`` scaled by how recently the chunk's
  period ended, derived at query time from the existing ``startdate``/``enddate``/
  ``status`` metadata through the shared rule in cvbot_core. No re-indexing is
  involved, and a chunk without any usable date scores zero rather than
  negative.

Because the bonuses are bounded, the retrieval stage over-fetches before this
stage runs, so a fresher or matching chunk can win from further down the
similarity ranking. Replacing this stage, e.g. by a cross-encoder reranker,
only means replacing this module.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from cvbot_core.metadata import period_end_year, split_values
from langchain_core.documents import Document

from cvbot_core.metadata import (
    current_year,
)

from .config import RankingConfig

LOGGER = logging.getLogger(__name__)


def rerank(
    candidates: Sequence[tuple[Document, float]],
    config: RankingConfig,
    boost: Mapping[str, Sequence[str]] | None = None,
) -> list[Document]:
    """Orders the candidates by decreasing total score.

    Args:
        candidates: Chunks with their vector distance; lower means closer.
            Expected in increasing distance order, which the store guarantees.
        config: Weights and window of the ranking.
        boost: Metadata fields mapped onto the values to boost.

    Returns:
        The chunks in ``candidates``, ordered by decreasing total score. Ties
        keep the order of the candidates, which is the similarity order.
    """
    now_year = current_year()
    scores = [
        _total_score(
            chunk,
            distance,
            boost=boost,
            config=config,
            now_year=now_year,
        )
        for chunk, distance in candidates
    ]
    # Python's sort is stable, so the similarity order survives as tie-breaker.
    ranked = [
        chunk
        for (chunk, _), _ in sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: -item[1],
        )
    ]
    boosted = sum(
        1
        for (_, distance), score in zip(candidates, scores, strict=True)
        if score > _relevance(distance)
    )
    LOGGER.info(
        "re-ranked %d candidate(s), %d with a bonus", len(candidates), boosted
    )
    return ranked


def _total_score(
    chunk: Document,
    distance: float,
    *,
    boost: Mapping[str, Sequence[str]] | None,
    config: RankingConfig,
    now_year: int,
) -> float:
    """Combines similarity, boost and recency into one score.

    Args:
        chunk: The retrieved chunk.
        distance: Its vector distance; lower means closer.
        boost: Metadata fields mapped onto the values to boost.
        config: Weights and window of the ranking.
        now_year: The current year, passed in so tests stay deterministic.

    Returns:
        The total ranking score.
    """
    score = _relevance(distance)
    if boost:
        score += config.filter_weight * _boost_factor(chunk, boost)
    score += config.recency_weight * _recency_factor(
        chunk.metadata, now_year, config.recency_window_years
    )
    return score


def _relevance(distance: float) -> float:
    """Maps a vector distance onto a relevance in ``(0, 1]``.

    Args:
        distance: The vector distance, lower means closer.

    Returns:
        The relevance, monotonically decreasing in the distance.
    """
    return 1.0 / (1.0 + max(distance, 0.0))


def _boost_factor(chunk: Document, boost: Mapping[str, Sequence[str]]) -> float:
    """Rates how many of the boost fields a chunk satisfies, from 0 to 1.

    The share rather than the raw count keeps the bonus on the same scale as
    the other signals, so ``filter_weight`` stays comparable to the similarity
    score no matter how many fields were extracted. A missing field scores
    zero rather than negative, which is what keeps chunks without metadata
    eligible.

    Args:
        chunk: The retrieved chunk.
        boost: Metadata fields mapped onto the values to boost.

    Returns:
        The matched share of the boost fields between ``0.0`` and ``1.0``.
    """
    matches = 0
    for name, wanted in boost.items():
        value = chunk.metadata.get(name)
        if not isinstance(value, str):
            continue
        if any(candidate in wanted for candidate in split_values(value)):
            matches += 1
    return matches / len(boost)


def _recency_factor(
    metadata: Mapping[str, Any], now_year: int, window_years: int
) -> float:
    """Rates how recently a chunk's period ended, on a scale from 0 to 1.

    The end year decays linearly: the current year scores ``1.0``, the year
    ``window_years`` back scores ``0.0``, anything older stays at ``0.0``. A
    chunk without any usable date is neutral rather than penalised, which is
    what keeps the undated sections of the documents competitive.

    Args:
        metadata: The metadata of the chunk.
        now_year: The current year, passed in so tests stay deterministic.
        window_years: How many years back the bonus decays to zero.

    Returns:
        The recency factor between ``0.0`` and ``1.0``.
    """
    end_year = period_end_year(metadata, now_year)
    if end_year is None:
        return 0.0
    factor = (end_year - (now_year - window_years)) / window_years
    return min(max(factor, 0.0), 1.0)

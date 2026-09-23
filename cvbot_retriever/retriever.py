"""Retrieval of the chunks that are relevant for a question."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from cvbot_core.metadata import (
    OPEN_PERIOD_MARKERS,
    PERIOD_END_KEY,
    PERIOD_START_KEY,
    STATUS_KEY,
    normalize_value,
    parse_period_year,
    split_values,
)
from cvbot_core.protocols import VectorStoreReader
from langchain_core.documents import Document

from .config import (
    DEFAULT_FILTER_WEIGHT,
    DEFAULT_OVERFETCH_FACTOR,
    DEFAULT_RECENCY_WEIGHT,
    DEFAULT_RECENCY_WINDOW_YEARS,
)

LOGGER = logging.getLogger(__name__)


def retrieve(
    store: VectorStoreReader,
    question: str,
    top_k: int,
    filters: Mapping[str, Sequence[str]] | None = None,
    *,
    overfetch_factor: int = DEFAULT_OVERFETCH_FACTOR,
    filter_weight: float = DEFAULT_FILTER_WEIGHT,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    recency_window_years: int = DEFAULT_RECENCY_WINDOW_YEARS,
) -> list[Document]:
    """Looks up the chunks that match a question.

    The store embeds the question with the model it was opened with and returns
    the nearest chunks together with their vector distance, including the
    metadata written by cvbot-embedder. The raw distance order is then re-ranked
    by a total score made of three additive parts:

    - ``relevance``: the distance mapped onto ``(0, 1]`` via ``1 / (1 + d)``,
      so a closer chunk always contributes more.
    - ``filter bonus``: ``filter_weight`` per metadata field the chunk
      satisfies, so a chunk matching the extracted filters moves up.
    - ``recency bonus``: ``recency_weight`` scaled by how recently the chunk's
      period ended, derived at query time from the existing ``from``/``to``/
      ``status`` metadata. No re-indexing is involved, and a chunk without any
      usable date scores zero rather than negative.

    Because the bonuses are bounded, over-fetching ``top_k`` by
    ``overfetch_factor`` before the re-rank lets a fresher or matching chunk win
    from further down the similarity ranking.

    Args:
        store: Source store the chunks are read from.
        question: The user question.
        top_k: Number of chunks to retrieve.
        filters: Metadata fields mapped onto the values to boost.
        overfetch_factor: How many times ``top_k`` is fetched before
            re-ranking.
        filter_weight: Score added per matching filter field.
        recency_weight: Largest score the recency bonus can add; ``0`` turns
            the bonus off entirely.
        recency_window_years: How many years back the recency bonus decays
            linearly to zero.

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
        question, k=top_k * max(overfetch_factor, 1)
    )
    now_year = _current_year()
    scores = [
        _total_score(
            chunk,
            distance,
            filters=filters,
            filter_weight=filter_weight,
            recency_weight=recency_weight,
            recency_window_years=recency_window_years,
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
        "retrieved %d chunk(s) for top_k=%d from %d candidate(s), %d with a bonus",
        len(ranked[:top_k]),
        top_k,
        len(candidates),
        boosted,
    )
    return ranked[:top_k]


def _total_score(
    chunk: Document,
    distance: float,
    *,
    filters: Mapping[str, Sequence[str]] | None,
    filter_weight: float,
    recency_weight: float,
    recency_window_years: int,
    now_year: int,
) -> float:
    """Combines similarity, filter boost and recency into one score.

    Args:
        chunk: The retrieved chunk.
        distance: Its vector distance; lower means closer.
        filters: Metadata fields mapped onto the values to boost.
        filter_weight: Score added per matching filter field.
        recency_weight: Largest score the recency bonus can add.
        recency_window_years: Years back the recency bonus decays to zero.
        now_year: The current year, passed in so tests stay deterministic.

    Returns:
        The total ranking score.
    """
    score = _relevance(distance)
    if filters:
        score += filter_weight * _filter_matches(chunk, filters)
    score += recency_weight * _recency_factor(
        chunk.metadata, now_year, recency_window_years
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


def _filter_matches(chunk: Document, filters: Mapping[str, Sequence[str]]) -> int:
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
    end_year = _end_year(metadata, now_year)
    if end_year is None:
        return 0.0
    factor = (end_year - (now_year - window_years)) / window_years
    return min(max(factor, 0.0), 1.0)


def _end_year(metadata: Mapping[str, Any], now_year: int) -> int | None:
    """Determines the last year a chunk speaks about, or ``None``.

    Mirrors the rules cvbot-embedder applied when it derived the ``years``
    field, so the ranking and the published filter values agree: an open-ended
    ``to`` (``now``, ``laufend``, absent) reaches into the present, a concrete
    one ends at its year, and ``status: current`` marks the undated sections as
    up to date.

    Args:
        metadata: The metadata of the chunk.
        now_year: The current year, passed in so tests stay deterministic.

    Returns:
        The end year, or ``None`` if the metadata carries no usable signal.
    """
    end_raw = metadata.get(PERIOD_END_KEY)
    end_text = end_raw if isinstance(end_raw, str) else None
    if end_text is not None and normalize_value(end_text) in OPEN_PERIOD_MARKERS:
        return now_year
    parsed = parse_period_year(end_text)
    if parsed is not None:
        return parsed

    start_raw = metadata.get(PERIOD_START_KEY)
    start = parse_period_year(start_raw if isinstance(start_raw, str) else None)
    if end_text is None and start is not None:
        # Absent end reads as "still running", exactly like the year list the
        # embedder derived from the same fields.
        return now_year
    if start is not None:
        # Unparsable end: conservative, like the embedder.
        return start

    status = metadata.get(STATUS_KEY)
    if isinstance(status, str) and normalize_value(status) in OPEN_PERIOD_MARKERS:
        return now_year
    return None


def _current_year() -> int:
    """Returns the current year; separated out so tests can pin it."""
    return datetime.now(UTC).year

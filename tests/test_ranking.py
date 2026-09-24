"""Tests for the re-ranking stage."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from cvbot_retriever import ranking
from cvbot_retriever.config import RankingConfig
from cvbot_retriever.ranking import rerank

NOW_YEAR = 2026


@pytest.fixture(autouse=True)
def pinned_year(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the current year the recency bonus decays against."""
    monkeypatch.setattr(ranking, "current_year", lambda: NOW_YEAR)


def build_candidates(
    *fields: dict[str, str], distances: list[float] | None = None
) -> list[tuple[Document, float]]:
    """Creates candidates carrying the given metadata and distances.

    Args:
        *fields: Metadata per candidate; the page content is ``Chunk <i>``.
        distances: Distance per candidate; defaults to ``0.1``, ``0.2``, ...

    Returns:
        Candidates in the shape ``rerank`` receives them from the store.
    """
    resolved = (
        distances
        if distances is not None
        else [0.1 * (index + 1) for index in range(len(fields))]
    )
    return [
        (
            Document(page_content=f"Chunk {index}", metadata=dict(metadata)),
            distance,
        )
        for index, (metadata, distance) in enumerate(zip(fields, resolved, strict=True))
    ]


def contents(candidates: list[tuple[Document, float]]) -> list[str]:
    """Extracts the page contents of the candidates in order."""
    return [chunk.page_content for chunk in rerank(candidates, RankingConfig())]


def test_closer_chunks_win_without_any_bonus() -> None:
    candidates = build_candidates({"enddate": "2010"}, {})

    assert contents(candidates) == ["Chunk 0", "Chunk 1"]


def test_empty_boost_keeps_the_plain_similarity_order() -> None:
    candidates = build_candidates({"status": "aktuell"}, {"status": "historisch"})

    assert [chunk.page_content for chunk in rerank(candidates, RankingConfig(), {})] == [
        "Chunk 0",
        "Chunk 1",
    ]


def test_a_matching_chunk_is_boosted_to_the_front() -> None:
    candidates = build_candidates(
        {"status": "historisch"}, {"status": "historisch"}, {"status": "aktuell"}
    )

    result = rerank(candidates, RankingConfig(), {"status": ["aktuell"]})

    assert [chunk.page_content for chunk in result] == [
        "Chunk 2",
        "Chunk 0",
        "Chunk 1",
    ]


def test_a_chunk_without_the_field_stays_eligible() -> None:
    candidates = build_candidates({}, {"status": "historisch"})

    result = rerank(candidates, RankingConfig(), {"status": ["aktuell"]})

    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_matching_more_fields_ranks_higher() -> None:
    candidates = build_candidates(
        {"status": "aktuell"}, {"status": "aktuell", "ort": "berlin"}
    )

    result = rerank(
        candidates, RankingConfig(), {"status": ["aktuell"], "ort": ["berlin"]}
    )

    assert [chunk.page_content for chunk in result] == ["Chunk 1", "Chunk 0"]


def test_a_boost_without_any_match_keeps_the_semantic_order() -> None:
    candidates = build_candidates({"status": "historisch"}, {"status": "historisch"})

    result = rerank(candidates, RankingConfig(), {"status": ["aktuell"]})

    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_multi_value_metadata_matches_a_single_boost_value() -> None:
    candidates = build_candidates({"tech": "Python, Java"}, {"tech": "Go"})

    result = rerank(candidates, RankingConfig(), {"tech": ["java"]})

    assert result[0].page_content == "Chunk 0"


def test_the_boost_bonus_is_bounded_by_its_weight() -> None:
    # Matching every one of several fields must not exceed filter_weight, so a
    # chunk cannot outrank a much closer one through field count alone. With
    # the old unbounded count this would add 3 * 0.2 and win.
    candidates = build_candidates(
        {}, {"a": "x", "b": "y", "c": "z"}, distances=[0.05, 0.5]
    )

    result = rerank(
        candidates,
        RankingConfig(filter_weight=0.2),
        {"a": ["x"], "b": ["y"], "c": ["z"]},
    )

    # The closer chunk keeps the lead: its relevance edge beats the capped bonus.
    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_a_newer_chunk_beats_a_slightly_closer_older_one() -> None:
    candidates = build_candidates(
        {"startdate": "2012", "enddate": "2014"},
        {"startdate": "2025", "enddate": "2026"},
        distances=[0.10, 0.11],
    )

    result = rerank(candidates, RankingConfig())

    assert [chunk.page_content for chunk in result] == ["Chunk 1", "Chunk 0"]


def test_an_open_ended_period_counts_as_current() -> None:
    candidates = build_candidates(
        {"enddate": "2016"}, {"enddate": "laufend"}, distances=[0.10, 0.11]
    )

    result = rerank(candidates, RankingConfig())

    assert [chunk.page_content for chunk in result] == ["Chunk 1", "Chunk 0"]


def test_a_chunk_without_any_date_stays_neutral() -> None:
    candidates = build_candidates({}, {"enddate": "now"}, distances=[0.05, 0.40])

    result = rerank(candidates, RankingConfig())

    # The undated chunk is closer and loses nothing by having no date.
    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_is_current_alone_carries_no_recency_bonus() -> None:
    # iscurrent is a boost field only; recency keeps reading the period, so an
    # undated chunk is not treated as recent by the marker.
    candidates = build_candidates({}, {"iscurrent": "true"}, distances=[0.05, 0.40])

    result = rerank(candidates, RankingConfig())

    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_a_period_beyond_the_window_gets_no_bonus() -> None:
    candidates = build_candidates(
        {"enddate": str(NOW_YEAR - 10)},
        {"enddate": str(NOW_YEAR - 11)},
        distances=[0.10, 0.11],
    )

    result = rerank(candidates, RankingConfig())

    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_a_zero_recency_weight_restores_the_pure_similarity_order() -> None:
    candidates = build_candidates(
        {"enddate": "2014"}, {"enddate": "now"}, distances=[0.10, 0.11]
    )

    result = rerank(candidates, RankingConfig(recency_weight=0.0))

    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_a_higher_filter_weight_outweighs_a_smaller_distance() -> None:
    candidates = build_candidates({"status": "historisch"}, {"status": "aktuell"})

    result = rerank(
        candidates, RankingConfig(filter_weight=0.5), {"status": ["aktuell"]}
    )

    assert [chunk.page_content for chunk in result] == ["Chunk 1", "Chunk 0"]

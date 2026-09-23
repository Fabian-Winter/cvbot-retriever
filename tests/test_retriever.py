"""Tests for the retrieval stage."""

from __future__ import annotations

import pytest

from cvbot_retriever.config import RankingConfig
from cvbot_retriever.retriever import retrieve
from tests.conftest import FakeStore, make_documents


def test_retrieve_overfetchs_before_re_ranking() -> None:
    chunks = make_documents("Studied computer science.", "Worked as a developer.")
    store = FakeStore(chunks)

    result = retrieve(store, "What did the candidate study?", 2, RankingConfig())

    assert store.queries == [("What did the candidate study?", 8)]
    assert result == chunks


def test_retrieve_honours_the_overfetch_factor_of_the_config() -> None:
    store = FakeStore(make_documents("A.", "B.", "C.", "D.", "E.", "F."))

    retrieve(store, "Frage?", 2, RankingConfig(overfetch_factor=3))

    assert store.queries == [("Frage?", 6)]


def test_retrieve_keeps_the_embedder_metadata() -> None:
    store = FakeStore(make_documents("Studied computer science.", source="cv.md"))

    [chunk] = retrieve(store, "Education?", 1, RankingConfig())

    assert chunk.metadata["source"] == "cv.md"
    assert chunk.metadata["chunk_index"] == 0


def test_retrieve_without_hits_returns_empty_list() -> None:
    store = FakeStore([])

    assert retrieve(store, "Anything?", 3, RankingConfig()) == []


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_raises(question: str) -> None:
    with pytest.raises(ValueError, match="question"):
        retrieve(FakeStore([]), question, 1, RankingConfig())


def test_non_positive_top_k_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve(FakeStore([]), "Education?", 0, RankingConfig())


def test_retrieve_returns_at_most_top_k_chunks() -> None:
    store = FakeStore(make_documents(*[f"Chunk {i}" for i in range(10)]))

    result = retrieve(store, "Frage?", 2, RankingConfig())

    assert len(result) == 2


def test_a_boosted_chunk_from_the_overfetch_wins_a_slot() -> None:
    # The matching chunk sits at distance 0.3, behind the two closer ones.
    # Without over-fetching and re-ranking it would never make top_k = 2.
    store = FakeStore(
        make_documents("A.", "B.", "Treffer."),
        distances=[0.1, 0.2, 0.3],
    )
    store.documents[2].metadata.update({"status": "aktuell"})

    result = retrieve(
        store, "Frage?", 2, RankingConfig(filter_weight=0.5), boost={"status": ["aktuell"]}
    )

    assert [chunk.page_content for chunk in result] == ["Treffer.", "A."]

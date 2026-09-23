"""Tests for the retriever."""

from __future__ import annotations

import pytest

from cvbot_retriever import retriever
from cvbot_retriever.retriever import retrieve
from tests.conftest import FakeStore, make_documents

NOW_YEAR = 2026


@pytest.fixture(autouse=True)
def pinned_year(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the current year the recency bonus decays against."""
    monkeypatch.setattr(retriever, "_current_year", lambda: NOW_YEAR)


def test_retrieve_overfetchs_before_re_ranking() -> None:
    chunks = make_documents("Studied computer science.", "Worked as a developer.")
    store = FakeStore(chunks)

    result = retrieve(store, "What did the candidate study?", top_k=2)

    assert store.queries == [("What did the candidate study?", 8)]
    assert result == chunks


def test_retrieve_keeps_the_embedder_metadata() -> None:
    store = FakeStore(make_documents("Studied computer science.", source="cv.md"))

    [chunk] = retrieve(store, "Education?", top_k=1)

    assert chunk.metadata["source"] == "cv.md"
    assert chunk.metadata["chunk_index"] == 0


def test_retrieve_without_hits_returns_empty_list() -> None:
    store = FakeStore([])

    assert retrieve(store, "Anything?", top_k=3) == []


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_raises(question: str) -> None:
    with pytest.raises(ValueError, match="question"):
        retrieve(FakeStore([]), question, top_k=1)


def test_non_positive_top_k_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve(FakeStore([]), "Education?", top_k=0)


def build_store(*fields: dict[str, str]) -> FakeStore:
    """Creates a store whose chunks carry the given metadata."""
    chunks = make_documents(*[f"Chunk {index}" for index in range(len(fields))])
    for chunk, metadata in zip(chunks, fields, strict=True):
        chunk.metadata.update(metadata)
    return FakeStore(chunks)


def test_closer_chunks_win_without_any_bonus() -> None:
    store = build_store({"to": "2010"}, {})

    result = retrieve(store, "Frage?", top_k=2)

    # Chunk 0 has the smaller distance, and its old end year only ever adds.
    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_empty_filters_keep_the_plain_search() -> None:
    store = build_store({"status": "aktuell"}, {"status": "historisch"})

    result = retrieve(store, "Frage?", top_k=2, filters={})

    assert store.queries == [("Frage?", 8)]
    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_a_matching_chunk_is_boosted_to_the_front() -> None:
    store = build_store(
        {"status": "historisch"}, {"status": "historisch"}, {"status": "aktuell"}
    )

    result = retrieve(store, "Frage?", top_k=2, filters={"status": ["aktuell"]})

    assert [chunk.page_content for chunk in result] == ["Chunk 2", "Chunk 0"]


def test_a_chunk_without_the_field_stays_eligible() -> None:
    store = build_store({}, {"status": "historisch"})

    result = retrieve(store, "Frage?", top_k=2, filters={"status": ["aktuell"]})

    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_matching_more_fields_ranks_higher() -> None:
    store = build_store(
        {"status": "aktuell"}, {"status": "aktuell", "ort": "berlin"}
    )

    result = retrieve(
        store, "Frage?", top_k=2, filters={"status": ["aktuell"], "ort": ["berlin"]}
    )

    assert [chunk.page_content for chunk in result] == ["Chunk 1", "Chunk 0"]


def test_a_filter_without_any_match_keeps_the_semantic_order() -> None:
    store = build_store({"status": "historisch"}, {"status": "historisch"})

    result = retrieve(store, "Frage?", top_k=2, filters={"status": ["aktuell"]})

    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_multi_value_metadata_matches_a_single_filter_value() -> None:
    store = build_store({"tech": "Python, Java"}, {"tech": "Go"})

    result = retrieve(store, "Frage?", top_k=2, filters={"tech": ["java"]})

    assert result[0].page_content == "Chunk 0"


def test_a_newer_chunk_beats_a_slightly_closer_older_one() -> None:
    store = FakeStore(
        make_documents("Alt.", "Neu."),
        distances=[0.10, 0.11],
    )
    store.documents[0].metadata.update({"from": "2012", "to": "2014"})
    store.documents[1].metadata.update({"from": "2025", "to": "2026"})

    result = retrieve(store, "Frage?", top_k=1)

    assert [chunk.page_content for chunk in result] == ["Neu."]


def test_an_open_ended_period_counts_as_current() -> None:
    store = FakeStore(
        make_documents("Vergangen.", "Laeuft."),
        distances=[0.10, 0.11],
    )
    store.documents[0].metadata.update({"to": "2016"})
    store.documents[1].metadata.update({"to": "laufend"})

    result = retrieve(store, "Frage?", top_k=1)

    assert [chunk.page_content for chunk in result] == ["Laeuft."]


def test_status_current_marks_an_undated_chunk_as_recent() -> None:
    store = FakeStore(
        make_documents("Ohne Datum.",),
        distances=[0.5],
    )
    store.documents[0].metadata.update({"status": "current"})

    result = retrieve(store, "Frage?", top_k=1)

    assert [chunk.page_content for chunk in result] == ["Ohne Datum."]


def test_a_chunk_without_any_date_stays_neutral() -> None:
    store = FakeStore(
        make_documents("Ohne Datum.", "Aktuell."),
        distances=[0.05, 0.40],
    )
    store.documents[1].metadata.update({"to": "now"})

    result = retrieve(store, "Frage?", top_k=2)

    # The undated chunk is closer and loses nothing by having no date.
    assert [chunk.page_content for chunk in result] == ["Ohne Datum.", "Aktuell."]


def test_a_period_beyond_the_window_gets_no_bonus() -> None:
    store = FakeStore(
        make_documents("Uralte.", "Knapp ausserhalb."),
        distances=[0.10, 0.11],
    )
    store.documents[0].metadata.update({"to": str(NOW_YEAR - 10)})
    store.documents[1].metadata.update({"to": str(NOW_YEAR - 11)})

    result = retrieve(store, "Frage?", top_k=2)

    assert [chunk.page_content for chunk in result] == ["Uralte.", "Knapp ausserhalb."]


def test_a_zero_recency_weight_restores_the_pure_similarity_order() -> None:
    store = FakeStore(
        make_documents("Aeltere, naeher.", "Neuere, ferner."),
        distances=[0.10, 0.11],
    )
    store.documents[0].metadata.update({"to": "2014"})
    store.documents[1].metadata.update({"to": "now"})

    result = retrieve(store, "Frage?", top_k=2, recency_weight=0.0)

    assert [chunk.page_content for chunk in result] == [
        "Aeltere, naeher.",
        "Neuere, ferner.",
    ]


def test_a_higher_filter_weight_outweighs_a_smaller_distance() -> None:
    store = build_store({"status": "historisch"}, {"status": "aktuell"})

    result = retrieve(
        store, "Frage?", top_k=1, filters={"status": ["aktuell"]}, filter_weight=0.5
    )

    assert [chunk.page_content for chunk in result] == ["Chunk 1"]

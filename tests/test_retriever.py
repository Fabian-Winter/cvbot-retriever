"""Tests for the retriever."""

from __future__ import annotations

import pytest

from cvbot_retriever.retriever import retrieve
from tests.conftest import FakeStore, make_documents


def test_retrieve_queries_the_store_with_top_k() -> None:
    chunks = make_documents("Studied computer science.", "Worked as a developer.")
    store = FakeStore(chunks)

    result = retrieve(store, "What did the candidate study?", top_k=2)

    assert store.queries == [("What did the candidate study?", 2)]
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


def test_empty_filters_keep_the_plain_search() -> None:
    store = build_store({"status": "aktuell"}, {"status": "historisch"})

    result = retrieve(store, "Frage?", top_k=2, filters={})

    assert store.queries == [("Frage?", 2)]
    assert [chunk.page_content for chunk in result] == ["Chunk 0", "Chunk 1"]


def test_filters_overfetch_before_re_ranking() -> None:
    store = build_store({"status": "aktuell"})

    retrieve(store, "Frage?", top_k=2, filters={"status": ["aktuell"]}, overfetch_factor=3)

    assert store.queries == [("Frage?", 6)]


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

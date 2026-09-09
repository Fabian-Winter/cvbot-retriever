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

"""Tests for the ChromaDB access layer."""

from __future__ import annotations

import pytest

from cvbot_retriever import vector_store
from cvbot_retriever.config import Settings
from tests.conftest import FakeEmbeddings


class RecordingChromaClient:
    """Client double that fails the test when the collection is modified."""

    def delete_collection(self, name: str) -> None:
        """Raises because the retriever must never modify the collection."""
        raise AssertionError(f"collection {name!r} must not be deleted")

    def create_collection(self, *args: object, **kwargs: object) -> None:
        """Raises because the retriever must never modify the collection."""
        raise AssertionError("collection must not be created")


def test_create_client_passes_connection_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        vector_store.chromadb, "HttpClient", lambda **kw: captured.update(kw)
    )
    settings = Settings(chroma_host="chroma.internal", chroma_port=8443)

    vector_store.create_client(settings)

    assert captured["host"] == "chroma.internal"
    assert captured["port"] == 8443


def test_open_collection_wires_client_and_embeddings(
    monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> None:
    monkeypatch.setattr(vector_store, "Chroma", lambda **kw: kw)
    client = RecordingChromaClient()

    store = vector_store.open_collection(client, "jobs", fake_embeddings)

    assert store["client"] is client
    assert store["collection_name"] == "jobs"
    assert store["embedding_function"] is fake_embeddings

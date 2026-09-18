"""Tests for the ChromaDB access layer."""

from __future__ import annotations

import cvbot_core.vector_store
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
        cvbot_core.vector_store.chromadb, "HttpClient", lambda **kw: captured.update(kw)
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
    assert store["create_collection_if_not_exists"] is False


class FakeCollection:
    def __init__(self, metadata: dict[str, object] | None) -> None:
        self.metadata = metadata


class FakeClientWithCollection:
    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection

    def get_collection(self, name: str) -> FakeCollection:
        self.requested_name = name
        return self._collection


def test_get_indexed_embedding_model_id_reads_the_collection_metadata() -> None:
    client = FakeClientWithCollection(
        FakeCollection({"embedding_model_id": "amazon.titan-embed-text-v2:0"})
    )

    model_id = vector_store.get_indexed_embedding_model_id(client, "jobs")

    assert model_id == "amazon.titan-embed-text-v2:0"
    assert client.requested_name == "jobs"


def test_get_indexed_embedding_model_id_raises_when_metadata_is_missing() -> None:
    client = FakeClientWithCollection(FakeCollection(None))

    with pytest.raises(RuntimeError, match="jobs"):
        vector_store.get_indexed_embedding_model_id(client, "jobs")


def test_get_indexed_metadata_schema_reads_the_collection_metadata() -> None:
    client = FakeClientWithCollection(
        FakeCollection({"metadata_schema": '{"status":["aktuell"]}'})
    )

    assert vector_store.get_indexed_metadata_schema(client, "jobs") == {
        "status": ["aktuell"]
    }


def test_get_indexed_metadata_schema_is_empty_for_an_older_collection() -> None:
    client = FakeClientWithCollection(
        FakeCollection({"embedding_model_id": "amazon.titan-embed-text-v2:0"})
    )

    assert vector_store.get_indexed_metadata_schema(client, "jobs") == {}

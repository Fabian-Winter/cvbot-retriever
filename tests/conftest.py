"""Shared fixtures and test doubles."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cvbot_retriever.config import Settings


class FakeEmbeddings(Embeddings):
    """Deterministic embedding model without network access.

    Produces reproducible vectors from a hash of the text so that similarity
    based logic can be tested without calling AWS.
    """

    def __init__(self, dimensions: int = 8) -> None:
        """Initializes the model.

        Args:
            dimensions: Length of the produced vectors.
        """
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Produces vectors for several texts.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text.
        """
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Produces a vector for a single text.

        Args:
            text: The text to embed.

        Returns:
            The vector.
        """
        self.calls.append([text])
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        """Maps a text deterministically onto a vector.

        Args:
            text: The text to embed.

        Returns:
            The vector of length ``dimensions``.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dimensions)]


class FakeStore:
    """Vector store double that records the queries it receives."""

    def __init__(self, documents: list[Document] | None = None) -> None:
        """Initializes the store.

        Args:
            documents: Chunks returned by every similarity search.
        """
        self.documents = list(documents or ())
        self.queries: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int) -> list[Document]:
        """Returns the configured chunks and records the call.

        Args:
            query: The question to search for.
            k: Number of requested chunks.

        Returns:
            At most ``k`` of the configured chunks.
        """
        self.queries.append((query, k))
        return self.documents[:k]


class FakeBedrockRuntime:
    """Bedrock runtime double that records ``converse`` calls."""

    def __init__(self, texts: list[str] | None = None) -> None:
        """Initializes the client.

        Args:
            texts: Text blocks the model answers with.
        """
        self.texts = list(texts) if texts is not None else ["A fake answer."]
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        """Answers a request without contacting AWS.

        Args:
            **kwargs: The request as passed to the real Converse API.

        Returns:
            A response in the shape of the Converse API.
        """
        self.calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": text} for text in self.texts],
                }
            }
        }


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """Provides a deterministic embedding model."""
    return FakeEmbeddings()


@pytest.fixture
def settings() -> Settings:
    """Provides a configuration that never points at real infrastructure."""
    return Settings(
        chroma_host="chroma.internal",
        chroma_port=8000,
        collection_name="test_collection",
        aws_region="eu-central-1",
        llm_model_id="test.model-v1:0",
        top_k=2,
    )


def make_documents(*contents: str, source: str = "cv.md") -> list[Document]:
    """Creates chunks with the metadata written by cvbot-embedder.

    Args:
        *contents: Text of the chunks.
        source: Value of the ``source`` metadata field.

    Returns:
        One document per content, numbered by ``chunk_index``.
    """
    return [
        Document(
            page_content=content,
            metadata={"source": source, "filename": source, "chunk_index": index},
        )
        for index, content in enumerate(contents)
    ]

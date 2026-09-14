"""Shared fixtures and test doubles."""

from __future__ import annotations

from typing import Any, Callable

import pytest
from cvbot_core.testing import FakeEmbeddings
from langchain_core.documents import Document

from cvbot_retriever.config import Settings
from cvbot_retriever.conversation import ConversationStore
from cvbot_retriever.pipeline import AnswerResult


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


class FakeEngine:
    """Conversation engine double that answers without AWS or ChromaDB.

    Writes both turns into the injected store exactly like the real engine, so
    that callers reading the store see the same history.
    """

    def __init__(
        self,
        settings: Settings,
        store: ConversationStore,
        responder: Callable[[str], str] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Initializes the engine.

        Args:
            settings: Runtime configuration.
            store: Storage the turns are written to.
            responder: Maps a question onto the answer; defaults to a constant.
            error: Raised by every call instead of answering.
        """
        self.settings = settings
        self.store = store
        self.calls: list[tuple[str, str]] = []
        self._responder = responder or (lambda question: "A fake answer.")
        self._error = error

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        """Answers a question as the next turn of a conversation.

        Args:
            conversation_id: Identifier of the conversation.
            question: The user question.

        Returns:
            The generated answer.

        Raises:
            Exception: The configured error, if one is set.
        """
        self.calls.append((conversation_id, question))
        if self._error is not None:
            raise self._error

        conversation = self.store.load(conversation_id)
        answer = self._responder(question)
        conversation.add_user(question)
        conversation.add_assistant(answer)
        self.store.save(conversation)
        return AnswerResult(
            question=question,
            answer=answer,
            chunks=[],
            conversation_id=conversation_id,
        )


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

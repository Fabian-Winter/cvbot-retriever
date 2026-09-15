"""Read-only access to the ChromaDB instance (container on AWS Fargate)."""

from __future__ import annotations

import logging

import chromadb
from cvbot_core.vector_store import create_chroma_client, read_embedding_model_id
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .config import Settings

LOGGER = logging.getLogger(__name__)


def create_client(settings: Settings) -> chromadb.ClientAPI:
    """Creates an HTTP client for the ChromaDB instance.

    Args:
        settings: Runtime configuration holding host and port.

    Returns:
        The connected Chroma client.
    """
    return create_chroma_client(
        host=settings.chroma_host, port=settings.chroma_port
    )


def get_indexed_embedding_model_id(
    client: chromadb.ClientAPI, collection_name: str
) -> str:
    """Reads the embedding model ID cvbot-embedder indexed the collection with.

    Args:
        client: The Chroma client.
        collection_name: Name of the collection.

    Returns:
        The Bedrock model ID to embed the question with.

    Raises:
        RuntimeError: If the collection does not exist or carries no
            embedding model metadata.
    """
    return read_embedding_model_id(client, collection_name)


def open_collection(
    client: chromadb.ClientAPI,
    collection_name: str,
    embeddings: Embeddings,
) -> Chroma:
    """Opens the collection filled by cvbot-embedder.

    The collection is only read; creating, clearing and deleting it stays the
    responsibility of the ingestion pipeline.

    Args:
        client: The Chroma client.
        collection_name: Name of the collection.
        embeddings: Embedding model used to embed the question.

    Returns:
        The vector store used for querying.
    """
    LOGGER.info("opening collection %r", collection_name)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )

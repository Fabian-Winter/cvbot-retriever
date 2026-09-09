"""Read-only access to the ChromaDB instance (container on AWS Fargate)."""

from __future__ import annotations

import logging

import chromadb
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
    LOGGER.info(
        "connecting to ChromaDB: %s:%d",
        settings.chroma_host,
        settings.chroma_port,
    )
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )


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
    )

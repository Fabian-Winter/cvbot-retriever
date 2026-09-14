"""Factory for the Bedrock embedding model."""

from __future__ import annotations

from cvbot_core.embeddings import build_bedrock_embeddings
from langchain_core.embeddings import Embeddings

from .config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    """Creates the embedding model used to embed the question.

    Must stay identical to the model cvbot-embedder indexed the chunks with,
    otherwise the query vectors do not match the stored ones.

    Args:
        settings: Runtime configuration holding model ID and region.

    Returns:
        The configured Bedrock embedding model.
    """
    return build_bedrock_embeddings(
        model_id=settings.embedding_model_id,
        region_name=settings.aws_region,
    )

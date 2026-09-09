"""Factory for the Bedrock embedding model."""

from __future__ import annotations

import logging

from langchain_aws import BedrockEmbeddings
from langchain_core.embeddings import Embeddings

from .config import Settings

LOGGER = logging.getLogger(__name__)


def build_embeddings(settings: Settings) -> Embeddings:
    """Creates the embedding model used to embed the question.

    Must stay identical to the model cvbot-embedder indexed the chunks with,
    otherwise the query vectors do not match the stored ones.

    AWS credentials are resolved through the usual boto3 chain (environment
    variables, profile, IAM role of the Fargate task).

    Args:
        settings: Runtime configuration holding model ID and region.

    Returns:
        The configured Bedrock embedding model.
    """
    LOGGER.info(
        "Bedrock embeddings: model_id=%s region=%s",
        settings.embedding_model_id,
        settings.aws_region,
    )
    return BedrockEmbeddings(
        model_id=settings.embedding_model_id,
        region_name=settings.aws_region,
    )

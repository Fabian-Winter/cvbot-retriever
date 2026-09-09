"""Tests for the Bedrock embedding factory."""

from __future__ import annotations

import pytest

from cvbot_retriever import embeddings
from cvbot_retriever.config import Settings


def test_build_embeddings_passes_model_and_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embeddings, "BedrockEmbeddings", lambda **kw: kw)
    settings = Settings(
        embedding_model_id="amazon.titan-embed-text-v2:0",
        aws_region="eu-west-1",
    )

    model = embeddings.build_embeddings(settings)

    assert model == {
        "model_id": "amazon.titan-embed-text-v2:0",
        "region_name": "eu-west-1",
    }

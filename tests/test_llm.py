"""Tests for the Bedrock LLM client."""

from __future__ import annotations

import pytest

from cvbot_retriever import llm
from cvbot_retriever.config import Settings
from tests.conftest import FakeBedrockRuntime


def test_creates_bedrock_runtime_client_for_the_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_client(service: str, **kwargs: object) -> FakeBedrockRuntime:
        captured["service"] = service
        captured.update(kwargs)
        return FakeBedrockRuntime()

    monkeypatch.setattr(llm.boto3, "client", fake_client)

    llm.BedrockLLMClient(Settings(aws_region="eu-west-1"))

    assert captured["service"] == "bedrock-runtime"
    assert captured["region_name"] == "eu-west-1"


def test_generate_sends_prompt_as_user_message(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(["The candidate studied computer science."])
    client = llm.BedrockLLMClient(settings, client=runtime)

    answer = client.generate("What did the candidate study?")

    [request] = runtime.calls
    assert request["modelId"] == "test.model-v1:0"
    assert request["messages"] == [
        {"role": "user", "content": [{"text": "What did the candidate study?"}]}
    ]
    assert answer == "The candidate studied computer science."


def test_generate_omits_system_block_when_not_given(settings: Settings) -> None:
    runtime = FakeBedrockRuntime()
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate("A question.")

    assert "system" not in runtime.calls[0]


def test_generate_passes_system_prompt_separately(settings: Settings) -> None:
    runtime = FakeBedrockRuntime()
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate("A question.", system="You are an assistant.")

    assert runtime.calls[0]["system"] == [{"text": "You are an assistant."}]


def test_generate_joins_multiple_text_blocks(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(["First part.", "Second part."])
    client = llm.BedrockLLMClient(settings, client=runtime)

    assert client.generate("A question.") == "First part.\nSecond part."


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_raises(settings: Settings, prompt: str) -> None:
    client = llm.BedrockLLMClient(settings, client=FakeBedrockRuntime())

    with pytest.raises(ValueError, match="prompt"):
        client.generate(prompt)

"""Tests for the Bedrock LLM client."""

from __future__ import annotations

import pytest

from cvbot_retriever import llm
from cvbot_retriever.config import Settings
from cvbot_retriever.conversation import Message
from tests.conftest import FakeBedrockRuntime

QUESTION = Message(role="user", content="What did the candidate study?")
SYSTEM = "You are an assistant."


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


def test_generate_sends_the_messages_as_user_turn(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(["The candidate studied computer science."])
    client = llm.BedrockLLMClient(settings, client=runtime)

    answer = client.generate([QUESTION], SYSTEM)

    [request] = runtime.calls
    assert request["modelId"] == "test.model-v1:0"
    assert request["messages"] == [
        {"role": "user", "content": [{"text": "What did the candidate study?"}]}
    ]
    assert answer == "The candidate studied computer science."


def test_generate_keeps_the_roles_of_a_history(settings: Settings) -> None:
    runtime = FakeBedrockRuntime()
    client = llm.BedrockLLMClient(settings, client=runtime)
    history = [
        Message(role="user", content="Erste Frage?"),
        Message(role="assistant", content="Erste Antwort."),
        QUESTION,
    ]

    client.generate(history, SYSTEM)

    assert [message["role"] for message in runtime.calls[0]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]


def test_generate_passes_system_prompt_separately(settings: Settings) -> None:
    runtime = FakeBedrockRuntime()
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate([QUESTION], SYSTEM)

    assert runtime.calls[0]["system"] == [{"text": SYSTEM}]
    assert all(
        SYSTEM not in block["text"]
        for message in runtime.calls[0]["messages"]
        for block in message["content"]
    )


def test_generate_joins_multiple_text_blocks(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(["First part.", "Second part."])
    client = llm.BedrockLLMClient(settings, client=runtime)

    assert client.generate([QUESTION], SYSTEM) == "First part.\nSecond part."


TOOL_CONFIG = {
    "tools": [{"toolSpec": {"name": "my_tool", "inputSchema": {"json": {}}}}],
    "toolChoice": {"any": {}},
}


def test_generate_tool_call_sends_the_tool_config(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(
        tool_calls=[{"name": "my_tool", "input": {"query": "q"}}]
    )
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate_tool_call([QUESTION], SYSTEM, TOOL_CONFIG)

    [request] = runtime.calls
    assert request["toolConfig"] == TOOL_CONFIG
    assert request["system"] == [{"text": SYSTEM}]


def test_generate_tool_call_returns_the_tool_use_block(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(
        tool_calls=[{"name": "my_tool", "input": {"query": "q", "filters": {}}}]
    )
    client = llm.BedrockLLMClient(settings, client=runtime)

    call = client.generate_tool_call([QUESTION], SYSTEM, TOOL_CONFIG)

    assert call is not None
    assert call.name == "my_tool"
    assert call.input == {"query": "q", "filters": {}}
    assert call.tool_use_id


def test_generate_tool_call_returns_none_for_a_text_answer(
    settings: Settings,
) -> None:
    runtime = FakeBedrockRuntime(["Just prose."])
    client = llm.BedrockLLMClient(settings, client=runtime)

    assert client.generate_tool_call([QUESTION], SYSTEM, TOOL_CONFIG) is None


def test_generate_tool_call_tolerates_a_non_mapping_input(
    settings: Settings,
) -> None:
    runtime = FakeBedrockRuntime(tool_calls=[{"name": "my_tool", "input": None}])
    client = llm.BedrockLLMClient(settings, client=runtime)

    call = client.generate_tool_call([QUESTION], SYSTEM, TOOL_CONFIG)

    assert call is not None
    assert call.input == {}


@pytest.mark.parametrize("system", ["", "   "])
def test_empty_system_prompt_raises(settings: Settings, system: str) -> None:
    client = llm.BedrockLLMClient(settings, client=FakeBedrockRuntime())

    with pytest.raises(ValueError, match="system prompt"):
        client.generate([QUESTION], system)


def test_missing_system_prompt_raises(settings: Settings) -> None:
    client = llm.BedrockLLMClient(settings, client=FakeBedrockRuntime())

    with pytest.raises(TypeError):
        client.generate([QUESTION])  # type: ignore[call-arg]


def test_empty_message_list_raises(settings: Settings) -> None:
    client = llm.BedrockLLMClient(settings, client=FakeBedrockRuntime())

    with pytest.raises(ValueError, match="messages"):
        client.generate([], SYSTEM)


def test_last_message_must_be_a_user_message(settings: Settings) -> None:
    client = llm.BedrockLLMClient(settings, client=FakeBedrockRuntime())

    with pytest.raises(ValueError, match="user message"):
        client.generate([Message(role="assistant", content="Eine Antwort.")], SYSTEM)

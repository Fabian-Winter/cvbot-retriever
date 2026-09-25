"""Tests for the Bedrock LLM client."""

from __future__ import annotations

import json

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


def test_generate_omits_the_inference_config_by_default(settings: Settings) -> None:
    runtime = FakeBedrockRuntime()
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate([QUESTION], SYSTEM)

    assert "inferenceConfig" not in runtime.calls[0]


def test_generate_forwards_the_inference_config(settings: Settings) -> None:
    runtime = FakeBedrockRuntime()
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate([QUESTION], SYSTEM, inference_config={"temperature": 0})

    assert runtime.calls[0]["inferenceConfig"] == {"temperature": 0}


JSON_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


def test_generate_json_sends_the_schema_as_structured_output(
    settings: Settings,
) -> None:
    runtime = FakeBedrockRuntime(json_responses=[{"query": "q"}])
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate_json(
        [QUESTION], SYSTEM, json_schema=JSON_SCHEMA, schema_name="my_schema"
    )

    [request] = runtime.calls
    assert request["system"] == [{"text": SYSTEM}]
    text_format = request["outputConfig"]["textFormat"]
    assert text_format["type"] == "json_schema"
    json_schema = text_format["structure"]["jsonSchema"]
    assert json_schema["name"] == "my_schema"
    assert json.loads(json_schema["schema"]) == JSON_SCHEMA


def test_generate_json_returns_the_parsed_object(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(
        json_responses=[{"query": "q", "filters": {"years": ["2013"]}}]
    )
    client = llm.BedrockLLMClient(settings, client=runtime)

    answer = client.generate_json(
        [QUESTION], SYSTEM, json_schema=JSON_SCHEMA, schema_name="my_schema"
    )

    assert answer == {"query": "q", "filters": {"years": ["2013"]}}


def test_generate_json_rejects_a_prose_answer(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(["Just prose."])
    client = llm.BedrockLLMClient(settings, client=runtime)

    with pytest.raises(ValueError, match="no valid JSON"):
        client.generate_json(
            [QUESTION], SYSTEM, json_schema=JSON_SCHEMA, schema_name="my_schema"
        )


def test_generate_json_rejects_a_json_array(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(["[1, 2]"])
    client = llm.BedrockLLMClient(settings, client=runtime)

    with pytest.raises(ValueError, match="no JSON object"):
        client.generate_json(
            [QUESTION], SYSTEM, json_schema=JSON_SCHEMA, schema_name="my_schema"
        )


def test_generate_json_forwards_the_inference_config(settings: Settings) -> None:
    runtime = FakeBedrockRuntime(json_responses=[{"query": "q"}])
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate_json(
        [QUESTION],
        SYSTEM,
        json_schema=JSON_SCHEMA,
        schema_name="my_schema",
        inference_config={"temperature": 0},
    )

    assert runtime.calls[0]["inferenceConfig"] == {"temperature": 0}


def test_generate_json_omits_the_inference_config_by_default(
    settings: Settings,
) -> None:
    runtime = FakeBedrockRuntime(json_responses=[{"query": "q"}])
    client = llm.BedrockLLMClient(settings, client=runtime)

    client.generate_json(
        [QUESTION], SYSTEM, json_schema=JSON_SCHEMA, schema_name="my_schema"
    )

    assert "inferenceConfig" not in runtime.calls[0]


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

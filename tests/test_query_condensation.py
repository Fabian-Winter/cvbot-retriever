"""Tests for condensing follow-up questions into standalone queries."""

from __future__ import annotations

from typing import Any

from cvbot_retriever import query_condensation
from cvbot_retriever.config import Settings
from cvbot_retriever.conversation import Message
from cvbot_retriever.llm import BedrockLLMClient
from tests.conftest import FakeBedrockRuntime

HISTORY = [
    Message(role="user", content="Wo hat die Person studiert?"),
    Message(role="assistant", content="An der LMU in München."),
]


class RaisingBedrockRuntime(FakeBedrockRuntime):
    """Bedrock double that always fails, to exercise the fallback path."""

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")


def test_first_turn_is_returned_unchanged_without_a_model_call(
    settings: Settings,
) -> None:
    runtime = FakeBedrockRuntime(["should never be used"])
    llm = BedrockLLMClient(settings, client=runtime)

    result = query_condensation.condense_question(llm, [], "Wo hat sie studiert?")

    assert result == "Wo hat sie studiert?"
    assert runtime.calls == []


def test_history_is_sent_to_the_model_and_the_condensed_text_is_returned(
    settings: Settings,
) -> None:
    runtime = FakeBedrockRuntime(["Wo hat die Person vorher gearbeitet?"])
    llm = BedrockLLMClient(settings, client=runtime)

    result = query_condensation.condense_question(llm, HISTORY, "Und davor?")

    assert result == "Wo hat die Person vorher gearbeitet?"
    [request] = runtime.calls
    assert [message["role"] for message in request["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert request["messages"][-1]["content"][0]["text"] == "Und davor?"
    assert request["system"] == [
        {"text": query_condensation.CONDENSATION_SYSTEM_PROMPT}
    ]


def test_failed_condensation_falls_back_to_the_raw_question(
    settings: Settings,
) -> None:
    llm = BedrockLLMClient(settings, client=RaisingBedrockRuntime())

    result = query_condensation.condense_question(llm, HISTORY, "Und davor?")

    assert result == "Und davor?"


def test_empty_condensation_result_falls_back_to_the_raw_question(
    settings: Settings,
) -> None:
    runtime = FakeBedrockRuntime(["   "])
    llm = BedrockLLMClient(settings, client=runtime)

    result = query_condensation.condense_question(llm, HISTORY, "Und davor?")

    assert result == "Und davor?"

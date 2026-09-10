"""Tests for the derivation of the LLM context from the full history."""

from __future__ import annotations

import pytest

from cvbot_retriever.context import build_context
from cvbot_retriever.conversation import Message
from cvbot_retriever.tokens import count_message_tokens, count_tokens

SYSTEM = "Du bist ein Assistent."
CURRENT = Message(role="user", content="Aktuelle Frage?")


def make_history(turns: int) -> list[Message]:
    """Creates alternating user and assistant messages.

    Args:
        turns: Number of user/assistant pairs.

    Returns:
        The messages in chronological order.
    """
    history: list[Message] = []
    for index in range(turns):
        history.append(Message(role="user", content=f"Frage {index}?"))
        history.append(Message(role="assistant", content=f"Antwort {index}."))
    return history


def budget_for(system: str, messages: list[Message], buffer: int) -> int:
    """Returns a ``max_context_tokens`` value that fits exactly ``messages``.

    Args:
        system: The system prompt counted against the budget.
        messages: The messages that must still fit.
        buffer: The response buffer to reserve.

    Returns:
        The matching ``max_context_tokens`` value.
    """
    return count_tokens(system) + count_message_tokens(messages) + buffer


def test_short_history_is_kept_completely() -> None:
    history = make_history(2)

    context = build_context(
        SYSTEM, history, CURRENT, budget_for(SYSTEM, [*history, CURRENT], 10), 10
    )

    assert context == [*history, CURRENT]


def test_oldest_messages_are_dropped_first() -> None:
    history = make_history(4)
    kept = history[4:]

    context = build_context(
        SYSTEM, history, CURRENT, budget_for(SYSTEM, [*kept, CURRENT], 10), 10
    )

    assert context == [*kept, CURRENT]
    assert history[0] not in context


def test_current_message_survives_any_truncation() -> None:
    history = make_history(10)

    context = build_context(
        SYSTEM, history, CURRENT, budget_for(SYSTEM, [CURRENT], 10), 10
    )

    assert context == [CURRENT]


def test_system_prompt_is_never_part_of_the_context() -> None:
    context = build_context(SYSTEM, make_history(2), CURRENT, 1000, 10)

    assert all(SYSTEM not in message.content for message in context)
    assert all(message.role in {"user", "assistant"} for message in context)


def test_system_prompt_counts_against_the_budget() -> None:
    history = make_history(3)
    max_tokens = budget_for(SYSTEM, [*history, CURRENT], 10)

    with_long_system = build_context(
        SYSTEM + " Zusätzliche Instruktionen." * 3,
        history,
        CURRENT,
        max_tokens,
        10,
    )

    assert len(with_long_system) < len(history) + 1


def test_full_history_is_not_modified() -> None:
    history = make_history(5)
    original = list(history)

    build_context(SYSTEM, history, CURRENT, budget_for(SYSTEM, [CURRENT], 10), 10)

    assert history == original


def test_context_starts_with_a_user_message() -> None:
    history = make_history(3)
    # Budget for the trailing assistant message only, which Converse rejects
    # as the first message.
    kept = history[-1:]

    context = build_context(
        SYSTEM, history, CURRENT, budget_for(SYSTEM, [*kept, CURRENT], 10), 10
    )

    assert context == [CURRENT]


def test_oversized_current_message_raises() -> None:
    huge = Message(role="user", content="Sehr langer Kontext. " * 200)

    with pytest.raises(ValueError, match="context budget"):
        build_context(SYSTEM, [], huge, 100, 10)


def test_buffer_larger_than_the_context_raises() -> None:
    with pytest.raises(ValueError, match="response_token_buffer"):
        build_context(SYSTEM, [], CURRENT, 100, 100)


@pytest.mark.parametrize("system", ["", "   "])
def test_empty_system_prompt_raises(system: str) -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        build_context(system, [], CURRENT, 1000, 10)

"""Tests for the token counting."""

from __future__ import annotations

from cvbot_retriever.conversation import Message
from cvbot_retriever.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    count_message_tokens,
    count_tokens,
)


def test_empty_text_counts_as_zero() -> None:
    assert count_tokens("") == 0


def test_longer_text_counts_more_tokens() -> None:
    short = count_tokens("Der Bewerber arbeitet als Platform Engineer.")
    long = count_tokens("Der Bewerber arbeitet als Platform Engineer. " * 10)

    assert 0 < short < long


def test_counting_is_deterministic() -> None:
    text = "Welche Projekte hat die Person umgesetzt?"

    assert count_tokens(text) == count_tokens(text)


def test_message_tokens_include_the_role_overhead() -> None:
    message = Message(role="user", content="Eine Frage.")

    expected = count_tokens("Eine Frage.") + MESSAGE_OVERHEAD_TOKENS
    assert count_message_tokens([message]) == expected


def test_message_tokens_sum_up_all_messages() -> None:
    messages = [
        Message(role="user", content="Eine Frage."),
        Message(role="assistant", content="Eine Antwort."),
    ]

    assert count_message_tokens(messages) == sum(
        count_message_tokens([message]) for message in messages
    )


def test_no_messages_count_as_zero() -> None:
    assert count_message_tokens([]) == 0

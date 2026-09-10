"""Tests for messages, conversations and the conversation store."""

from __future__ import annotations

import pytest

from cvbot_retriever.conversation import (
    Conversation,
    InMemoryConversationStore,
    Message,
)


def test_message_converts_to_the_converse_format() -> None:
    message = Message(role="user", content="Eine Frage?")

    assert message.to_converse() == {
        "role": "user",
        "content": [{"text": "Eine Frage?"}],
    }


def test_unknown_role_raises() -> None:
    with pytest.raises(ValueError, match="role"):
        Message(role="system", content="Ich bin ein System-Prompt.")


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_content_raises(content: str) -> None:
    with pytest.raises(ValueError, match="content"):
        Message(role="user", content=content)


def test_turns_are_appended_in_order() -> None:
    conversation = Conversation(conversation_id="c1")

    conversation.add_user("Erste Frage?")
    conversation.add_assistant("Erste Antwort.")
    conversation.add_user("Zweite Frage?")

    assert [message.content for message in conversation.messages] == [
        "Erste Frage?",
        "Erste Antwort.",
        "Zweite Frage?",
    ]
    assert [message.role for message in conversation.messages] == [
        "user",
        "assistant",
        "user",
    ]


def test_history_is_a_copy() -> None:
    conversation = Conversation(conversation_id="c1")
    conversation.add_user("Eine Frage?")

    conversation.history().clear()

    assert len(conversation.messages) == 1


def test_store_returns_an_empty_conversation_for_unknown_ids() -> None:
    store = InMemoryConversationStore()

    conversation = store.load("unknown")

    assert conversation.conversation_id == "unknown"
    assert conversation.messages == []


def test_store_round_trips_the_full_history() -> None:
    store = InMemoryConversationStore()
    conversation = store.load("c1")
    conversation.add_user("Eine Frage?")
    conversation.add_assistant("Eine Antwort.")

    store.save(conversation)

    assert store.load("c1").messages == conversation.messages


def test_store_keeps_conversations_apart() -> None:
    store = InMemoryConversationStore()
    first = store.load("c1")
    first.add_user("Frage in c1?")
    store.save(first)

    assert store.load("c2").messages == []


def test_stored_history_is_isolated_from_later_changes() -> None:
    store = InMemoryConversationStore()
    conversation = store.load("c1")
    conversation.add_user("Eine Frage?")
    store.save(conversation)

    conversation.add_assistant("Nicht gespeicherte Antwort.")

    assert len(store.load("c1").messages) == 1


def test_empty_conversation_id_raises() -> None:
    with pytest.raises(ValueError, match="conversation_id"):
        InMemoryConversationStore().load("  ")

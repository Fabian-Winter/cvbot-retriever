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


class FakeClock:
    """Monotonic clock that only moves when a test moves it."""

    def __init__(self) -> None:
        """Starts the clock at zero."""
        self.value = 0.0

    def __call__(self) -> float:
        """Returns the current reading."""
        return self.value

    def advance(self, seconds: float) -> None:
        """Moves the clock forward.

        Args:
            seconds: Amount of time that passes.
        """
        self.value += seconds


def store_with(clock: FakeClock, **kwargs: int) -> InMemoryConversationStore:
    """Builds a store running on the test clock.

    Args:
        clock: The clock driving the expiry.
        **kwargs: Further arguments of the store.

    Returns:
        The store.
    """
    return InMemoryConversationStore(time_source=clock, **kwargs)


def save_turn(store: InMemoryConversationStore, conversation_id: str) -> None:
    """Stores one turn of a conversation.

    Args:
        store: The store to write to.
        conversation_id: Identifier of the conversation.
    """
    conversation = store.load(conversation_id)
    conversation.add_user("Eine Frage?")
    store.save(conversation)


def test_a_conversation_survives_within_the_timeout() -> None:
    clock = FakeClock()
    store = store_with(clock, ttl_seconds=60)
    save_turn(store, "c1")

    clock.advance(59)

    assert store.load("c1").messages


def test_an_idle_conversation_is_dropped_after_the_timeout() -> None:
    clock = FakeClock()
    store = store_with(clock, ttl_seconds=60)
    save_turn(store, "c1")

    clock.advance(61)

    assert store.load("c1").messages == []


def test_reading_a_conversation_extends_its_lifetime() -> None:
    clock = FakeClock()
    store = store_with(clock, ttl_seconds=60)
    save_turn(store, "c1")

    clock.advance(40)
    store.load("c1")
    clock.advance(40)

    assert store.load("c1").messages


def test_an_expired_conversation_frees_its_memory() -> None:
    clock = FakeClock()
    store = store_with(clock, ttl_seconds=60)
    save_turn(store, "c1")

    clock.advance(61)
    store.load("c2")

    assert "c1" not in store._conversations


def test_an_expired_conversation_does_not_take_others_with_it() -> None:
    clock = FakeClock()
    store = store_with(clock, ttl_seconds=60)
    save_turn(store, "old")
    clock.advance(40)
    save_turn(store, "recent")

    clock.advance(30)

    assert store.load("old").messages == []
    assert store.load("recent").messages


def test_the_store_drops_the_least_recently_used_conversation_when_full() -> None:
    clock = FakeClock()
    store = store_with(clock, max_conversations=2)
    for conversation_id in ("c1", "c2"):
        save_turn(store, conversation_id)
        clock.advance(1)
    store.load("c1")

    save_turn(store, "c3")

    assert store.load("c2").messages == []
    assert store.load("c1").messages
    assert store.load("c3").messages


def test_the_number_of_stored_conversations_stays_bounded() -> None:
    clock = FakeClock()
    store = store_with(clock, max_conversations=3)

    for index in range(20):
        save_turn(store, f"c{index}")

    assert len(store._conversations) == 3


@pytest.mark.parametrize(
    "kwargs", [{"ttl_seconds": 0}, {"max_conversations": 0}]
)
def test_invalid_bounds_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        InMemoryConversationStore(**kwargs)

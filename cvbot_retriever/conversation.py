"""Conversation state: full history and the store it is kept in.

A ``Conversation`` is the single source of truth and always holds the complete,
unmodified history - it is what a UI displays later. The trimmed view that is
sent to the model is derived from it in :mod:`cvbot_retriever.context` and never
changes the conversation itself.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol
from .config import DEFAULT_CONVERSATION_TTL_SECONDS, DEFAULT_MAX_CONVERSATIONS

LOGGER = logging.getLogger(__name__)

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

_VALID_ROLES = frozenset({ROLE_USER, ROLE_ASSISTANT})


@dataclass(frozen=True)
class Message:
    """A single turn of a conversation.

    Attributes:
        role: Either ``user`` or ``assistant``. The system prompt is not a
            message: it is passed to Bedrock in its own Converse block and is
            therefore never part of a history.
        content: The text of the turn.
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        """Validates the message.

        Raises:
            ValueError: If the role is unknown or the content is empty.
        """
        if self.role not in _VALID_ROLES:
            raise ValueError(f"unknown role: {self.role!r}")
        if not self.content.strip():
            raise ValueError("content must not be empty")

    def to_converse(self) -> dict[str, Any]:
        """Converts the message into the Bedrock Converse format.

        Returns:
            The message as expected in the ``messages`` list of a request.
        """
        return {"role": self.role, "content": [{"text": self.content}]}


@dataclass
class Conversation:
    """The complete history of one conversation.

    Attributes:
        conversation_id: Identifier separating concurrent conversations.
        messages: All turns in chronological order. This list is never
            truncated by the context management.
    """

    conversation_id: str
    messages: list[Message] = field(default_factory=list)

    def add_user(self, content: str) -> Message:
        """Appends a user turn.

        Args:
            content: The text of the turn.

        Returns:
            The appended message.
        """
        return self._add(Message(role=ROLE_USER, content=content))

    def add_assistant(self, content: str) -> Message:
        """Appends an assistant turn.

        Args:
            content: The text of the turn.

        Returns:
            The appended message.
        """
        return self._add(Message(role=ROLE_ASSISTANT, content=content))

    def history(self) -> list[Message]:
        """Returns the complete history.

        Returns:
            A copy of all turns, so that callers cannot modify the
            conversation by accident.
        """
        return list(self.messages)

    def _add(self, message: Message) -> Message:
        """Appends a message to the history.

        Args:
            message: The message to append.

        Returns:
            The appended message.
        """
        self.messages.append(message)
        return message


class ConversationStore(Protocol):
    """Storage of conversations across requests.

    Implemented in-memory for now; a shared backend (for example DynamoDB) can
    be added later without touching the callers.
    """

    def load(self, conversation_id: str) -> Conversation:
        """Reads a conversation.

        Args:
            conversation_id: Identifier of the conversation.

        Returns:
            The stored conversation, or a new empty one if it is unknown.
        """
        ...

    def save(self, conversation: Conversation) -> None:
        """Writes a conversation back.

        Args:
            conversation: The conversation to store.
        """
        ...


@dataclass
class _Entry:
    """One stored conversation together with the time it was last used.

    Attributes:
        conversation: The stored conversation.
        touched_at: Reading of the store clock at the last access.
    """

    conversation: Conversation
    touched_at: float


class InMemoryConversationStore:
    """Process-local conversation store with an idle timeout.

    Sufficient for tests and the command line; it is explicitly not shared
    between several application instances.

    Conversations are kept only as long as they are in use: an entry that was
    not touched within ``ttl_seconds`` is dropped, and the store never holds
    more than ``max_conversations`` entries. Both bounds keep the memory of a
    long running task from growing with every visitor and limit how long
    conversation content exists at all.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_CONVERSATION_TTL_SECONDS,
        max_conversations: int = DEFAULT_MAX_CONVERSATIONS,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initializes an empty store.

        Args:
            ttl_seconds: Idle time after which a conversation is dropped.
            max_conversations: Upper bound of conversations kept at once.
            time_source: Monotonic clock, replaced in tests.

        Raises:
            ValueError: If a bound is not positive.
        """
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be positive: {ttl_seconds}")
        if max_conversations < 1:
            raise ValueError(
                f"max_conversations must be positive: {max_conversations}"
            )

        self._ttl_seconds = ttl_seconds
        self._max_conversations = max_conversations
        self._now = time_source
        self._conversations: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    def load(self, conversation_id: str) -> Conversation:
        """Reads a conversation.

        An expired conversation is treated like an unknown one, so a returning
        visitor simply starts over instead of seeing an error.

        Args:
            conversation_id: Identifier of the conversation.

        Returns:
            A copy of the stored conversation, or a new empty one.

        Raises:
            ValueError: If the identifier is empty.
        """
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")

        with self._lock:
            self._expire()
            entry = self._conversations.get(conversation_id)
            if entry is None:
                LOGGER.debug("new conversation %s", conversation_id)
                return Conversation(conversation_id=conversation_id)

            entry.touched_at = self._now()
            self._conversations.move_to_end(conversation_id)
            stored = entry.conversation
            return replace(stored, messages=list(stored.messages))

    def save(self, conversation: Conversation) -> None:
        """Writes a conversation back.

        Args:
            conversation: The conversation to store.
        """
        with self._lock:
            self._expire()
            self._conversations[conversation.conversation_id] = _Entry(
                conversation=replace(
                    conversation, messages=list(conversation.messages)
                ),
                touched_at=self._now(),
            )
            self._conversations.move_to_end(conversation.conversation_id)
            while len(self._conversations) > self._max_conversations:
                dropped, _ = self._conversations.popitem(last=False)
                LOGGER.debug("dropped conversation %s: store is full", dropped)

    def _expire(self) -> None:
        """Drops conversations that were idle for longer than the timeout.

        Called while the lock is held, on every access, so that no background
        thread is needed.
        """
        horizon = self._now() - self._ttl_seconds
        while self._conversations:
            conversation_id, entry = next(iter(self._conversations.items()))
            if entry.touched_at > horizon:
                break
            del self._conversations[conversation_id]
            LOGGER.debug("dropped conversation %s: idle", conversation_id)

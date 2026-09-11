"""Request and response schemas of the JSON API.

The schemas deliberately expose only what a client needs: the conversation
identifier, the generated answer and the visible history. Retrieved chunks and
the system prompt never leave the process.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field, field_validator

from .conversation import Message

MAX_QUESTION_LENGTH = 2000


class MessageOut(BaseModel):
    """A single visible turn of a conversation.

    Attributes:
        role: Either ``user`` or ``assistant``.
        content: The text of the turn.
    """

    role: str
    content: str


class ChatRequest(BaseModel):
    """A question asked within a conversation.

    Attributes:
        question: The user question.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Rejects questions that consist of whitespace only.

        Args:
            value: The submitted question.

        Returns:
            The question without surrounding whitespace.

        Raises:
            ValueError: If the question is blank.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class ChatResponse(BaseModel):
    """The answer to a question plus the resulting conversation state.

    Attributes:
        conversation_id: Identifier of the conversation.
        answer: The generated answer.
        messages: The complete visible history including the current turn.
    """

    conversation_id: str
    answer: str
    messages: list[MessageOut]


class ConversationResponse(BaseModel):
    """The visible history of a conversation.

    Attributes:
        conversation_id: Identifier of the conversation.
        messages: The complete visible history.
    """

    conversation_id: str
    messages: list[MessageOut]


class NewConversationResponse(BaseModel):
    """The identifier of a freshly created conversation.

    Attributes:
        conversation_id: Identifier to use for subsequent requests.
    """

    conversation_id: str


class ErrorResponse(BaseModel):
    """A user facing error message.

    Attributes:
        detail: Explanation that can be shown to the user as-is.
    """

    detail: str


def to_messages_out(messages: Sequence[Message]) -> list[MessageOut]:
    """Converts conversation turns into their API representation.

    Args:
        messages: The turns of a conversation.

    Returns:
        The turns as response models.
    """
    return [
        MessageOut(role=message.role, content=message.content)
        for message in messages
    ]

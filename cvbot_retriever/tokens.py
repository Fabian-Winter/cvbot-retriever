"""Token counting for the context budget.

Uses the same ``cl100k_base`` encoding as the chunking in cvbot-embedder. That
is an approximation for the Bedrock models, but it keeps both repositories
consistent and makes the budget deterministic and testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from .conversation import Message

ENCODING_NAME = "cl100k_base"

# Rough allowance for the role framing the model adds around every message.
MESSAGE_OVERHEAD_TOKENS = 4


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Loads the encoding once per process.

    Returns:
        The ``cl100k_base`` encoding.
    """
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Counts the tokens of a text.

    Args:
        text: The text to measure.

    Returns:
        The token count according to ``cl100k_base``.
    """
    if not text:
        return 0
    return len(_encoding().encode(text))


def count_message_tokens(messages: Iterable[Message]) -> int:
    """Counts the tokens of several messages including their role framing.

    Args:
        messages: The messages to measure.

    Returns:
        The token count of all message contents plus a per-message overhead.
    """
    return sum(
        count_tokens(message.content) + MESSAGE_OVERHEAD_TOKENS
        for message in messages
    )

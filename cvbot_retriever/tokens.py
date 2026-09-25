"""Token counting for the context budget.

Wraps the shared ``cl100k_base`` counting of cvbot-core with the per-message
overhead that the Converse API adds around every turn, and with a safety
margin: ``cl100k_base`` is a tokenizer from a different model family, so its
counts are only an approximation of what the configured Bedrock model charges.
The margin keeps the budget conservative until the approximation is refined.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from cvbot_core.tokens import ENCODING_NAME, get_encoding
from cvbot_core.tokens import count_tokens as _count_core_tokens

if TYPE_CHECKING:
    from .conversation import Message

__all__ = [
    "ENCODING_NAME",
    "MESSAGE_OVERHEAD_TOKENS",
    "TOKEN_SAFETY_FACTOR",
    "count_message_tokens",
    "count_tokens",
    "get_encoding",
]

# Rough allowance for the role framing the model adds around every message.
MESSAGE_OVERHEAD_TOKENS = 4

# Multiplier on every approximate count, so a tokenizer from another model
# family cannot underestimate the real prompt and overflow the context window.
TOKEN_SAFETY_FACTOR = 1.2


def count_tokens(text: str) -> int:
    """Counts the tokens of a text with a safety margin.

    Args:
        text: The text to measure.

    Returns:
        The approximate token count, rounded up with the safety factor
        applied.
    """
    return math.ceil(_count_core_tokens(text) * TOKEN_SAFETY_FACTOR)


def count_message_tokens(messages: Iterable[Message]) -> int:
    """Counts the tokens of several messages including their role framing.

    Args:
        messages: The messages to measure.

    Returns:
        The token count of all message contents plus a per-message overhead,
        with the safety factor applied to both so the framing is scaled the
        same way as the content it wraps.
    """
    return sum(
        math.ceil(
            (_count_core_tokens(message.content) + MESSAGE_OVERHEAD_TOKENS)
            * TOKEN_SAFETY_FACTOR
        )
        for message in messages
    )

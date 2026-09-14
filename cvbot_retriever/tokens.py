"""Token counting for the context budget.

Wraps the shared ``cl100k_base`` counting of cvbot-core with the per-message
overhead that the Converse API adds around every turn.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from cvbot_core.tokens import ENCODING_NAME, count_tokens, get_encoding

if TYPE_CHECKING:
    from .conversation import Message

__all__ = [
    "ENCODING_NAME",
    "MESSAGE_OVERHEAD_TOKENS",
    "count_message_tokens",
    "count_tokens",
    "get_encoding",
]

# Rough allowance for the role framing the model adds around every message.
MESSAGE_OVERHEAD_TOKENS = 4


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

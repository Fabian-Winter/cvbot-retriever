"""Derivation of the LLM context from the full conversation history.

The context sent to Bedrock must fit into ``max_context_tokens`` minus the
buffer reserved for the answer. The system prompt and the current user message
(question plus retrieved chunks) are mandatory; if the budget is still
exceeded, the oldest history messages are dropped one by one. The conversation
itself is never modified - only the derived view is shorter.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .conversation import ROLE_USER, Message
from .tokens import count_message_tokens, count_tokens

LOGGER = logging.getLogger(__name__)


def build_context(
    system_prompt: str,
    history: Sequence[Message],
    current_message: Message,
    max_context_tokens: int,
    response_token_buffer: int,
) -> list[Message]:
    """Builds the message list that fits into the context budget.

    Args:
        system_prompt: The system prompt. It is counted against the budget but
            never removed, because it is sent in its own Converse block.
        history: The full history up to (but excluding) the current question.
        current_message: The current user message including the retrieved
            chunks.
        max_context_tokens: Upper bound for the whole context.
        response_token_buffer: Tokens kept free for the answer.

    Returns:
        The history messages that still fit, followed by ``current_message``.

    Raises:
        ValueError: If the system prompt is empty, if the budget is not
            positive or if system prompt and current message alone already
            exceed it.
    """
    if not system_prompt.strip():
        raise ValueError("system_prompt must not be empty")

    budget = max_context_tokens - response_token_buffer
    if budget < 1:
        raise ValueError(
            "max_context_tokens must exceed response_token_buffer: "
            f"{max_context_tokens} <= {response_token_buffer}"
        )

    fixed_tokens = count_tokens(system_prompt) + count_message_tokens(
        [current_message]
    )
    if fixed_tokens > budget:
        raise ValueError(
            "system prompt and current question exceed the context budget: "
            f"{fixed_tokens} > {budget}"
        )

    kept = list(history)
    available = budget - fixed_tokens
    while kept and count_message_tokens(kept) > available:
        kept.pop(0)

    # Converse rejects a message list that does not start with a user turn.
    while kept and kept[0].role != ROLE_USER:
        kept.pop(0)

    dropped = len(history) - len(kept)
    if dropped:
        LOGGER.info(
            "dropped %d of %d history message(s) to stay within %d token(s)",
            dropped,
            len(history),
            budget,
        )
    return [*kept, current_message]

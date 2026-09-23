"""Bedrock client for the answer generation."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import boto3

from .config import Settings
from .conversation import ROLE_USER, Message

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation the model requested through the Converse API.

    Attributes:
        name: Name of the tool the model called.
        tool_use_id: Converse identifier of this invocation.
        input: The tool input the model generated, already parsed into a
            mapping by the SDK. Empty when the model sent no usable input.
    """

    name: str
    tool_use_id: str
    input: dict[str, Any] = field(default_factory=dict)


class BedrockLLMClient:
    """Generates text with a Bedrock model through the Converse API.

    The client knows nothing about prompt construction: it receives a finished
    message list and passes it on unchanged. System prompt, retrieved context
    and truncation of the history are handled by the calling pipeline.
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        """Initializes the client.

        AWS credentials are resolved through the usual boto3 chain (environment
        variables, profile, IAM role of the Fargate task).

        Args:
            settings: Runtime configuration holding model ID and region.
            client: Optional ``bedrock-runtime`` client used instead of a newly
                created one (for tests).
        """
        self._model_id = settings.llm_model_id
        self._client = client or boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )
        LOGGER.info(
            "Bedrock LLM: model_id=%s region=%s",
            settings.llm_model_id,
            settings.aws_region,
        )

    def generate(
        self,
        messages: Sequence[Message],
        system: str,
        inference_config: dict[str, Any] | None = None,
    ) -> str:
        """Sends a conversation to the model and returns its answer.

        Args:
            messages: The turns of the conversation in chronological order,
                ending with the current user message.
            system: The system prompt, sent as a separate Converse block so
                that the model can tell it apart from the user input. It is
                mandatory: without it the persona and the injection guardrails
                would silently be missing.
            inference_config: Optional ``inferenceConfig`` block (for example
                ``{"temperature": 0}``) applied to this single request. Left
                unset, the model answers with its own default, which is what
                the answer generation relies on for a natural style.

        Returns:
            The generated text.

        Raises:
            ValueError: If the system prompt is empty, if no message is given
                or if the last one is not a user turn.
        """
        text, _ = self._converse(messages, system, inference_config=inference_config)
        return text

    def generate_tool_call(
        self,
        messages: Sequence[Message],
        system: str,
        tool_config: dict[str, Any],
        inference_config: dict[str, Any] | None = None,
    ) -> ToolCall | None:
        """Sends a conversation that forces the model to call a tool.

        Used where the answer must be structured: with ``toolChoice`` set to
        ``any``, the model cannot reply with free text and has to fill the
        tool's input schema instead, which the Converse API returns already
        parsed. The caller is responsible for validating the input, since the
        schema can only describe the shape, not the allowed values.

        Args:
            messages: The turns of the conversation in chronological order,
                ending with the current user message.
            system: The system prompt, sent as a separate Converse block.
            tool_config: The ``toolConfig`` block of the Converse request,
                holding the tool definitions and the tool choice.
            inference_config: Optional ``inferenceConfig`` block for this
                single request; extraction calls pass ``{"temperature": 0}``
                so that the same question yields the same tool input.

        Returns:
            The tool call the model requested, or ``None`` if the response
            carried no usable tool use despite the forced tool choice.

        Raises:
            ValueError: If the system prompt is empty, if no message is given
                or if the last one is not a user turn.
        """
        _, response = self._converse(
            messages, system, tool_config=tool_config, inference_config=inference_config
        )
        return _extract_tool_call(response)

    def _converse(
        self,
        messages: Sequence[Message],
        system: str,
        tool_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Runs a Converse request and returns text plus raw response.

        Args:
            messages: The turns of the conversation, ending with a user turn.
            system: The system prompt.
            tool_config: Optional ``toolConfig`` block forcing tool use.
            inference_config: Optional ``inferenceConfig`` block, applied to
                this request only, so that other callers of the same client
                keep the model defaults.

        Returns:
            The generated text and the untouched Converse response, so that
            callers can look for content blocks the text extraction ignores.

        Raises:
            ValueError: If the system prompt is empty, if no message is given
                or if the last one is not a user turn.
        """
        if not system.strip():
            raise ValueError("system prompt must not be empty")
        if not messages:
            raise ValueError("messages must not be empty")
        if messages[-1].role != ROLE_USER:
            raise ValueError("the last message must be a user message")

        request: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": [message.to_converse() for message in messages],
            "system": [{"text": system}],
        }
        if tool_config is not None:
            request["toolConfig"] = tool_config
        if inference_config is not None:
            request["inferenceConfig"] = inference_config

        LOGGER.debug(
            "invoking %s with %d message(s), tool_config=%s",
            self._model_id,
            len(messages),
            tool_config is not None,
        )
        response = self._client.converse(**request)
        _log_response(response)
        return _extract_text(response), response


def _log_response(response: dict[str, Any]) -> None:
    """Logs the Converse response for diagnosis.

    Args:
        response: The response returned by the Converse API.
    """
    try:
        message = response.get("output", {}).get("message", {})
        content = message.get("content") or []
        block_types = sorted(
            {
                key
                for block in content
                if isinstance(block, dict)
                for key in block
                if key != "text"
            }
        )
        text = _extract_text(response)
        stop_reason = message.get("stopReason", "")
        usage = response.get("usage", {})
        LOGGER.debug(
            "bedrock response:"
            "stop_reason=%s block_types=%s tokens=%s/%s text_len=%d"
            " text=%r tool_input=%s",
            stop_reason or "?",
            ",".join(block_types) or "text",
            usage.get("inputTokens", "?"),
            usage.get("outputTokens", "?"),
            len(text),
            " ".join(text.split()),
            _toolUse_to_log(content),
        )
    except Exception:
        # Diagnosis must never be the reason a request fails.
        LOGGER.debug("could not log bedrock response shape", exc_info=True)


def _toolUse_to_log(content: list[Any], limit: int = 400) -> str:
    """Renders the input of the first tool use block for a single log line.

    ``_extract_text`` ignores tool use blocks, so without this the structured
    answer a forced tool call produced would be invisible in the log - which
    makes it impossible to tell an empty model answer from a filter that a
    later validation step dropped.

    Args:
        content: The content blocks of the Converse response.
        limit: Maximum number of characters to keep.

    Returns:
        The tool input as compact JSON, or ``-`` if the answer has no tool use
        block.
    """
    for block in content:
        if not isinstance(block, dict) or "toolUse" not in block:
            continue
        rendered = json.dumps(
            block["toolUse"].get("input"), ensure_ascii=False, default=str
        )
        collapsed = " ".join(rendered.split())
        return collapsed[:limit] + "…" if len(collapsed) > limit else collapsed
    return "-"


def _extract_text(response: dict[str, Any]) -> str:
    """Reads the generated text out of a Converse response.

    Args:
        response: The response returned by the Converse API.

    Returns:
        All text blocks of the answer, joined by newlines.
    """
    content = response["output"]["message"]["content"]
    return "\n".join(block["text"] for block in content if "text" in block)


def _extract_tool_call(response: dict[str, Any]) -> ToolCall | None:
    """Reads the first tool use block out of a Converse response.

    Args:
        response: The response returned by the Converse API.

    Returns:
        The requested tool call, or ``None`` if the answer holds no tool use
        block or its input is not a mapping.
    """
    content = response.get("output", {}).get("message", {}).get("content") or []
    for block in content:
        tool_use = block.get("toolUse") if isinstance(block, dict) else None
        if not tool_use:
            continue
        tool_input = tool_use.get("input")
        return ToolCall(
            name=str(tool_use.get("name", "")),
            tool_use_id=str(tool_use.get("toolUseId", "")),
            input=tool_input if isinstance(tool_input, dict) else {},
        )
    return None

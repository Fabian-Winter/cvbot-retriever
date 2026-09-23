"""Local web application: chat UI and JSON API on top of the engine.

The application owns the ``ConversationStore`` and injects it into the
``ConversationEngine``. The engine is built lazily on the first request, so the
application starts even while ChromaDB is unavailable, and a failed attempt is
retried on the next request instead of being cached.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import botocore.exceptions
import chromadb.errors
import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from .config import Settings
from .conversation import ConversationStore, InMemoryConversationStore
from .pipeline import ConversationEngine
from .ratelimit import SlidingWindowRateLimiter
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ErrorResponse,
    NewConversationResponse,
    to_messages_out,
    MAX_QUESTION_LENGTH,
)

LOGGER = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

KNOWLEDGE_BASE_UNAVAILABLE = (
    "Die Wissensdatenbank ist derzeit nicht erreichbar. "
    "Bitte versuche es in einem Moment noch einmal."
)
ANSWER_SERVICE_UNAVAILABLE = (
    "Der Antwortdienst ist derzeit nicht erreichbar. "
    "Bitte versuche es in einem Moment noch einmal."
)
UNEXPECTED_ERROR = (
    "Die Frage konnte nicht beantwortet werden. "
    "Bitte versuche es in einem Moment noch einmal."
)
INVALID_QUESTION = f"Bitte gib eine Frage ein (maximal {MAX_QUESTION_LENGTH} Zeichen)."
INVALID_CONVERSATION_ID = "Unbekannte Konversations-ID."
RATE_LIMITED = (
    "Zu viele Anfragen in kurzer Zeit. "
    "Bitte warte einen Moment und versuche es dann erneut."
)

UNKNOWN_CLIENT = "unknown"

_MAX_TRACKED_LOCKS = 10_000

_KNOWLEDGE_BASE_ERRORS = (chromadb.errors.ChromaError, httpx.HTTPError)
_ANSWER_SERVICE_ERRORS = (
    botocore.exceptions.BotoCoreError,
    botocore.exceptions.ClientError,
)

EngineFactory = Callable[[Settings, ConversationStore], ConversationEngine]


def _is_valid_conversation_id(conversation_id: str) -> bool:
    """Checks whether an identifier is a UUID4.

    Restricting the identifier keeps foreign conversations from being guessed
    through the URL.

    Args:
        conversation_id: Identifier taken from the request path.

    Returns:
        ``True`` if the identifier is a UUID4.
    """
    try:
        parsed = uuid.UUID(conversation_id)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == conversation_id


def _client_key(request: Request, trust_forwarded_for: bool) -> str:
    """Determines which client a request is charged to.

    Behind the API Gateway every request reaches the task through the same VPC
    link interface, so the peer address would put all callers into one bucket.
    The original address is then taken from ``X-Forwarded-For``, whose first
    entry is the client. The header is only trusted when the deployment puts a
    proxy in front of the application, because a direct caller can forge it.

    Args:
        request: The incoming request.
        trust_forwarded_for: Whether the forwarding header may be used.

    Returns:
        The identifier the rate limit is counted against.
    """
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        client = forwarded.split(",")[0].strip()
        if client:
            return client
    if request.client is not None and request.client.host:
        return request.client.host
    return UNKNOWN_CLIENT


class _EngineProvider:
    """Builds the engine on demand and shares it between requests.

    Constructing the engine opens the ChromaDB connection and can therefore
    fail. A failed attempt is not cached, so the next request tries again.
    """

    def __init__(
        self,
        settings: Settings,
        store: ConversationStore,
        engine_factory: EngineFactory,
    ) -> None:
        """Initializes the provider without building the engine.

        Args:
            settings: Runtime configuration.
            store: Storage shared by every engine instance.
            engine_factory: Callable creating an engine from settings and
                store.
        """
        self._settings = settings
        self._store = store
        self._engine_factory = engine_factory
        self._engine: ConversationEngine | None = None
        self._lock = threading.Lock()

    @property
    def store(self) -> ConversationStore:
        """Returns the store holding the histories."""
        return self._store

    def get(self) -> ConversationEngine:
        """Returns the engine, building it on first use.

        Returns:
            The shared engine instance.

        Raises:
            Exception: Whatever the factory raises if the backends cannot be
                reached.
        """
        with self._lock:
            if self._engine is None:
                LOGGER.info("building the conversation engine")
                self._engine = self._engine_factory(self._settings, self._store)
            return self._engine


class _ConversationLocks:
    """Per-conversation locks around the load-modify-save cycle.

    Turns of one conversation are serialized while different conversations stay
    fully concurrent. The registry is bounded so that it cannot outgrow the
    conversation store it guards.
    """

    def __init__(self, max_locks: int = _MAX_TRACKED_LOCKS) -> None:
        """Initializes an empty lock registry.

        Args:
            max_locks: Upper bound of remembered locks.
        """
        self._locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._max_locks = max_locks
        self._guard = threading.Lock()

    def get(self, conversation_id: str) -> threading.Lock:
        """Returns the lock of a conversation.

        Args:
            conversation_id: Identifier of the conversation.

        Returns:
            The lock guarding this conversation.
        """
        with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[conversation_id] = lock
            self._locks.move_to_end(conversation_id)
            while len(self._locks) > self._max_locks:
                # Only drops locks of the least recently used conversations,
                # which are no longer in the store either.
                self._locks.popitem(last=False)
            return lock


def create_app(
    settings: Settings | None = None,
    engine_factory: EngineFactory = ConversationEngine,
) -> FastAPI:
    """Builds the web application.

    Args:
        settings: Runtime configuration; defaults to the environment.
        engine_factory: Callable creating the engine; replaced in tests.

    Returns:
        The configured application.
    """
    effective = settings or Settings.from_env()
    store: ConversationStore = InMemoryConversationStore(
        ttl_seconds=effective.conversation_ttl_seconds,
        max_conversations=effective.max_conversations,
    )
    provider = _EngineProvider(effective, store, engine_factory)
    locks = _ConversationLocks()
    limiter = SlidingWindowRateLimiter(
        per_minute=effective.rate_limit_per_minute,
        per_hour=effective.rate_limit_per_hour,
    )
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    app = FastAPI(title="cvbot", docs_url="/api/docs", redoc_url=None)

    # Vendored client libraries (marked, DOMPurify) for Markdown rendering in
    # the chat page. Served as plain files, so no API surface is added.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # An empty allow list emits no CORS headers at all, which keeps the JSON
    # API same-origin unless origins are configured explicitly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(effective.cors_allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    def _rate_limited(request: Request) -> JSONResponse | None:
        """Charges a request to its client and builds the refusal if needed.

        Args:
            request: The incoming request.

        Returns:
            The ``429`` response, or ``None`` if the request may proceed.
        """
        decision = limiter.check(
            _client_key(request, effective.trust_forwarded_for)
        )
        if decision.allowed:
            return None
        return _error(
            429,
            RATE_LIMITED,
            headers={"Retry-After": str(decision.retry_after)},
        )

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Replaces the field-level default response with one friendly text."""
        # Only the failing fields, never exc.errors(): those carry the input.
        fields = [
            ".".join(str(part) for part in error["loc"]) for error in exc.errors()
        ]
        LOGGER.info(
            "rejected request to %s: invalid %s",
            request.url.path,
            ", ".join(fields) or "payload",
        )
        return _error(400, INVALID_QUESTION)

    @app.get("/", response_class=RedirectResponse)
    def index() -> RedirectResponse:
        """Starts a new conversation and redirects to its chat page."""
        return RedirectResponse(f"/c/{uuid.uuid4()}", status_code=303)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Reports that the process is alive, without touching the backends."""
        return {"status": "ok"}

    @app.get("/c/{conversation_id}", response_class=HTMLResponse)
    def chat_page(request: Request, conversation_id: str) -> HTMLResponse:
        """Renders the chat page with the visible history.

        The history is read straight from the store, so the page also works
        while the backends are unavailable.

        Args:
            request: The incoming request, required by the template engine.
            conversation_id: Identifier taken from the path.

        Returns:
            The rendered page, or a redirect to a new conversation if the
            identifier is malformed.
        """
        if not _is_valid_conversation_id(conversation_id):
            return RedirectResponse("/", status_code=303)

        conversation = store.load(conversation_id)
        return templates.TemplateResponse(
            request=request,
            name="chat.html",
            context={
                "conversation_id": conversation_id,
                "messages": conversation.history(),
            },
        )

    @app.post("/api/conversations", response_model=NewConversationResponse)
    def create_conversation(
        request: Request,
    ) -> NewConversationResponse | JSONResponse:
        """Hands out an identifier for a new conversation.

        Args:
            request: The incoming request, used to identify the client.

        Returns:
            The new identifier, or an error response if the client is over its
            budget.
        """
        refusal = _rate_limited(request)
        if refusal is not None:
            return refusal
        return NewConversationResponse(conversation_id=str(uuid.uuid4()))

    @app.get(
        "/api/conversations/{conversation_id}",
        response_model=ConversationResponse,
    )
    def read_conversation(
        conversation_id: str,
    ) -> ConversationResponse | JSONResponse:
        """Returns the visible history of a conversation.

        Args:
            conversation_id: Identifier taken from the path.

        Returns:
            The history, or an error response for a malformed identifier.
        """
        if not _is_valid_conversation_id(conversation_id):
            return _error(400, INVALID_CONVERSATION_ID)

        conversation = store.load(conversation_id)
        return ConversationResponse(
            conversation_id=conversation_id,
            messages=to_messages_out(conversation.history()),
        )

    @app.post(
        "/api/conversations/{conversation_id}/messages",
        response_model=ChatResponse,
    )
    def post_message(
        request: Request, conversation_id: str, payload: ChatRequest
    ) -> ChatResponse | JSONResponse:
        """Answers a question and returns the resulting conversation state.

        Args:
            request: The incoming request, used to identify the client.
            conversation_id: Identifier taken from the path.
            payload: The question to answer.

        Returns:
            The answer with the full visible history, or an error response
            carrying a message that can be shown to the user.
        """
        if not _is_valid_conversation_id(conversation_id):
            return _error(400, INVALID_CONVERSATION_ID)

        # Checked before the engine is touched: every answer costs two
        # Bedrock calls.
        refusal = _rate_limited(request)
        if refusal is not None:
            return refusal

        try:
            engine = provider.get()
        except Exception:
            # chromadb reports an unreachable server as a plain ValueError, so
            # the type of a construction failure says nothing about its cause.
            LOGGER.exception("building the conversation engine failed")
            return _error(503, KNOWLEDGE_BASE_UNAVAILABLE)

        try:
            with locks.get(conversation_id):
                result = engine.answer(conversation_id, payload.question)
        except _KNOWLEDGE_BASE_ERRORS:
            LOGGER.exception("the vector store did not answer")
            return _error(503, KNOWLEDGE_BASE_UNAVAILABLE)
        except _ANSWER_SERVICE_ERRORS:
            LOGGER.exception("bedrock did not answer")
            return _error(503, ANSWER_SERVICE_UNAVAILABLE)
        except Exception:
            LOGGER.exception("answering failed")
            return _error(500, UNEXPECTED_ERROR)

        conversation = store.load(conversation_id)
        return ChatResponse(
            conversation_id=conversation_id,
            answer=result.answer,
            messages=to_messages_out(conversation.history()),
        )

    return app


def _error(
    status_code: int, detail: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Builds an error response without leaking internals.

    Args:
        status_code: HTTP status code of the response.
        detail: Message that is shown to the user.
        headers: Additional response headers, for example ``Retry-After``.

    Returns:
        The response carrying only the message.
    """
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(detail=detail).model_dump(),
        headers=headers,
    )

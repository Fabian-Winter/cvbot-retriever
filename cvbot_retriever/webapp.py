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
from pathlib import Path
from typing import Callable

import botocore.exceptions
import chromadb.errors
import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .config import Settings
from .conversation import ConversationStore, InMemoryConversationStore
from .pipeline import ConversationEngine
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
    fully concurrent.
    """

    def __init__(self) -> None:
        """Initializes an empty lock registry."""
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, conversation_id: str) -> threading.Lock:
        """Returns the lock of a conversation.

        Args:
            conversation_id: Identifier of the conversation.

        Returns:
            The lock guarding this conversation.
        """
        with self._guard:
            return self._locks.setdefault(conversation_id, threading.Lock())


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
    store: ConversationStore = InMemoryConversationStore()
    provider = _EngineProvider(effective, store, engine_factory)
    locks = _ConversationLocks()
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    app = FastAPI(title="cvbot", docs_url="/api/docs", redoc_url=None)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Replaces the field-level default response with one friendly text."""
        LOGGER.info("rejected request to %s: %s", request.url.path, exc.errors())
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
    def create_conversation() -> NewConversationResponse:
        """Hands out an identifier for a new conversation."""
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
        conversation_id: str, payload: ChatRequest
    ) -> ChatResponse | JSONResponse:
        """Answers a question and returns the resulting conversation state.

        Args:
            conversation_id: Identifier taken from the path.
            payload: The question to answer.

        Returns:
            The answer with the full visible history, or an error response
            carrying a message that can be shown to the user.
        """
        if not _is_valid_conversation_id(conversation_id):
            return _error(400, INVALID_CONVERSATION_ID)

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


def _error(status_code: int, detail: str) -> JSONResponse:
    """Builds an error response without leaking internals.

    Args:
        status_code: HTTP status code of the response.
        detail: Message that is shown to the user.

    Returns:
        The response carrying only the message.
    """
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(detail=detail).model_dump(),
    )

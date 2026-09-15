"""Command line entry point: ``python -m cvbot_retriever``."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

import uvicorn
from cvbot_core.logging_config import VALID_LOG_LEVELS, configure_logging

from .config import Settings
from .pipeline import ConversationEngine
from .webapp import create_app

LOGGER = logging.getLogger("cvbot_retriever")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line arguments.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv``.

    Returns:
        The parsed arguments. Options that are not given are ``None`` and leave
        the value from the environment untouched.
    """
    parser = argparse.ArgumentParser(
        prog="cvbot_retriever",
        description=(
            "Answers questions from the documents indexed in ChromaDB by "
            "cvbot-embedder, either once on the command line or as a web "
            "application."
        ),
    )
    parser.add_argument(
        "question", nargs="?", help="the question to answer (omit with --serve)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start the chat UI and the JSON API instead of answering once",
    )
    parser.add_argument(
        "--host", help="interface to bind to (default: WEB_HOST or 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, help="port to listen on (default: WEB_PORT or 8080)"
    )
    parser.add_argument("--top-k", type=int, help="number of chunks to retrieve")
    parser.add_argument(
        "--log-level",
        choices=sorted(VALID_LOG_LEVELS),
        help="verbosity of the log output (default: LOG_LEVEL or INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Answers one question or serves the web application.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    args = _parse_args(argv)
    if not args.serve and args.question is None:
        LOGGER.error("a question is required unless --serve is given")
        return 1

    try:
        settings = Settings.from_env().with_overrides(
            top_k=args.top_k,
            log_level=args.log_level,
            web_host=args.host,
            web_port=args.port,
        )
        configure_logging(settings.log_level)
        if args.serve:
            return _serve(settings)

        engine = ConversationEngine(settings)
        result = engine.answer(str(uuid.uuid4()), args.question)
    except Exception:
        LOGGER.exception("answering failed")
        return 1

    print(result.answer)
    return 0


def _serve(settings: Settings) -> int:
    """Runs the web application until the process is stopped.

    Args:
        settings: Runtime configuration.

    Returns:
        ``0`` once the server has shut down.
    """
    LOGGER.info(
        "serving on http://%s:%d", settings.web_host, settings.web_port
    )
    uvicorn.run(
        create_app(settings),
        host=settings.web_host,
        port=settings.web_port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

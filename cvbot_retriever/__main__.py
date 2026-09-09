"""Command line entry point: ``python -m cvbot_retriever``."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Settings
from .pipeline import answer_question

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
            "Answers a single question from the documents indexed in ChromaDB "
            "by cvbot-embedder."
        ),
    )
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--top-k", type=int, help="number of chunks to retrieve")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="verbosity of the log output (default: LOG_LEVEL or INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Answers one question and prints it.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    args = _parse_args(argv)

    try:
        settings = Settings.from_env().with_overrides(
            top_k=args.top_k,
            log_level=args.log_level,
        )
        logging.basicConfig(
            level=settings.log_level,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        )
        result = answer_question(settings, args.question)
    except Exception:
        LOGGER.exception("answering failed")
        return 1

    print(result.answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Per-client rate limiting for the public endpoints.

Answering a question costs one Bedrock embedding call and one Bedrock LLM
call, so an unthrottled public endpoint is a direct cost risk. The limiter
keeps the timestamps of the recent requests per client and rejects a client
once it exceeds the configured budget.

The state is process-local, which matches the deployment: the web application
runs as exactly one ECS task.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

LOGGER = logging.getLogger(__name__)

MINUTE_SECONDS = 60
HOUR_SECONDS = 3600

DEFAULT_MAX_CLIENTS = 10_000


@dataclass(frozen=True)
class RateLimitDecision:
    """Outcome of a limit check.

    Attributes:
        allowed: Whether the request may be served.
        retry_after: Whole seconds the client should wait before retrying;
            zero if the request is allowed.
    """

    allowed: bool
    retry_after: int = 0


class SlidingWindowRateLimiter:
    """Counts requests per client in a sliding minute and hour window.

    Both windows are checked; the stricter one wins. Only allowed requests are
    recorded, so a client that keeps hammering a closed window does not push
    its own recovery further away.
    """

    def __init__(
        self,
        per_minute: int,
        per_hour: int,
        time_source: Callable[[], float] = time.monotonic,
        max_clients: int = DEFAULT_MAX_CLIENTS,
    ) -> None:
        """Initializes the limiter.

        Args:
            per_minute: Allowed requests within 60 seconds.
            per_hour: Allowed requests within 3600 seconds.
            time_source: Monotonic clock, replaced in tests.
            max_clients: Upper bound of tracked clients, so that the limiter
                itself cannot be used to exhaust the memory of the task.

        Raises:
            ValueError: If a limit is not positive or the hour budget is
                smaller than the minute budget.
        """
        if per_minute < 1:
            raise ValueError(f"per_minute must be positive: {per_minute}")
        if per_hour < per_minute:
            raise ValueError(
                f"per_hour must not be smaller than per_minute: {per_hour} < "
                f"{per_minute}"
            )
        if max_clients < 1:
            raise ValueError(f"max_clients must be positive: {max_clients}")

        self._per_minute = per_minute
        self._per_hour = per_hour
        self._now = time_source
        self._max_clients = max_clients
        self._clients: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, client_key: str) -> RateLimitDecision:
        """Checks and, if allowed, records a request of a client.

        Args:
            client_key: Identifier of the calling client.

        Returns:
            The decision including the retry hint for a rejected request.

        Raises:
            ValueError: If the identifier is empty.
        """
        if not client_key.strip():
            raise ValueError("client_key must not be empty")

        now = self._now()
        with self._lock:
            hits = self._bucket(client_key, now)
            self._drop_before(hits, now - HOUR_SECONDS)

            in_hour = len(hits)
            in_minute = sum(1 for hit in hits if hit > now - MINUTE_SECONDS)

            if in_minute >= self._per_minute:
                oldest = hits[in_hour - in_minute]
                return self._reject(client_key, oldest + MINUTE_SECONDS - now)
            if in_hour >= self._per_hour:
                return self._reject(client_key, hits[0] + HOUR_SECONDS - now)

            hits.append(now)
            return RateLimitDecision(allowed=True)

    def _bucket(self, client_key: str, now: float) -> deque[float]:
        """Returns the timestamps of a client, creating them on first use.

        Args:
            client_key: Identifier of the calling client.
            now: Current reading of the clock.

        Returns:
            The deque of recent request timestamps.
        """
        hits = self._clients.get(client_key)
        if hits is None:
            self._evict(now)
            hits = deque()
            self._clients[client_key] = hits
        self._clients.move_to_end(client_key)
        return hits

    def _evict(self, now: float) -> None:
        """Frees room for a new client before it is tracked.

        Drops clients whose window has fully expired and, if that is not
        enough, the least recently seen one.

        Args:
            now: Current reading of the clock.
        """
        if len(self._clients) < self._max_clients:
            return

        horizon = now - HOUR_SECONDS
        stale = [
            key
            for key, hits in self._clients.items()
            if not hits or hits[-1] <= horizon
        ]
        for key in stale:
            del self._clients[key]

        while len(self._clients) >= self._max_clients:
            self._clients.popitem(last=False)

    def _reject(self, client_key: str, retry_after: float) -> RateLimitDecision:
        """Builds the decision for a client that is over its budget.

        Args:
            client_key: Identifier of the calling client.
            retry_after: Seconds until the oldest recorded request expires.

        Returns:
            The rejecting decision with at least one second of wait time.
        """
        LOGGER.warning("rate limit reached for client %s", client_key)
        return RateLimitDecision(
            allowed=False, retry_after=max(1, int(retry_after) + 1)
        )

    @staticmethod
    def _drop_before(hits: deque[float], horizon: float) -> None:
        """Removes timestamps that left the largest window.

        Args:
            hits: Timestamps of one client, oldest first.
            horizon: Point in time before which entries no longer count.
        """
        while hits and hits[0] <= horizon:
            hits.popleft()

"""Tests of the per-client rate limiter."""

from __future__ import annotations

import pytest

from cvbot_retriever.ratelimit import SlidingWindowRateLimiter


class FakeClock:
    """Monotonic clock that only moves when a test moves it."""

    def __init__(self) -> None:
        """Starts the clock at zero."""
        self.value = 0.0

    def __call__(self) -> float:
        """Returns the current reading."""
        return self.value

    def advance(self, seconds: float) -> None:
        """Moves the clock forward.

        Args:
            seconds: Amount of time that passes.
        """
        self.value += seconds


@pytest.fixture
def clock() -> FakeClock:
    """Provides a clock the test controls."""
    return FakeClock()


def make_limiter(
    clock: FakeClock, per_minute: int = 3, per_hour: int = 10, **kwargs: int
) -> SlidingWindowRateLimiter:
    """Builds a limiter running on the test clock.

    Args:
        clock: The clock driving the windows.
        per_minute: Allowed requests per minute.
        per_hour: Allowed requests per hour.
        **kwargs: Further arguments of the limiter.

    Returns:
        The limiter.
    """
    return SlidingWindowRateLimiter(
        per_minute=per_minute, per_hour=per_hour, time_source=clock, **kwargs
    )


def test_requests_below_the_limit_are_allowed(clock: FakeClock) -> None:
    limiter = make_limiter(clock)

    decisions = [limiter.check("1.2.3.4") for _ in range(3)]

    assert all(decision.allowed for decision in decisions)


def test_the_request_beyond_the_minute_limit_is_rejected(
    clock: FakeClock,
) -> None:
    limiter = make_limiter(clock)
    for _ in range(3):
        limiter.check("1.2.3.4")

    decision = limiter.check("1.2.3.4")

    assert not decision.allowed
    assert decision.retry_after > 0


def test_the_window_reopens_after_a_minute(clock: FakeClock) -> None:
    limiter = make_limiter(clock)
    for _ in range(3):
        limiter.check("1.2.3.4")
    assert not limiter.check("1.2.3.4").allowed

    clock.advance(61)

    assert limiter.check("1.2.3.4").allowed


def test_the_window_slides_instead_of_resetting(clock: FakeClock) -> None:
    limiter = make_limiter(clock, per_minute=2)
    limiter.check("1.2.3.4")
    clock.advance(30)
    limiter.check("1.2.3.4")

    # The first hit has expired, the second one has not.
    clock.advance(31)

    assert limiter.check("1.2.3.4").allowed
    assert not limiter.check("1.2.3.4").allowed


def test_the_hour_limit_also_applies(clock: FakeClock) -> None:
    limiter = make_limiter(clock, per_minute=2, per_hour=4)

    for _ in range(2):
        assert limiter.check("1.2.3.4").allowed
        clock.advance(61)
    assert limiter.check("1.2.3.4").allowed
    assert limiter.check("1.2.3.4").allowed

    clock.advance(61)
    decision = limiter.check("1.2.3.4")

    assert not decision.allowed
    assert decision.retry_after > 60


def test_the_hour_window_reopens(clock: FakeClock) -> None:
    limiter = make_limiter(clock, per_minute=2, per_hour=2)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")

    clock.advance(3601)

    assert limiter.check("1.2.3.4").allowed


def test_clients_have_separate_budgets(clock: FakeClock) -> None:
    limiter = make_limiter(clock)
    for _ in range(3):
        limiter.check("1.2.3.4")

    assert not limiter.check("1.2.3.4").allowed
    assert limiter.check("5.6.7.8").allowed


def test_a_rejected_request_does_not_extend_the_block(clock: FakeClock) -> None:
    limiter = make_limiter(clock, per_minute=1)
    limiter.check("1.2.3.4")

    clock.advance(30)
    assert not limiter.check("1.2.3.4").allowed

    clock.advance(31)
    assert limiter.check("1.2.3.4").allowed


def test_expired_clients_are_forgotten(clock: FakeClock) -> None:
    limiter = make_limiter(clock, max_clients=2)
    limiter.check("1.2.3.4")
    limiter.check("5.6.7.8")

    clock.advance(3601)
    limiter.check("9.9.9.9")

    assert limiter._clients.keys() == {"9.9.9.9"}


def test_the_number_of_tracked_clients_stays_bounded(clock: FakeClock) -> None:
    limiter = make_limiter(clock, max_clients=5)

    for index in range(50):
        limiter.check(f"10.0.0.{index}")

    assert len(limiter._clients) <= 5


def test_an_empty_client_key_raises(clock: FakeClock) -> None:
    limiter = make_limiter(clock)

    with pytest.raises(ValueError, match="client_key"):
        limiter.check("   ")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"per_minute": 0, "per_hour": 10},
        {"per_minute": 5, "per_hour": 4},
        {"per_minute": 1, "per_hour": 1, "max_clients": 0},
    ],
)
def test_invalid_limits_raise(clock: FakeClock, kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(time_source=clock, **kwargs)

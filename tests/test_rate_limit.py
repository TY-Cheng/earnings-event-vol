from __future__ import annotations

import pytest

from earnings_event_vol import rate_limit


@pytest.fixture(autouse=True)
def reset_rate_limit_state() -> None:
    rate_limit._NEXT_REQUEST_MONOTONIC = 0.0


def test_throttle_requests_per_minute_ignores_none_and_nonpositive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "earnings_event_vol.rate_limit.time.sleep",
        lambda seconds: pytest.fail(f"unexpected sleep: {seconds}"),
    )

    rate_limit.throttle_requests_per_minute(None)
    rate_limit.throttle_requests_per_minute(0)
    rate_limit.throttle_requests_per_minute(-1)

    assert rate_limit._NEXT_REQUEST_MONOTONIC == 0.0


def test_throttle_requests_per_minute_schedules_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("earnings_event_vol.rate_limit.time.monotonic", lambda: 100.0)
    monkeypatch.setattr(
        "earnings_event_vol.rate_limit.time.sleep",
        lambda seconds: pytest.fail(f"unexpected sleep: {seconds}"),
    )

    rate_limit.throttle_requests_per_minute(120)

    assert rate_limit._NEXT_REQUEST_MONOTONIC == 100.5


def test_throttle_requests_per_minute_waits_until_global_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_values = iter([100.0, 101.0])
    sleeps: list[float] = []
    rate_limit._NEXT_REQUEST_MONOTONIC = 101.0
    monkeypatch.setattr(
        "earnings_event_vol.rate_limit.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr("earnings_event_vol.rate_limit.time.sleep", sleeps.append)

    rate_limit.throttle_requests_per_minute(60)

    assert sleeps == [1.0]
    assert rate_limit._NEXT_REQUEST_MONOTONIC == 102.0

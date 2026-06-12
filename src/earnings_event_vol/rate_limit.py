from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_NEXT_REQUEST_MONOTONIC = 0.0


def throttle_requests_per_minute(requests_per_minute: int | None) -> None:
    if requests_per_minute is None:
        return
    rate = int(requests_per_minute)
    if rate <= 0:
        return
    interval_seconds = 60.0 / float(rate)
    global _NEXT_REQUEST_MONOTONIC
    with _LOCK:
        now = time.monotonic()
        wait_seconds = max(0.0, _NEXT_REQUEST_MONOTONIC - now)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
        _NEXT_REQUEST_MONOTONIC = max(_NEXT_REQUEST_MONOTONIC, now) + interval_seconds

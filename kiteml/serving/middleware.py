"""serving/middleware.py — Request logging middleware for KiteML serving."""

import time
from typing import Callable


def latency_logging_middleware(request_handler: Callable, log: bool = True) -> Callable:
    """
    Wrap a request handler to log latency and request counts.
    Compatible with plain functions; for FastAPI use the ASGI middleware.
    """
    call_count = [0]

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = request_handler(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        call_count[0] += 1
        if log:
            print(f"  [MW] Request #{call_count[0]} | {elapsed_ms:.2f}ms")
        return result

    return wrapper

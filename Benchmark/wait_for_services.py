"""Polls each benchmark backend's health endpoint until ready (or a timeout elapses).

Run directly: python -m Benchmark.wait_for_services
"""
import sys
import time

import httpx

DEFAULT_TIMEOUT_S = 120
DEFAULT_POLL_INTERVAL_S = 2

CHECKS = {
    "postgres": lambda: _tcp_ok("localhost", 5433),
    "qdrant": lambda: _http_ok("http://localhost:6333/healthz"),
    "weaviate": lambda: _http_ok("http://localhost:8080/v1/.well-known/ready"),
    "meilisearch": lambda: _http_ok("http://localhost:7700/health"),
}


def _http_ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code < 400
    except Exception:
        return False


def _tcp_ok(host: str, port: int) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


def wait_for_all(timeout_s: int = DEFAULT_TIMEOUT_S, poll_interval_s: int = DEFAULT_POLL_INTERVAL_S) -> bool:
    pending = set(CHECKS)
    start = time.perf_counter()
    while pending and (time.perf_counter() - start) < timeout_s:
        for name in list(pending):
            if CHECKS[name]():
                print(f"  {name}: ready ({time.perf_counter() - start:.1f}s)")
                pending.discard(name)
        if pending:
            time.sleep(poll_interval_s)
    if pending:
        print(f"Timed out waiting for: {sorted(pending)}")
        return False
    print("All backend services ready.")
    return True


if __name__ == "__main__":
    ok = wait_for_all()
    sys.exit(0 if ok else 1)

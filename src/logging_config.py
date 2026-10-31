"""Central logging to the terminal (stderr), with a per-request trace id.

Every log line carries `[rid=...]` so one chatbot question can be followed end to end
across Streamlit / FastAPI -> agent -> LLM calls -> tools in the run-time output.

Level is set with LOG_LEVEL (default INFO).
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
import uuid
from contextlib import contextmanager

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | [rid=%(request_id)s] %(message)s"

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    """Idempotent: safe to call from every entry point (Streamlit reruns, FastAPI, CLI)."""
    root = logging.getLogger("credit_risk")
    if root.handlers:
        return
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    root.propagate = False
    fmt = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(_RequestIdFilter())
    root.addHandler(console)

    # keep third-party noise down
    for noisy in ("httpx", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"credit_risk.{name}")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:8]
    request_id_var.set(rid)
    return rid


@contextmanager
def timed(logger: logging.Logger, what: str, level: int = logging.INFO):
    """Log the duration of a block; logs the exception (and re-raises) if it fails."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("%s FAILED after %.2fs", what, time.perf_counter() - start)
        raise
    logger.log(level, "%s done in %.2fs", what, time.perf_counter() - start)

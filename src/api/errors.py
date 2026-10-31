"""Shared error handling for the API routers."""

import functools

from fastapi import HTTPException

from src.logging_config import get_logger

log = get_logger("api")


def handle_errors(endpoint: str):
    """Log + convert unexpected exceptions into a 500 with a reason; keep HTTPExceptions as-is."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except HTTPException as e:
                log.warning("%s -> HTTP %d: %s", endpoint, e.status_code, e.detail)
                raise
            except (ValueError, KeyError) as e:
                log.warning("%s -> bad request: %s", endpoint, e)
                raise HTTPException(400, {"error_code": "bad_request", "message": str(e)}) from e
            except Exception as e:
                log.exception("%s -> unhandled error: %s", endpoint, e)
                raise HTTPException(500, {"error_code": "internal_error",
                                          "message": f"{type(e).__name__}: {e}"}) from e
        return wrapper
    return deco


def raise_if_tool_error(out: dict, status: int = 400) -> dict:
    """Tools return {"error": ...} instead of raising; surface that as an HTTP error."""
    if isinstance(out, dict) and "error" in out:
        raise HTTPException(status, {"error_code": "tool_error", "message": out["error"], "hint": out.get("hint")})
    return out

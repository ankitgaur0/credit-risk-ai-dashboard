"""FastAPI service exposing the same analytics, model, anomaly and agent logic as the dashboard.

Run:  uv run uvicorn src.api.main:app --reload     (docs at http://127.0.0.1:8000/docs)
"""

import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src import data_store as ds
from src.agent.agent_model.portfolio_agent import get_api_key
from src.api.routers import agent, anomaly, model, portfolio
from src.logging_config import get_logger, new_request_id, request_id_var

load_dotenv()
log = get_logger("api")



@asynccontextmanager
async def lifespan(_: FastAPI):
    # warm the caches so the first request is fast
    log.info("API startup: warming caches (llm_configured=%s)", bool(get_api_key()))
    ds.get_dataset()
    ds.get_anomalies()
    log.info("API ready")
    yield
    log.info("API shutdown")


app = FastAPI(lifespan=lifespan, title="Credit Risk Portfolio API", version="1.0.0",
              description="Portfolio analytics, default model scoring, anomaly review and Ask the Portfolio (Gemini).")



@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Assign a trace id, log method/path/status/duration, return the id in X-Request-ID."""
    rid = request.headers.get("X-Request-ID") or new_request_id()
    request_id_var.set(rid)
    start = time.perf_counter()
    log.info("--> %s %s", request.method, request.url.path + (f"?{request.url.query}" if request.url.query else ""))
    try:
        response = await call_next(request)
    except Exception as e:  # last-resort safety net
        log.exception("<-- %s %s 500 unhandled: %s", request.method, request.url.path, e)
        response = JSONResponse(status_code=500, content={"error_code": "internal_error", "message": str(e),
                                                          "request_id": rid})
    log.info("<-- %s %s %d in %.2fs", request.method, request.url.path, response.status_code,
             time.perf_counter() - start)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]
    log.warning("%s %s -> 422 validation failed: %s", request.method, request.url.path, errors)
    return JSONResponse(status_code=422, content={"error_code": "validation_error", "errors": errors})


app.include_router(portfolio.router)
app.include_router(model.router)
app.include_router(anomaly.router)
app.include_router(agent.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "accounts": len(ds.get_dataset()), "llm_configured": bool(get_api_key())}

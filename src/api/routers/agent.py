"""'Ask the Portfolio' LLM endpoint."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.agent.agent_model.portfolio_agent import AI_FEATURE_TEXT, AgentError, ask
from src.api.errors import log
from src.schemas.api import AskRequest, AskResponse

router = APIRouter(prefix="/agent", tags=["ask-the-portfolio"])


@router.post("/ask", response_model=AskResponse,
             responses={429: {"description": "LLM rate limited"}, 503: {"description": "LLM not configured / overloaded"}})
def ask_portfolio(req: AskRequest):
    try:
        return ask(req.question, req.history)
    except AgentError as e:  # already logged with full detail inside ask()
        log.warning("POST /agent/ask -> HTTP %d %s", e.status, e.code)
        return JSONResponse(status_code=e.status, content=e.to_dict())


@router.get("/about")
def about():
    return AI_FEATURE_TEXT

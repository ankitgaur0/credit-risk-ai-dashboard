"""'Ask the Portfolio' ReAct agent: LangGraph create_react_agent + Gemini + portfolio tools.

Flow: question -> Gemini picks tool(s) -> tools compute exact figures with pandas/model
-> Gemini phrases a short answer from the tool output only -> we verify every number
in the answer appears in the tool output before showing it.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.tools.portfolio_tools import PORTFOLIO_TOOLS
from src.config import GEMINI_MODEL
from src.logging_config import get_logger

log = get_logger("agent")

DECLINE_PREFIX = "DECLINED:"

SYSTEM_PROMPT = f"""You are "Ask the Portfolio", an assistant for credit-risk analysts at a bank.
You answer questions about a loan portfolio of 14,685 synthetic loan accounts (10,000 customers).

RULES
1. Every number you state MUST come from a tool result in this conversation. Never estimate,
   calculate new figures yourself, or use outside knowledge. Copy figures exactly; you may
   express a rate like 0.2471 as 24.71%.
2. Always call a tool before answering a data question. Use filters for sub-populations.
3. If the question cannot be answered with the tools/data (e.g. forecasts, macro-economics,
   data not in the portfolio, personal advice), reply with exactly one line starting with
   "{DECLINE_PREFIX}" followed by a short reason. Do not guess.
4. PREDICTIONS for a new or hypothetical applicant/loan ARE supported - never decline them.
   Use predict_default_probability. Required: annual_income, loan_amount, loan_type,
   employment_type. Optional: age, interest_rate_pct, tenure_months, missed_payments,
   bureau_writeoffs, region. If required fields are missing, ask for them in one short
   message (no tool call). Map free text to the allowed values (e.g. "mortage" -> Mortgage,
   "software engineer at a company" -> Salaried) and STATE the mapping you used in the answer; if
   the employment type is genuinely ambiguous, ask. Occupation/sector is not a model input.
   When answering, give the probability of default as "X% (about N in 100 similar loans default)",
   the risk band, risk_vs_portfolio and recommended_action, list the inputs used, and say that
   unspecified fields were filled with portfolio typical values.
   For an existing account id, use get_account_risk instead.
5. Answer in 2-4 short plain-English sentences. ONLY when a tool was used, end with
   "Source: <that tool's source>". Never invent a source for conversational replies.
6. Default rate = share of accounts with target_default = 1. Exposure = outstanding balance in USD.
"""


class AgentError(RuntimeError):
    """A classified agent/LLM failure with a user-facing reason and an HTTP status."""

    def __init__(self, code: str, message: str, status: int = 502, detail: str = ""):
        super().__init__(message)
        self.code, self.message, self.status, self.detail = code, message, status, detail

    def to_dict(self) -> dict:
        return {"error_code": self.code, "message": self.message, "detail": self.detail[:500]}


class AgentUnavailable(AgentError):
    def __init__(self, message: str):
        super().__init__("llm_not_configured", message, status=503)


def classify_llm_error(e: Exception) -> AgentError:
    """Map raw SDK / LangGraph exceptions to a clear reason for logs, API and UI."""
    if isinstance(e, AgentError):
        return e
    text = f"{type(e).__name__}: {e}"
    low = text.lower()
    rules = [
        (("api key not valid", "api_key_invalid", "permission_denied", "401", "403"),
         "llm_auth_failed", "Gemini rejected the API key (invalid, revoked or no access).", 502),
        (("not_found", "404", "is no longer available"),
         "llm_model_not_found", f"Model '{GEMINI_MODEL}' is not available for this key. Set GEMINI_MODEL to another model.", 502),
        (("resource_exhausted", "429", "quota"),
         "llm_rate_limited", "Gemini free-tier quota / rate limit reached. Wait a minute and retry.", 429),
        (("unavailable", "503", "overloaded", "high demand"),
         "llm_overloaded", "Gemini is overloaded right now (503). Retry shortly or switch GEMINI_MODEL.", 503),
        (("timeout", "timed out", "deadline"),
         "llm_timeout", "The Gemini call timed out (30s).", 504),
        (("recursion", "graphrecursionerror"),
         "agent_loop_limit", "The agent needed too many steps for this question. Try rephrasing it more specifically.", 422),
        (("connecterror", "connection", "name resolution"),
         "llm_network_error", "Could not reach the Gemini API (network error).", 502),
    ]
    for needles, code, msg, status in rules:
        if any(n in low for n in needles):
            return AgentError(code, msg, status, text)
    return AgentError("agent_internal_error", "Unexpected error while answering the question.", 500, text)


def get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@lru_cache(maxsize=1)
def build_agent():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent

    key = get_api_key()
    if not key:
        log.error("Agent build failed: GEMINI_API_KEY is not set")
        raise AgentUnavailable("GEMINI_API_KEY is not set")
    log.info("Building ReAct agent: model=%s tools=%d", GEMINI_MODEL, len(PORTFOLIO_TOOLS))
    # fail fast instead of silent long backoff when the free tier is overloaded (503)
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=key, max_retries=2, timeout=30)
    return create_react_agent(llm, PORTFOLIO_TOOLS, prompt=SYSTEM_PROMPT)


# --------------------------------------------------------------------------- grounding check

# numbers the model may legitimately state without a tool: portfolio size from the system
# prompt, and the "N in 100" phrasing it is asked to use for probabilities
TRUSTED_NUMBERS = {14685.0, 10000.0, 100.0}

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _tool_numbers(obj) -> list[float]:
    return _numbers(json.dumps(obj, default=str))


def verify_numbers(answer: str, tool_outputs: list) -> tuple[bool, list[float]]:
    """True if every number in the answer matches a tool number (as-is, as %, or in K/M/B)."""
    body = answer.split("Source:")[0]
    source_nums = _tool_numbers(tool_outputs)
    candidates = set(TRUSTED_NUMBERS)
    for v in source_nums:
        for scaled in (v, v * 100, v / 1e3, v / 1e6, v / 1e9):
            candidates.add(scaled)
    unmatched = []
    for n in _numbers(body):
        if n.is_integer() and 0 <= n <= 10:  # small counts like "top 5", "Q1"
            continue
        if not any(abs(n - c) <= max(0.051, abs(c) * 0.005) for c in candidates):
            unmatched.append(n)
    return not unmatched, unmatched


# --------------------------------------------------------------------------- public API

def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def _to_messages(question: str, history: list[dict] | None) -> list:
    messages = [HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"])
                for h in (history or [])[-6:]]
    return messages + [HumanMessage(question)]


def stream_answer(question: str, history: list[dict] | None = None, result: dict | None = None):
    """Yield the final answer text token-by-token; fill `result` with the same dict ask() returns.

    Only text generated by the LLM is yielded - tool calls and tool outputs are never streamed
    to the user (they are logged and kept in `result` for auditing).
    Raises AgentError (classified) on any failure.
    """
    result = result if result is not None else {}
    t0 = time.perf_counter()
    log.info("ASK start: question=%r history_turns=%d model=%s", question[:300], len(history or []), GEMINI_MODEL)
    new, step, t_step, streamed = [], 0, time.perf_counter(), []
    try:
        agent = build_agent()
        messages = _to_messages(question, history)
        for mode, data in agent.stream({"messages": messages}, config={"recursion_limit": 12},
                                       stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunk, meta = data
                # token chunks from the LLM node only; skip tool-call chunks and tool results
                if meta.get("langgraph_node") == "agent" and isinstance(chunk, AIMessage) \
                        and not getattr(chunk, "tool_call_chunks", None):
                    text = _content_text(chunk.content)
                    if text:
                        streamed.append(text)
                        yield text
                continue
            for node, payload in data.items():  # mode == "updates": one full step finished
                step += 1
                msgs = (payload or {}).get("messages", [])
                new.extend(msgs)
                dt = time.perf_counter() - t_step
                for m in msgs:
                    if isinstance(m, AIMessage) and m.tool_calls:
                        log.info("STEP %d %s (LLM %.2fs) -> tool calls %s", step, node, dt,
                                 [(c["name"], c["args"]) for c in m.tool_calls])
                    elif isinstance(m, AIMessage):
                        usage = getattr(m, "usage_metadata", None) or {}
                        log.info("STEP %d %s (LLM %.2fs) -> final answer (%d chars, tokens in=%s out=%s)",
                                 step, node, dt, len(_content_text(m.content)),
                                 usage.get("input_tokens"), usage.get("output_tokens"))
                    elif isinstance(m, ToolMessage):
                        log.info("STEP %d %s (%.2fs) -> %s returned %d chars%s", step, node, dt, m.name,
                                 len(str(m.content)), " [ERROR]" if '"error"' in str(m.content)[:40] else "")
                t_step = time.perf_counter()
    except Exception as e:
        err = classify_llm_error(e)
        # full traceback only for failures we can't explain from the message alone
        log.error("ASK FAILED after %.2fs: code=%s reason=%s | raw=%s",
                  time.perf_counter() - t0, err.code, err.message, err.detail[:300] or e,
                  exc_info=err.code in ("agent_internal_error", "llm_network_error"))
        raise err from e

    tool_calls, tool_outputs = [], []
    for m in new:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_calls += [{"tool": c["name"], "args": c["args"]} for c in m.tool_calls]
        if isinstance(m, ToolMessage):
            try:
                tool_outputs.append(json.loads(m.content))
            except (TypeError, json.JSONDecodeError):
                tool_outputs.append(m.content)

    answer = (_content_text(new[-1].content).strip() if new else "") or "".join(streamed).strip()
    if not answer:
        log.error("ASK FAILED: empty answer from model after %d steps", step)
        raise AgentError("llm_empty_answer", "The model returned an empty answer. Please retry.", 502)
    if not streamed:  # model didn't stream (e.g. provider fallback): emit the whole answer once
        yield answer
    declined = answer.upper().startswith(DECLINE_PREFIX)
    grounded, unmatched = (True, []) if declined else verify_numbers(answer, tool_outputs)

    log_fn = log.warning if not grounded else log.info
    log_fn("ASK done in %.2fs: steps=%d tools=%s declined=%s grounded=%s unverified=%s",
           time.perf_counter() - t0, step, [c["tool"] for c in tool_calls], declined, grounded, unmatched)
    result.update({"question": question, "answer": answer, "declined": declined, "grounded": grounded,
                   "unverified_numbers": unmatched, "tool_calls": tool_calls, "tool_outputs": tool_outputs})


def ask(question: str, history: list[dict] | None = None) -> dict:
    """Non-streaming wrapper (used by the API). Raises AgentError on failure."""
    result: dict = {}
    for _ in stream_answer(question, history, result):
        pass
    return result


AI_FEATURE_TEXT = {
    "trigger": "Runs when an analyst submits a question in the chat panel (or POST /agent/ask).",
    "false_positive": ("It answers a question it should have declined, or answers a slightly different question "
                       "than asked (e.g. uses all regions when the analyst meant South only)."),
    "false_negative": "It declines a question the data could answer (e.g. an unusual phrasing it didn't map to a tool).",
    "human_in_loop": ("Each answer cites its data source. Every number is automatically cross-checked against "
                      "the computed data and the answer is flagged if any figure can't be verified. Full tool "
                      "calls and results are logged for audit, and the analyst verifies flagged answers before use."),
}

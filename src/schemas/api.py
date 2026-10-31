"""Pydantic request / response models for the FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Which loan type has the highest default rate?"])
    history: list[dict[str, str]] = Field(default_factory=list,
                                          description='[{"role": "user"|"assistant", "content": "..."}]')


class AskResponse(BaseModel):
    question: str
    answer: str
    declined: bool
    grounded: bool
    unverified_numbers: list[float]
    tool_calls: list[dict[str, Any]]
    tool_outputs: list[Any]


class ApplicantRequest(BaseModel):
    annual_income: float = Field(..., gt=0, examples=[45000])
    loan_amount: float = Field(..., gt=0, examples=[20000])
    loan_type: Literal["Personal Loan", "Auto Loan", "Credit Card", "Mortgage", "Business Loan"]
    employment_type: Literal["Salaried", "Self-Employed", "Business Owner", "Contract", "Unemployed"]
    age: int | None = Field(None, ge=18, le=100)
    interest_rate_pct: float | None = None
    tenure_months: int | None = None
    missed_payments: int | None = Field(None, ge=0)
    bureau_writeoffs: int | None = Field(None, ge=0)
    region: Literal["North", "South", "East", "West", "Central"] | None = None


class DecisionRequest(BaseModel):
    decision: Literal["Confirm", "Dismiss"]
    note: str = ""
    reviewer: str = "analyst"

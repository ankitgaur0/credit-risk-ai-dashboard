"""Anomaly review-list endpoints with analyst decisions."""

from fastapi import APIRouter, HTTPException

from src import data_store as ds
from src.anomaly import TRIGGER_TEXT
from src.api.errors import handle_errors, log
from src.schemas.api import DecisionRequest

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("")
@handle_errors("GET /anomalies")
def flagged(limit: int = 50, flag_type: str | None = None):
    a = ds.get_anomalies()
    f = a[a["flagged"]]
    if flag_type:
        if flag_type not in ("Own history", "Peer segment"):
            raise HTTPException(400, {"error_code": "bad_request",
                                      "message": "flag_type must be 'Own history' or 'Peer segment'"})
        f = f[f["flag_type"] == flag_type]
    f = f.merge(ds.latest_decisions()[["account_id", "decision"]], on="account_id", how="left")
    cols = ["account_id", "loan_type", "flag_type", "anomaly_score", "pd_score", "exposure", "reasons", "decision"]
    page = f[cols].head(limit)
    log.info("Anomaly list served: %d of %d flagged", len(page), len(f))
    return {"flagged_total": len(f), "about": TRIGGER_TEXT,
            "rows": page.astype(object).where(page.notna(), None).to_dict("records")}


@router.post("/{account_id}/decision")
@handle_errors("POST /anomalies/{id}/decision")
def decide(account_id: str, req: DecisionRequest):
    a = ds.get_anomalies()
    if account_id not in set(a.loc[a["flagged"], "account_id"]):
        raise HTTPException(404, {"error_code": "not_flagged",
                                  "message": f"{account_id} is not on the anomaly review list"})
    return ds.save_decision(account_id, req.decision, req.note, req.reviewer)


@router.get("/decisions")
@handle_errors("GET /anomalies/decisions")
def decisions():
    return ds.load_decisions().fillna("").to_dict("records")

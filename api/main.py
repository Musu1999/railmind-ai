"""RailMind AI — REST API (FastAPI).

Run:   uvicorn api.main:app --reload
Docs:  http://localhost:8000/docs     Auth header:  X-API-Key: <key>  (see railmind/security.py)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from railmind import faults  # noqa: E402
from railmind.governance import monitoring, registry, simulator  # noqa: E402
from railmind.governance.store import Store  # noqa: E402
from railmind.governance.supervisor import SupervisorAgent, apply_approval  # noqa: E402
from railmind.passenger.base import GovernanceBlocked  # noqa: E402
from railmind.passenger.confirmation import ConfirmationPredictor  # noqa: E402
from railmind.passenger.delay import DelayPredictor  # noqa: E402
from railmind.passenger.fare import FarePredictor  # noqa: E402
from railmind.passenger.fraud import FraudDetector  # noqa: E402
from railmind.passenger.planner import PlannerAgent, book  # noqa: E402
from railmind.passenger.recommender import TrainRecommender  # noqa: E402
from railmind.passenger.support import SupportAssistant  # noqa: E402
from railmind.security import allowed, key_fingerprint, role_for_key  # noqa: E402

app = FastAPI(title="RailMind AI", version="1.0.0",
              description="Intelligent railway travel services with an autonomous AI-governance layer.")
_PLANS: dict[str, dict] = {}
_assistant: SupportAssistant | None = None


def requires(permission: str):
    def dep(x_api_key: str | None = Header(default=None)) -> str:
        role = role_for_key(x_api_key)
        if role is None:
            raise HTTPException(401, "missing or invalid X-API-Key")
        if not allowed(role, permission):
            Store.get().audit(f"key:{key_fingerprint(x_api_key)}", "access.denied", permission, {"role": role}, "user")
            raise HTTPException(403, f"role '{role}' lacks permission '{permission}'")
        return role
    return dep


def clean(obj):
    """JSON-safe: NaN/inf -> None, numpy/pandas scalars -> Python, DataFrames -> records."""
    import math

    import numpy as np
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        return clean(obj.to_dict("records"))
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (pd.Timestamp, date)):
        return str(obj)
    return obj


def _explain(r: dict) -> dict:
    r = dict(r)
    if "explanation" in r:
        r["explanation"] = r["explanation"].head(6)[["label", "value", "contribution"]].to_dict("records")
    r.pop("record", None)
    return r


def _guard(fn):
    try:
        return fn()
    except GovernanceBlocked as e:
        raise HTTPException(422, {"blocked_by": "data_quality_gate", "score": e.result.score, "issues": e.result.issues})
    except KeyError as e:
        raise HTTPException(404, str(e))


# ------------------------------------------------------------------ schemas
class ConfirmationIn(BaseModel):
    train_no: str = Field(examples=["12089"])
    travel_class: str = Field("SL", examples=["SL"])
    journey_date: date
    position: int = Field(ge=1, le=1000)
    status: Literal["WL", "RAC"] = "WL"
    quota: Literal["GN", "TQ", "LD"] = "GN"


class RecommendIn(BaseModel):
    source: str = Field(examples=["Mumbai"])
    destination: str = Field(examples=["Delhi"])
    journey_date: date
    budget: float | None = None
    max_duration_h: float | None = None
    budget_level: Literal["low", "mid", "high"] = "mid"
    preferred_class: str = "3A"
    preferred_time: Literal["morning", "afternoon", "night"] = "night"
    punctuality_weight: float = Field(0.6, ge=0, le=1)


class DelayIn(BaseModel):
    train_no: str
    journey_date: date
    weather: Literal["clear", "rain", "storm", "fog"] | None = None
    departure_delay_min: float = 0
    maintenance_block: bool = False


class FareIn(BaseModel):
    train_no: str
    travel_class: str
    journey_date: date


class PlanIn(BaseModel):
    query: str = Field(examples=["I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM."])


class BookIn(BaseModel):
    run_id: str
    approval_token: str


class ChatIn(BaseModel):
    message: str
    session: str | None = None


class FraudIn(BaseModel):
    account_age_days: float
    bookings_last_24h: int
    passengers_per_booking: int
    session_duration_s: float
    avg_click_interval_ms: float
    form_fill_time_s: float
    captcha_failures: int
    payment_failures: int
    distinct_cards_7d: int
    accounts_per_device: int
    ip_reputation: float = Field(ge=0, le=1)
    is_tatkal: int
    booking_hour: int = Field(ge=0, le=23)
    geo_mismatch: int
    amount: float


class ApprovalIn(BaseModel):
    approve: bool


# ---------------------------------------------------------------- passenger
@app.get("/health")
def health():
    return {"status": "ok", "models": {m: registry.production_version(m) for m in registry.list_models()}}


@app.post("/v1/confirmation/predict")
def confirmation(body: ConfirmationIn, role: str = Depends(requires("passenger"))):
    return clean(_guard(lambda: _explain(ConfirmationPredictor().predict(body.train_no, body.travel_class, body.journey_date,
                                                                   body.position, body.status, body.quota))))


@app.post("/v1/recommendations")
def recommend(body: RecommendIn, role: str = Depends(requires("passenger"))):
    user = body.model_dump(include={"budget_level", "preferred_class", "preferred_time", "punctuality_weight"})
    res = TrainRecommender().recommend(body.source, body.destination, body.journey_date, user, body.budget,
                                       max_duration_h=body.max_duration_h)
    items = res["items"]
    cols = ["train_no", "train_name", "class", "fare", "dep_time", "arr_time", "duration_min", "availability_label",
            "confirm_chance", "punctuality", "score", "why"]
    return clean({"version": res.get("version"), "items": items[cols].to_dict("records") if len(items) else [],
            "message": res.get("message")})


@app.post("/v1/delay/predict")
def delay(body: DelayIn, role: str = Depends(requires("passenger"))):
    return clean(_guard(lambda: _explain(DelayPredictor().predict(body.train_no, body.journey_date, body.weather,
                                                            body.departure_delay_min, int(body.maintenance_block)))))


@app.post("/v1/fares/forecast")
def fares(body: FareIn, role: str = Depends(requires("passenger"))):
    r = _guard(lambda: FarePredictor().forecast(body.train_no, body.travel_class, body.journey_date))
    r = _explain(r)
    r["curve"] = r["curve"].assign(date=r["curve"]["date"].astype(str)).to_dict("records")
    return clean(r)


@app.post("/v1/planner/plan")
def plan(body: PlanIn, role: str = Depends(requires("passenger"))):
    r = PlannerAgent().plan(body.query)
    _PLANS[r["run_id"]] = r
    best = r["best"]
    return clean({"run_id": r["run_id"], "outcome": r["outcome"], "message": r["message"], "valid": r["valid"],
            "anomalies": r["anomalies"], "approval_token": r["approval_token"],
            "best": {**best, "departure": str(best["departure"]), "scheduled_arrival": str(best["scheduled_arrival"]),
                     "expected_arrival": str(best["expected_arrival"])} if best else None,
            "trace": r["trace"].to_dict("records")})


@app.post("/v1/planner/book")
def plan_book(body: BookIn, role: str = Depends(requires("passenger"))):
    r = _PLANS.get(body.run_id)
    if not r:
        raise HTTPException(404, "unknown run_id")
    try:
        return clean({"bookings": book(r, approved_by=role, approval_token=body.approval_token)})
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.post("/v1/support/chat")
def chat(body: ChatIn, role: str = Depends(requires("passenger"))):
    global _assistant
    _assistant = _assistant or SupportAssistant()
    return clean(_assistant.answer(body.message, session=body.session, channel="api"))


@app.post("/v1/fraud/score")
def fraud(body: FraudIn, role: str = Depends(requires("ops.view"))):
    return clean(_guard(lambda: _explain(FraudDetector().risk(body.model_dump()))))


# --------------------------------------------------------------- governance
@app.get("/v1/governance/health")
def gov_health(minutes: int = 60, role: str = Depends(requires("governance.view"))):
    return clean(monitoring.health_table(minutes).to_dict("records"))


@app.get("/v1/governance/models")
def gov_models(role: str = Depends(requires("governance.view"))):
    return clean(registry.list_models())


@app.get("/v1/governance/alerts")
def gov_alerts(status: str | None = None, role: str = Depends(requires("governance.view"))):
    return clean(Store.get().alerts(status).to_dict("records"))


@app.post("/v1/governance/faults/{name}")
def gov_fault(name: str, on: bool = True, role: str = Depends(requires("governance.act"))):
    if name not in faults.SCENARIOS:
        raise HTTPException(404, f"unknown scenario; choose from {list(faults.SCENARIOS)}")
    faults.set_fault(name, on)
    return clean({"active": faults.all_active()})


@app.post("/v1/governance/traffic")
def gov_traffic(minutes: int = 30, n: int = 300, role: str = Depends(requires("governance.act"))):
    return clean(simulator.run_traffic(minutes, n))


@app.post("/v1/governance/cycle")
def gov_cycle(window_minutes: int = 60, role: str = Depends(requires("governance.act"))):
    return clean(SupervisorAgent(window_minutes).run_cycle())


@app.get("/v1/governance/approvals")
def gov_approvals(role: str = Depends(requires("governance.view"))):
    return clean(Store.get().approvals("pending").to_dict("records"))


@app.post("/v1/governance/approvals/{approval_id}")
def gov_decide(approval_id: int, body: ApprovalIn, role: str = Depends(requires("approve"))):
    return clean(apply_approval(approval_id, body.approve, role))


@app.get("/v1/governance/audit")
def gov_audit(limit: int = 200, role: str = Depends(requires("audit"))):
    ok, bad = Store.get().verify_audit_chain()
    log = Store.get().query("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
    return clean({"chain_intact": ok, "first_bad_record": bad, "records": log.to_dict("records")})

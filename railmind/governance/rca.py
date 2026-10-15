"""Autonomous Root-Cause Analysis over the AI dependency graph."""
from __future__ import annotations

import time

from . import registry

# data source / infra -> model edges, and model -> consumer edges
FEATURE_SOURCES = {
    "delay_predictor": {"weather": "IMD weather feed", "station_congestion": "Station footfall sensors",
                        "departure_delay_min": "NTES running-status feed", "maintenance_block": "Track maintenance plan",
                        "month": "Calendar", "festival_intensity": "Festival calendar"},
    "confirmation_predictor": {"festival_intensity": "Festival calendar", "route_demand_index": "PRS booking stream",
                               "initial_position": "PRS booking stream", "booking_lead_days": "PRS booking stream",
                               "hist_cancel_rate": "Cancellation warehouse", "month": "Calendar"},
    "fare_predictor": {"occupancy_pct": "Seat inventory (PRS)", "days_before": "PRS booking stream",
                       "base_fare": "Fare master", "festival_intensity": "Festival calendar"},
    "fraud_detector": {"avg_click_interval_ms": "Clickstream", "form_fill_time_s": "Clickstream",
                       "accounts_per_device": "Device fingerprinting", "ip_reputation": "IP reputation service",
                       "payment_failures": "Payment gateway", "bookings_last_24h": "PRS booking stream"},
    "train_recommender": {"availability": "Seat inventory (PRS)", "punctuality": "NTES running-status feed"},
    "support_assistant": {"_kb": "Policy knowledge base", "_llm": "LLM provider"},
    "travel_planner_agent": {"_tools": "Agent tool layer"},
}
CONSUMERS = {
    "confirmation_predictor": ["travel_planner_agent", "train_recommender", "Passenger app"],
    "delay_predictor": ["travel_planner_agent", "support_assistant", "Passenger alerts"],
    "fare_predictor": ["travel_planner_agent", "Passenger app"],
    "fraud_detector": ["Payment & booking gateway"],
    "train_recommender": ["Passenger app"],
    "support_assistant": ["Chat / WhatsApp / IVR channels"],
    "travel_planner_agent": ["Passenger app", "support_assistant"],
}

PLAYBOOK = {
    "data_pipeline": ("block_bad_data", "Quarantine malformed records at the DQ gate and notify the data owner"),
    "covariate_drift": ("retrain", "Retrain on recent labelled data (champion/challenger)"),
    "concept_drift": ("retrain", "Retrain with recent ground truth; review feature set"),
    "bad_deployment": ("rollback", "Roll back to the previous production version"),
    "infrastructure": ("scale_out", "Scale out inference replicas and enable response cache"),
    "genai_hallucination": ("refresh_knowledge_base", "Disable free-form generation, re-index KB, enforce grounded templates"),
    "pricing_logic": ("revert_release", "Revert the faulty pricing-service release and clamp quotes to policy bounds"),
    "agent_behaviour": ("circuit_breaker", "Enable tool-call circuit breaker and roll back agent config"),
    "new_threat_pattern": ("retrain", "Retrain on newly confirmed fraud labels; threshold change needs approval"),
    "fairness": ("human_review", "Send fairness report to the AI ethics board"),
}


def analyse(model: str, findings: list[dict]) -> dict:
    """Combine all evidence about a model into ranked root-cause hypotheses."""
    kinds = {f["kind"] for f in findings}
    ev = {f["kind"]: f for f in findings}
    hyps = []
    srcs = FEATURE_SOURCES.get(model, {})

    def add(cause, conf, why, sources=None):
        action, desc = PLAYBOOK[cause]
        hyps.append({"cause": cause, "confidence": round(conf, 2), "evidence": why, "upstream": sources or [],
                     "remediation": action, "remediation_detail": desc})

    if "data_quality" in kinds:
        bad = ev["data_quality"].get("details", {}).get("features", [])
        add("data_pipeline", 0.9, f"{ev['data_quality']['message']}",
            sorted({srcs.get(f, "upstream feed") for f in bad}))
    perf = ev.get("performance")
    drift = ev.get("data_drift")
    if "pricing_anomaly" in kinds:
        perf = None  # the error is explained by the serving-layer pricing fault, not by the model
    if perf and drift:
        top = drift.get("details", {}).get("top_features", [])
        add("covariate_drift", 0.8, f"{perf['message']} together with {drift['message']}",
            sorted({srcs.get(f, "upstream feed") for f in top}))
    elif perf and not drift:
        recent_deploy = _recent_promotion(model)
        if recent_deploy:
            add("bad_deployment", 0.75, f"{perf['message']} shortly after promotion to {recent_deploy}")
        add("concept_drift", 0.65 if recent_deploy else 0.8,
            f"{perf['message']} without input drift → relationship between features and outcome changed")
    elif drift and not perf:
        add("covariate_drift", 0.45, f"{drift['message']} (no confirmed performance loss yet — watch)")
    if "fraud_miss" in kinds:
        add("new_threat_pattern", 0.85, ev["fraud_miss"]["message"], ["Clickstream", "Device fingerprinting"])
    if "sla" in kinds:
        add("infrastructure", 0.85, ev["sla"]["message"], ["Inference cluster"])
    if "hallucination" in kinds:
        add("genai_hallucination", 0.9, ev["hallucination"]["message"], ["LLM provider", "Policy knowledge base"])
    if "pricing_anomaly" in kinds:
        add("pricing_logic", 0.85, ev["pricing_anomaly"]["message"], ["Fare service"])
    if "agent_anomaly" in kinds:
        add("agent_behaviour", 0.85, ev["agent_anomaly"]["message"], ["Agent tool layer"])
    if "fairness" in kinds:
        add("fairness", 0.6, ev["fairness"]["message"])
    hyps.sort(key=lambda h: -h["confidence"])
    return {"model": model, "hypotheses": hyps, "primary": hyps[0] if hyps else None,
            "blast_radius": CONSUMERS.get(model, []), "dependencies": sorted(set(srcs.values()))}


def _recent_promotion(model: str, within_s: int = 3 * 3600) -> str | None:
    hist = registry.list_models().get(model, {}).get("history", [])
    if len(hist) < 2:
        return None
    last = hist[-1]
    try:
        ts = time.mktime(time.strptime(last["ts"][:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except ValueError:
        return None
    return last["to"] if time.time() - ts < within_s and "ROLLBACK" not in last.get("reason", "") else None


def dependency_edges() -> list[tuple[str, str]]:
    edges = set()
    for m, fs in FEATURE_SOURCES.items():
        for s in fs.values():
            edges.add((s, m))
    for m, cs in CONSUMERS.items():
        for c in cs:
            edges.add((m, c))
    return sorted(edges)

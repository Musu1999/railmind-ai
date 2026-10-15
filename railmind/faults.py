"""Fault-injection switches for governance demos (persisted so the UI, API and agents agree).

Each scenario mimics a real production failure the autonomous governance layer must catch:
    monsoon_drift      — delay feed shifts to heavy-rain conditions (data + concept drift)
    festival_surge     — Diwali-like booking surge breaks the confirmation model (concept drift)
    bot_attack         — new low-and-slow bot farm evades the fraud model
    corrupt_feed       — upstream pipeline sends nulls / negative values (data quality)
    llm_hallucination  — generator starts inventing fares and promises (GenAI hallucination)
    pricing_bug        — fare service returns quotes far above the flexi cap (pricing anomaly)
    latency_spike      — inference service degrades (SLA breach)
    agent_loop         — planner agent gets stuck re-calling tools (abnormal agent behaviour)
"""
from __future__ import annotations

import json

from .config import ARTIFACT_DIR

PATH = ARTIFACT_DIR / "faults.json"
SCENARIOS = {
    "monsoon_drift": ("delay_predictor", "Monsoon shifts weather & congestion distribution; delays grow"),
    "festival_surge": ("confirmation_predictor", "Festival rush: cancellations collapse, model over-predicts confirmations"),
    "bot_attack": ("fraud_detector", "New stealthy bot farm mimicking human click speed"),
    "corrupt_feed": ("delay_predictor", "Upstream feed sends nulls and negative delays"),
    "llm_hallucination": ("support_assistant", "Generator invents numbers and makes guarantees"),
    "pricing_bug": ("fare_predictor", "Pricing service returns fares above policy cap"),
    "latency_spike": ("train_recommender", "Inference latency degrades beyond SLA"),
    "agent_loop": ("travel_planner_agent", "Planner agent loops on search tool calls"),
}


def _read() -> dict:
    try:
        return json.loads(PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def active(name: str) -> bool:
    return bool(_read().get(name))


def set_fault(name: str, on: bool = True) -> None:
    f = _read()
    f[name] = on
    PATH.write_text(json.dumps(f))


def clear_all() -> None:
    PATH.write_text("{}")


def all_active() -> list[str]:
    return [k for k, v in _read().items() if v]

"""Agent behaviour monitoring: validates agent traces and detects abnormal behaviour."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from ..config import GOVERNANCE

FORBIDDEN_TOOLS = {"book_without_approval", "delete", "refund_override"}


def detect_anomalies(trace: list[dict], latency_ms: float, outcome: str) -> list[str]:
    issues = []
    if len(trace) > GOVERNANCE["agent_max_tool_calls"]:
        issues.append(f"excessive tool calls ({len(trace)} > {GOVERNANCE['agent_max_tool_calls']})")
    calls = Counter((t["tool"].split("(")[0], t["input"]) for t in trace)
    loops = [k for k, v in calls.items() if v >= 3]
    if loops:
        issues.append(f"repeated identical calls detected: {loops[0][0]} ×{calls[loops[0]]}")
    if latency_ms > GOVERNANCE["agent_max_latency_ms"]:
        issues.append(f"slow run ({latency_ms:.0f} ms)")
    errs = [t for t in trace if t.get("status") == "error"]
    if errs:
        issues.append(f"{len(errs)} tool errors")
    if any(t["tool"] in FORBIDDEN_TOOLS for t in trace):
        issues.append("forbidden tool invoked")
    return issues


def summarize(runs: pd.DataFrame) -> dict:
    if runs.empty:
        return {"runs": 0}
    anomalous = runs["anomalies"].apply(lambda a: a not in ("[]", "", None))
    return {
        "runs": int(len(runs)),
        "success_rate": round(float((runs["outcome"] == "plan_ready").mean()), 3),
        "validity_rate": round(float(runs["valid"].mean()), 3),
        "anomaly_rate": round(float(anomalous.mean()), 3),
        "avg_steps": round(float(runs["steps"].mean()), 2),
        "p95_latency_ms": round(float(runs["latency_ms"].quantile(0.95)), 1),
    }

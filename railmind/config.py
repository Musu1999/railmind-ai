"""Central configuration: paths, seeds, SLA targets and governance policy thresholds."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# minimal .env support (no extra dependency)
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
DATA_DIR = Path(os.getenv("RAILMIND_DATA_DIR", ROOT / "data" / "generated"))
ARTIFACT_DIR = Path(os.getenv("RAILMIND_ARTIFACT_DIR", ROOT / "artifacts"))
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"
DB_PATH = Path(os.getenv("RAILMIND_DB", ARTIFACT_DIR / "railmind.db"))

SEED = 42

for _d in (DATA_DIR, ARTIFACT_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- SLAs
SLA = {
    "p95_latency_ms": 250.0,      # near real-time inference
    "error_rate": 0.01,           # max 1 % failed inferences
    "availability": 0.995,
}

# ------------------------------------------------- governance thresholds
GOVERNANCE = {
    "psi_warn": 0.10,             # population stability index
    "psi_alert": 0.25,
    "ks_pvalue": 0.01,
    "perf_drop_alert": 0.05,      # absolute drop vs. registered baseline (scores)
    "error_increase_alert": 0.35, # relative increase vs. baseline (error metrics such as MAE)
    "disparate_impact_min": 0.80, # four-fifths rule
    "parity_diff_max": 0.10,
    "grounding_min": 0.35,        # hallucination guard for GenAI answers
    "dq_block_score": 0.70,       # block inference when data-quality score below
    "fraud_fpr_max": 0.05,
    "fraud_recall_min": 0.85,
    "agent_max_tool_calls": 25,
    "agent_max_latency_ms": 5000,
}

# Actions the supervisor may execute without a human; everything else is queued.
AUTO_REMEDIATION = {
    "retrain": True,
    "rollback": True,
    "block_bad_data": True,
    "refresh_knowledge_base": True,
    "tighten_threshold": False,   # changes fraud decisions -> human approval
}

# Business criticality used for alert prioritisation (1 = low, 5 = critical)
MODEL_CRITICALITY = {
    "confirmation_predictor": 4,
    "train_recommender": 3,
    "delay_predictor": 4,
    "fare_predictor": 3,
    "support_assistant": 3,
    "fraud_detector": 5,
    "travel_planner_agent": 4,
}

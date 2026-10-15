"""Production-traffic simulator.

Replays realistic inference traffic (with delayed ground-truth labels) through the
production models and writes it to telemetry, optionally under an injected fault.
This is what lets the governance layer be demonstrated end-to-end on a laptop.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import faults
from ..config import ARTIFACT_DIR
from ..passenger.confirmation import ConfirmationPredictor
from ..passenger.delay import DelayPredictor
from ..passenger.fare import FarePredictor
from ..passenger.fraud import FraudDetector
from ..passenger.recommender import TrainRecommender
from .store import Store

FEEDBACK_DIR = ARTIFACT_DIR / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

BASE_LATENCY = {"confirmation_predictor": 38, "delay_predictor": 45, "fare_predictor": 32, "fraud_detector": 22,
                "train_recommender": 95}

SUPPORT_QUESTIONS = [
    "What is the cancellation charge for sleeper class?", "When does tatkal booking open?",
    "How long does a refund take?", "Can I travel on a waiting list e-ticket?", "What does RAC mean?",
    "तत्काल बुकिंग कब खुलती है?", "रिफंड कितने दिन में आता है?", "ওয়েটিং লিস্ট টিকিট বাতিল করলে কত টাকা কাটে?",
    "How much luggage can I carry in AC 2 tier?", "Money deducted but ticket not booked, what to do?",
    "Is train 12089 running late?", "What is GNWL?",
]
PLANNER_QUERIES = [
    "I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.",
    "Mumbai to Goa on Saturday in chair car", "Kolkata to Patna tomorrow cheapest",
    "Bengaluru to Chennai tomorrow morning fastest",
]


def services():
    return {s.name: s for s in (ConfirmationPredictor(), DelayPredictor(), FarePredictor(), FraudDetector(),
                                TrainRecommender())}


# ----------------------------------------------------------------- scenarios
def apply_fault(model: str, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if model == "delay_predictor" and faults.active("monsoon_drift"):
        idx = rng.random(n) < 0.8
        w = rng.choice(["rain", "storm"], size=idx.sum(), p=[0.65, 0.35])
        df.loc[idx, "weather"] = w
        df.loc[idx, "month"] = rng.choice([7, 8], size=idx.sum())
        df.loc[idx, "station_congestion"] = np.clip(df.loc[idx, "station_congestion"] + 0.2, 0, 1)
        df.loc[idx, "departure_delay_min"] = df.loc[idx, "departure_delay_min"] + rng.gamma(2, 15, idx.sum())
        extra = np.where(w == "storm", rng.normal(110, 15, idx.sum()), rng.normal(50, 8, idx.sum()))
        df.loc[idx, "arrival_delay_min"] = df.loc[idx, "arrival_delay_min"] + extra
    if model == "delay_predictor" and faults.active("corrupt_feed"):
        idx = rng.random(n) < 0.3
        df.loc[idx, "station_congestion"] = np.nan
        idx2 = rng.random(n) < 0.15
        df.loc[idx2, "departure_delay_min"] = -rng.uniform(30, 300, idx2.sum())
    if model == "confirmation_predictor" and faults.active("festival_surge"):
        idx = rng.random(n) < 0.85
        df.loc[idx, "festival_intensity"] = rng.uniform(0.85, 1.0, idx.sum())
        df.loc[idx, "route_demand_index"] = rng.uniform(1.45, 1.75, idx.sum())   # unprecedented demand regime
        df.loc[idx, "month"] = rng.choice([10, 11], size=idx.sum())
        flip = idx & (df["confirmed"] == 1) & (rng.random(n) < 0.9)   # cancellations collapse
        df.loc[flip, "confirmed"] = 0
    if model == "fraud_detector" and faults.active("bot_attack"):
        k = int(0.2 * n)
        idx = rng.choice(n, size=k, replace=False)
        # aged accounts behind residential proxies, human-speed clicks: looks legitimate per-session
        new = {"avg_click_interval_ms": rng.uniform(1200, 2200, k), "form_fill_time_s": rng.uniform(20, 60, k),
               "bookings_last_24h": rng.integers(1, 3, k), "accounts_per_device": np.ones(k, int),
               "ip_reputation": rng.uniform(0.65, 0.9, k), "is_tatkal": np.ones(k, int),
               "booking_hour": rng.choice([10, 11], k), "captcha_failures": np.zeros(k, int),
               "passengers_per_booking": rng.integers(3, 5, k), "account_age_days": rng.uniform(200, 900, k),
               "payment_failures": np.zeros(k, int), "distinct_cards_7d": np.ones(k, int),
               "session_duration_s": rng.uniform(60, 110, k), "amount": rng.uniform(3500, 6000, k),
               "geo_mismatch": np.zeros(k, int), "is_fraud": np.ones(k, int)}
        for c, v in new.items():
            df.loc[df.index[idx], c] = v
        df.loc[df.index[idx], "fraud_type"] = "stealth_bot_farm"
    return df


def _segment(svc, row) -> str | None:
    return str(row[svc.segment_col]) if svc.segment_col and svc.segment_col in row else None


def generate(model: str, n: int, start_epoch: float, end_epoch: float, use_faults: bool = True,
             seed: int | None = None, save_feedback: bool = True) -> dict:
    svc = services()[model]
    bundle, version = svc.bundle()
    reference, _ = svc.bundle("v1")     # traffic is always drawn from the original, clean reference data
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % 2**32)
    src = reference["holdout"]
    df = src.sample(n, replace=len(src) < n, random_state=int(rng.integers(0, 1e9))).reset_index(drop=True)
    if use_faults:
        df = apply_fault(model, df, rng)
    feats = svc.features
    X = df[feats]
    # vectorised data-quality gate (same rules as the per-request gate)
    blocked = X.isna().any(axis=1)
    for c, (lo, hi) in svc.hard.items():
        if c in X:
            v = pd.to_numeric(X[c], errors="coerce")
            blocked |= (v < lo) | (v > hi)
    dq_score = 1 - X.isna().mean(axis=1) * 0.5 - blocked.astype(float) * 0.3
    preds = np.full(len(df), np.nan)
    ok = ~blocked
    if ok.any():
        preds[ok.values] = svc._raw_predict(bundle["pipeline"], X[ok])
    if model == "fare_predictor" and use_faults and faults.active("pricing_bug"):
        bug = ok.values & (rng.random(len(df)) < 0.3)
        preds[bug] *= rng.uniform(1.7, 2.2, bug.sum())
    lat = BASE_LATENCY.get(model, 40) * rng.lognormal(0, 0.25, len(df))
    if use_faults and faults.active("latency_spike") and model == "train_recommender":
        lat = rng.uniform(350, 950, len(df))
    epochs = np.sort(rng.uniform(start_epoch, end_epoch, len(df)))
    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        status = "blocked" if blocked.iloc[i] else "ok"
        inp = {c: (None if pd.isna(row[c]) else (row[c].item() if hasattr(row[c], "item") else row[c])) for c in feats}
        ts = datetime.fromtimestamp(epochs[i], timezone.utc).isoformat(timespec="seconds")
        label = float(row[svc.target]) if status == "ok" else None
        rows.append((ts, float(epochs[i]), model, version, float(lat[i]), status, float(dq_score.iloc[i]),
                     json.dumps(inp, default=str), json.dumps({"prediction": None if np.isnan(preds[i]) else float(preds[i])}),
                     None if np.isnan(preds[i]) else float(preds[i]), label, _segment(svc, row), None))
    Store.get().log_inferences_bulk(rows)
    if save_feedback:
        fb = df[ok.values].copy()
        fb.to_csv(FEEDBACK_DIR / f"{model}.csv", index=False)
    return {"model": model, "version": version, "requests": len(df), "blocked": int(blocked.sum())}


def genai_traffic(n_support: int = 12, n_planner: int = 4) -> dict:
    from ..passenger.planner import PlannerAgent
    from ..passenger.support import SupportAssistant
    a = SupportAssistant()
    rng = np.random.default_rng()
    for q in rng.choice(SUPPORT_QUESTIONS, size=n_support, replace=n_support > len(SUPPORT_QUESTIONS)):
        a.answer(str(q), channel="simulated")
    for q in PLANNER_QUERIES[:n_planner]:
        PlannerAgent().plan(q)
    return {"support": n_support, "planner": n_planner}


def seed_history(hours: int = 24, n: int = 700) -> list[dict]:
    """Healthy background traffic for the last `hours` (ending one hour ago)."""
    now = time.time()
    out = []
    for i, m in enumerate(BASE_LATENCY):
        out.append(generate(m, n, now - hours * 3600, now - 3600, use_faults=False, seed=100 + i, save_feedback=False))
    return out


def _last_fix(model: str) -> float:
    try:
        return float(json.loads((ARTIFACT_DIR / "governance_state.json").read_text()).get(model, 0))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.0


def run_traffic(minutes: int = 30, n: int = 350, include_genai: bool = True) -> list[dict]:
    """Current traffic for the last `minutes`, under whatever faults are active.
    Traffic is never back-dated before a model's last remediation (so it is evaluated as new traffic)."""
    now = time.time()
    out = []
    for m in BASE_LATENCY:
        last = Store.get().query("SELECT MAX(epoch) AS e FROM inference_log WHERE model=?", (m,))["e"].iloc[0]
        last = float(last) if last is not None and last == last else 0.0
        start = min(max(now - minutes * 60, _last_fix(m) + 1, last + 1), now - 1)  # never overlap older traffic
        out.append(generate(m, n, start, now))
    if include_genai:
        out.append(genai_traffic())
    Store.get().audit("simulator", "traffic.generated", "", {"minutes": minutes, "n": n,
                                                             "faults": faults.all_active()})
    return out


def load_feedback(model: str) -> pd.DataFrame:
    p = FEEDBACK_DIR / f"{model}.csv"
    return pd.read_csv(p, dtype={"train_no": str}) if p.exists() else pd.DataFrame()

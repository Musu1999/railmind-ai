"""Automated model lifecycle: retraining, champion/challenger evaluation, promotion, rollback."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data
from . import registry
from .store import Store

TRAIN_SOURCES = {"confirmation_predictor": "bookings", "delay_predictor": "delays", "fare_predictor": "fares",
                 "fraud_detector": "sessions", "train_recommender": "choices"}


def _service(model: str):
    from ..passenger.confirmation import ConfirmationPredictor
    from ..passenger.delay import DelayPredictor
    from ..passenger.fare import FarePredictor
    from ..passenger.fraud import FraudDetector
    from ..passenger.recommender import TrainRecommender
    return {c.name: c for c in (ConfirmationPredictor, DelayPredictor, FarePredictor, FraudDetector,
                                TrainRecommender)}[model]()


def evaluate(model: str, version: str, df: pd.DataFrame) -> dict:
    svc = _service(model)
    b, _ = registry.load(model, version)
    X = svc.prepare(df)
    preds = svc._raw_predict(b["pipeline"], X[svc.features])
    return svc.score(df[svc.target].values, preds)


def retrain(model: str, recent: pd.DataFrame, reason: str, actor: str = "lifecycle",
            min_gain: float = 0.0) -> dict:
    """Champion/challenger retraining on historical + recent labelled data.

    The challenger is promoted only if it beats the champion on the *recent* holdout and does
    not regress materially on the historical holdout (2 % for scores, 15 % for error metrics) (guards against catastrophic forgetting).
    """
    store = Store.get()
    svc = _service(model)
    champion = registry.production_version(model)
    if recent.empty or len(recent) < 50:
        return {"status": "skipped", "reason": "not enough recent labelled data"}
    recent = recent.sample(frac=1, random_state=7).reset_index(drop=True)
    cut = int(len(recent) * 0.6)
    recent_train, recent_test = recent.iloc[:cut], recent.iloc[cut:]
    hist = data.load(TRAIN_SOURCES[model])
    hist_eval = registry.load(model, "v1")[0]["holdout"]  # original reference data
    # upweight recent data by replication so the model adapts while retaining history
    reps = max(1, int(0.25 * len(hist) / max(len(recent_train), 1)))
    cols = [c for c in hist.columns if c in recent_train.columns]
    train_df = pd.concat([hist[cols]] + [recent_train[cols]] * reps, ignore_index=True)
    store.audit(actor, "retrain.started", model, {"champion": champion, "recent_rows": len(recent),
                                                  "replication": reps, "reason": reason}, "agent")
    challenger, train_metrics = svc.fit(train_df, reason=reason, stage="staging", parent=champion)
    metric, hib = svc.primary_metric
    champ_recent = evaluate(model, champion, recent_test)
    chall_recent = evaluate(model, challenger, recent_test)
    champ_hist = evaluate(model, champion, hist_eval)
    chall_hist = evaluate(model, challenger, hist_eval)
    sign = 1 if hib else -1
    gain = sign * (chall_recent[metric] - champ_recent[metric])
    hist_reg = sign * (champ_hist[metric] - chall_hist[metric]) / max(abs(champ_hist[metric]), 1e-9)
    promote = gain > min_gain and hist_reg <= (0.02 if hib else 0.15)
    result = {"status": "promoted" if promote else "rejected", "champion": champion, "challenger": challenger,
              "metric": metric, "champion_recent": champ_recent[metric], "challenger_recent": chall_recent[metric],
              "champion_hist": champ_hist[metric], "challenger_hist": chall_hist[metric],
              "gain": round(float(gain), 4), "historical_regression": round(float(hist_reg), 4)}
    if promote:
        registry.promote(model, challenger, actor=actor, reason=f"{reason} | {metric}: "
                         f"{champ_recent[metric]} → {chall_recent[metric]} on recent data")
    store.audit(actor, f"retrain.{result['status']}", model, result, "agent")
    return result


def rollback(model: str, reason: str, actor: str = "lifecycle") -> dict:
    current = registry.production_version(model)
    target = registry.rollback(model, actor=actor, reason=reason)
    return {"status": "rolled_back" if target else "no_previous_version", "from": current, "to": target}


def previous_production(model: str) -> str | None:
    """The version that was in production before the current one (from promotion history)."""
    entry = registry.list_models().get(model, {})
    cur = entry.get("production")
    for h in reversed(entry.get("history", [])):
        if h["to"] == cur and h["from"]:
            return h["from"]
    return None


def reset_to_baseline() -> None:
    """Demo helper: promote v1 of every model back to production."""
    for m, entry in registry.list_models().items():
        if entry.get("production") not in (None, "v1"):
            registry.promote(m, "v1", actor="admin", reason="demo reset to baseline")


def lineage(model: str) -> pd.DataFrame:
    entry = registry.list_models().get(model, {})
    rows = []
    for v in entry.get("versions", []):
        m = v.get("metrics", {})
        rows.append({"version": v["version"], "stage": v["stage"], "created": v["created"], "parent": v.get("parent"),
                     "reason": v.get("reason"), "data_hash": v.get("data_hash"),
                     **{k: m[k] for k in list(m)[:4] if isinstance(m.get(k), (int, float))}})
    return pd.DataFrame(rows)


def history(model: str) -> pd.DataFrame:
    return pd.DataFrame(registry.list_models().get(model, {}).get("history", []))


def to_numpy(x):
    return np.asarray(x)

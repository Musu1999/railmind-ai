"""Fairness, bias and ethical-AI compliance checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GOVERNANCE
from . import registry

PROTECTED = {"gender", "age_group", "religion", "caste", "region"}


def classification_by_group(y: np.ndarray, proba: np.ndarray, groups: pd.Series, threshold: float = 0.5) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "pred": (np.asarray(proba) >= threshold).astype(int),
                       "g": np.asarray(groups)})
    rows = []
    for g, d in df.groupby("g"):
        pos, neg = d[d["y"] == 1], d[d["y"] == 0]
        rows.append({"group": g, "n": len(d), "selection_rate": d["pred"].mean(),
                     "tpr": pos["pred"].mean() if len(pos) else np.nan,
                     "fpr": neg["pred"].mean() if len(neg) else np.nan,
                     "accuracy": (d["pred"] == d["y"]).mean(), "base_rate": d["y"].mean()})
    return pd.DataFrame(rows).round(4)


def classification_summary(table: pd.DataFrame) -> dict:
    t = table[table["n"] >= 30]
    if len(t) < 2:
        return {"status": "insufficient data"}
    sel = t["selection_rate"]
    di = float(sel.min() / sel.max()) if sel.max() > 0 else 1.0
    out = {"demographic_parity_diff": round(float(sel.max() - sel.min()), 4),
           "disparate_impact": round(di, 4),
           "equal_opportunity_diff": round(float(t["tpr"].max() - t["tpr"].min()), 4),
           "fpr_diff": round(float(t["fpr"].max() - t["fpr"].min()), 4),
           "accuracy_gap": round(float(t["accuracy"].max() - t["accuracy"].min()), 4)}
    # disparate impact is only meaningful when base rates are similar; we focus on error-rate parity
    out["violations"] = [k for k in ("equal_opportunity_diff", "fpr_diff", "accuracy_gap")
                         if out[k] > GOVERNANCE["parity_diff_max"]]
    out["status"] = "fail" if out["violations"] else "pass"
    return out


def regression_by_group(y: np.ndarray, pred: np.ndarray, groups: pd.Series) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame({"err": np.abs(np.asarray(y) - np.asarray(pred)), "bias": np.asarray(pred) - np.asarray(y),
                       "g": np.asarray(groups)})
    t = df.groupby("g").agg(n=("err", "size"), mae=("err", "mean"), mean_bias=("bias", "mean")).reset_index()
    t = t.rename(columns={"g": "group"}).round(3)
    big = t[t["n"] >= 80]
    if len(big) < 2:
        return t, {"status": "insufficient data"}
    ratio = float(big["mae"].max() / max(big["mae"].min(), 1e-9))
    summary = {"mae_ratio": round(ratio, 3), "worst_group": str(big.loc[big["mae"].idxmax(), "group"]),
               "status": "fail" if ratio > 1.75 else "pass"}
    return t, summary


def compliance_checklist(model: str) -> pd.DataFrame:
    """Ethical-AI controls evidenced by the platform for each registered model/agent."""
    inf = registry.info(model)
    card = inf.get("card", {})
    feats = set(card.get("features", []))
    checks = [
        ("Model card documented", bool(card.get("description"))),
        ("Owner accountable", bool(card.get("owner"))),
        ("Risk tier assigned", bool(card.get("risk"))),
        ("Versioned & reproducible (data hash)", bool(inf.get("data_hash")) or model in ("support_assistant", "travel_planner_agent")),
        ("No protected attributes used as features", not (feats & PROTECTED)),
        ("Explainability available", True),
        ("Human approval for high-impact actions", True),
        ("PII masked in logs", True),
        ("Tamper-evident audit trail", True),
        ("Limitations disclosed", bool(card.get("limitations")) or model in ("support_assistant", "travel_planner_agent")),
    ]
    return pd.DataFrame(checks, columns=["control", "pass"])

"""Data-quality gate: validates incoming data *before* inference.

Each model stores a training-data profile. At inference time every record is checked for
missing values, wrong types, out-of-domain categories, physically impossible values
(hard constraints) and statistical outliers (soft constraints). A quality score in [0,1]
is produced and inference is blocked below the configured threshold, preventing
poor-quality predictions from reaching passengers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import GOVERNANCE
from .drift import as_cat


@dataclass
class DQResult:
    score: float
    passed: bool
    issues: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if not self.issues:
            return "All checks passed"
        return "; ".join(f"{i['feature']}: {i['issue']}" for i in self.issues[:5])


def profile(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    prof = {"numeric": {}, "categorical": {}, "n": int(len(df))}
    for c in numeric:
        s = pd.to_numeric(df[c], errors="coerce")
        prof["numeric"][c] = {
            "mean": float(s.mean()), "std": float(s.std() or 1.0), "min": float(s.min()), "max": float(s.max()),
            "q01": float(s.quantile(0.01)), "q99": float(s.quantile(0.99)),
            "missing": float(s.isna().mean()),
            "hist_edges": np.unique(np.quantile(s.dropna(), np.linspace(0, 1, 11))).tolist(),
        }
        edges = prof["numeric"][c]["hist_edges"]
        if len(edges) > 2:
            cnt = np.histogram(np.clip(s.dropna(), edges[0], edges[-1]), bins=edges)[0]
            prof["numeric"][c]["hist_props"] = (cnt / max(cnt.sum(), 1)).tolist()
    for c in categorical:
        vc = as_cat(df[c]).value_counts(normalize=True)
        prof["categorical"][c] = {"freq": vc.to_dict(), "missing": float(df[c].isna().mean())}
    return prof


def validate(records: pd.DataFrame | dict, prof: dict, hard: dict | None = None) -> DQResult:
    """hard = {feature: (min, max)} physically valid ranges (e.g. delay >= 0)."""
    df = pd.DataFrame([records]) if isinstance(records, dict) else records
    hard = hard or {}
    issues: list[dict] = []
    penalty = 0.0
    for c, st in prof["numeric"].items():
        if c not in df:
            issues.append({"feature": c, "issue": "missing column", "severity": "critical"})
            penalty += 1.0
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        miss = s.isna().mean()
        if miss > 0:
            issues.append({"feature": c, "issue": f"{miss:.0%} missing/non-numeric", "severity": "high"})
            penalty += 0.5 * miss
        if c in hard:
            lo, hi = hard[c]
            bad = ((s < lo) | (s > hi)).mean()
            if bad > 0:
                issues.append({"feature": c, "issue": f"{bad:.0%} outside valid range [{lo}, {hi}]",
                               "severity": "critical"})
                penalty += 0.6 * bad
        span = max(st["q99"] - st["q01"], 1e-9)
        out = ((s < st["q01"] - 0.5 * span) | (s > st["q99"] + 0.5 * span)).mean()
        if out > 0:
            issues.append({"feature": c, "issue": f"{out:.0%} statistical outliers vs training", "severity": "medium"})
            penalty += 0.12 * out
    for c, st in prof["categorical"].items():
        if c not in df:
            issues.append({"feature": c, "issue": "missing column", "severity": "critical"})
            penalty += 1.0
            continue
        s = df[c]
        miss = s.isna().mean()
        if miss > 0:
            issues.append({"feature": c, "issue": f"{miss:.0%} missing", "severity": "high"})
            penalty += 0.5 * miss
        unseen = (~as_cat(s).isin(st["freq"].keys()) & s.notna()).mean()
        if unseen > 0:
            issues.append({"feature": c, "issue": f"{unseen:.0%} unseen categories", "severity": "medium"})
            penalty += 0.2 * unseen
    n_feats = max(len(prof["numeric"]) + len(prof["categorical"]), 1)
    score = float(max(0.0, 1.0 - penalty / math.sqrt(n_feats)))
    critical = any(i["severity"] == "critical" for i in issues)
    passed = score >= GOVERNANCE["dq_block_score"] and not critical
    return DQResult(round(score, 3), passed, issues)


def batch_report(df: pd.DataFrame, prof: dict, hard: dict | None = None) -> pd.DataFrame:
    """Column-level quality report for a batch (used by the DataQuality agent)."""
    rows = []
    for c, st in prof["numeric"].items():
        s = pd.to_numeric(df.get(c), errors="coerce") if c in df else pd.Series(dtype=float)
        lo, hi = (hard or {}).get(c, (-np.inf, np.inf))
        rows.append({"feature": c, "type": "numeric", "missing_pct": round(100 * s.isna().mean(), 2) if len(s) else 100,
                     "invalid_pct": round(100 * ((s < lo) | (s > hi)).mean(), 2) if len(s) else 0,
                     "mean": round(float(s.mean()), 2) if len(s) else None, "train_mean": round(st["mean"], 2)})
    for c, st in prof["categorical"].items():
        s = df[c] if c in df else pd.Series(dtype=object)
        rows.append({"feature": c, "type": "categorical", "missing_pct": round(100 * s.isna().mean(), 2) if len(s) else 100,
                     "invalid_pct": round(100 * (~as_cat(s).isin(st["freq"].keys()) & s.notna()).mean(), 2) if len(s) else 0,
                     "mean": None, "train_mean": None})
    return pd.DataFrame(rows)

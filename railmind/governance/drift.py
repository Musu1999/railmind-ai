"""Drift & performance-degradation detection.

* Data drift      — PSI (population stability index) per feature + KS / chi-square tests.
* Prediction drift — PSI of model outputs vs. holdout predictions.
* Concept drift   — performance on freshly-labelled traffic vs. registered baseline,
                    plus a Page-Hinkley change detector on the error stream.
* Behaviour drift — shift in categorical usage mix (e.g. support intents, languages).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import GOVERNANCE

EPS = 1e-4


def as_cat(s: pd.Series) -> pd.Series:
    """String view of a categorical column; missing -> 'NA' (stable across pandas 2 and 3)."""
    return s.map(lambda v: "NA" if pd.isna(v) else str(v))


def psi_numeric(expected_props: list[float], edges: list[float], actual: pd.Series) -> float:
    a = pd.to_numeric(actual, errors="coerce").dropna()
    if len(a) == 0 or len(edges) < 3:
        return 0.0
    cnt = np.histogram(np.clip(a, edges[0], edges[-1]), bins=edges)[0]
    act = cnt / max(cnt.sum(), 1)
    exp = np.asarray(expected_props)
    exp, act = np.clip(exp, EPS, None), np.clip(act, EPS, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def psi_categorical(expected_freq: dict, actual: pd.Series) -> float:
    act = as_cat(actual).value_counts(normalize=True).to_dict()
    keys = set(expected_freq) | set(act)
    return float(sum((act.get(k, 0) + EPS - expected_freq.get(k, 0) - EPS)
                     * np.log((act.get(k, 0) + EPS) / (expected_freq.get(k, 0) + EPS)) for k in keys))


def psi_status(v: float) -> str:
    return "drift" if v >= GOVERNANCE["psi_alert"] else ("watch" if v >= GOVERNANCE["psi_warn"] else "stable")


def data_drift(profile: dict, reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, st in profile["numeric"].items():
        if c not in current or "hist_props" not in st:
            continue
        v = psi_numeric(st["hist_props"], st["hist_edges"], current[c])
        cur = pd.to_numeric(current[c], errors="coerce").dropna()
        ref = pd.to_numeric(reference[c], errors="coerce").dropna() if c in reference else pd.Series(dtype=float)
        p = float(stats.ks_2samp(ref, cur).pvalue) if len(cur) > 5 and len(ref) > 5 else 1.0
        rows.append({"feature": c, "type": "numeric", "psi": round(v, 4), "test": "KS", "p_value": round(p, 5),
                     "train_mean": f"{st['mean']:.3f}", "current_mean": f"{cur.mean():.3f}" if len(cur) else "—",
                     "status": psi_status(v)})
    for c, st in profile["categorical"].items():
        if c not in current:
            continue
        v = psi_categorical(st["freq"], current[c].dropna())
        p = 1.0
        if c in reference and len(current) > 5:
            cats = sorted(set(as_cat(reference[c])) | set(as_cat(current[c])))
            obs = np.array([[ (as_cat(reference[c]) == k).sum() for k in cats],
                            [ (as_cat(current[c]) == k).sum() for k in cats]]) + 1
            p = float(stats.chi2_contingency(obs)[1])
        top = as_cat(current[c]).value_counts(normalize=True).head(1)
        rows.append({"feature": c, "type": "categorical", "psi": round(v, 4), "test": "Chi²", "p_value": round(p, 5),
                     "train_mean": "—", "current_mean": f"{top.index[0]} ({top.iloc[0]:.0%})" if len(top) else "—",
                     "status": psi_status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def prediction_drift(reference_preds: np.ndarray, current_preds: np.ndarray) -> dict:
    ref = pd.Series(reference_preds).dropna()
    cur = pd.Series(current_preds).dropna()
    if len(cur) < 10:
        return {"psi": 0.0, "status": "insufficient data"}
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return {"psi": 0.0, "status": "stable"}
    props = np.histogram(ref, bins=edges)[0] / len(ref)
    v = psi_numeric(props.tolist(), edges.tolist(), cur)
    return {"psi": round(v, 4), "ref_mean": round(float(ref.mean()), 3), "cur_mean": round(float(cur.mean()), 3),
            "status": psi_status(v)}


def page_hinkley(errors: np.ndarray, delta: float = 0.005, lamb: float | None = None) -> dict:
    """Detects an upward shift in the mean of an error stream."""
    e = np.asarray(errors, dtype=float)
    if len(e) < 20:
        return {"change": False, "index": None}
    lamb = lamb if lamb is not None else 5 * float(np.std(e[: len(e) // 3]) or 1.0) * np.sqrt(len(e)) / 4
    mean, cum, mn = 0.0, 0.0, 0.0
    for i, x in enumerate(e, start=1):
        mean += (x - mean) / i
        cum += x - mean - delta
        mn = min(mn, cum)
        if cum - mn > lamb:
            return {"change": True, "index": i - 1}
    return {"change": False, "index": None}


def performance_drop(baseline: float, current: float, higher_is_better: bool, relative_scale: float = 1.0) -> float:
    """Positive = worse. For regression errors (MAE) the drop is relative to baseline."""
    if higher_is_better:
        return baseline - current
    return (current - baseline) / max(baseline, 1e-9) * relative_scale


def behaviour_shift(baseline: pd.Series, current: pd.Series) -> dict:
    b = as_cat(baseline).value_counts(normalize=True).to_dict()
    v = psi_categorical(b, as_cat(current))
    return {"psi": round(v, 4), "status": psi_status(v),
            "current_mix": as_cat(current).value_counts(normalize=True).round(3).to_dict()}

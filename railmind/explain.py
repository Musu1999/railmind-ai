"""Model-agnostic explainability.

* local_explanation: per-prediction attributions by marginal feature ablation
  (replace one feature with values drawn from the training background and measure the
  average change in the prediction — a fast, SHAP-like main-effect estimate).
* global_importance: permutation importance on a holdout set.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

LABELS = {
    "booking_lead_days": "days booked before journey", "initial_position": "waiting-list position",
    "class_capacity": "seats in class", "festival_intensity": "festival / peak-season demand",
    "route_demand_index": "route demand", "hist_cancel_rate": "historical cancellation rate",
    "quota": "booking quota", "initial_status": "initial status (WL/RAC)", "class": "travel class",
    "train_type": "train type", "distance_km": "distance", "day_of_week": "day of week", "month": "month",
    "weather": "weather", "station_congestion": "station congestion", "departure_delay_min": "delay at origin",
    "maintenance_block": "track maintenance block", "punctuality_score": "train's punctuality record",
    "sched_arr_hour": "scheduled arrival hour", "zone": "railway zone", "days_before": "days before journey",
    "occupancy_pct": "seats already sold", "base_fare": "base fare", "position_ratio": "WL position vs seats",
    "bookings_last_24h": "bookings in last 24h", "avg_click_interval_ms": "click speed (ms)",
    "form_fill_time_s": "form fill time", "captcha_failures": "CAPTCHA failures",
    "payment_failures": "payment failures", "distinct_cards_7d": "cards used in 7 days",
    "accounts_per_device": "accounts on same device", "ip_reputation": "IP reputation",
    "geo_mismatch": "location mismatch", "account_age_days": "account age", "amount": "amount",
    "is_tatkal": "Tatkal booking", "booking_hour": "booking hour", "passengers_per_booking": "passengers",
    "session_duration_s": "session length",
}


def label(f: str) -> str:
    return LABELS.get(f, f.replace("_", " "))


def local_explanation(predict: Callable[[pd.DataFrame], np.ndarray], x: pd.DataFrame,
                      background: pd.DataFrame, features: list[str], n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_pred = float(predict(x)[0])
    bg = background.sample(min(n, len(background)), random_state=int(rng.integers(0, 1e6)))
    rows = []
    batch, owners = [], []
    for f in features:
        X = pd.concat([x] * len(bg), ignore_index=True)
        X[f] = bg[f].values
        batch.append(X)
        owners.extend([f] * len(bg))
    preds = predict(pd.concat(batch, ignore_index=True))
    s = pd.Series(preds, index=owners)
    for f in features:
        contrib = base_pred - float(s.loc[f].mean())
        rows.append({"feature": f, "label": label(f), "value": x.iloc[0][f], "contribution": contrib})
    out = pd.DataFrame(rows)
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def narrate(expl: pd.DataFrame, unit: str = "pts", scale: float = 100.0, top: int = 3) -> list[str]:
    lines = []
    for _, r in expl.head(top).iterrows():
        c = r["contribution"] * scale
        if abs(c) < 0.05:
            continue
        v = r["value"]
        v = f"{v:.2f}" if isinstance(v, float) and not float(v).is_integer() else v
        direction = "raises" if c > 0 else "lowers"
        lines.append(f"{r['label']} = {v} {direction} the estimate by {abs(c):.1f} {unit}")
    return lines


def global_importance(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, y: np.ndarray,
                      features: list[str], metric: Callable, higher_is_better: bool = True,
                      repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = metric(y, predict(X))
    rows = []
    for f in features:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            sc = metric(y, predict(Xp))
            drops.append((base - sc) if higher_is_better else (sc - base))
        rows.append({"feature": f, "label": label(f), "importance": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)

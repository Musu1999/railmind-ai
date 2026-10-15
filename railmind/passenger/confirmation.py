"""1. Intelligent Ticket Confirmation Predictor (WL / RAC)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .. import data
from ..config import SEED
from ..explain import narrate
from ..reference import festival_intensity, festival_name
from .base import TabularService


class ConfirmationPredictor(TabularService):
    name = "confirmation_predictor"
    task = "classification"
    numeric = ["booking_lead_days", "initial_position", "class_capacity", "position_ratio", "festival_intensity",
               "route_demand_index", "hist_cancel_rate", "distance_km", "day_of_week", "month"]
    categorical = ["quota", "initial_status", "class", "train_type"]
    target = "confirmed"
    hard = {"booking_lead_days": (0, 120), "initial_position": (1, 1000), "class_capacity": (1, 2000),
            "festival_intensity": (0, 1), "hist_cancel_rate": (0, 1)}
    segment_col = "class"
    description = "Probability that a Waiting-List/RAC ticket is confirmed by chart preparation."
    risk_tier = "high"

    @property
    def primary_metric(self):
        # calibrated probabilities matter more than ranking for WL decisions -> Brier score
        return ("brier", False)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["position_ratio"] = df["initial_position"] / df["class_capacity"].clip(lower=1)
        return df

    def extra_artifacts(self, pipe, tr: pd.DataFrame) -> dict:
        # multi-class model for final allocation status CNF / RAC / WL
        status_pipe = self.make_pipeline()
        status_pipe.steps[-1] = ("model", HistGradientBoostingClassifier(max_iter=200, random_state=SEED))
        status_pipe.fit(tr[self.features], tr["final_status"])
        # reliability table -> calibrated confidence per probability bin
        proba = pipe.predict_proba(tr[self.features])[:, 1]
        bins = np.clip((proba * 10).astype(int), 0, 9)
        rel = pd.DataFrame({"bin": bins, "y": tr[self.target].values, "p": proba})
        rel["correct"] = ((rel["p"] >= 0.5).astype(int) == rel["y"]).astype(int)
        table = rel.groupby("bin").agg(observed=("y", "mean"), predicted=("p", "mean"),
                                       accuracy=("correct", "mean"), n=("y", "size")).reset_index()
        return {"status_model": status_pipe, "reliability": table}

    # ------------------------------------------------------------------ API
    def build_record(self, train_no: str, cls: str, journey: date, position: int, status: str = "WL",
                     quota: str = "GN", booked_on: date | None = None) -> dict:
        t = data.train(train_no)
        booked_on = booked_on or date.today()
        fest = festival_intensity(journey)
        return {
            "booking_lead_days": max((journey - booked_on).days, 0), "initial_position": int(position),
            "class_capacity": int(t["capacity"].get(cls, 500)), "festival_intensity": fest,
            "route_demand_index": round(0.55 + 0.6 * fest + (0.1 if journey.weekday() in (4, 5, 6) else 0), 3),
            "hist_cancel_rate": float(t["cancel_rate_base"]), "distance_km": int(t["distance_km"]),
            "day_of_week": journey.weekday(), "month": journey.month, "quota": quota,
            "initial_status": status, "class": cls, "train_type": t["train_type"],
        }

    def predict(self, train_no: str, cls: str, journey: date, position: int, status: str = "WL",
                quota: str = "GN", booked_on: date | None = None, log: bool = True) -> dict:
        rec = self.build_record(train_no, cls, journey, position, status, quota, booked_on)
        out = self.infer(rec, log=log)
        b, _ = self.bundle()
        X = self.prepare(pd.DataFrame([rec]))[self.features]
        sm = b["extra"]["status_model"]
        status_p = dict(zip(sm.classes_, sm.predict_proba(X)[0].round(3)))
        p = out["prediction"]
        rel = b["extra"]["reliability"]
        row = rel[rel["bin"] == min(int(p * 10), 9)]
        conf = float(row["accuracy"].iloc[0]) if not row.empty else 0.7
        exp_cancel = rec["class_capacity"] * rec["hist_cancel_rate"] * (1 - 0.45 * rec["festival_intensity"])
        out.update({
            "probability": round(p, 3),
            "verdict": "Likely to confirm" if p >= 0.7 else ("Uncertain" if p >= 0.4 else "Unlikely to confirm"),
            "confidence": round(conf, 3),
            "status_distribution": status_p,
            "expected_final": max(status_p, key=status_p.get),
            "expected_cancellations": int(exp_cancel),
            "festival": festival_name(journey),
            "reasons": narrate(out["explanation"]),
            "record": rec,
        })
        return out


def historical_trends(train_no: str | None = None, cls: str | None = None) -> dict[str, pd.DataFrame]:
    b = data.load("bookings")
    if train_no:
        sub = b[b["train_no"] == str(train_no)]
        b = sub if len(sub) > 50 else b
    if cls:
        sub = b[b["class"] == cls]
        b = sub if len(sub) > 50 else b
    b["lead_bucket"] = pd.cut(b["booking_lead_days"], [-1, 2, 7, 15, 30, 60, 120],
                              labels=["0-2", "3-7", "8-15", "16-30", "31-60", "60+"])
    b["pos_bucket"] = pd.cut(b["initial_position"], [0, 10, 25, 50, 100, 200, 1000],
                             labels=["1-10", "11-25", "26-50", "51-100", "101-200", "200+"])
    return {
        "by_lead": b.groupby("lead_bucket", observed=True)["confirmed"].agg(["mean", "size"]).reset_index(),
        "by_position": b.groupby("pos_bucket", observed=True)["confirmed"].agg(["mean", "size"]).reset_index(),
        "by_month": b.groupby("month")["confirmed"].mean().reset_index(),
        "by_status": b["final_status"].value_counts(normalize=True).reset_index(),
    }


def train(reason: str = "initial training") -> tuple[str, dict]:
    return ConfirmationPredictor().fit(data.load("bookings"), reason=reason)

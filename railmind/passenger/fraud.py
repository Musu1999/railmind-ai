"""7. Fraud, Security & Bot Detection: hybrid supervised + anomaly + rule engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .. import data
from ..config import SEED
from ..explain import narrate
from .base import TabularService

RULES = [
    ("bot_click_speed", "Clicks faster than humanly possible (<300 ms avg)", lambda r: r["avg_click_interval_ms"] < 300),
    ("bulk_booking", "More than 5 bookings in 24 h", lambda r: r["bookings_last_24h"] > 5),
    ("form_autofill", "Booking form filled in under 5 s", lambda r: r["form_fill_time_s"] < 5),
    ("card_testing", "3+ failed payments or 4+ cards in a week", lambda r: r["payment_failures"] >= 3 or r["distinct_cards_7d"] >= 4),
    ("device_sharing", "4+ accounts on one device", lambda r: r["accounts_per_device"] >= 4),
    ("geo_anomaly", "Login location mismatch with low IP reputation", lambda r: r["geo_mismatch"] == 1 and r["ip_reputation"] < 0.5),
    ("captcha_abuse", "Repeated CAPTCHA failures", lambda r: r["captcha_failures"] >= 3),
]


class FraudDetector(TabularService):
    name = "fraud_detector"
    task = "classification"
    numeric = ["account_age_days", "bookings_last_24h", "passengers_per_booking", "session_duration_s",
               "avg_click_interval_ms", "form_fill_time_s", "captcha_failures", "payment_failures",
               "distinct_cards_7d", "accounts_per_device", "ip_reputation", "is_tatkal", "booking_hour",
               "geo_mismatch", "amount"]
    categorical: list[str] = []
    target = "is_fraud"
    hard = {"ip_reputation": (0, 1), "avg_click_interval_ms": (0, 600000), "amount": (0, 500000)}
    description = "Risk-scores booking sessions for bots, card testing, account takeover and agent misuse."
    risk_tier = "critical"
    review_threshold = 0.5
    block_threshold = 0.8

    def make_estimator(self):
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, class_weight="balanced",
                                              random_state=SEED)

    def score(self, y_true, y_pred) -> dict:
        y_true = np.asarray(y_true)
        proba = np.asarray(y_pred)
        hard = (proba >= self.review_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, hard, labels=[0, 1]).ravel()
        return {"auc": round(float(roc_auc_score(y_true, proba)), 4),
                "precision": round(float(precision_score(y_true, hard, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, hard, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, hard, zero_division=0)), 4),
                "fpr": round(float(fp / max(fp + tn, 1)), 4), "fnr": round(float(fn / max(fn + tp, 1)), 4),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    @property
    def primary_metric(self):
        return ("f1", True)

    def extra_artifacts(self, pipe, tr: pd.DataFrame) -> dict:
        legit = tr[tr[self.target] == 0][self.numeric]
        iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=SEED).fit(legit)
        s = -iso.score_samples(tr[self.numeric])
        return {"iforest": iso, "iso_lo": float(np.quantile(s, 0.05)), "iso_hi": float(np.quantile(s, 0.995))}

    def risk(self, record: dict, log: bool = True, label: int | None = None) -> dict:
        out = self.infer(record, log=log, label=label)
        b, _ = self.bundle()
        ex = b["extra"]
        X = pd.DataFrame([record])[self.numeric]
        iso_raw = float(-ex["iforest"].score_samples(X)[0])
        iso = float(np.clip((iso_raw - ex["iso_lo"]) / (ex["iso_hi"] - ex["iso_lo"]), 0, 1))
        fired = [(k, msg) for k, msg, fn in RULES if fn(record)]
        rule_score = min(len(fired) / 2.0, 1.0)
        sup = out["prediction"]
        risk = 100 * (0.70 * sup + 0.18 * iso + 0.12 * rule_score)
        decision = "BLOCK" if risk >= 100 * self.block_threshold else ("REVIEW" if risk >= 100 * self.review_threshold else "ALLOW")
        out.update({"risk_score": round(risk, 1), "decision": decision, "supervised_prob": round(sup, 3),
                    "anomaly_score": round(iso, 3), "rules_fired": [m for _, m in fired],
                    "rule_ids": [k for k, _ in fired], "reasons": narrate(out["explanation"])})
        return out

    def batch_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        b, _ = self.bundle()
        ex = b["extra"]
        sup = self._raw_predict(b["pipeline"], df[self.numeric])
        iso_raw = -ex["iforest"].score_samples(df[self.numeric])
        iso = np.clip((iso_raw - ex["iso_lo"]) / (ex["iso_hi"] - ex["iso_lo"]), 0, 1)
        rules = df.apply(lambda r: sum(fn(r) for _, _, fn in RULES), axis=1)
        out = df.copy()
        out["risk_score"] = 100 * (0.70 * sup + 0.18 * iso + 0.12 * np.minimum(rules / 2.0, 1.0))
        out["decision"] = np.where(out["risk_score"] >= 80, "BLOCK", np.where(out["risk_score"] >= 50, "REVIEW", "ALLOW"))
        return out


def train(reason: str = "initial training", df: pd.DataFrame | None = None) -> tuple[str, dict]:
    return FraudDetector().fit(df if df is not None else data.load("sessions"), reason=reason)

"""Base class shared by every tabular ML service.

Each service is *governance-native*: training registers a versioned bundle (pipeline +
training profile + background sample + metrics) in the registry, and every inference
passes through the data-quality gate and is logged to telemetry for monitoring,
drift detection and audit.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import SEED
from ..explain import local_explanation
from ..governance import data_quality as dq
from ..governance import registry
from ..governance.store import Store


class GovernanceBlocked(Exception):
    def __init__(self, result: dq.DQResult):
        super().__init__(f"Inference blocked by data-quality gate: {result.summary()}")
        self.result = result


class TabularService:
    name: str = "base"
    task: str = "classification"          # or "regression"
    numeric: list[str] = []
    categorical: list[str] = []
    target: str = ""
    hard: dict = {}
    segment_col: str | None = None        # column used for fairness slicing
    description: str = ""
    owner: str = "AI Platform Team"
    risk_tier: str = "medium"

    # ---------------------------------------------------------- building
    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical

    def make_estimator(self):
        if self.task == "classification":
            return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                                  l2_regularization=0.1, random_state=SEED)
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                             l2_regularization=0.1, random_state=SEED)

    def make_pipeline(self) -> Pipeline:
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical),
            ("num", "passthrough", self.numeric),
        ])
        return Pipeline([("pre", pre), ("model", self.make_estimator())])

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering hook — override in subclasses."""
        return df

    # ------------------------------------------------------------ metrics
    def score(self, y_true, y_pred) -> dict:
        if self.task == "classification":
            proba = np.asarray(y_pred)
            hard = (proba >= 0.5).astype(int)
            return {"auc": round(float(roc_auc_score(y_true, proba)), 4),
                    "accuracy": round(float(accuracy_score(y_true, hard)), 4),
                    "f1": round(float(f1_score(y_true, hard)), 4),
                    "brier": round(float(brier_score_loss(y_true, proba)), 4)}
        return {"mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
                "r2": round(float(r2_score(y_true, y_pred)), 4)}

    @property
    def primary_metric(self) -> tuple[str, bool]:
        """(metric name, higher_is_better)."""
        return ("auc", True) if self.task == "classification" else ("mae", False)

    # ------------------------------------------------------------ training
    def fit(self, df: pd.DataFrame, reason: str = "initial training", stage: str = "production",
            parent: str | None = None) -> tuple[str, dict]:
        df = self.prepare(df)
        tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
        pipe = self.make_pipeline()
        pipe.fit(tr[self.features], tr[self.target])
        preds = self._raw_predict(pipe, te[self.features])
        metrics = self.score(te[self.target].values, preds)
        bundle = self._bundle(pipe, tr, te, metrics)
        version = registry.register(
            self.name, bundle, metrics, params={"estimator": type(pipe[-1]).__name__, "n_train": len(tr)},
            data_hash=registry.fingerprint(df), stage=stage, reason=reason, parent=parent, card=self.card())
        return version, metrics

    def _bundle(self, pipe, tr: pd.DataFrame, te: pd.DataFrame, metrics: dict) -> dict:
        return {
            "pipeline": pipe,
            "features": self.features,
            "profile": dq.profile(tr, self.numeric, self.categorical),
            "background": tr[self.features + [self.target]].sample(min(400, len(tr)), random_state=SEED),
            "holdout": te.sample(min(3000, len(te)), random_state=SEED),
            "metrics": metrics,
            "extra": self.extra_artifacts(pipe, tr),
        }

    def extra_artifacts(self, pipe, tr: pd.DataFrame) -> dict:
        return {}

    def card(self) -> dict:
        return {"description": self.description, "owner": self.owner, "risk": self.risk_tier, "task": self.task,
                "features": self.features, "target": self.target,
                "limitations": "Trained on synthetic data for demonstration; recalibrate on real IRCTC/CRIS data."}

    # ---------------------------------------------------------- inference
    def _raw_predict(self, pipe, X: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return pipe.predict_proba(X)[:, 1]
        return pipe.predict(X)

    def bundle(self, version: str | None = None) -> tuple[dict, str]:
        return registry.load(self.name, version)

    def predict_frame(self, X: pd.DataFrame, version: str | None = None) -> np.ndarray:
        b, _ = self.bundle(version)
        return self._raw_predict(b["pipeline"], self.prepare(X)[self.features])

    def infer(self, record: dict, *, label: float | None = None, explain: bool = True,
              enforce_quality: bool = True, segment: str | None = None, log: bool = True) -> dict[str, Any]:
        """Single governed prediction: DQ gate -> predict -> explain -> telemetry."""
        t0 = time.perf_counter()
        b, version = self.bundle()
        X = self.prepare(pd.DataFrame([record]))
        result = dq.validate(X[[c for c in self.features if c in X]], b["profile"], self.hard)
        rid = uuid.uuid4().hex[:12]
        store = Store.get()
        if enforce_quality and not result.passed:
            if log:
                store.log_inference(self.name, version, (time.perf_counter() - t0) * 1000, "blocked",
                                    result.score, record, {"issues": result.issues}, request_id=rid)
            raise GovernanceBlocked(result)
        try:
            pred = float(self._raw_predict(b["pipeline"], X[self.features])[0])
        except Exception as exc:  # noqa: BLE001
            if log:
                store.log_inference(self.name, version, (time.perf_counter() - t0) * 1000, "error",
                                    result.score, record, {"error": str(exc)}, request_id=rid)
            raise
        out: dict[str, Any] = {"prediction": pred, "version": version, "dq_score": result.score,
                               "dq_issues": result.issues, "request_id": rid}
        if explain:
            predict = lambda Z: self._raw_predict(b["pipeline"], Z[self.features])  # noqa: E731
            out["explanation"] = local_explanation(predict, X[self.features], b["background"], self.features)
        latency = (time.perf_counter() - t0) * 1000
        out["latency_ms"] = round(latency, 1)
        if log:
            seg = segment or (str(record.get(self.segment_col)) if self.segment_col else None)
            store.log_inference(self.name, version, latency, "ok", result.score, record, {"prediction": pred},
                                prediction=pred, label=label, segment=seg, request_id=rid)
        return out

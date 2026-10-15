"""5. Dynamic Fare Prediction Engine: fare movement, demand surges, optimal booking window."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_percentage_error

from .. import data
from ..config import SEED
from ..explain import narrate
from ..reference import base_fare, festival_intensity, festival_name
from .base import TabularService

TATKAL_MAX = {"2S": 15, "SL": 200, "CC": 225, "3E": 400, "3A": 400, "2A": 500, "EC": 500, "1A": 0}


class FarePredictor(TabularService):
    name = "fare_predictor"
    task = "regression"
    numeric = ["distance_km", "days_before", "occupancy_pct", "festival_intensity", "day_of_week", "month", "base_fare"]
    categorical = ["train_type", "class", "quota"]
    target = "fare"
    hard = {"days_before": (0, 120), "occupancy_pct": (0, 100), "base_fare": (1, 20000), "festival_intensity": (0, 1)}
    segment_col = "class"
    description = "Forecasts the fare a passenger will pay given time-to-journey and demand (flexi/Tatkal pricing)."

    def extra_artifacts(self, pipe, tr: pd.DataFrame) -> dict:
        occ_feats = ["days_before", "festival_intensity", "day_of_week", "month", "distance_km"]
        occ = HistGradientBoostingRegressor(max_iter=200, random_state=SEED)
        occ.fit(tr[occ_feats], tr["occupancy_pct"])
        dem = data.load("demand")
        dem["corridor"] = dem["src"] + "-" + dem["dst"]
        codes = {c: i for i, c in enumerate(sorted(dem["corridor"].unique()))}
        dem["corr_code"] = dem["corridor"].map(codes)
        d0 = pd.to_datetime(dem["date"]).min()
        dem["t"] = (pd.to_datetime(dem["date"]) - d0).dt.days
        dfeats = ["corr_code", "day_of_week", "month", "festival_intensity", "t"]
        cut = dem["t"].max() - 60
        dm = HistGradientBoostingRegressor(max_iter=300, random_state=SEED, categorical_features=[0])
        dm.fit(dem[dem["t"] <= cut][dfeats], dem[dem["t"] <= cut]["bookings"])
        test = dem[dem["t"] > cut]
        mape = mean_absolute_percentage_error(test["bookings"], dm.predict(test[dfeats]))
        dm.fit(dem[dfeats], dem["bookings"])
        resid = tr[self.target] - pipe.predict(tr[self.features])
        iso = IsolationForest(n_estimators=150, contamination=0.01, random_state=SEED)
        iso.fit(np.c_[tr[["days_before", "occupancy_pct"]].values, (tr[self.target] / tr["base_fare"]).values])
        return {"occupancy_model": occ, "occ_feats": occ_feats, "demand_model": dm, "demand_feats": dfeats,
                "corridor_codes": codes, "demand_t0": str(d0.date()), "demand_mape": round(float(mape), 4),
                "resid_std": float(resid.std()), "pricing_iforest": iso}

    # ----------------------------------------------------------- forecasts
    def _record(self, t: dict, cls: str, journey: date, days_before: int, occ: float, quota: str = "GN") -> dict:
        return {"distance_km": int(t["distance_km"]), "days_before": int(days_before), "occupancy_pct": float(occ),
                "festival_intensity": festival_intensity(journey), "day_of_week": journey.weekday(),
                "month": journey.month, "base_fare": base_fare(cls, t["distance_km"], t["train_type"]),
                "train_type": t["train_type"], "class": cls, "quota": quota}

    def forecast(self, train_no: str, cls: str, journey: date, today: date | None = None, log: bool = True) -> dict:
        today = today or date.today()
        t = data.train(train_no)
        b, version = self.bundle()
        ex = b["extra"]
        lead_now = max((journey - today).days, 0)
        days = list(range(min(lead_now, 60), -1, -1))
        occ_X = pd.DataFrame([{"days_before": d, "festival_intensity": festival_intensity(journey),
                               "day_of_week": journey.weekday(), "month": journey.month,
                               "distance_km": t["distance_km"]} for d in days])
        occ = np.clip(ex["occupancy_model"].predict(occ_X[ex["occ_feats"]]), 0, 100)
        recs = [self._record(t, cls, journey, d, o) for d, o in zip(days, occ)]
        X = pd.DataFrame(recs)
        fares = self.predict_frame(X)
        curve = pd.DataFrame({"days_before": days, "date": [journey - timedelta(days=d) for d in days],
                              "occupancy_pct": occ.round(1), "fare": np.round(fares, 0)})
        curve["fare_low"] = (curve["fare"] - 1.28 * ex["resid_std"]).round(0)
        curve["fare_high"] = (curve["fare"] + 1.28 * ex["resid_std"]).round(0)
        curve["sellout_risk"] = (curve["occupancy_pct"] >= 97).astype(int)
        # best window: cheapest day before sell-out risk, prefer earlier when within ₹10
        ok = curve[curve["sellout_risk"] == 0]
        if ok.empty:
            ok = curve
        best_fare = ok["fare"].min()
        best = ok[ok["fare"] <= best_fare + 10].iloc[0]
        now_row = curve.iloc[0]
        last = curve.iloc[-1]
        out = self.infer(recs[0], log=log)
        tatkal = None
        if cls in TATKAL_MAX and TATKAL_MAX[cls] and t["train_type"] in ("Superfast", "Express", "Mail", "Duronto"):
            tq = self._record(t, cls, journey, 1, float(occ[-2] if len(occ) > 1 else occ[-1]), quota="TQ")
            tatkal = round(float(self.predict_frame(pd.DataFrame([tq]))[0]), 0)
        trend = "rising" if last["fare"] > now_row["fare"] * 1.03 else ("stable" if abs(last["fare"] - now_row["fare"]) <= now_row["fare"] * 0.03 else "falling")
        out.update({
            "fare_now": float(now_row["fare"]), "fare_at_journey": float(last["fare"]),
            "base_fare": recs[0]["base_fare"], "trend": trend, "curve": curve,
            "best_booking_date": best["date"].isoformat(), "best_fare": float(best["fare"]),
            "saving_vs_waiting": round(float(curve["fare"].max() - best["fare"]), 0),
            "tatkal_fare_estimate": tatkal, "festival": festival_name(journey),
            "reasons": narrate(out["explanation"], unit="₹", scale=1.0), "version": version,
        })
        return out

    def demand_forecast(self, src: str, dst: str, start: date | None = None, days: int = 45) -> pd.DataFrame:
        b, _ = self.bundle()
        ex = b["extra"]
        start = start or date.today()
        code = ex["corridor_codes"].get(f"{src}-{dst}")
        if code is None:
            return pd.DataFrame()
        t0 = date.fromisoformat(ex["demand_t0"])
        rows = [{"corr_code": code, "day_of_week": (start + timedelta(i)).weekday(),
                 "month": (start + timedelta(i)).month, "festival_intensity": festival_intensity(start + timedelta(i)),
                 "t": (start + timedelta(i) - t0).days, "date": start + timedelta(i)} for i in range(days)]
        df = pd.DataFrame(rows)
        df["forecast_bookings"] = ex["demand_model"].predict(df[ex["demand_feats"]]).round(0)
        base = float(df["forecast_bookings"].median())
        df["surge"] = df["forecast_bookings"] > 1.3 * base
        df["festival"] = df["date"].apply(festival_name)
        return df[["date", "forecast_bookings", "surge", "festival"]]

    def pricing_anomalies(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """Flags quotes outside policy bounds or statistically unusual (governance hook)."""
        b, _ = self.bundle()
        q = quotes.copy()
        tat = q["class"].map(TATKAL_MAX).fillna(0)
        upper = q["base_fare"] * 1.55 + np.where(q["quota"] == "TQ", tat, 0) + 25
        lower = q["base_fare"] * 0.9
        q["policy_violation"] = (q["fare"] > upper) | (q["fare"] < lower)
        ratio = (q["fare"] / q["base_fare"]).values
        q["statistical_outlier"] = b["extra"]["pricing_iforest"].predict(
            np.c_[q[["days_before", "occupancy_pct"]].values, ratio]) == -1
        q["anomaly"] = q["policy_violation"] | q["statistical_outlier"]
        return q


def train(reason: str = "initial training") -> tuple[str, dict]:
    svc = FarePredictor()
    version, metrics = svc.fit(data.load("fares"), reason=reason)
    return version, metrics

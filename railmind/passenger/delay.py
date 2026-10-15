"""3. Train Delay & Disruption Prediction (arrival delay, station congestion, alerts)."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .. import data
from ..config import SEED
from ..explain import narrate
from ..reference import STATION_INDEX, festival_intensity, fog_season, monsoon
from .base import TabularService


def likely_weather(d: date, zone: str) -> str:
    north = zone in ("NR", "NWR", "NCR", "ECR", "NFR")
    if fog_season(d) and north:
        return "fog"
    if monsoon(d):
        return "rain"
    return "clear"


class DelayPredictor(TabularService):
    name = "delay_predictor"
    task = "regression"
    numeric = ["distance_km", "month", "day_of_week", "sched_arr_hour", "festival_intensity", "station_congestion",
               "departure_delay_min", "maintenance_block", "punctuality_score"]
    categorical = ["train_type", "weather", "zone"]
    target = "arrival_delay_min"
    hard = {"departure_delay_min": (0, 1440), "station_congestion": (0, 1), "festival_intensity": (0, 1),
            "punctuality_score": (0, 1), "distance_km": (1, 5000)}
    segment_col = "zone"
    description = "Predicts minutes of arrival delay at destination with 80% prediction interval."
    risk_tier = "high"

    def extra_artifacts(self, pipe, tr: pd.DataFrame) -> dict:
        quantiles = {}
        for q in (0.1, 0.9):
            p = self.make_pipeline()
            p.steps[-1] = ("model", HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200,
                                                                  random_state=SEED))
            p.fit(tr[self.features], tr[self.target])
            quantiles[q] = p
        # station congestion model trained on footfall data
        load = data.load("station_load")
        cong = HistGradientBoostingRegressor(max_iter=200, random_state=SEED, categorical_features=[0])
        Xc = load[["station", "hour", "day_of_week", "month", "festival_intensity"]].copy()
        codes = {c: i for i, c in enumerate(sorted(load["station"].unique()))}
        Xc["station"] = Xc["station"].map(codes)
        cong.fit(Xc, load["footfall_index"])
        cong_mae = mean_absolute_error(load["footfall_index"].tail(5000), cong.predict(Xc.tail(5000)))
        return {"quantiles": quantiles, "congestion_model": cong, "station_codes": codes,
                "congestion_mae": round(float(cong_mae), 2)}

    # ------------------------------------------------------------------ API
    def station_congestion(self, station: str, when: datetime) -> dict:
        b, _ = self.bundle()
        ex = b["extra"]
        hour = min((6, 9, 12, 15, 18, 21), key=lambda h: abs(h - when.hour))
        X = pd.DataFrame([{"station": ex["station_codes"].get(station, 0), "hour": hour,
                           "day_of_week": when.weekday(), "month": when.month,
                           "festival_intensity": festival_intensity(when.date())}])
        idx = float(ex["congestion_model"].predict(X)[0])
        level = "High" if idx >= 85 else ("Moderate" if idx >= 60 else "Low")
        return {"station": station, "footfall_index": round(idx, 1), "level": level}

    def build_record(self, train_no: str, journey: date, weather: str | None = None,
                     departure_delay: float = 0.0, maintenance_block: int = 0) -> dict:
        t = data.train(train_no)
        zone = STATION_INDEX[t["dst"]][5]
        arr = data.arrival_datetime(t, journey)
        cong = self.station_congestion(t["dst"], arr)
        return {
            "distance_km": int(t["distance_km"]), "month": arr.month, "day_of_week": journey.weekday(),
            "sched_arr_hour": arr.hour, "festival_intensity": festival_intensity(journey),
            "station_congestion": round(min(cong["footfall_index"] / 110.0, 1.0), 3),
            "departure_delay_min": float(departure_delay), "maintenance_block": int(maintenance_block),
            "punctuality_score": float(t["punctuality"]), "train_type": t["train_type"],
            "weather": weather or likely_weather(journey, zone), "zone": zone,
        }

    def predict(self, train_no: str, journey: date, weather: str | None = None, departure_delay: float = 0.0,
                maintenance_block: int = 0, log: bool = True, explain: bool = True) -> dict:
        rec = self.build_record(train_no, journey, weather, departure_delay, maintenance_block)
        out = self.infer(rec, log=log, explain=explain)
        b, _ = self.bundle()
        X = pd.DataFrame([rec])[self.features]
        q10 = float(b["extra"]["quantiles"][0.1].predict(X)[0])
        q90 = float(b["extra"]["quantiles"][0.9].predict(X)[0])
        mean = max(out["prediction"], 0.0)
        lo, hi = max(min(q10, mean), 0.0), max(q90, mean)
        sigma = max((hi - lo) / 2.563, 3.0)
        on_time = 0.5 * (1 + math.erf((15 - mean) / (sigma * math.sqrt(2))))
        t = data.train(train_no)
        sched_arr = data.arrival_datetime(t, journey)
        exp_arr = sched_arr + timedelta(minutes=mean)
        alert = None
        if mean >= 90:
            alert = ("severe", f"Expected to run ~{mean:.0f} min late. Consider an alternative train.")
        elif mean >= 45:
            alert = ("warning", f"Likely delay of about {mean:.0f} min — plan connections with buffer.")
        width = hi - lo
        out.update({
            "delay_min": round(mean, 1), "interval_80": [round(lo, 1), round(hi, 1)],
            "on_time_probability": round(on_time, 3),
            "confidence": round(float(np.clip(1 - width / (mean + 60), 0.2, 0.95)), 3),
            "scheduled_arrival": sched_arr.isoformat(timespec="minutes"),
            "expected_arrival": exp_arr.isoformat(timespec="minutes"),
            "destination_congestion": self.station_congestion(t["dst"], exp_arr),
            "alert": alert, "record": rec,
            "reasons": narrate(out["explanation"], unit="min", scale=1.0) if explain else [],
        })
        return out

    def alternatives(self, train_no: str, journey: date, limit: int = 4) -> pd.DataFrame:
        t = data.train(train_no)
        src_city, dst_city = STATION_INDEX[t["src"]][2], STATION_INDEX[t["dst"]][2]
        rows = []
        for d in (journey, journey + timedelta(days=1)):
            for _, r in data.trains_between(src_city, dst_city, d).iterrows():
                if r["train_no"] == str(train_no):
                    continue
                p = self.predict(r["train_no"], d, log=False, explain=False)
                rows.append({"train_no": r["train_no"], "train_name": r["train_name"], "date": d.isoformat(),
                             "dep_time": r["dep_time"], "predicted_delay_min": p["delay_min"],
                             "expected_arrival": p["expected_arrival"], "on_time_prob": p["on_time_probability"]})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(["expected_arrival"]).head(limit).reset_index(drop=True)


def train(reason: str = "initial training") -> tuple[str, dict]:
    return DelayPredictor().fit(data.load("delays"), reason=reason)

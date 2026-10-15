"""2. Smart Train Recommendation Engine — learning-to-rank on historical choice sessions."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .. import data, inventory
from ..config import SEED
from ..explain import local_explanation
from ..governance import data_quality as dq
from ..governance import registry
from .base import TabularService

BUDGET_PER_KM = {"low": 1.0, "mid": 2.2, "high": 4.0}


def time_bucket(dep_min: int) -> str:
    h = int(dep_min) // 60
    return "morning" if 5 <= h < 12 else ("afternoon" if 12 <= h < 18 else "night")


def ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = 3) -> float:
    vals = []
    for _, g in df.groupby("session_id"):
        g = g.sort_values(score_col, ascending=False).head(k)
        gains = g["chosen"].values / np.log2(np.arange(2, len(g) + 2))
        vals.append(gains.sum())
    return float(np.mean(vals))


def hit_rate_at_1(df: pd.DataFrame, score_col: str) -> float:
    return float(df.loc[df.groupby("session_id")[score_col].idxmax(), "chosen"].mean())


class TrainRecommender(TabularService):
    name = "train_recommender"
    task = "classification"
    numeric = ["over_budget", "fare_ratio", "duration_ratio", "duration_h", "punctuality", "availability",
               "class_match", "time_match", "punctuality_weight", "type_affinity"]
    categorical = ["train_type", "budget_level"]
    target = "chosen"
    hard = {"availability": (0, 1), "punctuality": (0, 1), "fare_ratio": (1, 50), "duration_ratio": (1, 50)}
    segment_col = "budget_level"
    description = "Ranks train+class options for a passenger by predicted probability of being the best choice."

    # --------------------------------------------------------------- features
    @staticmethod
    def featurize(c: pd.DataFrame, user: dict, affinity: dict | None = None) -> pd.DataFrame:
        c = c.copy()
        budget_km = user.get("budget_per_km") or BUDGET_PER_KM.get(user.get("budget_level", "mid"), 2.2)
        c["over_budget"] = np.maximum(0, c["fare"] / c["distance_km"] - budget_km)
        c["fare_ratio"] = c["fare"] / c["fare"].min()
        c["duration_ratio"] = c["duration_min"] / c["duration_min"].min()
        c["duration_h"] = c["duration_min"] / 60
        c["class_match"] = (c["class"] == user.get("preferred_class")).astype(int)
        c["time_match"] = (c["dep_min"].apply(time_bucket) == user.get("preferred_time")).astype(int)
        c["punctuality_weight"] = float(user.get("punctuality_weight", 0.6))
        c["budget_level"] = user.get("budget_level", "mid")
        aff = (affinity or {}).get(user.get("user_id"), {})
        c["type_affinity"] = c["train_type"].map(lambda t: aff.get(t, 0.0))
        return c

    def _session_features(self, ch: pd.DataFrame, users: pd.DataFrame, counts: pd.DataFrame,
                          leave_one_out: bool) -> pd.DataFrame:
        """Vectorised featurisation of many sessions at once."""
        c = ch.merge(users, on="user_id", how="left")
        c["budget_per_km"] = c["budget_level"].map(BUDGET_PER_KM)
        c["over_budget"] = np.maximum(0, c["fare"] / c["distance_km"] - c["budget_per_km"])
        c["fare_ratio"] = c["fare"] / c.groupby("session_id")["fare"].transform("min")
        c["duration_ratio"] = c["duration_min"] / c.groupby("session_id")["duration_min"].transform("min")
        c["duration_h"] = c["duration_min"] / 60
        c["class_match"] = (c["class"] == c["preferred_class"]).astype(int)
        c["time_match"] = (c["dep_min"].apply(time_bucket) == c["preferred_time"]).astype(int)
        # historical behaviour: share of the user's past choices that were this train type
        n_user = counts.sum(axis=1)
        key_cnt = [counts.at[u, t] if (u in counts.index and t in counts.columns) else 0
                   for u, t in zip(c["user_id"], c["train_type"])]
        c["_cnt"] = np.array(key_cnt, dtype=float)
        c["_n"] = c["user_id"].map(n_user).fillna(0).astype(float)
        if leave_one_out:  # exclude the session's own choice to avoid target leakage
            own = c.groupby("session_id")["chosen"].transform("max")  # every session has one choice
            c["_n"] -= own
            chosen_type = c[c["chosen"] == 1].set_index("session_id")["train_type"]
            c["_cnt"] -= (c["session_id"].map(chosen_type) == c["train_type"]).astype(float)
        c["type_affinity"] = np.where(c["_n"] > 0, c["_cnt"] / c["_n"].clip(lower=1), 0.0)
        return c.drop(columns=["_cnt", "_n"])

    def fit(self, df: pd.DataFrame | None = None, reason: str = "initial training", stage: str = "production",
            parent: str | None = None):
        ch = df if df is not None else data.load("choices")
        users = data.load("users")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_idx, te_idx = next(gss.split(ch, groups=ch["session_id"]))
        tr_raw, te_raw = ch.iloc[tr_idx], ch.iloc[te_idx]
        chosen = tr_raw[tr_raw["chosen"] == 1]
        counts = chosen.groupby(["user_id", "train_type"]).size().unstack(fill_value=0)
        affinity = (counts.div(counts.sum(axis=1), axis=0)).to_dict("index")
        tr = self._session_features(tr_raw, users, counts, leave_one_out=True)
        te = self._session_features(te_raw, users, counts, leave_one_out=False)
        pipe = self.make_pipeline()
        pipe.fit(tr[self.features], tr[self.target])
        te["score"] = self._raw_predict(pipe, te[self.features])
        te["baseline"] = -te["fare_ratio"]  # "cheapest first" baseline
        metrics = {"ndcg@3": round(ndcg_at_k(te, "score"), 4), "hit@1": round(hit_rate_at_1(te, "score"), 4),
                   "baseline_hit@1": round(hit_rate_at_1(te, "baseline"), 4),
                   **{k: v for k, v in super().score(te[self.target], te["score"]).items() if k == "auc"}}
        bundle = {"pipeline": pipe, "features": self.features,
                  "profile": dq.profile(tr, self.numeric, self.categorical),
                  "background": tr[self.features + [self.target]].sample(400, random_state=SEED),
                  "holdout": te, "metrics": metrics, "extra": {"affinity": affinity}}
        version = registry.register(self.name, bundle, metrics, params={"n_sessions": int(ch["session_id"].nunique())},
                                    data_hash=registry.fingerprint(ch), stage=stage, reason=reason, parent=parent,
                                    card=self.card())
        return version, metrics

    @property
    def primary_metric(self):
        return ("hit@1", True)

    # -------------------------------------------------------------------- API
    def candidates(self, src: str, dst: str, journey: date, classes: list[str] | None = None) -> pd.DataFrame:
        rows = []
        for _, t in data.trains_between(src, dst, journey).iterrows():
            for cls in t["classes"].split(","):
                if classes and cls not in classes:
                    continue
                av = inventory.availability(t["train_no"], cls, journey)
                rows.append({"train_no": t["train_no"], "train_name": t["train_name"], "train_type": t["train_type"],
                             "class": cls, "fare": inventory.fare(t["train_no"], cls),
                             "duration_min": t["duration_min"], "distance_km": t["distance_km"],
                             "dep_min": t["dep_min"], "dep_time": t["dep_time"], "arr_time": t["arr_time"],
                             "day_offset": t["day_offset"], "punctuality": t["punctuality"], "src": t["src"],
                             "dst": t["dst"], "availability_label": av["label"], "avail_status": av["status"],
                             "avail_count": av["count"]})
        return pd.DataFrame(rows)

    def _availability_scores(self, c: pd.DataFrame, journey: date) -> np.ndarray:
        from .confirmation import ConfirmationPredictor
        cp = ConfirmationPredictor()
        scores = np.ones(len(c))
        wl = c["avail_status"] != "AVAILABLE"
        if wl.any():
            recs = [cp.build_record(r["train_no"], r["class"], journey, int(r["avail_count"]),
                                    "RAC" if r["avail_status"] == "RAC" else "WL")
                    for r in c.loc[wl].to_dict("records")]
            scores[wl.values] = cp.predict_frame(pd.DataFrame(recs))
        return scores

    def recommend(self, src: str, dst: str, journey: date, user: dict | None = None, budget: float | None = None,
                  classes: list[str] | None = None, max_duration_h: float | None = None, k: int = 5,
                  log: bool = True) -> dict:
        user = dict(user or {"budget_level": "mid", "preferred_class": "3A", "preferred_time": "night",
                             "punctuality_weight": 0.6})
        c = self.candidates(src, dst, journey, classes)
        if c.empty:
            return {"items": pd.DataFrame(), "message": "No trains run on this route on the chosen date."}
        if budget:
            user["budget_per_km"] = budget / c["distance_km"].median()
            c = c[c["fare"] <= budget * 1.15]
        if max_duration_h:
            c = c[c["duration_min"] <= max_duration_h * 60]
        if c.empty:
            return {"items": pd.DataFrame(), "message": "No option satisfies the budget/duration constraints."}
        c = c.reset_index(drop=True)
        c["availability"] = self._availability_scores(c, journey)
        b, version = self.bundle()
        X = self.featurize(c, user, b["extra"]["affinity"])
        q = dq.validate(X[self.features], b["profile"], self.hard)
        X["score"] = self._raw_predict(b["pipeline"], X[self.features])
        top = X.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        predict = lambda Z: self._raw_predict(b["pipeline"], Z[self.features])  # noqa: E731
        rationales = []
        for i in range(len(top)):
            ex = local_explanation(predict, top.iloc[[i]][self.features].reset_index(drop=True), b["background"],
                                   self.features, n=30)
            rationales.append(self._rationale(top.iloc[i], ex))
        top["why"] = rationales
        top["confirm_chance"] = top["availability"].round(2)
        if log:
            from ..governance.store import Store
            Store.get().log_inference(self.name, version, 0.0, "ok", q.score,
                                      {"src": src, "dst": dst, "date": str(journey), "user": user.get("user_id")},
                                      {"top": top["train_no"].tolist()}, prediction=float(top["score"].iloc[0]),
                                      segment=user.get("budget_level"))
        return {"items": top, "version": version, "dq_score": q.score, "n_candidates": len(c)}

    @staticmethod
    def _rationale(row: pd.Series, ex: pd.DataFrame) -> str:
        nice = {"over_budget": "fits budget" if row["over_budget"] == 0 else "over budget",
                "fare_ratio": "cheapest option" if row["fare_ratio"] <= 1.01 else f"{row['fare_ratio']:.1f}× cheapest fare",
                "duration_ratio": "fastest" if row["duration_ratio"] <= 1.01 else f"{row['duration_h']:.1f} h journey",
                "duration_h": f"{row['duration_h']:.1f} h journey",
                "punctuality": f"{row['punctuality']:.0%} on-time record",
                "availability": f"{row['availability']:.0%} chance of a confirmed berth",
                "class_match": "your preferred class" if row["class_match"] else "not your usual class",
                "time_match": "departs at your preferred time" if row["time_match"] else "departure outside preferred time",
                "type_affinity": "train type you often choose" if row["type_affinity"] > 0.3 else "new train type for you",
                "train_type": row["train_type"], "budget_level": f"{row['budget_level']} budget profile",
                "punctuality_weight": "you value punctuality"}
        pos = [nice[f] for f, cval in zip(ex["feature"], ex["contribution"]) if cval > 0.005][:3]
        neg = [nice[f] for f, cval in zip(ex["feature"], ex["contribution"]) if cval < -0.005][:1]
        s = "; ".join(pos) if pos else "balanced trade-off"
        return s + (f" (trade-off: {neg[0]})" if neg else "")


def fairness_audit(n_users: int = 120, seed: int = 0) -> dict:
    """Counterfactual + group fairness of recommendations (used by the Fairness agent)."""
    rec = TrainRecommender()
    b, _ = rec.bundle()
    te = b["holdout"].copy()
    by_gender = te.groupby("gender").apply(lambda g: hit_rate_at_1(g, "score"), include_groups=False)
    by_region = te.groupby("region").apply(lambda g: hit_rate_at_1(g, "score"), include_groups=False)
    top = te.loc[te.groupby("session_id")["score"].idxmax()]
    premium = top["train_type"].isin(["Rajdhani", "Vande Bharat", "Shatabdi", "Duronto"])
    exp_gender = premium.groupby(top["gender"]).mean()
    fare_gender = top.groupby("gender")["fare_ratio"].mean()
    return {
        "quality_by_gender": by_gender.round(3).to_dict(),
        "quality_by_region": by_region.round(3).to_dict(),
        "premium_exposure_by_gender": exp_gender.round(3).to_dict(),
        "fare_ratio_by_gender": fare_gender.round(3).to_dict(),
        "quality_gap_gender": round(float(by_gender.max() - by_gender.min()), 3),
        "quality_gap_region": round(float(by_region.max() - by_region.min()), 3),
        "exposure_gap_gender": round(float(exp_gender.max() - exp_gender.min()), 3),
        "protected_attributes_used": [f for f in rec.features if f in ("gender", "region", "age_group")],
    }


def train(reason: str = "initial training") -> tuple[str, dict]:
    return TrainRecommender().fit(reason=reason)

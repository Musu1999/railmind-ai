"""Agentic AI Supervisory Layer — "AI agents monitoring AI agents".

Specialist governance agents inspect telemetry and emit findings; the SupervisorAgent
prioritises alerts, runs root-cause analysis, executes policy-approved remediations
(retrain / rollback / quarantine / circuit-breaker / KB refresh), queues high-impact
actions for human approval, verifies the fix on fresh traffic and closes the loop.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .. import faults
from ..config import ARTIFACT_DIR, AUTO_REMEDIATION, GOVERNANCE, MODEL_CRITICALITY
from . import agent_monitor, drift, fairness, lifecycle, monitoring, rca, registry
from .store import Store, to_epoch

STATE = ARTIFACT_DIR / "governance_state.json"
SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3, "critical": 4}
TABULAR = ["confirmation_predictor", "delay_predictor", "fare_predictor", "fraud_detector", "train_recommender"]


@dataclass
class Finding:
    agent: str
    model: str
    kind: str
    severity: str
    message: str
    confidence: float = 0.8
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    details: dict = field(default_factory=dict)

    @property
    def priority(self) -> float:
        crit = MODEL_CRITICALITY.get(self.model, 3)
        blast = len(rca.CONSUMERS.get(self.model, []))
        return round(SEVERITY_WEIGHT[self.severity] * crit * self.confidence * (1 + 0.1 * blast), 2)


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s))


def since(model: str) -> float:
    return float(_state().get(model, 0))


def _svc(model):
    return lifecycle._service(model)


# =========================================================== specialist agents
class HealthAgent:
    name = "HealthAgent"

    def run(self, window: int) -> list[Finding]:
        out = []
        for m in monitoring.MODELS:
            h = monitoring.health(m, window, since(m))
            if h["requests"] and not h["sla_ok"]:
                sev = "critical" if h["availability"] < 0.99 else "high"
                out.append(Finding(self.name, m, "sla", sev, "SLA breach: " + "; ".join(h["breaches"]), 0.95,
                                   "p95_ms", h["p95_ms"], None, {"health": h}))
        return out


class DataQualityAgent:
    name = "DataQualityAgent"

    def run(self, window: int) -> list[Finding]:
        out = []
        for m in TABULAR:
            h = monitoring.health(m, window, since(m))
            if h["requests"] < 20 or h["blocked_rate"] < 0.05:
                continue
            bad = monitoring.recent_inputs(m, window, since(m), status="blocked")
            svc = _svc(m)
            feats = []
            for c in svc.features:
                if c not in bad:
                    continue
                v = pd.to_numeric(bad[c], errors="coerce") if c in svc.numeric else bad[c]
                miss = v.isna().mean()
                lo, hi = svc.hard.get(c, (-np.inf, np.inf))
                inval = ((v < lo) | (v > hi)).mean() if c in svc.numeric else 0
                if miss > 0.05 or inval > 0.05:
                    feats.append(c)
            out.append(Finding(self.name, m, "data_quality", "high",
                               f"{h['blocked_rate']:.0%} of requests blocked by the data-quality gate "
                               f"(bad fields: {', '.join(feats) or 'various'})", 0.95, "blocked_rate",
                               h["blocked_rate"], 0.05, {"features": feats}))
        return out


class DriftAgent:
    """Data drift, prediction drift and performance (concept) drift on labelled traffic."""
    name = "DriftAgent"

    def run(self, window: int) -> list[Finding]:
        out = []
        for m in TABULAR:
            cur = monitoring.recent_inputs(m, window, since(m))
            if len(cur) < 60:
                continue
            svc = _svc(m)
            b, version = registry.load(m)
            ref = b["holdout"]
            dd = drift.data_drift(b["profile"], ref, cur)
            drifted = dd[dd["status"] == "drift"] if not dd.empty else dd
            if not drifted.empty:
                top = drifted["feature"].head(4).tolist()
                out.append(Finding(self.name, m, "data_drift", "medium",
                                   f"input drift on {', '.join(top)} (max PSI {drifted['psi'].max():.2f})", 0.8,
                                   "max_psi", float(drifted["psi"].max()), GOVERNANCE["psi_alert"],
                                   {"top_features": top, "table": dd.head(8).to_dict("records")}))
            # performance on the most recent labelled requests (labels arrive with a delay)
            lab = cur.dropna(subset=["_label", "_prediction"]).sort_values("_epoch").tail(350)
            if len(lab) < 50:
                continue
            y, p = lab["_label"].astype(float).values, lab["_prediction"].astype(float).values
            if svc.task == "classification" and len(np.unique(y)) < 2:
                continue
            now = svc.score(y, p) if m != "train_recommender" else {"auc": svc.score(y, p)["auc"]}
            base = registry.info(m, version)["metrics"]
            if m == "fraud_detector":
                if now["recall"] < GOVERNANCE["fraud_recall_min"] and now["tp"] + now["fn"] >= 25:
                    out.append(Finding(self.name, m, "fraud_miss", "critical",
                                       f"fraud recall fell to {now['recall']:.0%} (baseline {base['recall']:.0%}); "
                                       f"{now['fn']} frauds missed", 0.9, "recall", now["recall"],
                                       GOVERNANCE["fraud_recall_min"], {"current": now, "baseline": base}))
                if now["fpr"] > GOVERNANCE["fraud_fpr_max"]:
                    out.append(Finding(self.name, m, "performance", "high", f"false-positive rate {now['fpr']:.1%}",
                                       0.85, "fpr", now["fpr"], GOVERNANCE["fraud_fpr_max"], {"current": now}))
                continue
            metric, hib = ("auc", True) if svc.task == "classification" else ("mae", False)
            if metric not in base:
                continue
            drop = drift.performance_drop(base[metric], now[metric], hib)
            limit = GOVERNANCE["perf_drop_alert"] if hib else GOVERNANCE["error_increase_alert"]
            if svc.task == "classification" and "accuracy" in base and base["accuracy"] - now["accuracy"] > limit:
                metric, drop = "accuracy", max(drop, base["accuracy"] - now["accuracy"]) + limit
            errs = np.abs(y - p) if svc.task == "regression" else np.abs(y - p)
            ph = drift.page_hinkley(errs)
            if drop > limit:
                out.append(Finding(self.name, m, "performance", "high",
                                   f"{metric.upper()} degraded from {base[metric]} to {now[metric]}"
                                   + (" (Page-Hinkley change detected)" if ph["change"] else ""), 0.9, metric,
                                   now[metric], base[metric], {"current": now, "baseline": base, "page_hinkley": ph}))
        return out


class FairnessAgent:
    name = "FairnessAgent"

    def run(self, window: int) -> list[Finding]:
        out = []
        conf = monitoring.recent_inputs("confirmation_predictor", window, since("confirmation_predictor")).dropna(
            subset=["_label"])
        if len(conf) > 150 and conf["_label"].nunique() == 2:
            grp = np.where(conf["class"].isin(["SL", "2S"]), "non-AC (SL/2S)", "AC classes")
            t = fairness.classification_by_group(conf["_label"], conf["_prediction"], pd.Series(grp))
            s = fairness.classification_summary(t)
            if s.get("status") == "fail":
                out.append(Finding(self.name, "confirmation_predictor", "fairness", "medium",
                                   f"error-rate disparity across travel classes ({', '.join(s['violations'])})", 0.6,
                                   "equal_opportunity_diff", s["equal_opportunity_diff"], GOVERNANCE["parity_diff_max"],
                                   {"table": t.to_dict("records"), "summary": s}))
        fr = monitoring.recent_inputs("fraud_detector", window, since("fraud_detector")).dropna(subset=["_label"])
        if len(fr) > 150:
            grp = np.where(fr["account_age_days"].astype(float) < 60, "new account (<60d)", "established")
            t = fairness.classification_by_group(fr["_label"], fr["_prediction"], pd.Series(grp))
            legit = t.dropna(subset=["fpr"])
            if len(legit) == 2 and legit["fpr"].max() - legit["fpr"].min() > 0.08:
                out.append(Finding(self.name, "fraud_detector", "fairness", "medium",
                                   "new users are flagged as fraud disproportionately", 0.6, "fpr_diff",
                                   float(legit["fpr"].max() - legit["fpr"].min()), 0.05, {"table": t.to_dict("records")}))
        dl = monitoring.recent_inputs("delay_predictor", window, since("delay_predictor")).dropna(subset=["_label"])
        if len(dl) > 150:
            t, s = fairness.regression_by_group(dl["_label"].astype(float), dl["_prediction"].astype(float), dl["zone"])
            if s.get("status") == "fail":
                out.append(Finding(self.name, "delay_predictor", "fairness", "low",
                                   f"prediction error {s['mae_ratio']}× higher for zone {s['worst_group']}", 0.5,
                                   "mae_ratio", s["mae_ratio"], 1.75, {"table": t.to_dict("records")}))
        conv = Store.get().conversations(2000)
        if not conv.empty:
            conv = conv[to_epoch(conv["ts"]) >= monitoring.window_start(window, since("support_assistant"))]
            by_lang = conv.groupby("lang")["quality"].agg(["mean", "size"])
            by_lang = by_lang[by_lang["size"] >= 3]
            if len(by_lang) >= 2 and by_lang["mean"].max() - by_lang["mean"].min() > 0.2:
                out.append(Finding(self.name, "support_assistant", "fairness", "medium",
                                   f"answer quality gap across languages ({by_lang['mean'].round(2).to_dict()})", 0.6,
                                   "quality_gap", float(by_lang["mean"].max() - by_lang["mean"].min()), 0.2))
        return out


class PricingAgent:
    name = "PricingAgent"

    def run(self, window: int) -> list[Finding]:
        cur = monitoring.recent_inputs("fare_predictor", window, since("fare_predictor"))
        if len(cur) < 30:
            return []
        from ..passenger.fare import FarePredictor
        q = cur.rename(columns={"_prediction": "fare"}).dropna(subset=["fare"])
        q = FarePredictor().pricing_anomalies(q)
        rate = float(q["policy_violation"].mean())
        if rate > 0.05:
            return [Finding(self.name, "fare_predictor", "pricing_anomaly", "high",
                            f"{rate:.0%} of fare quotes violate the flexi-fare cap / floor", 0.9, "violation_rate",
                            rate, 0.05, {"examples": q[q["policy_violation"]].head(5)[["class", "base_fare", "fare"]].to_dict("records")})]
        return []


class GenAIGuardAgent:
    name = "GenAIGuardAgent"

    def run(self, window: int) -> list[Finding]:
        conv = Store.get().conversations(5000)
        if conv.empty:
            return []
        start = monitoring.window_start(window, since("support_assistant"))
        conv = conv[to_epoch(conv["ts"]) >= start]
        if len(conv) < 5:
            return []
        rate = float(conv["hallucination"].mean())
        if rate > 0.1:
            return [Finding(self.name, "support_assistant", "hallucination", "high",
                            f"{rate:.0%} of generated answers failed grounding checks (blocked before delivery)", 0.9,
                            "hallucination_rate", rate, 0.1,
                            {"avg_grounding": round(float(conv["grounding"].mean()), 3),
                             "avg_quality": round(float(conv["quality"].mean()), 3)})]
        return []


class AgentBehaviourAgent:
    name = "AgentBehaviourAgent"

    def run(self, window: int) -> list[Finding]:
        runs = Store.get().agent_runs(2000)
        if runs.empty:
            return []
        start = monitoring.window_start(window, since("travel_planner_agent"))
        runs = runs[to_epoch(runs["ts"]) >= start]
        if len(runs) < 2:
            return []
        s = agent_monitor.summarize(runs)
        if s["anomaly_rate"] > 0.2 or s["validity_rate"] < 0.95:
            ex = runs[runs["anomalies"] != "[]"]["anomalies"].head(1).tolist()
            return [Finding(self.name, "travel_planner_agent", "agent_anomaly", "high",
                            f"{s['anomaly_rate']:.0%} of planner runs abnormal (e.g. {ex[0] if ex else 'invalid plans'})",
                            0.9, "anomaly_rate", s["anomaly_rate"], 0.2, {"summary": s})]
        return []


AGENTS = [HealthAgent(), DataQualityAgent(), DriftAgent(), FairnessAgent(), PricingAgent(), GenAIGuardAgent(),
          AgentBehaviourAgent()]

# remediation -> fault switches it resolves in the simulated world
RESOLVES = {"block_bad_data": ["corrupt_feed"], "scale_out": ["latency_spike"],
            "refresh_knowledge_base": ["llm_hallucination"], "circuit_breaker": ["agent_loop"],
            "revert_release": ["pricing_bug"]}


# ================================================================= supervisor
class SupervisorAgent:
    name = "SupervisorAgent"

    def __init__(self, window_minutes: int = 60, auto_remediate: bool = True):
        self.window = window_minutes
        self.auto = auto_remediate
        self.store = Store.get()

    def detect(self) -> list[Finding]:
        findings = []
        for a in AGENTS:
            try:
                findings.extend(a.run(self.window))
            except Exception as exc:  # noqa: BLE001 — a failing monitor must not stop the others
                self.store.audit(a.name, "agent.error", "", {"error": str(exc)}, "agent")
        return sorted(findings, key=lambda f: -f.priority)

    def _execute(self, model: str, action: str, hyp: dict, findings: list[Finding]) -> dict:
        policy_ok = AUTO_REMEDIATION.get(action, action in ("scale_out", "circuit_breaker", "revert_release"))
        if action == "human_review" or not self.auto or not policy_ok:
            pend = self.store.approvals("pending")
            same = pend[(pend["action"] == action) & (pend["model"] == model)] if not pend.empty else pend
            aid = int(same["id"].iloc[0]) if not same.empty else \
                self.store.request_approval(action, model, {"hypothesis": hyp}, hyp["evidence"], self.name)
            return {"action": action, "status": "awaiting_approval", "approval_id": aid}
        if action == "retrain":
            from .simulator import load_feedback
            fb = load_feedback(model)
            if model == "train_recommender":
                return {"action": action, "status": "skipped", "detail": "recommender retrains on choice logs nightly"}
            # regime reverted? if the previous production version is better on current data, roll back instead
            prev = lifecycle.previous_production(model)
            if prev and len(fb) >= 100:
                cur_v = registry.production_version(model)
                svc = _svc(model)
                metric, hib = svc.primary_metric
                old, new = lifecycle.evaluate(model, prev, fb)[metric], lifecycle.evaluate(model, cur_v, fb)[metric]
                better = (old > new + 0.01) if hib else (old < new * 0.9)
                if better:
                    res = lifecycle.rollback(model, reason=f"regime reverted: {prev} {metric}={old} beats {cur_v} "
                                                           f"{metric}={new} on current traffic", actor=self.name)
                    return {"action": "rollback", **res, "detail": f"{metric}: {cur_v} {new} vs {prev} {old}"}
            res = lifecycle.retrain(model, fb, reason=f"auto-retrain: {hyp['cause']}", actor=self.name)
            out = {"action": action, **res}
            if model == "fraud_detector":
                aid = self.store.request_approval(
                    "tighten_threshold", model, {"review_threshold": 0.4},
                    "New attack pattern — consider lowering review threshold from 0.50 to 0.40", self.name)
                out["follow_up_approval"] = aid
            return out
        if action == "rollback":
            res = lifecycle.rollback(model, reason=hyp["evidence"], actor=self.name)
            return {"action": action, **res}
        resolved = [f for f in RESOLVES.get(action, []) if faults.active(f)]
        for f in resolved:
            faults.set_fault(f, False)
        detail = {"block_bad_data": "malformed records quarantined; pipeline failed over to backup feed",
                  "scale_out": "inference replicas scaled 3 → 6; response cache enabled",
                  "refresh_knowledge_base": "free-form generation disabled, KB re-indexed, grounded templates enforced",
                  "circuit_breaker": "tool-call budget enforced; agent config rolled back",
                  "revert_release": "faulty pricing-service release reverted; quotes clamped to policy bounds"}.get(action, "")
        self.store.audit(self.name, f"remediation.{action}", model, {"detail": detail, "resolved": resolved}, "agent")
        return {"action": action, "status": "executed", "detail": detail}

    def _verify(self, model: str, t_fix: float, prev_version: str | None = None) -> dict:
        """Fresh post-remediation traffic must meet thresholds again."""
        from . import simulator
        if model in TABULAR:
            simulator.generate(model, 250, t_fix, t_fix + 1, save_feedback=False)
        elif model == "support_assistant":
            simulator.genai_traffic(n_support=8, n_planner=0)
        elif model == "travel_planner_agent":
            simulator.genai_traffic(n_support=0, n_planner=2)
        state = _state()
        state[model] = t_fix
        _save_state(state)
        remaining = [f for a in AGENTS for f in a.run(self.window) if f.model == model and f.kind not in ("fairness", "data_drift")]
        notes = []
        current = registry.production_version(model)
        for f in list(remaining):
            if f.kind != "performance" or not prev_version or prev_version == current:
                continue
            # the world itself changed (e.g. monsoon): judge the new model against the old one on the
            # *same* post-fix traffic instead of against a baseline from a different regime
            cur = monitoring.recent_inputs(model, self.window, t_fix).dropna(subset=["_label"])
            svc = _svc(model)
            if len(cur) < 50:
                continue
            cur = cur.rename(columns={"_label": svc.target})
            old = lifecycle.evaluate(model, prev_version, cur)
            new = lifecycle.evaluate(model, current, cur)
            m = f.metric if f.metric in new else svc.primary_metric[0]
            hib = m in ("auc", "accuracy", "f1", "recall", "hit@1")
            gain = (new[m] - old[m]) if hib else (old[m] - new[m]) / max(old[m], 1e-9)
            if gain >= (0.02 if hib else 0.10):
                remaining.remove(f)
                notes.append(f"{m}: {prev_version} {old[m]} → {current} {new[m]} on identical post-fix traffic")
        return {"verified": not remaining, "remaining": [f.message for f in remaining], "notes": notes}

    def run_cycle(self) -> dict:
        t0 = time.time()
        findings = self.detect()
        self.store.audit(self.name, "cycle.started", "", {"findings": len(findings)}, "agent")
        alert_ids = {}
        for f in findings:
            alert_ids[id(f)] = self.store.add_alert(f.agent, f.model, f.kind, f.severity, f.priority, f.message,
                                                    {**asdict(f), "priority": f.priority})
        by_model: dict[str, list[Finding]] = {}
        for f in findings:
            by_model.setdefault(f.model, []).append(f)
        incidents = []
        for model, fs in sorted(by_model.items(), key=lambda kv: -max(f.priority for f in kv[1])):
            if all(f.kind == "fairness" for f in fs) and max(SEVERITY_WEIGHT[f.severity] for f in fs) < 2:
                continue
            analysis = rca.analyse(model, [asdict(f) for f in fs])
            if not analysis["primary"]:
                continue
            iid = self.store.open_incident(model, analysis["primary"]["evidence"][:160], analysis)
            for f in fs:
                self.store.set_alert_status(alert_ids[id(f)], "acknowledged", iid)
            prev_version = registry.production_version(model)
            actions, done = [], set()
            for hyp in analysis["hypotheses"]:
                act = hyp["remediation"]
                if act in done or hyp["confidence"] < 0.5:
                    continue
                done.add(act)
                actions.append(self._execute(model, act, hyp, fs))
            t_fix = time.time()
            if any(a.get("status") in ("executed", "promoted", "rolled_back") for a in actions):
                verification = self._verify(model, t_fix, prev_version)
                status = "resolved" if verification["verified"] else "mitigating"
            elif any(a.get("status") == "awaiting_approval" for a in actions):
                verification, status = {"verified": False, "remaining": ["awaiting human approval"]}, "awaiting_approval"
            elif not actions:
                verification, status = {"verified": False, "remaining": ["low-confidence signal — monitoring"]}, "monitoring"
            else:
                verification, status = {"verified": False, "remaining": [a.get("reason", a.get("status")) for a in actions]}, "escalated"
            self.store.close_incident(iid, actions + [{"verification": verification}], status)
            if status == "resolved":
                for f in fs:
                    self.store.set_alert_status(alert_ids[id(f)], "resolved", iid)
            incidents.append({"incident_id": iid, "model": model, "rca": analysis, "actions": actions,
                              "verification": verification, "status": status})
        result = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": round(time.time() - t0, 1),
                  "findings": [{**asdict(f), "priority": f.priority} for f in findings], "incidents": incidents}
        from .reports import governance_report
        result["report_path"] = str(governance_report(result))
        self.store.audit(self.name, "cycle.completed", "", {"findings": len(findings), "incidents": len(incidents),
                                                           "resolved": sum(i["status"] == "resolved" for i in incidents)},
                         "agent")
        return result


def apply_approval(approval_id: int, approve: bool, user: str) -> dict:
    """Human decision on a queued action; approved actions execute immediately."""
    store = Store.get()
    rec = store.decide_approval(approval_id, approve, user)
    if not approve or not rec:
        return rec
    payload = json.loads(rec["payload"])
    if rec["action"] == "tighten_threshold" and rec["model"] == "fraud_detector":
        from ..passenger.fraud import FraudDetector
        FraudDetector.review_threshold = float(payload.get("review_threshold", 0.4))
        store.audit(user, "threshold.changed", "fraud_detector", payload, "user")
    return rec

"""Governance persistence: a single SQLite database holding telemetry, alerts,
incidents, approvals, conversations and a tamper-evident (hash-chained) audit log.

In production these tables map to Kafka topics + a time-series store + an
immutable ledger; the interface stays the same.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, actor_type TEXT,
    action TEXT, resource TEXT, details TEXT, prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS inference_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, epoch REAL, model TEXT, version TEXT,
    latency_ms REAL, status TEXT, dq_score REAL, input TEXT, output TEXT,
    prediction REAL, label REAL, segment TEXT, request_id TEXT);
CREATE INDEX IF NOT EXISTS ix_inf_model ON inference_log(model, epoch);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, agent TEXT, model TEXT, kind TEXT,
    severity TEXT, priority REAL, title TEXT, details TEXT, status TEXT, incident_id INTEGER);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_open TEXT, ts_close TEXT, model TEXT,
    title TEXT, root_cause TEXT, actions TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, action TEXT, model TEXT, payload TEXT,
    reason TEXT, status TEXT, requested_by TEXT, decided_by TEXT, decided_ts TEXT);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session TEXT, channel TEXT, lang TEXT,
    intent TEXT, user_msg TEXT, bot_msg TEXT, grounding REAL, hallucination INTEGER,
    quality REAL, csat REAL, sources TEXT);
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, agent TEXT, run_id TEXT, query TEXT,
    steps INTEGER, latency_ms REAL, outcome TEXT, valid INTEGER, anomalies TEXT, trace TEXT);
"""

_PII_PATTERNS = [
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)"), "[PHONE]"),
]


def mask_pii(text: str) -> str:
    """Privacy-by-design: strip personal identifiers before anything is persisted."""
    if not isinstance(text, str):
        return text
    for pat, repl in _PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


def mask_obj(obj):
    """Mask PII inside string values only — never touch numbers or JSON structure."""
    if isinstance(obj, str):
        return mask_pii(obj)
    if isinstance(obj, dict):
        return {k: mask_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [mask_obj(v) for v in obj]
    return obj


def to_epoch(ts: pd.Series) -> pd.Series:
    """ISO-8601 timestamps -> unix seconds (independent of pandas datetime resolution)."""
    return ts.map(lambda x: datetime.fromisoformat(x).timestamp())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    _instance: "Store | None" = None
    _lock = threading.Lock()

    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @classmethod
    def get(cls) -> "Store":
        with cls._lock:
            if cls._instance is None or not cls._instance.path.exists():
                cls._instance = Store()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Delete the database (used by tests and the 'reset demo' button)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.conn.close()
                cls._instance = None
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(DB_PATH) + suffix)
                if p.exists():
                    p.unlink()

    # ------------------------------------------------------------ helpers
    def execute(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self.conn, params=params)

    # -------------------------------------------------------------- audit
    def audit(self, actor: str, action: str, resource: str = "", details: dict | None = None,
              actor_type: str = "system") -> str:
        payload = json.dumps(mask_obj(details or {}), default=str, sort_keys=True)
        ts = now_iso()
        with self._lock:
            row = self.conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
            prev = row[0] if row else "GENESIS"
            digest = hashlib.sha256(f"{prev}|{ts}|{actor}|{action}|{resource}|{payload}".encode()).hexdigest()
            self.conn.execute(
                "INSERT INTO audit_log(ts, actor, actor_type, action, resource, details, prev_hash, hash)"
                " VALUES (?,?,?,?,?,?,?,?)", (ts, actor, actor_type, action, resource, payload, prev, digest))
            self.conn.commit()
        return digest

    def verify_audit_chain(self) -> tuple[bool, int | None]:
        """Recompute every hash; returns (ok, first_bad_id)."""
        rows = self.conn.execute(
            "SELECT id, ts, actor, action, resource, details, prev_hash, hash FROM audit_log ORDER BY id").fetchall()
        prev = "GENESIS"
        for rid, ts, actor, action, resource, details, prev_hash, h in rows:
            calc = hashlib.sha256(f"{prev}|{ts}|{actor}|{action}|{resource}|{details}".encode()).hexdigest()
            if prev_hash != prev or calc != h:
                return False, rid
            prev = h
        return True, None

    # ---------------------------------------------------------- telemetry
    def log_inference(self, model: str, version: str, latency_ms: float, status: str, dq_score: float,
                      inp: dict, out: dict, prediction: float | None = None, label: float | None = None,
                      segment: str | None = None, request_id: str | None = None, epoch: float | None = None) -> int:
        ep = epoch if epoch is not None else time.time()
        ts = datetime.fromtimestamp(ep, timezone.utc).isoformat(timespec="seconds")
        return self.execute(
            "INSERT INTO inference_log(ts, epoch, model, version, latency_ms, status, dq_score, input, output,"
            " prediction, label, segment, request_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, ep, model, version, latency_ms, status, dq_score,
             json.dumps(mask_obj(inp), default=str), json.dumps(out, default=str),
             prediction, label, segment, request_id))

    def log_inferences_bulk(self, rows: list[tuple]) -> None:
        self.executemany(
            "INSERT INTO inference_log(ts, epoch, model, version, latency_ms, status, dq_score, input, output,"
            " prediction, label, segment, request_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def inferences(self, model: str, since_epoch: float = 0, limit: int = 20000) -> pd.DataFrame:
        return self.query("SELECT * FROM inference_log WHERE model=? AND epoch>=? ORDER BY epoch DESC LIMIT ?",
                          (model, since_epoch, limit))

    # ------------------------------------------------------------- alerts
    def add_alert(self, agent: str, model: str, kind: str, severity: str, priority: float, title: str,
                  details: dict) -> int:
        aid = self.execute(
            "INSERT INTO alerts(ts, agent, model, kind, severity, priority, title, details, status)"
            " VALUES (?,?,?,?,?,?,?,?, 'open')",
            (now_iso(), agent, model, kind, severity, priority, title, json.dumps(details, default=str)))
        self.audit(agent, "alert.raised", model, {"alert_id": aid, "kind": kind, "severity": severity,
                                                  "title": title}, actor_type="agent")
        return aid

    def alerts(self, status: str | None = None) -> pd.DataFrame:
        if status:
            return self.query("SELECT * FROM alerts WHERE status=? ORDER BY priority DESC, id DESC", (status,))
        return self.query("SELECT * FROM alerts ORDER BY id DESC")

    def set_alert_status(self, alert_id: int, status: str, incident_id: int | None = None) -> None:
        self.execute("UPDATE alerts SET status=?, incident_id=COALESCE(?, incident_id) WHERE id=?",
                     (status, incident_id, alert_id))

    # ---------------------------------------------------------- incidents
    def open_incident(self, model: str, title: str, root_cause: dict) -> int:
        iid = self.execute("INSERT INTO incidents(ts_open, model, title, root_cause, actions, status)"
                           " VALUES (?,?,?,?, '[]', 'open')",
                           (now_iso(), model, title, json.dumps(root_cause, default=str)))
        self.audit("supervisor", "incident.opened", model, {"incident_id": iid, "title": title}, "agent")
        return iid

    def close_incident(self, iid: int, actions: list[dict], status: str = "resolved") -> None:
        self.execute("UPDATE incidents SET ts_close=?, actions=?, status=? WHERE id=?",
                     (now_iso(), json.dumps(actions, default=str), status, iid))
        self.audit("supervisor", f"incident.{status}", "", {"incident_id": iid, "actions": actions}, "agent")

    def incidents(self) -> pd.DataFrame:
        return self.query("SELECT * FROM incidents ORDER BY id DESC")

    # ---------------------------------------------------------- approvals
    def request_approval(self, action: str, model: str, payload: dict, reason: str, requested_by: str) -> int:
        aid = self.execute("INSERT INTO approvals(ts, action, model, payload, reason, status, requested_by)"
                           " VALUES (?,?,?,?,?, 'pending', ?)",
                           (now_iso(), action, model, json.dumps(payload, default=str), reason, requested_by))
        self.audit(requested_by, "approval.requested", model, {"approval_id": aid, "action": action}, "agent")
        return aid

    def decide_approval(self, approval_id: int, approve: bool, decided_by: str) -> dict:
        status = "approved" if approve else "rejected"
        self.execute("UPDATE approvals SET status=?, decided_by=?, decided_ts=? WHERE id=?",
                     (status, decided_by, now_iso(), approval_id))
        self.audit(decided_by, f"approval.{status}", "", {"approval_id": approval_id}, "user")
        row = self.query("SELECT * FROM approvals WHERE id=?", (approval_id,))
        return row.iloc[0].to_dict() if not row.empty else {}

    def approvals(self, status: str | None = None) -> pd.DataFrame:
        if status:
            return self.query("SELECT * FROM approvals WHERE status=? ORDER BY id DESC", (status,))
        return self.query("SELECT * FROM approvals ORDER BY id DESC")

    # ------------------------------------------------------ conversations
    def log_conversation(self, session: str, channel: str, lang: str, intent: str, user_msg: str,
                         bot_msg: str, grounding: float, hallucination: bool, quality: float,
                         csat: float | None, sources: list[str]) -> int:
        return self.execute(
            "INSERT INTO conversations(ts, session, channel, lang, intent, user_msg, bot_msg, grounding,"
            " hallucination, quality, csat, sources) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (now_iso(), session, channel, lang, intent, mask_pii(user_msg), mask_pii(bot_msg), grounding,
             int(hallucination), quality, csat, json.dumps(sources)))

    def conversations(self, limit: int = 5000) -> pd.DataFrame:
        return self.query("SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,))

    # --------------------------------------------------------- agent runs
    def log_agent_run(self, agent: str, run_id: str, query: str, steps: int, latency_ms: float, outcome: str,
                      valid: bool, anomalies: list[str], trace: list[dict]) -> int:
        return self.execute(
            "INSERT INTO agent_runs(ts, agent, run_id, query, steps, latency_ms, outcome, valid, anomalies, trace)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now_iso(), agent, run_id, mask_pii(query), steps, latency_ms, outcome, int(valid),
             json.dumps(anomalies), json.dumps(trace, default=str)))

    def agent_runs(self, limit: int = 2000) -> pd.DataFrame:
        return self.query("SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,))

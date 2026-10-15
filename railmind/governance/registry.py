"""Model registry: versioning, stage management (staging/production/archived),
promotion and rollback. Backed by joblib files + a JSON index (MLflow-compatible concepts)."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any

import joblib

from ..config import MODEL_DIR
from .store import Store

INDEX = MODEL_DIR / "registry.json"
_lock = threading.RLock()
_cache: dict[tuple[str, str], Any] = {}


def _read() -> dict:
    if INDEX.exists():
        return json.loads(INDEX.read_text())
    return {}


def _write(idx: dict) -> None:
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, indent=2, default=str))
    tmp.replace(INDEX)


def fingerprint(df) -> str:
    """Hash of training data shape + head for lineage."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(df.head(200).to_csv(index=False).encode())
    return h.hexdigest()[:16]


def register(name: str, obj: Any, metrics: dict, *, params: dict | None = None, data_hash: str = "",
             stage: str = "staging", reason: str = "initial training", parent: str | None = None,
             card: dict | None = None) -> str:
    with _lock:
        idx = _read()
        versions = idx.setdefault(name, {"versions": [], "production": None})
        version = f"v{len(versions['versions']) + 1}"
        path = MODEL_DIR / name / f"{version}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        versions["versions"].append({
            "version": version, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": metrics, "params": params or {}, "data_hash": data_hash, "stage": stage,
            "reason": reason, "parent": parent, "path": str(path.relative_to(MODEL_DIR)), "card": card or {},
        })
        _write(idx)
    Store.get().audit("registry", "model.registered", name,
                      {"version": version, "metrics": metrics, "reason": reason, "stage": stage})
    if stage == "production":
        promote(name, version, actor="registry", reason=reason)
    return version


def promote(name: str, version: str, actor: str = "lifecycle", reason: str = "") -> None:
    with _lock:
        idx = _read()
        entry = idx[name]
        for v in entry["versions"]:
            if v["version"] == version:
                v["stage"] = "production"
            elif v["stage"] == "production":
                v["stage"] = "archived"
        prev = entry.get("production")
        entry["production"] = version
        entry.setdefault("history", []).append(
            {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from": prev, "to": version,
             "actor": actor, "reason": reason})
        _write(idx)
    Store.get().audit(actor, "model.promoted", name, {"from": prev, "to": version, "reason": reason}, "agent")


def rollback(name: str, actor: str = "lifecycle", reason: str = "") -> str | None:
    """Restore the previous production version recorded in history."""
    idx = _read()
    hist = idx.get(name, {}).get("history", [])
    current = idx.get(name, {}).get("production")
    for h in reversed(hist):
        if h["to"] == current and h["from"]:
            target = h["from"]
            promote(name, target, actor=actor, reason=f"ROLLBACK: {reason}")
            Store.get().audit(actor, "model.rolled_back", name, {"from": current, "to": target, "reason": reason},
                              "agent")
            return target
    return None


def production_version(name: str) -> str | None:
    return _read().get(name, {}).get("production")


def info(name: str, version: str | None = None) -> dict:
    entry = _read().get(name)
    if not entry:
        return {}
    version = version or entry.get("production")
    for v in entry["versions"]:
        if v["version"] == version:
            return v
    return {}


def load(name: str, version: str | None = None) -> tuple[Any, str]:
    version = version or production_version(name)
    if version is None:
        raise FileNotFoundError(f"No production model for '{name}'. Run: python scripts/train_all.py")
    key = (name, version)
    if key not in _cache:
        _cache[key] = joblib.load(MODEL_DIR / info(name, version)["path"])
    return _cache[key], version


def list_models() -> dict:
    return _read()


def clear_cache() -> None:
    _cache.clear()

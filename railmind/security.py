"""Role-based access control shared by the UI and the REST API."""
from __future__ import annotations

import hashlib
import os

PERMISSIONS = {
    "passenger": "Use passenger-facing AI services",
    "ops.view": "View operational dashboards (fraud, delays, demand)",
    "governance.view": "View model health, drift, bias and audit data",
    "governance.act": "Inject scenarios, run governance cycles, retrain or roll back",
    "approve": "Approve or reject high-impact actions queued by agents",
    "audit": "Export audit trail, reports and model cards",
}

ROLES = {
    "Passenger": {"passenger"},
    "Operations Analyst": {"passenger", "ops.view", "governance.view"},
    "ML Engineer": {"passenger", "ops.view", "governance.view", "governance.act"},
    "Governance Officer": {"passenger", "ops.view", "governance.view", "approve", "audit"},
    "Administrator": set(PERMISSIONS),
}

# Demo API keys (override with RAILMIND_API_KEYS="key1:Role,key2:Role")
_DEFAULT_KEYS = {"passenger-demo-key": "Passenger", "analyst-demo-key": "Operations Analyst",
                 "mlops-demo-key": "ML Engineer", "governance-demo-key": "Governance Officer",
                 "admin-demo-key": "Administrator"}


def api_keys() -> dict[str, str]:
    env = os.getenv("RAILMIND_API_KEYS")
    if not env:
        return _DEFAULT_KEYS
    return dict(pair.split(":", 1) for pair in env.split(",") if ":" in pair)


def role_for_key(key: str | None) -> str | None:
    return api_keys().get(key or "")


def allowed(role: str, permission: str) -> bool:
    return permission in ROLES.get(role, set())


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:10]

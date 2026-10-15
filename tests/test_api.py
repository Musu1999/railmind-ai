from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
P = {"X-API-Key": "passenger-demo-key"}
ADMIN = {"X-API-Key": "admin-demo-key"}


def test_health_open():
    assert client.get("/health").json()["status"] == "ok"


def test_auth_required_and_rbac():
    assert client.post("/v1/support/chat", json={"message": "hi"}).status_code == 401
    assert client.post("/v1/governance/cycle", headers=P).status_code == 403


def test_plan_and_book_via_api():
    r = client.post("/v1/planner/plan", headers=P,
                    json={"query": "Pune to Delhi tomorrow under 1500 reach before 10 am"}).json()
    assert r["outcome"] == "plan_ready"
    bad = client.post("/v1/planner/book", headers=P, json={"run_id": r["run_id"], "approval_token": "forged"})
    assert bad.status_code == 403
    ok = client.post("/v1/planner/book", headers=P, json={"run_id": r["run_id"], "approval_token": r["approval_token"]})
    assert ok.status_code == 200 and ok.json()["bookings"]


def test_governance_endpoints():
    assert client.get("/v1/governance/health", headers=ADMIN).status_code == 200
    audit = client.get("/v1/governance/audit", headers=ADMIN).json()
    assert "chain_intact" in audit

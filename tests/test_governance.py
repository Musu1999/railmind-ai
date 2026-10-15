import numpy as np
import pandas as pd

from railmind import faults
from railmind.governance import drift, hallucination, simulator
from railmind.governance.store import Store, mask_pii
from railmind.governance.supervisor import SupervisorAgent
from railmind.passenger.support import SupportAssistant, detect_language


def test_audit_chain_detects_tampering():
    s = Store.get()
    s.audit("test", "a", "r", {"x": 1})
    s.audit("test", "b", "r", {"x": 2})
    assert s.verify_audit_chain()[0]
    last = s.query("SELECT MAX(id) AS id FROM audit_log")["id"].iloc[0]
    s.execute("UPDATE audit_log SET details='{\"x\": 999}' WHERE id=?", (int(last),))
    ok, bad = s.verify_audit_chain()
    assert not ok and bad == last
    s.execute("DELETE FROM audit_log WHERE id=?", (int(last),))  # restore a valid chain for other tests


def test_pii_is_masked():
    out = mask_pii("call 9876543210 or mail a.b@x.com, card 4111 1111 1111 1111")
    assert "9876543210" not in out and "a.b@x.com" not in out and "4111" not in out


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    edges = np.quantile(ref, np.linspace(0, 1, 11))
    props = np.histogram(ref, edges)[0] / len(ref)
    assert drift.psi_numeric(props, edges, pd.Series(rng.normal(0, 1, 2000))) < 0.1
    assert drift.psi_numeric(props, edges, pd.Series(rng.normal(1.5, 1, 2000))) > 0.25


def test_hallucination_guard():
    ev = ["Rs 120 for Sleeper and Rs 60 for Second Class."]
    assert not hallucination.check("The charge is Rs 120 for Sleeper class.", ev).hallucination
    bad = hallucination.check("The charge is Rs 450 for Sleeper and confirmation is guaranteed for your berth.", ev)
    assert bad.hallucination and "450" in bad.unsupported_numbers and bad.policy_violations


def test_support_multilingual_grounded():
    a = SupportAssistant()
    assert detect_language("तत्काल बुकिंग कब खुलती है?") == "hi"
    assert detect_language("ওয়েটিং লিস্ট") == "bn"
    r = a.answer("What is the cancellation charge for sleeper class?")
    assert "120" in r["answer"] and not r["hallucination_blocked"]


def test_support_blocks_injected_hallucination():
    faults.set_fault("llm_hallucination", True)
    try:
        r = SupportAssistant().answer("What is the cancellation charge for sleeper class?")
    finally:
        faults.clear_all()
    assert r["hallucination_blocked"]
    assert "guaranteed" not in r["answer"]


def test_closed_loop_governance_cycle():
    faults.clear_all()
    simulator.seed_history(hours=6, n=200)
    faults.set_fault("pricing_bug", True)
    faults.set_fault("latency_spike", True)
    faults.set_fault("corrupt_feed", True)
    simulator.run_traffic(minutes=20, n=250, include_genai=False)
    res = SupervisorAgent(window_minutes=60).run_cycle()
    kinds = {(f["model"], f["kind"]) for f in res["findings"]}
    assert ("fare_predictor", "pricing_anomaly") in kinds
    assert ("train_recommender", "sla") in kinds
    assert ("delay_predictor", "data_quality") in kinds
    status = {i["model"]: i["status"] for i in res["incidents"]}
    assert status["fare_predictor"] == "resolved"
    assert status["train_recommender"] == "resolved"
    assert not faults.all_active()

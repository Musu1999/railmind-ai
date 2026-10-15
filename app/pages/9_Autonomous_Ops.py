from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from common import pill, setup

from railmind import faults
from railmind.governance import rca, simulator
from railmind.governance.store import Store
from railmind.governance.supervisor import SupervisorAgent, apply_approval
from railmind.security import allowed

role = setup("Autonomous AI Ops", "🧠", "AI agents monitoring AI agents: inject real-world failures, then watch the "
             "supervisor detect, prioritise, diagnose, remediate, verify — and ask humans when policy requires.",
             permission="governance.view")
store = Store.get()
can_act = allowed(role, "governance.act")
can_approve = allowed(role, "approve")

st.subheader("① Simulate production")
c1, c2, c3 = st.columns(3)
if c1.button("Seed 24 h of healthy history", disabled=not can_act, width="stretch"):
    with st.spinner("Replaying traffic…"):
        simulator.seed_history()
    st.success("Background traffic written to telemetry.")
if c2.button("Generate live traffic (last 30 min)", disabled=not can_act, type="primary", width="stretch"):
    with st.spinner("Serving requests through production models…"):
        out = simulator.run_traffic()
    st.success(f"Served {sum(o.get('requests', 0) for o in out if 'requests' in o):,} model requests "
               f"+ GenAI conversations under faults: {', '.join(faults.all_active()) or 'none'}")
if c3.button("Clear all faults", disabled=not can_act, width="stretch"):
    faults.clear_all()
    st.rerun()
with st.expander("Reset demo"):
    st.caption("Clears faults, telemetry, alerts, incidents and the audit log, and restores every model to v1.")
    if st.button("Reset demo to baseline", disabled=not allowed(role, "governance.act")):
        from railmind.config import ARTIFACT_DIR
        from railmind.governance import lifecycle
        faults.clear_all()
        lifecycle.reset_to_baseline()
        (ARTIFACT_DIR / "governance_state.json").unlink(missing_ok=True)
        Store.reset()
        st.session_state.pop("cycle", None)
        st.rerun()

st.subheader("② Inject failure scenarios")
cols = st.columns(4)
for i, (name, (model, desc)) in enumerate(faults.SCENARIOS.items()):
    with cols[i % 4]:
        on = st.toggle(name.replace("_", " "), value=faults.active(name), key=f"f_{name}", disabled=not can_act,
                       help=f"{desc} (affects {model})")
        if on != faults.active(name):
            faults.set_fault(name, on)
st.caption("Tip: toggle one or more scenarios → *Generate live traffic* → *Run governance cycle*.")
if not can_act:
    st.caption("🔒 Your role can observe but not inject faults or run cycles (needs `governance.act`).")

st.subheader("③ Run the autonomous governance cycle")
window = st.slider("Look-back window (minutes)", 15, 180, 60, step=15)
auto = st.toggle("Allow autonomous remediation (policy-approved actions only)", value=True)
if st.button("🧠 Run governance cycle", type="primary", disabled=not can_act):
    with st.spinner("Governance agents inspecting telemetry, running RCA and remediating…"):
        st.session_state["cycle"] = SupervisorAgent(window, auto).run_cycle()

cyc = st.session_state.get("cycle")
if cyc:
    f = pd.DataFrame(cyc["findings"])
    k = st.columns(4)
    k[0].metric("Findings", len(f))
    k[1].metric("Incidents", len(cyc["incidents"]))
    k[2].metric("Resolved autonomously", sum(i["status"] == "resolved" for i in cyc["incidents"]))
    k[3].metric("Cycle time", f"{cyc['duration_s']} s")
    if f.empty:
        st.success("All AI services healthy — no findings.")
    else:
        st.markdown("**Prioritised alerts** (severity × model criticality × confidence × blast radius)")
        st.dataframe(f[["priority", "severity", "model", "agent", "kind", "message"]], width="stretch", hide_index=True)
    for inc in cyc["incidents"]:
        kind = {"resolved": "ok", "awaiting_approval": "warn", "monitoring": "info"}.get(inc["status"], "bad")
        with st.expander(f"Incident #{inc['incident_id']} · {inc['model']} · {inc['status']}", expanded=True):
            p = inc["rca"]["primary"]
            st.markdown(pill(inc["status"], kind) + pill(f"root cause: {p['cause']}", "info")
                        + pill(f"confidence {p['confidence']}", "info"), unsafe_allow_html=True)
            st.markdown(f"**Evidence:** {p['evidence']}")
            st.markdown(f"**Upstream implicated:** {', '.join(p['upstream']) or '—'} · "
                        f"**Blast radius:** {', '.join(inc['rca']['blast_radius'])}")
            st.markdown("**Hypotheses considered**")
            st.dataframe(pd.DataFrame(inc["rca"]["hypotheses"])[["cause", "confidence", "remediation", "evidence"]],
                         width="stretch", hide_index=True)
            st.markdown("**Actions**")
            for a in inc["actions"]:
                st.markdown(f"- `{a['action']}` → **{a.get('status')}** "
                            + " ".join(f"{k}={v}" for k, v in a.items() if k not in ("action", "status")))
            v = inc["verification"]
            st.markdown(f"**Verification on fresh traffic:** {'✅ passed' if v['verified'] else '⏳ ' + '; '.join(v['remaining'])}")
            for n in v.get("notes", []):
                st.caption(n)
    rp = Path(cyc["report_path"])
    if rp.exists():
        st.download_button("Download governance report (Markdown)", rp.read_text(encoding="utf-8"), file_name=rp.name)

st.subheader("④ Human-in-the-loop approvals")
pend = store.approvals("pending")
if pend.empty:
    st.caption("No pending approvals.")
for _, a in pend.iterrows():
    with st.container(border=True):
        st.markdown(f"**#{a['id']} · {a['action']}** on `{a['model']}` — requested by {a['requested_by']}")
        st.caption(a["reason"])
        c1, c2, _ = st.columns([1, 1, 5])
        if c1.button("Approve", key=f"ap{a['id']}", disabled=not can_approve):
            apply_approval(int(a["id"]), True, role)
            st.rerun()
        if c2.button("Reject", key=f"rj{a['id']}", disabled=not can_approve):
            apply_approval(int(a["id"]), False, role)
            st.rerun()
if not can_approve:
    st.caption("🔒 Only Governance Officers / Administrators can approve.")

st.subheader("⑤ Dependency graph used for root-cause analysis")
edges = rca.dependency_edges()
dot = "digraph D { rankdir=LR; bgcolor=transparent; node [shape=box, style=\"rounded,filled\", fillcolor=\"#F4F1EA\", fontname=Helvetica, fontsize=10];\n"
models = set(rca.CONSUMERS)
for m in models:
    dot += f'"{m}" [fillcolor="#FFE8D6"];\n'
for a, b in edges:
    dot += f'"{a}" -> "{b}";\n'
st.graphviz_chart(dot + "}", width="stretch")

st.subheader("Incident history")
inc = store.incidents()
if not inc.empty:
    st.dataframe(inc[["id", "ts_open", "ts_close", "model", "status", "title"]], width="stretch", hide_index=True)

"""RailMind AI — home / overview."""
from __future__ import annotations

import time

import streamlit as st

from common import pill, setup

from railmind.governance import monitoring, registry
from railmind.governance.store import Store

setup("RailMind AI", "🚆", "AI-powered intelligent railway travel with an autonomous AI-for-AI governance layer")

store = Store.get()
models = registry.list_models()
alerts = store.alerts("open")
inc = store.incidents()
day = store.query("SELECT COUNT(*) AS n FROM inference_log WHERE epoch >= ?", (time.time() - 86400,))["n"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("AI services in production", sum(1 for v in models.values() if v.get("production")))
c2.metric("Governed inferences (24 h)", f"{int(day):,}")
c3.metric("Open alerts", len(alerts))
c4.metric("Incidents auto-resolved", int((inc["status"] == "resolved").sum()) if not inc.empty else 0)

st.subheader("Platform architecture")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#1B1F3B"];
  subgraph cluster_p { label="Passenger AI services"; style="rounded"; color="#E8590C"; fontname="Helvetica";
    conf [label="1 · Ticket confirmation\\npredictor", fillcolor="#FFF4E5"];
    rec  [label="2 · Smart train\\nrecommender", fillcolor="#FFF4E5"];
    dly  [label="3 · Delay & disruption\\nprediction", fillcolor="#FFF4E5"];
    agt  [label="4 · Travel planner\\n(multi-agent)", fillcolor="#FFE8D6"];
    fare [label="5 · Dynamic fare\\nprediction", fillcolor="#FFF4E5"];
    sup  [label="6 · Multilingual support\\nassistant (RAG)", fillcolor="#FFE8D6"];
    frd  [label="7 · Fraud, security &\\nbot detection", fillcolor="#FFF4E5"]; }
  subgraph cluster_g { label="8 · Autonomous AI governance layer"; style="rounded"; color="#1B1F3B"; fontname="Helvetica";
    dq  [label="Data-quality gate", fillcolor="#EAF2FB"];
    tel [label="Telemetry &\\nSLA monitor", fillcolor="#EAF2FB"];
    ga  [label="Governance agents\\n(health · drift · bias ·\\nhallucination · pricing · agent)", fillcolor="#DDE7F7"];
    sv  [label="Supervisor agent\\n(prioritise · RCA · remediate · verify)", fillcolor="#C9D8F2"];
    lc  [label="Lifecycle: registry ·\\nretrain · promote · rollback", fillcolor="#EAF2FB"];
    au  [label="Hash-chained audit ·\\nmodel cards · reports", fillcolor="#EAF2FB"];
    hil [label="Human approvals\\n(RBAC)", fillcolor="#F3E8FD"]; }
  {conf rec dly fare frd} -> dq [color="#868E96"]; agt -> conf; agt -> dly; agt -> fare; sup -> agt; sup -> dly;
  dq -> tel -> ga -> sv -> lc; sv -> hil; sv -> au; lc -> {conf dly fare frd rec} [style=dashed, color="#2F9E44"];
}
""", width="stretch")

st.subheader("Live service health (last 60 min)")
h = monitoring.health_table(60)
h["status"] = h["status"].map({"healthy": "🟢 healthy", "degraded": "🟠 degraded", "down": "🔴 down", "idle": "⚪ idle"})
st.dataframe(h[["model", "status", "requests", "p95_ms", "error_rate", "blocked_rate", "avg_dq", "versions"]],
             width="stretch", hide_index=True)
if (h["requests"] == 0).all():
    st.info("No traffic yet — open **Autonomous Ops** and click *Generate traffic* to simulate production load.")

st.subheader("Try it")
cols = st.columns(3)
cols[0].markdown("**Plan a journey** — open *Travel Planner Agent* and ask: "
                 "_“I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.”_")
cols[1].markdown("**Ask support in Hindi or Bengali** — _“तत्काल बुकिंग कब खुलती है?”_")
cols[2].markdown("**Break something on purpose** — inject *monsoon drift* or *LLM hallucination* in "
                 "*Autonomous Ops* and watch the governance agents detect, diagnose and fix it.")
st.markdown(pill("synthetic data", "warn") + pill("works offline · no API key", "ok")
            + pill("optional LLM: Gemini / OpenAI / Claude / Ollama", "info"), unsafe_allow_html=True)

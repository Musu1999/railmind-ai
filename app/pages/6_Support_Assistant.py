from __future__ import annotations

import uuid

import streamlit as st

from common import pill, setup

from railmind import data, llm
from railmind.passenger.support import SupportAssistant, record_feedback

setup("AI Customer Support Assistant", "💬", "24×7 multilingual help (English · हिन्दी · বাংলা) for PNR status, "
      "refunds, cancellations, delays and journey planning — every answer grounded and hallucination-checked.")


@st.cache_resource
def assistant() -> SupportAssistant:
    return SupportAssistant()


a = assistant()
st.session_state.setdefault("chat", [])
st.session_state.setdefault("session", uuid.uuid4().hex[:8])

pn = data.load("pnrs")
sample_pnr = pn[pn["status"] == "CAN"]["pnr"].iloc[0]
suggest = ["What is the cancellation charge for sleeper class?", f"Refund status for PNR {sample_pnr}",
           "तत्काल बुकिंग कब खुलती है?", "ওয়েটিং লিস্ট টিকিট বাতিল করলে কত টাকা কাটে?", "Is train 12089 running late?",
           "Plan a trip from Pune to Delhi tomorrow under 1500"]
cols = st.columns(3)
clicked = None
for i, s in enumerate(suggest):
    if cols[i % 3].button(s, key=f"s{i}", width="stretch"):
        clicked = s
st.caption(f"Generation mode: **{llm.provider() or 'grounded templates (no API key)'}** · "
           "set `RAILMIND_LLM` + an API key to enable LLM rephrasing — the guard still checks every word.")

for m in st.session_state["chat"]:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("meta"):
            st.markdown(m["meta"], unsafe_allow_html=True)

prompt = st.chat_input("Ask about your ticket, refund, train or trip…") or clicked
if prompt:
    st.session_state["chat"].append({"role": "user", "text": prompt})
    r = a.answer(prompt, session=st.session_state["session"])
    meta = (pill(f"lang {r['lang']}", "info") + pill(f"intent {r['intent']}", "info")
            + pill(f"grounding {r['grounding']:.2f}", "ok" if r["grounding"] >= 0.35 else "warn")
            + pill(f"quality {r['quality']:.2f}", "ok" if r["quality"] >= 0.7 else "warn")
            + pill(f"{r['latency_ms']:.0f} ms", "info"))
    if r["hallucination_blocked"]:
        meta += pill("⚠ unsafe draft blocked: " + "; ".join(r["guard_reasons"]), "bad")
    st.session_state["chat"].append({"role": "assistant", "text": r["answer"], "meta": meta, "cid": r["conversation_id"]})
    st.rerun()

last = next((m for m in reversed(st.session_state["chat"]) if m["role"] == "assistant"), None)
if last:
    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("👍 Helpful"):
        record_feedback(last["cid"], 5)
        st.toast("Thanks — feedback logged for CSAT monitoring")
    if c2.button("👎 Not helpful"):
        record_feedback(last["cid"], 1)
        st.toast("Thanks — this conversation is flagged for quality review")
if st.session_state["chat"] and st.button("Clear chat"):
    st.session_state["chat"] = []
    st.rerun()

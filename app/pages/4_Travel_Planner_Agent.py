from __future__ import annotations

import pandas as pd
import streamlit as st

from common import pill, setup

from railmind.passenger.planner import PlannerAgent, book

role = setup("AI Travel Planner Agent", "🤖", "Tell the agent your trip in plain language. A team of sub-agents "
             "searches, predicts availability, delays and fares, optimises, validates — and books only after you approve.")

EXAMPLES = ["I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.",
            "Mumbai to Goa on Saturday in chair car for 2 people",
            "Kolkata to Patna tomorrow, cheapest option",
            "Bengaluru to Udupi next Friday, AC, reach before 8 am"]

ex = st.selectbox("Example requests", ["(type your own)"] + EXAMPLES)
query = st.text_area("Your travel request", value=EXAMPLES[0] if ex == "(type your own)" else ex, height=80)

if st.button("Plan my journey", type="primary"):
    with st.spinner("Agents at work…"):
        st.session_state["plan"] = PlannerAgent().plan(query)
    st.session_state.pop("booking", None)

r = st.session_state.get("plan")
if r:
    req = r["request"]
    st.markdown("**Understood request** " + "".join(pill(x, "info") for x in [
        f"{(req.src or '?').title()} → {(req.dst or '?').title()}", f"{req.journey:%a %d %b %Y}" if req.journey else "date ?",
        f"budget ₹{req.budget:.0f}" if req.budget else "no budget", f"arrive by {req.arrive_by}" if req.arrive_by else "any arrival",
        f"{req.passengers} pax", f"class {','.join(req.classes)}" if req.classes else ("AC only" if req.ac_only else "any class"),
        f"objective: {req.objective}"]), unsafe_allow_html=True)
    for a in req.assumptions:
        st.caption(f"Assumption: {a}")
    box = st.success if r["outcome"] == "plan_ready" else st.warning
    box(r["message"])
    if r["anomalies"]:
        st.error("🛡️ Agent-behaviour monitor flagged: " + "; ".join(r["anomalies"]))
    gov = pill("validator: passed" if r["valid"] else "validator: FAILED", "ok" if r["valid"] else "bad") \
        + pill(f"{len(r['trace'])} tool calls", "info") + pill(f"{r['latency_ms']:.0f} ms", "info") \
        + pill(f"run {r['run_id']}", "info")
    st.markdown(gov, unsafe_allow_html=True)

    if r["best"] is not None:
        st.subheader("Recommended itinerary")
        legs = pd.DataFrame(r["best"]["legs"])
        st.dataframe(legs.rename(columns={"pred_delay_min": "predicted delay (min)", "confirm_prob": "confirm prob"}),
                     width="stretch", hide_index=True)
        bk = st.session_state.get("booking")
        if bk:
            st.success("✅ Booked after your approval: " + ", ".join(f"PNR {b['pnr']} ({b['status']}, ₹{b['amount']:.0f})" for b in bk))
        else:
            c1, c2 = st.columns([1, 3])
            if c1.button("✅ Approve & book", type="primary"):
                st.session_state["booking"] = book(r, approved_by=role, approval_token=r["approval_token"])
                st.rerun()
            c2.caption("The agent cannot book without this explicit approval — attempts without the token are "
                       "blocked and written to the audit trail.")

    if not r["options"].empty:
        st.subheader("All feasible options (ranked)")
        cols = ["type", "trains", "name", "class", "total_fare", "departure", "scheduled_arrival", "expected_arrival",
                "duration_h", "confirm_prob", "expected_delay_min", "deadline_risk", "score"]
        st.dataframe(r["options"][cols].assign(score=lambda d: d["score"].round(3)), width="stretch", hide_index=True)
    if not r["rejected"].empty:
        with st.expander(f"Options rejected by constraints ({len(r['rejected'])})"):
            rj = r["rejected"].copy()
            rj["violations"] = rj["violations"].apply("; ".join)
            st.dataframe(rj[["trains", "name", "class", "total_fare", "scheduled_arrival", "violations"]],
                         width="stretch", hide_index=True)
    with st.expander("Agent trace (audit trail of every tool call)", expanded=False):
        st.dataframe(r["trace"], width="stretch", hide_index=True)

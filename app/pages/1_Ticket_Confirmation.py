from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import ACCENT, explanation_chart, governance_footer, pill, setup, train_picker, date_picker

from railmind import data
from railmind.passenger.base import GovernanceBlocked
from railmind.passenger.confirmation import ConfirmationPredictor, historical_trends

setup("Ticket Confirmation Predictor", "🎫", "Will my Waiting-List / RAC ticket get confirmed? Probability, expected "
      "final status, confidence and the reasons behind it.")

c1, c2, c3 = st.columns(3)
with c1:
    tn = train_picker()
    t = data.train(tn)
    cls = st.selectbox("Class", t["classes"].split(","), index=min(1, len(t["classes"].split(",")) - 1))
with c2:
    jd = date_picker(offset=12)
    status = st.radio("Current status", ["WL", "RAC"], horizontal=True)
with c3:
    pos = st.number_input(f"{status} position", 1, 600, 24 if status == "WL" else 8)
    quota = st.selectbox("Quota", ["GN", "TQ", "LD"], format_func={"GN": "General", "TQ": "Tatkal", "LD": "Ladies"}.get)

if st.button("Predict confirmation", type="primary"):
    try:
        r = ConfirmationPredictor().predict(tn, cls, jd, int(pos), status, quota)
    except GovernanceBlocked as e:
        st.error(f"🛡️ Prediction withheld by the data-quality gate: {e.result.summary()}")
        st.stop()
    p = r["probability"]
    kind = "ok" if p >= 0.7 else ("warn" if p >= 0.4 else "bad")
    a, b, c, d = st.columns(4)
    a.metric("Confirmation probability", f"{p:.0%}")
    b.metric("Expected final status", r["expected_final"])
    c.metric("Model confidence", f"{r['confidence']:.0%}", help="Holdout accuracy for predictions in this probability band")
    d.metric("Expected cancellations", r["expected_cancellations"])
    st.markdown(pill(r["verdict"], kind) + (pill(f"peak season: {r['festival']}", "warn") if r["festival"] else ""),
                unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Final allocation outlook**")
        sd = r["status_distribution"]
        fig = px.bar(x=list(sd.keys()), y=list(sd.values()), labels={"x": "final status", "y": "probability"},
                     color=list(sd.keys()), color_discrete_map={"CNF": "#2F9E44", "RAC": ACCENT, "WL": "#B42318"})
        fig.update_layout(height=260, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig)
    with right:
        st.markdown("**Why this prediction** (feature contributions, percentage points)")
        st.plotly_chart(explanation_chart(r["explanation"]))
    for line in r["reasons"]:
        st.markdown(f"- {line}")
    governance_footer(r)

st.divider()
st.subheader("Historical booking & cancellation trends")
tr = historical_trends(tn, None)
c1, c2 = st.columns(2)
fig = px.bar(tr["by_lead"], x="lead_bucket", y="mean", labels={"lead_bucket": "days booked before journey",
             "mean": "confirmation rate"}, color_discrete_sequence=[ACCENT])
fig.update_layout(height=280, margin=dict(t=10, b=10))
c1.plotly_chart(fig)
fig = px.line(tr["by_position"], x="pos_bucket", y="mean", markers=True,
              labels={"pos_bucket": "initial WL position", "mean": "confirmation rate"})
fig.update_layout(height=280, margin=dict(t=10, b=10))
c2.plotly_chart(fig)
fig = px.line(tr["by_month"], x="month", y="confirmed", markers=True, labels={"confirmed": "confirmation rate"})
fig.update_layout(height=240, margin=dict(t=10, b=10))
st.plotly_chart(fig)
st.caption("Accuracy is monitored continuously against chart-preparation outcomes; the governance layer retrains "
           "this model automatically when its calibration (Brier score) degrades — see *Autonomous Ops*.")

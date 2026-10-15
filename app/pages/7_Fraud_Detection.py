from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import ACCENT, INK, explanation_chart, governance_footer, pill, setup

from railmind import data
from railmind.passenger.fraud import FraudDetector

setup("Fraud, Security & Bot Detection", "🛡️", "Hybrid risk engine (supervised model + anomaly detector + "
      "explainable rules) for bulk-booking bots, card testing, account takeover and agent misuse.",
      permission="ops.view")

svc = FraudDetector()
PRESETS = {
    "Normal passenger": dict(account_age_days=540, bookings_last_24h=1, passengers_per_booking=2, session_duration_s=260,
                             avg_click_interval_ms=1800, form_fill_time_s=45, captcha_failures=0, payment_failures=0,
                             distinct_cards_7d=1, accounts_per_device=1, ip_reputation=0.85, is_tatkal=0,
                             booking_hour=14, geo_mismatch=0, amount=1450),
    "Tatkal bulk-booking bot": dict(account_age_days=12, bookings_last_24h=24, passengers_per_booking=6,
                                    session_duration_s=40, avg_click_interval_ms=150, form_fill_time_s=2,
                                    captcha_failures=3, payment_failures=0, distinct_cards_7d=2, accounts_per_device=1,
                                    ip_reputation=0.2, is_tatkal=1, booking_hour=10, geo_mismatch=0, amount=6400),
    "Card testing": dict(account_age_days=3, bookings_last_24h=1, passengers_per_booking=1, session_duration_s=300,
                         avg_click_interval_ms=1200, form_fill_time_s=30, captcha_failures=0, payment_failures=8,
                         distinct_cards_7d=9, accounts_per_device=1, ip_reputation=0.6, is_tatkal=0, booking_hour=3,
                         geo_mismatch=0, amount=120),
    "Account takeover": dict(account_age_days=900, bookings_last_24h=1, passengers_per_booking=3, session_duration_s=45,
                             avg_click_interval_ms=1500, form_fill_time_s=25, captcha_failures=0, payment_failures=0,
                             distinct_cards_7d=1, accounts_per_device=1, ip_reputation=0.25, is_tatkal=0,
                             booking_hour=2, geo_mismatch=1, amount=9800),
}
preset = st.selectbox("Load a session profile", list(PRESETS))
v = PRESETS[preset]
cols = st.columns(5)
rec = {}
for i, (k, val) in enumerate(v.items()):
    c = cols[i % 5]
    if isinstance(val, float) or k == "ip_reputation":
        rec[k] = c.number_input(k, value=float(val), step=0.05 if k == "ip_reputation" else 10.0, key=f"{preset}{k}")
    else:
        rec[k] = c.number_input(k, value=int(val), step=1, key=f"{preset}{k}")

if st.button("Score session", type="primary"):
    r = svc.risk(rec)
    a, b, c, d = st.columns(4)
    a.metric("Risk score", f"{r['risk_score']:.0f} / 100")
    b.metric("Decision", r["decision"])
    c.metric("Model probability", f"{r['supervised_prob']:.0%}")
    d.metric("Anomaly score", f"{r['anomaly_score']:.2f}")
    kind = {"ALLOW": "ok", "REVIEW": "warn", "BLOCK": "bad"}[r["decision"]]
    st.markdown(pill(r["decision"], kind) + "".join(pill(x, "bad") for x in r["rules_fired"]), unsafe_allow_html=True)
    st.markdown("**Explainable alert — top drivers**")
    st.plotly_chart(explanation_chart(r["explanation"]))
    governance_footer(r)

st.divider()
st.subheader("Security operations view (recent sessions)")
s = data.load("sessions").tail(4000)
sc = svc.batch_scores(s)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Sessions scored", f"{len(sc):,}")
k2.metric("Blocked", int((sc["decision"] == "BLOCK").sum()))
k3.metric("Sent to review", int((sc["decision"] == "REVIEW").sum()))
tp = ((sc["decision"] != "ALLOW") & (sc["is_fraud"] == 1)).sum()
fp = ((sc["decision"] != "ALLOW") & (sc["is_fraud"] == 0)).sum()
fn = ((sc["decision"] == "ALLOW") & (sc["is_fraud"] == 1)).sum()
k4.metric("False positives / negatives", f"{fp} / {fn}")
c1, c2 = st.columns(2)
fig = px.histogram(sc, x="risk_score", color=sc["is_fraud"].map({0: "legit", 1: "fraud"}), nbins=40,
                   color_discrete_map={"legit": INK, "fraud": ACCENT}, barmode="overlay", log_y=True)
fig.update_layout(height=300, margin=dict(t=10, b=10), legend_title="")
c1.plotly_chart(fig)
by = sc[sc["is_fraud"] == 1].groupby("fraud_type")["decision"].apply(lambda d: (d != "ALLOW").mean()).reset_index()
fig = px.bar(by, x="fraud_type", y="decision", labels={"decision": "detection rate"}, color_discrete_sequence=[ACCENT])
fig.update_layout(height=300, margin=dict(t=10, b=10), yaxis_range=[0, 1])
c2.plotly_chart(fig)
st.caption("False-positive and false-negative rates are tracked continuously by the governance layer; a drop in "
           "recall triggers automatic retraining, while threshold changes require Governance Officer approval.")

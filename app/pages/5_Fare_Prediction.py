from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import ACCENT, INK, date_picker, governance_footer, pill, setup, train_picker

from railmind import data
from railmind.passenger.fare import FarePredictor

setup("Dynamic Fare Prediction", "💸", "Forecast fare movement up to the journey date, spot demand surges and "
      "find the cheapest safe day to book.")

svc = FarePredictor()
c1, c2, c3 = st.columns(3)
with c1:
    tn = train_picker(default_no="12001")
t = data.train(tn)
cls = c2.selectbox("Class", t["classes"].split(","))
with c3:
    jd = date_picker(offset=30)

if st.button("Forecast fares", type="primary"):
    r = svc.forecast(tn, cls, jd)
    a, b, c, d = st.columns(4)
    a.metric("Fare if booked today", f"₹{r['fare_now']:.0f}")
    b.metric("Expected fare near departure", f"₹{r['fare_at_journey']:.0f}",
             f"{r['fare_at_journey'] - r['fare_now']:+.0f}", delta_color="inverse")
    c.metric("Best day to book", r["best_booking_date"], f"₹{r['best_fare']:.0f}", delta_color="off")
    d.metric("Max saving vs. worst day", f"₹{r['saving_vs_waiting']:.0f}")
    st.markdown(pill(f"trend: {r['trend']}", {"rising": "warn", "stable": "ok", "falling": "ok"}[r["trend"]])
                + pill(f"base fare ₹{r['base_fare']}", "info")
                + (pill(f"Tatkal estimate ₹{r['tatkal_fare_estimate']:.0f}", "info") if r["tatkal_fare_estimate"] else "")
                + (pill(f"peak: {r['festival']}", "warn") if r["festival"] else ""), unsafe_allow_html=True)
    cur = r["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cur["date"], y=cur["fare_high"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cur["date"], y=cur["fare_low"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(232,89,12,.15)", name="80% band"))
    fig.add_trace(go.Scatter(x=cur["date"], y=cur["fare"], line=dict(color=ACCENT, width=3), name="forecast fare"))
    fig.add_trace(go.Scatter(x=cur["date"], y=cur["occupancy_pct"], yaxis="y2", line=dict(color=INK, dash="dot"),
                             name="seats sold %"))
    fig.update_layout(height=360, yaxis=dict(title="fare (₹)"), margin=dict(t=10, b=10),
                      yaxis2=dict(title="seats sold %", overlaying="y", side="right", range=[0, 105]))
    st.plotly_chart(fig)
    st.markdown("**Why the fare is what it is today:** " + "; ".join(r["reasons"]))
    governance_footer(r)

st.divider()
st.subheader("Demand surge forecast (next 45 days)")
d = svc.demand_forecast(t["src"], t["dst"])
if not d.empty:
    fig = px.bar(d, x="date", y="forecast_bookings", color="surge", color_discrete_map={True: ACCENT, False: INK},
                 hover_data=["festival"], labels={"forecast_bookings": "forecast bookings / day"})
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    surges = d[d["surge"]]
    if not surges.empty:
        st.warning(f"Demand surge expected on {len(surges)} day(s), e.g. {surges['date'].iloc[0]:%d %b}"
                   + (f" ({surges['festival'].iloc[0]})" if surges['festival'].iloc[0] else "")
                   + " — book early for these dates.")
b, _ = svc.bundle()
st.caption(f"Demand model MAPE on last 60 days: {b['extra']['demand_mape']:.1%} · fare model MAE ₹{b['metrics']['mae']:.1f}. "
           "Pricing outputs are checked against flexi-fare policy bounds by the PricingAgent.")

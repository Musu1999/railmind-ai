from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import ACCENT, date_picker, explanation_chart, governance_footer, pill, setup, train_picker

from railmind import data
from railmind.passenger.base import GovernanceBlocked
from railmind.passenger.delay import DelayPredictor, likely_weather
from railmind.reference import STATION_INDEX, STATIONS

setup("Delay & Disruption Prediction", "⏱️", "Arrival-delay forecasts with 80% prediction intervals, station "
      "congestion, proactive alerts and alternative trains.")

svc = DelayPredictor()
c1, c2, c3, c4 = st.columns(4)
with c1:
    tn = train_picker()
with c2:
    jd = date_picker(offset=0)
t = data.train(tn)
zone = STATION_INDEX[t["dst"]][5]
weathers = ["clear", "rain", "storm", "fog"]
w = c3.selectbox("Weather on route", weathers, index=weathers.index(likely_weather(jd, zone)),
                 help="Defaults to the seasonal most-likely condition; plug in an IMD feed in production.")
dep_delay = c4.number_input("Current delay at origin (min)", 0, 600, 0)
block = st.toggle("Track maintenance block on route")

if st.button("Predict delay", type="primary"):
    try:
        r = svc.predict(tn, jd, w, dep_delay, int(block))
    except GovernanceBlocked as e:
        st.error(f"🛡️ Prediction withheld by the data-quality gate: {e.result.summary()}")
        st.stop()
    a, b, c, d = st.columns(4)
    a.metric("Predicted arrival delay", f"{r['delay_min']:.0f} min")
    b.metric("80% interval", f"{r['interval_80'][0]:.0f}–{r['interval_80'][1]:.0f} min")
    c.metric("On-time probability (<15 min)", f"{r['on_time_probability']:.0%}")
    d.metric("Confidence", f"{r['confidence']:.0%}")
    st.markdown(f"Scheduled **{r['scheduled_arrival'].replace('T', ' ')}** → expected **{r['expected_arrival'].replace('T', ' ')}** at "
                f"{STATION_INDEX[t['dst']][1]}  " + pill(f"destination congestion: {r['destination_congestion']['level']}",
                                                            {"High": "bad", "Moderate": "warn", "Low": "ok"}[r['destination_congestion']['level']]),
                unsafe_allow_html=True)
    if r["alert"]:
        (st.error if r["alert"][0] == "severe" else st.warning)(f"🔔 Proactive alert: {r['alert'][1]}")
        alt = svc.alternatives(tn, jd)
        if not alt.empty:
            st.markdown("**Alternative travel options**")
            st.dataframe(alt, width="stretch", hide_index=True)
    l, rgt = st.columns(2)
    with l:
        st.markdown("**Explainability report** (contribution in minutes)")
        st.plotly_chart(explanation_chart(r["explanation"], unit="min", scale=1.0))
    with rgt:
        lo, hi = r["interval_80"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["delay"], y=[hi - lo], base=[lo], marker_color="rgba(232,89,12,.25)", name="80% interval"))
        fig.add_trace(go.Scatter(x=["delay"], y=[r["delay_min"]], mode="markers", marker=dict(size=18, color=ACCENT),
                                 name="point forecast"))
        fig.update_layout(height=300, yaxis_title="minutes", margin=dict(t=10, b=10))
        st.plotly_chart(fig)
    for line in r["reasons"]:
        st.markdown(f"- {line}")
    governance_footer(r)

st.divider()
st.subheader("Station congestion forecast")
s1, s2 = st.columns([1, 3])
station = s1.selectbox("Station", [s[0] for s in STATIONS], format_func=lambda c: f"{STATION_INDEX[c][1]} ({c})")
rows = []
for d in range(3):
    day = pd.Timestamp.today().normalize() + pd.Timedelta(days=d)
    for h in (6, 9, 12, 15, 18, 21):
        cg = svc.station_congestion(station, (day + pd.Timedelta(hours=h)).to_pydatetime())
        rows.append({"time": day + pd.Timedelta(hours=h), "footfall index": cg["footfall_index"], "level": cg["level"]})
fig = px.bar(pd.DataFrame(rows), x="time", y="footfall index", color="level",
             color_discrete_map={"High": "#B42318", "Moderate": ACCENT, "Low": "#2F9E44"})
fig.update_layout(height=300, margin=dict(t=10, b=10))
s2.plotly_chart(fig)

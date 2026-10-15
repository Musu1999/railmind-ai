from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import city_options, pill, setup

from railmind import data
from railmind.governance import registry
from railmind.passenger.recommender import TrainRecommender, fairness_audit
from railmind.reference import CLASS_INFO

setup("Smart Train Recommender", "🚆", "Personalised ranking of train + class options using budget, duration, "
      "availability, punctuality, travel history and preferences — with a reason for every pick.")

cities = city_options()
c1, c2, c3 = st.columns(3)
src = c1.selectbox("From", cities, index=cities.index("Mumbai"))
dst = c2.selectbox("To", cities, index=cities.index("Delhi"))
jd = c3.date_input("Journey date", value=pd.Timestamp.today().date() + pd.Timedelta(days=14))

with st.expander("Passenger profile & preferences", expanded=True):
    p1, p2, p3, p4 = st.columns(4)
    users = data.load("users")
    uid = p1.selectbox("Known passenger (history)", ["— new passenger —"] + users["user_id"].head(300).tolist())
    profile = users.set_index("user_id").loc[uid].to_dict() if uid != "— new passenger —" else {}
    budget_level = p2.selectbox("Budget profile", ["low", "mid", "high"],
                                index=["low", "mid", "high"].index(profile.get("budget_level", "mid")))
    pref_cls = p3.selectbox("Preferred class", list(CLASS_INFO), index=list(CLASS_INFO).index(profile.get("preferred_class", "3A")))
    pref_time = p4.selectbox("Preferred departure", ["morning", "afternoon", "night"],
                             index=["morning", "afternoon", "night"].index(profile.get("preferred_time", "night")))
    q1, q2, q3 = st.columns(3)
    budget = q1.number_input("Max fare per passenger (₹, 0 = no limit)", 0, 10000, 0, step=100)
    max_h = q2.number_input("Max journey hours (0 = no limit)", 0, 72, 0)
    punct = q3.slider("How much do you value punctuality?", 0.0, 1.0, float(profile.get("punctuality_weight", 0.6)))

if st.button("Recommend trains", type="primary"):
    user = {"user_id": uid if profile else None, "budget_level": budget_level, "preferred_class": pref_cls,
            "preferred_time": pref_time, "punctuality_weight": punct}
    res = TrainRecommender().recommend(src, dst, jd, user, budget=budget or None, max_duration_h=max_h or None)
    items = res["items"]
    if items.empty:
        st.warning(res.get("message", "No options."))
    else:
        st.caption(f"Ranked {res['n_candidates']} train-class options · model {res['version']} · "
                   f"input quality {res['dq_score']:.2f}")
        for i, r in items.iterrows():
            with st.container(border=True):
                a, b, c, d, e = st.columns([3, 1.2, 1.2, 1.2, 1.2])
                a.markdown(f"**#{i + 1} · {r['train_name']}** ({r['train_no']}) · `{r['class']}`  \n"
                           f"{r['dep_time']} → {r['arr_time']}{' (+' + str(r['day_offset']) + 'd)' if r['day_offset'] else ''}"
                           f" · {r['duration_min'] / 60:.1f} h")
                b.metric("Fare", f"₹{r['fare']:.0f}")
                c.metric("Availability", r["availability_label"])
                d.metric("Confirm chance", f"{r['confirm_chance']:.0%}")
                e.metric("On-time record", f"{r['punctuality']:.0%}")
                st.markdown(f"<span class='rm-small'>Why: {r['why']}</span>", unsafe_allow_html=True)

st.divider()
st.subheader("Recommendation governance")
b, v = registry.load("train_recommender")
m = b["metrics"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Hit@1 (top pick = chosen)", f"{m['hit@1']:.0%}", f"{(m['hit@1'] - m['baseline_hit@1']) * 100:+.0f} pts vs cheapest-first")
k2.metric("NDCG@3", f"{m['ndcg@3']:.3f}")
k3.metric("AUC", f"{m['auc']:.3f}")
k4.metric("Version", v)
fa = fairness_audit()
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Relevance parity across passenger groups** (Hit@1 by group)")
    df = pd.DataFrame([{"group": f"gender {k}", "hit@1": v} for k, v in fa["quality_by_gender"].items()]
                      + [{"group": f"region {k}", "hit@1": v} for k, v in fa["quality_by_region"].items()])
    fig = px.bar(df, x="hit@1", y="group", orientation="h", color_discrete_sequence=["#1B1F3B"])
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
with c2:
    st.markdown("**Bias checks**")
    ok_attr = not fa["protected_attributes_used"]
    st.markdown(pill("no protected attributes in features" if ok_attr else "protected attribute used!", "ok" if ok_attr else "bad"),
                unsafe_allow_html=True)
    st.markdown(f"- Quality gap by gender: **{fa['quality_gap_gender']:.3f}**")
    st.markdown(f"- Quality gap by region: **{fa['quality_gap_region']:.3f}**")
    st.markdown(f"- Premium-train exposure gap by gender: **{fa['exposure_gap_gender']:.3f}** "
                f"(do we push expensive trains to one group?)")
    st.caption("Gaps above 0.10 raise a fairness finding for human review in the governance layer.")

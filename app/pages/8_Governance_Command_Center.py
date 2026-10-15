from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import ACCENT, INK, pill, setup

from railmind.config import GOVERNANCE, SLA
from railmind.governance import agent_monitor, data_quality, drift, fairness, lifecycle, monitoring, registry
from railmind.governance.simulator import load_feedback
from railmind.governance.store import Store
from railmind.governance.supervisor import TABULAR
from railmind.security import allowed

role = setup("Governance Command Center", "📊", "Centralised monitoring of every AI model and agent: health & SLA, "
             "data quality, drift, bias, hallucinations, versions and lifecycle.", permission="governance.view")
store = Store.get()
window = st.sidebar.slider("Monitoring window (minutes)", 15, 24 * 60, 120, step=15)

tabs = st.tabs(["Model health & SLA", "Data quality", "Drift & performance", "Bias & hallucination",
                "Registry & lifecycle", "Agent oversight"])

# ---------------------------------------------------------------- health
with tabs[0]:
    h = monitoring.health_table(window)
    st.caption(f"SLA targets: p95 ≤ {SLA['p95_latency_ms']:.0f} ms · error rate ≤ {SLA['error_rate']:.0%} · "
               f"availability ≥ {SLA['availability']:.1%}")
    view = h.copy()
    view["SLA"] = np.where(view["requests"] == 0, "—", np.where(view["sla_ok"], "✅ met", "❌ breached"))
    st.dataframe(view[["model", "status", "SLA", "requests", "rpm", "p50_ms", "p95_ms", "error_rate", "blocked_rate",
                       "avg_dq", "availability", "versions"]], width="stretch", hide_index=True)
    res = monitoring.resource_utilization()
    r1, r2, r3 = st.columns(3)
    r1.metric("CPU utilisation", f"{res['cpu_pct']}%")
    r2.metric("Memory utilisation", f"{res.get('mem_pct', '—')}%")
    r3.metric("Open alerts", len(store.alerts("open")))
    m = st.selectbox("Time series for", monitoring.MODELS, key="ts_model")
    ts = monitoring.timeseries(m, hours=max(window // 60, 24))
    if ts.empty:
        st.info("No telemetry yet — generate traffic in Autonomous Ops.")
    else:
        c1, c2 = st.columns(2)
        fig = px.line(ts, x="bucket", y="p95_ms", markers=True, color_discrete_sequence=[INK])
        fig.add_hline(y=SLA["p95_latency_ms"], line_dash="dash", line_color=ACCENT, annotation_text="SLA")
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        c1.plotly_chart(fig)
        ycol = "mean_abs_error" if "mean_abs_error" in ts else "requests"
        fig = px.bar(ts, x="bucket", y=ycol, color_discrete_sequence=[ACCENT])
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        c2.plotly_chart(fig)

# ----------------------------------------------------------- data quality
with tabs[1]:
    m = st.selectbox("Model", TABULAR, key="dq_model")
    svc = lifecycle._service(m)
    b, v = registry.load(m)
    allx = monitoring.recent_inputs(m, window, status=None)
    hh = monitoring.health(m, window)
    c1, c2, c3 = st.columns(3)
    c1.metric("Requests in window", hh["requests"])
    c2.metric("Blocked before inference", f"{hh['blocked_rate']:.1%}")
    c3.metric("Average quality score", hh["avg_dq"] if hh["avg_dq"] is not None else "—")
    if allx.empty:
        st.info("No traffic in this window.")
    else:
        rep = data_quality.batch_report(allx, b["profile"], svc.hard)
        st.dataframe(rep, width="stretch", hide_index=True)
    st.markdown("**Validate a record before inference** (paste JSON)")
    sample = b["holdout"][svc.features].iloc[0].to_dict()
    txt = st.text_area("Record", json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in sample.items()}, indent=1),
                       height=200)
    if st.button("Run data-quality gate"):
        try:
            res = data_quality.validate(json.loads(txt), b["profile"], svc.hard)
            (st.success if res.passed else st.error)(f"score {res.score} · {'PASS' if res.passed else 'BLOCKED'}")
            if res.issues:
                st.dataframe(pd.DataFrame(res.issues), width="stretch", hide_index=True)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

# --------------------------------------------------------- drift & perf
with tabs[2]:
    m = st.selectbox("Model", TABULAR, key="drift_model")
    svc = lifecycle._service(m)
    b, v = registry.load(m)
    cur = monitoring.recent_inputs(m, window)
    if len(cur) < 30:
        st.info("Not enough recent traffic for drift tests (need ≥ 30 requests).")
    else:
        dd = drift.data_drift(b["profile"], b["holdout"], cur)
        st.markdown(f"**Data drift** — PSI thresholds: watch ≥ {GOVERNANCE['psi_warn']}, drift ≥ {GOVERNANCE['psi_alert']}")
        fig = px.bar(dd, x="psi", y="feature", color="status", orientation="h",
                     color_discrete_map={"stable": "#2F9E44", "watch": ACCENT, "drift": "#B42318"})
        fig.update_layout(height=40 + 26 * len(dd), margin=dict(t=10, b=10))
        st.plotly_chart(fig)
        st.dataframe(dd, width="stretch", hide_index=True)
        ref_pred = svc._raw_predict(b["pipeline"], svc.prepare(b["holdout"])[svc.features])
        pdrift = drift.prediction_drift(ref_pred, cur["_prediction"].astype(float).values)
        lab = cur.dropna(subset=["_label"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction drift (PSI)", pdrift["psi"], pdrift["status"], delta_color="off")
        base = registry.info(m, v)["metrics"]
        if len(lab) >= 30 and (svc.task == "regression" or lab["_label"].nunique() == 2):
            now = svc.score(lab["_label"].astype(float).values, lab["_prediction"].astype(float).values)
            key = "mae" if svc.task == "regression" else ("recall" if m == "fraud_detector" else "auc")
            c2.metric(f"Live {key.upper()} (labelled)", now[key], f"{now[key] - base[key]:+.3f} vs baseline",
                      delta_color="inverse" if key == "mae" else "normal")
            ph = drift.page_hinkley(np.abs(lab["_label"].astype(float).values - lab["_prediction"].astype(float).values))
            c3.metric("Concept-drift detector (Page-Hinkley)", "change detected" if ph["change"] else "no change")
        beh = cur["_segment"].dropna()
        if len(beh):
            st.markdown(f"**User-behaviour mix** (segment: `{svc.segment_col}`)")
            st.bar_chart(beh.value_counts(normalize=True))

# ----------------------------------------------------- bias & hallucination
with tabs[3]:
    st.markdown("**Fairness on recent labelled traffic**")
    conf = monitoring.recent_inputs("confirmation_predictor", window).dropna(subset=["_label"])
    if len(conf) > 50 and conf["_label"].nunique() == 2:
        grp = np.where(conf["class"].isin(["SL", "2S"]), "non-AC (SL/2S)", "AC classes")
        t = fairness.classification_by_group(conf["_label"], conf["_prediction"], pd.Series(grp))
        s = fairness.classification_summary(t)
        st.markdown("Confirmation predictor — error-rate parity between non-AC and AC passengers "
                    + pill(s.get("status", "n/a"), "ok" if s.get("status") == "pass" else "warn"), unsafe_allow_html=True)
        st.dataframe(t, width="stretch", hide_index=True)
    fr = monitoring.recent_inputs("fraud_detector", window).dropna(subset=["_label"])
    if len(fr) > 50:
        grp = np.where(fr["account_age_days"].astype(float) < 60, "new account (<60d)", "established")
        t = fairness.classification_by_group(fr["_label"], fr["_prediction"], pd.Series(grp))
        st.markdown("Fraud detector — false-positive rate for new vs. established accounts")
        st.dataframe(t, width="stretch", hide_index=True)
    dl = monitoring.recent_inputs("delay_predictor", window).dropna(subset=["_label"])
    if len(dl) > 50:
        t, s = fairness.regression_by_group(dl["_label"].astype(float), dl["_prediction"].astype(float), dl["zone"])
        st.markdown("Delay predictor — error by railway zone " + pill(s.get("status", "n/a"), "ok" if s.get("status") == "pass" else "warn"),
                    unsafe_allow_html=True)
        st.dataframe(t, width="stretch", hide_index=True)
    st.markdown("**Generative AI quality**")
    conv = store.conversations(3000)
    if conv.empty:
        st.info("No conversations yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Conversations", len(conv))
        c2.metric("Hallucinations blocked", f"{conv['hallucination'].mean():.1%}")
        c3.metric("Avg grounding", f"{conv['grounding'].mean():.2f}")
        c4.metric("Avg CSAT (1–5)", f"{conv['csat'].mean():.2f}")
        by = conv.groupby("lang").agg(conversations=("id", "size"), quality=("quality", "mean"),
                                      grounding=("grounding", "mean"), csat=("csat", "mean")).round(3).reset_index()
        st.dataframe(by, width="stretch", hide_index=True)
        st.dataframe(conv[["ts", "lang", "intent", "user_msg", "bot_msg", "grounding", "hallucination", "quality", "csat"]].head(50),
                     width="stretch", hide_index=True)

# ------------------------------------------------------ registry & lifecycle
with tabs[4]:
    allm = registry.list_models()
    m = st.selectbox("Model / agent", list(allm), key="reg_model")
    st.markdown(f"Production version: **{allm[m].get('production')}**")
    st.dataframe(lifecycle.lineage(m), width="stretch", hide_index=True)
    hist = lifecycle.history(m)
    if not hist.empty:
        st.markdown("**Promotion / rollback history**")
        st.dataframe(hist, width="stretch", hide_index=True)
    if m in TABULAR:
        can_act = allowed(role, "governance.act")
        c1, c2 = st.columns(2)
        if c1.button("Retrain on recent labelled traffic (champion/challenger)", disabled=not can_act):
            with st.spinner("Training challenger…"):
                res = lifecycle.retrain(m, load_feedback(m), reason=f"manual retrain by {role}", actor=role)
            st.json(res)
        if c2.button("Roll back to previous version", disabled=not can_act):
            st.json(lifecycle.rollback(m, reason=f"manual rollback by {role}", actor=role))
        if not can_act:
            st.caption("🔒 Retrain/rollback requires the `governance.act` permission (ML Engineer / Administrator).")

# ------------------------------------------------------------ agent oversight
with tabs[5]:
    runs = store.agent_runs(500)
    if runs.empty:
        st.info("No agent runs yet.")
    else:
        s = agent_monitor.summarize(runs)
        c = st.columns(5)
        c[0].metric("Planner runs", s["runs"])
        c[1].metric("Plans delivered", f"{s['success_rate']:.0%}")
        c[2].metric("Validator pass rate", f"{s['validity_rate']:.0%}")
        c[3].metric("Abnormal runs", f"{s['anomaly_rate']:.0%}")
        c[4].metric("p95 latency", f"{s['p95_latency_ms']:.0f} ms")
        st.dataframe(runs[["ts", "run_id", "query", "steps", "latency_ms", "outcome", "valid", "anomalies"]],
                     width="stretch", hide_index=True)
        rid = st.selectbox("Inspect trace", runs["run_id"].head(30))
        st.dataframe(pd.DataFrame(json.loads(runs[runs["run_id"] == rid]["trace"].iloc[0])), width="stretch", hide_index=True)

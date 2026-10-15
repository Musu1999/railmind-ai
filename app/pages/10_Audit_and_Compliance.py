from __future__ import annotations

import pandas as pd
import streamlit as st

from common import setup

from railmind.config import REPORT_DIR
from railmind.governance import fairness, registry
from railmind.governance.reports import model_card
from railmind.governance.store import Store, mask_pii
from railmind.security import PERMISSIONS, ROLES, allowed

role = setup("Audit & Compliance", "📜", "Tamper-evident audit trail, model cards, ethical-AI controls, access "
             "control and governance reports for regulators and internal audit.", permission="governance.view")
store = Store.get()

t1, t2, t3, t4 = st.tabs(["Audit trail", "Model cards & controls", "Access control & privacy", "Reports"])

with t1:
    ok, bad = store.verify_audit_chain()
    st.markdown(("✅ " if ok else "❌ ") + f"Hash chain {'intact' if ok else f'broken at record {bad}'} — every "
                "record stores the SHA-256 of the previous one, so any edit or deletion is detectable.")
    log = store.query("SELECT id, ts, actor, actor_type, action, resource, details, hash FROM audit_log ORDER BY id DESC LIMIT 2000")
    c1, c2, c3 = st.columns(3)
    actors = c1.multiselect("Actor type", sorted(log["actor_type"].unique()))
    acts = c2.text_input("Action contains")
    res = c3.text_input("Resource contains")
    f = log
    if actors:
        f = f[f["actor_type"].isin(actors)]
    if acts:
        f = f[f["action"].str.contains(acts, case=False)]
    if res:
        f = f[f["resource"].str.contains(res, case=False, na=False)]
    f = f.assign(hash=f["hash"].str[:12] + "…")
    st.dataframe(f, width="stretch", hide_index=True, height=420)
    if allowed(role, "audit"):
        st.download_button("Export audit trail (CSV)", log.to_csv(index=False), "railmind_audit_trail.csv")
    else:
        st.caption("🔒 Export requires the `audit` permission.")

with t2:
    models = list(registry.list_models())
    rows = []
    for m in models:
        cl = fairness.compliance_checklist(m)
        rows.append({"model": m, "version": registry.production_version(m), "risk": registry.info(m).get("card", {}).get("risk"),
                     "controls passed": f"{int(cl['pass'].sum())}/{len(cl)}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    m = st.selectbox("Model card", models)
    card = model_card(m)
    st.markdown(card)
    st.download_button("Download model card", card, file_name=f"model_card_{m}.md")

with t3:
    st.markdown("**Role-based access control matrix**")
    mat = pd.DataFrame([{**{"role": r}, **{p: ("✔" if p in perms else "") for p in PERMISSIONS}} for r, perms in ROLES.items()])
    st.dataframe(mat, width="stretch", hide_index=True)
    st.markdown("**Privacy by design — PII masking before anything is logged**")
    sample = st.text_input("Try it", "My PNR 4521367890, phone 9876543210, email asha@example.com, card 4111 1111 1111 1111")
    st.code(mask_pii(sample))
    st.caption("Masking covers phone numbers, e-mail, Aadhaar-like 12-digit IDs and card numbers. PNRs are retained "
               "as operational identifiers.")

with t4:
    reps = sorted(REPORT_DIR.glob("*.md"), reverse=True)
    if not reps:
        st.info("No governance reports yet — run a cycle in Autonomous Ops.")
    for r in reps[:15]:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"`{r.name}`")
        c2.download_button("Download", r.read_text(encoding="utf-8"), file_name=r.name, key=r.name)
    if reps:
        with st.expander("Latest report", expanded=True):
            st.markdown(reps[0].read_text(encoding="utf-8"))

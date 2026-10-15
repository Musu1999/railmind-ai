"""Shared UI helpers: page setup, theme, RBAC gate, bootstrap and pickers."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from railmind import data  # noqa: E402
from railmind.governance import registry  # noqa: E402
from railmind.reference import STATIONS  # noqa: E402
from railmind.security import PERMISSIONS, ROLES, allowed  # noqa: E402

ACCENT = "#E8590C"
INK = "#1B1F3B"
PLOT_COLORS = ["#1B1F3B", "#E8590C", "#2F9E44", "#1971C2", "#AE3EC9", "#868E96"]

CSS = """
<style>
  .block-container {padding-top: 1.6rem; max-width: 1250px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .rm-hero {padding: 1.1rem 1.3rem; border-radius: 14px; background: linear-gradient(120deg,#1B1F3B 0%,#2B3270 60%,#E8590C 140%);
            color: #fff; margin-bottom: 1rem;}
  .rm-hero h1 {color:#fff; margin:0; font-size: 1.7rem;}
  .rm-hero p {color:#E9ECF5; margin:.3rem 0 0 0;}
  .rm-pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
            margin-right:6px; border:1px solid rgba(0,0,0,.08);}
  .ok {background:#E6F4EA; color:#1E7B34;} .warn {background:#FFF4E5; color:#B35C00;}
  .bad {background:#FDECEC; color:#B42318;} .info {background:#EAF2FB; color:#1A5FB4;}
  .rm-card {border:1px solid rgba(27,31,59,.12); border-radius:12px; padding: .9rem 1rem; background: rgba(255,255,255,.6);}
  div[data-testid="stMetric"] {border:1px solid rgba(27,31,59,.10); border-radius:12px; padding:.6rem .8rem;}
  .rm-small {font-size:.85rem; color:#5C6370;}
</style>
"""


def setup(title: str, icon: str, subtitle: str = "", permission: str = "passenger") -> str:
    st.set_page_config(page_title=f"{title} · RailMind AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    role = sidebar()
    st.markdown(f"<div class='rm-hero'><h1>{icon} {title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)
    if not allowed(role, permission):
        st.error(f"🔒 Your role **{role}** does not have the `{permission}` permission "
                 f"({PERMISSIONS[permission]}). Switch role in the sidebar.")
        st.stop()
    ensure_ready()
    return role


def sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🚆 RailMind AI")
        st.caption("Intelligent railway travel + autonomous AI governance")
        role = st.selectbox("Signed in as (role)", list(ROLES), index=len(ROLES) - 1, key="role")
        st.caption("Permissions: " + ", ".join(sorted(ROLES[role])))
        st.divider()
        st.caption("All train data is synthetic, generated for demonstration.")
    return role


def ensure_ready() -> None:
    if registry.production_version("confirmation_predictor") is None:
        st.warning("Models are not trained yet. This takes about 30 seconds the first time.")
        if st.button("Generate data & train all models", type="primary"):
            import subprocess
            with st.spinner("Training…"):
                subprocess.run([sys.executable, str(ROOT / "scripts" / "train_all.py")], check=True)
            st.rerun()
        st.stop()


@st.cache_data
def city_options() -> list[str]:
    return sorted({s[2] for s in STATIONS})


@st.cache_data
def train_options() -> list[str]:
    t = data.trains()
    return [f"{r.train_no} · {r.train_name} ({r.dep_time}→{r.arr_time})" for r in t.itertuples()]


def train_picker(label: str = "Train", default_no: str = "12089", key: str = "train") -> str:
    opts = train_options()
    idx = next((i for i, o in enumerate(opts) if o.startswith(default_no)), 0)
    return st.selectbox(label, opts, index=idx, key=key).split(" · ")[0]


def date_picker(label: str = "Journey date", offset: int = 1, key: str = "date") -> date:
    return st.date_input(label, value=date.today() + timedelta(days=offset),
                         min_value=date.today(), max_value=date.today() + timedelta(days=60), key=key)


def pill(text: str, kind: str = "info") -> str:
    return f"<span class='rm-pill {kind}'>{text}</span>"


def explanation_chart(expl, unit: str = "pts", scale: float = 100.0, top: int = 8):
    import plotly.graph_objects as go
    e = expl.head(top).iloc[::-1]
    vals = e["contribution"] * scale
    fig = go.Figure(go.Bar(x=vals, y=[f"{l} = {v}" for l, v in zip(e["label"], e["value"])], orientation="h",
                           marker_color=[ACCENT if v > 0 else INK for v in vals]))
    fig.update_layout(height=40 + 32 * len(e), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=f"contribution ({unit})", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def governance_footer(result: dict) -> None:
    dq = result.get("dq_score")
    st.markdown(
        pill(f"model {result.get('version', '-')}", "info")
        + pill(f"data quality {dq:.2f}" if dq is not None else "data quality n/a", "ok" if (dq or 0) >= 0.9 else "warn")
        + pill(f"latency {result.get('latency_ms', 0):.0f} ms", "info")
        + pill(f"request {result.get('request_id', '-')}", "info"), unsafe_allow_html=True)

"""Governance reporting: per-cycle incident reports and model cards (Markdown, regulator-friendly)."""
from __future__ import annotations

import time
from pathlib import Path

from ..config import REPORT_DIR
from . import fairness, registry
from .store import Store


def governance_report(cycle: dict) -> Path:
    lines = ["# RailMind AI — Governance Cycle Report", "",
             f"*Generated {cycle['ts']} · duration {cycle['duration_s']} s*", "",
             f"**Findings:** {len(cycle['findings'])} · **Incidents:** {len(cycle['incidents'])} · "
             f"**Resolved autonomously:** {sum(i['status'] == 'resolved' for i in cycle['incidents'])}", ""]
    if cycle["findings"]:
        lines += ["## Prioritised findings", "", "| Priority | Severity | Model | Agent | Finding |", "|---|---|---|---|---|"]
        for f in cycle["findings"]:
            lines.append(f"| {f['priority']} | {f['severity']} | {f['model']} | {f['agent']} | {f['message']} |")
        lines.append("")
    for inc in cycle["incidents"]:
        r = inc["rca"]
        lines += [f"## Incident #{inc['incident_id']} — {inc['model']} ({inc['status']})", "",
                  f"**Primary root cause:** {r['primary']['cause']} (confidence {r['primary']['confidence']})", "",
                  f"Evidence: {r['primary']['evidence']}", "",
                  f"Upstream dependencies implicated: {', '.join(r['primary']['upstream']) or '—'}", "",
                  f"Blast radius: {', '.join(r['blast_radius']) or '—'}", "", "**Actions taken**", ""]
        for a in inc["actions"]:
            desc = ", ".join(f"{k}={v}" for k, v in a.items() if k != "action")
            lines.append(f"- `{a['action']}` — {desc}")
        v = inc["verification"]
        lines += ["", f"**Verification:** {'passed' if v['verified'] else 'pending — ' + '; '.join(v['remaining'])}", ""]
    ok, bad = Store.get().verify_audit_chain()
    lines += ["## Audit integrity", "", f"Hash-chained audit log verified: **{'intact' if ok else f'TAMPERED at id {bad}'}**", ""]
    path = REPORT_DIR / f"governance_cycle_{time.strftime('%Y%m%d_%H%M%S')}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def model_card(model: str) -> str:
    inf = registry.info(model)
    if not inf:
        return f"# {model}\n\nNot registered."
    card = inf.get("card", {})
    checks = fairness.compliance_checklist(model)
    lines = [f"# Model card — {model} ({inf['version']})", "",
             f"**Description:** {card.get('description', '-')}", "",
             f"**Owner:** {card.get('owner', '-')} · **Risk tier:** {card.get('risk', '-')} · "
             f"**Stage:** {inf['stage']} · **Created:** {inf['created']}", "",
             f"**Reason for this version:** {inf.get('reason')}", "",
             "## Metrics (holdout)", ""] + [f"- {k}: {v}" for k, v in inf.get("metrics", {}).items()] + [
             "", "## Inputs", "", ", ".join(card.get("features", [])) or "n/a", "",
             "## Limitations", "", card.get("limitations", "Configuration-based component; see platform docs."), "",
             "## Ethical-AI controls", ""] + [f"- [{'x' if p else ' '}] {c}" for c, p in checks.values]
    return "\n".join(lines)

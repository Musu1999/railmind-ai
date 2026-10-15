"""Hallucination & response-quality guard for Generative AI outputs.

Every assistant/agent answer is checked against the evidence it was *allowed* to use
(retrieved knowledge-base passages + tool results):

1. Grounding  — each sentence must be supported by some evidence passage (lexical
   + character n-gram similarity, robust across English/Hindi/Bengali).
2. Numeric fidelity — every number/amount/time in the answer must appear in evidence.
3. Entity fidelity — train numbers and PNRs must come from tool results.
4. Policy & safety — no requests for OTP/password, no guarantees of confirmation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import GOVERNANCE

NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)(?![\w])")
TRAIN_RE = re.compile(r"\b(1\d{4}|2\d{4})\b")
PNR_RE = re.compile(r"\b\d{10}\b")
UNSAFE = [
    (re.compile(r"\b(share|send|tell|give)\b.{0,30}\b(otp|password|pin|cvv)\b", re.I), "asks for credentials"),
    (re.compile(r"\bguarantee[ds]?\b.{0,40}\b(confirm|berth|seat)", re.I), "guarantees confirmation"),
    (re.compile(r"\b100\s?%\s*(sure|certain|confirm)", re.I), "overconfident claim"),
]
SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


@dataclass
class GuardResult:
    grounding: float
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_entities: list[str] = field(default_factory=list)
    policy_violations: list[str] = field(default_factory=list)
    weak_sentences: list[str] = field(default_factory=list)

    @property
    def hallucination(self) -> bool:
        return (self.grounding < GOVERNANCE["grounding_min"] or bool(self.unsupported_numbers)
                or bool(self.unsupported_entities) or bool(self.policy_violations))

    @property
    def risk(self) -> float:
        r = max(0.0, GOVERNANCE["grounding_min"] + 0.25 - self.grounding)
        r += 0.25 * len(self.unsupported_numbers) + 0.3 * len(self.unsupported_entities)
        r += 0.5 * len(self.policy_violations)
        return round(min(r, 1.0), 3)

    def reasons(self) -> list[str]:
        out = []
        if self.grounding < GOVERNANCE["grounding_min"]:
            out.append(f"low grounding score {self.grounding:.2f}")
        if self.unsupported_numbers:
            out.append("numbers not found in sources: " + ", ".join(self.unsupported_numbers[:5]))
        if self.unsupported_entities:
            out.append("unknown trains/PNRs: " + ", ".join(self.unsupported_entities[:5]))
        out.extend(self.policy_violations)
        return out


def _norm_num(s: str) -> str:
    s = s.replace(",", "")
    return s[:-2] if s.endswith(".0") else s


def check(answer: str, evidence: list[str], allow_numbers: set[str] | None = None) -> GuardResult:
    evidence = [e for e in evidence if e and e.strip()]
    sents = [s.strip() for s in SENT_SPLIT.split(answer) if len(s.strip()) > 12]
    if not sents:
        return GuardResult(grounding=1.0)
    if not evidence:
        return GuardResult(grounding=0.0, weak_sentences=sents)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    mat = vec.fit_transform(evidence + sents)
    ev, se = mat[: len(evidence)], mat[len(evidence):]
    sims = cosine_similarity(se, ev).max(axis=1)
    weak = [s for s, v in zip(sents, sims) if v < GOVERNANCE["grounding_min"]]
    grounding = float(sims.mean())

    ev_text = " ".join(evidence)
    ev_nums = {_norm_num(n) for n in NUM_RE.findall(ev_text)} | (allow_numbers or set())
    nums = [_norm_num(n) for n in NUM_RE.findall(answer)]
    unsupported = sorted({n for n in nums if n not in ev_nums and len(n) > 1 and n not in {"24", "139"}})
    ents = set(TRAIN_RE.findall(answer)) | set(PNR_RE.findall(answer))
    bad_ents = sorted(e for e in ents if e not in ev_text)
    unsupported = [n for n in unsupported if n not in bad_ents]
    policy = [msg for pat, msg in UNSAFE if pat.search(answer)]
    return GuardResult(round(grounding, 3), unsupported, bad_ents, policy, weak)


def response_quality(query: str, answer: str, lang_expected: str, lang_answer: str, latency_ms: float,
                     guard: GuardResult, resolved: bool) -> float:
    """Composite 0-1 quality: relevance, grounding, language match, brevity, latency, resolution."""
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit([query, answer])
        m = vec.transform([query, answer])
        relevance = float(cosine_similarity(m[0], m[1])[0, 0])
    except ValueError:
        relevance = 0.0
    length_ok = 1.0 if 20 <= len(answer) <= 900 else 0.6
    lang_ok = 1.0 if lang_expected == lang_answer else 0.5
    speed = 1.0 if latency_ms < 1500 else 0.7
    q = (0.20 * min(relevance * 2.5, 1) + 0.35 * min(guard.grounding * 1.6, 1) + 0.15 * lang_ok + 0.1 * length_ok
         + 0.1 * speed + 0.1 * (1.0 if resolved else 0.4))
    if guard.hallucination:
        q *= 0.5
    return round(q, 3)


def predicted_csat(quality: float, resolved: bool, escalated: bool) -> float:
    """Proxy CSAT on a 1-5 scale until explicit feedback arrives."""
    base = 1.5 + 3.3 * quality
    if not resolved:
        base -= 0.6
    if escalated:
        base -= 0.3
    return round(max(1.0, min(5.0, base)), 2)

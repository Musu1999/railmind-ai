"""6. AI-Powered Customer Support Assistant — multilingual (English / हिन्दी / বাংলা), 24×7.

Retrieval-augmented and tool-augmented: answers come from the policy knowledge base or
live tools (PNR, refund, delay, planner). An optional LLM may rephrase the grounded draft,
and every outgoing message passes the hallucination guard before it is shown.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
import uuid
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .. import data, faults, inventory, llm
from ..governance import hallucination as hg
from ..governance import registry
from ..governance.store import Store

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "kb.json"
NAME = "support_assistant"

T = {
    "en": {"greet": "Namaste! I'm RailMind, your 24×7 rail assistant. I can check PNR status, track refunds, explain cancellation rules, predict delays and plan journeys. How can I help?",
           "pnr": "PNR {pnr}: {train_no} {train_name}, {src} → {dst} on {journey_date}, class {cls}, {passengers} passenger(s). Current status: {status_detail}.",
           "pnr_missing": "I couldn't find PNR {pnr}. Please check the 10-digit number on your ticket.",
           "ask_pnr": "Please share your 10-digit PNR number and I'll check it for you.",
           "refund": "Refund for PNR {pnr}: {refund_status}. Booking amount was Rs {amount}.",
           "refund_none": "PNR {pnr} is not cancelled ({status_detail}), so no refund is due yet.",
           "delay": "Train {train_no} ({train_name}) on {d}: predicted arrival delay about {delay} minutes (likely range {lo}–{hi} min); expected arrival {eta}.",
           "plan": "{msg}",
           "fallback": "I'm not fully sure about that. Here is the closest policy I found: {text}",
           "escalate": "I've passed this to a human agent who will follow up. You can also call 139.",
           "source": "Source", "out_of_scope": "I can help with train travel — tickets, PNR, refunds, cancellations, delays and journey planning. Could you rephrase your question about your rail journey?"},
    "hi": {"greet": "नमस्ते! मैं RailMind हूँ, आपका 24×7 रेल सहायक। मैं PNR स्थिति, रिफंड, रद्दीकरण नियम, देरी का अनुमान और यात्रा योजना में मदद कर सकता हूँ।",
           "pnr": "PNR {pnr}: ट्रेन {train_no} {train_name}, {src} → {dst}, यात्रा तिथि {journey_date}, श्रेणी {cls}, {passengers} यात्री। वर्तमान स्थिति: {status_detail}।",
           "pnr_missing": "PNR {pnr} नहीं मिला। कृपया टिकट पर 10 अंकों का नंबर जाँचें।",
           "ask_pnr": "कृपया अपना 10 अंकों का PNR नंबर बताएं।",
           "refund": "PNR {pnr} का रिफंड: {refund_status}। बुकिंग राशि Rs {amount} थी।",
           "refund_none": "PNR {pnr} रद्द नहीं है ({status_detail}), इसलिए अभी कोई रिफंड देय नहीं है।",
           "delay": "ट्रेन {train_no} ({train_name}) {d} को: अनुमानित देरी लगभग {delay} मिनट (संभावित {lo}–{hi} मिनट); अनुमानित आगमन {eta}।",
           "plan": "{msg}",
           "fallback": "मुझे इसकी पूरी जानकारी नहीं है। सबसे नज़दीकी नियम: {text}",
           "escalate": "मैंने यह मानव एजेंट को भेज दिया है। आप 139 पर भी कॉल कर सकते हैं।",
           "source": "स्रोत", "out_of_scope": "मैं रेल यात्रा से जुड़े सवालों में मदद कर सकता हूँ — टिकट, PNR, रिफंड, रद्दीकरण, देरी और यात्रा योजना। कृपया अपना प्रश्न दोबारा पूछें।"},
    "bn": {"greet": "নমস্কার! আমি RailMind, আপনার 24×7 রেল সহায়ক। PNR স্ট্যাটাস, রিফান্ড, বাতিলের নিয়ম, দেরির পূর্বাভাস ও যাত্রা পরিকল্পনায় সাহায্য করতে পারি।",
           "pnr": "PNR {pnr}: ট্রেন {train_no} {train_name}, {src} → {dst}, যাত্রার তারিখ {journey_date}, শ্রেণি {cls}, {passengers} জন যাত্রী। বর্তমান অবস্থা: {status_detail}।",
           "pnr_missing": "PNR {pnr} পাওয়া যায়নি। টিকিটের 10 সংখ্যার নম্বরটি দেখুন।",
           "ask_pnr": "অনুগ্রহ করে আপনার 10 সংখ্যার PNR নম্বর দিন।",
           "refund": "PNR {pnr}-এর রিফান্ড: {refund_status}। বুকিংয়ের পরিমাণ ছিল Rs {amount}।",
           "refund_none": "PNR {pnr} বাতিল হয়নি ({status_detail}), তাই এখন কোনো রিফান্ড প্রাপ্য নয়।",
           "delay": "ট্রেন {train_no} ({train_name}) {d}: আনুমানিক দেরি প্রায় {delay} মিনিট (সম্ভাব্য {lo}–{hi} মিনিট); আনুমানিক আগমন {eta}।",
           "plan": "{msg}",
           "fallback": "এ বিষয়ে আমি নিশ্চিত নই। সবচেয়ে কাছের নিয়ম: {text}",
           "escalate": "আমি এটি একজন মানব এজেন্টকে পাঠিয়েছি। আপনি 139-এ ফোনও করতে পারেন।",
           "source": "সূত্র", "out_of_scope": "আমি রেল যাত্রা সংক্রান্ত প্রশ্নে সাহায্য করতে পারি — টিকিট, PNR, রিফান্ড, বাতিল, দেরি ও যাত্রা পরিকল্পনা। অনুগ্রহ করে প্রশ্নটি আবার করুন।"},
}

HINGLISH = re.compile(r"\b(kya|mera|meri|kab|kaise|kitna|hai|nahi|chahiye|batao|kripya|jana|paisa|wapas)\b", re.I)
INTENTS = [
    ("greeting", re.compile(r"^\s*(hi|hello|hey|namaste|namaskar|नमस्ते|নমস্কার)\b", re.I)),
    ("refund_status", re.compile(r"(refund|रिफंड|রিফান্ড|paisa wapas).*\d{10}|\d{10}.*(refund|रिफंड|রিফান্ড)", re.I)),
    ("pnr_status", re.compile(r"\b\d{10}\b|\bpnr\b|पीएनआर", re.I)),
    ("delay", re.compile(r"(running|late|delay|delayed|देरी|लेट|দেরি).*\b[12]\d{4}\b|\b[12]\d{4}\b.*(running|late|delay|देरी|लेट|দেরি)", re.I)),
    ("journey_planning", re.compile(r"\b(from|se)\b.+\b(to|tak)\b|\bplan\b|\btrain[s]? (to|for)\b|recommend.*train|best train", re.I)),
]


def detect_language(text: str) -> str:
    if re.search(r"[ঀ-৿]", text):
        return "bn"
    if re.search(r"[ऀ-ॿ]", text) or len(HINGLISH.findall(text)) >= 2:
        return "hi"
    return "en"


class SupportAssistant:
    def __init__(self):
        self.kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
        corpus = [" ".join([d["title"], " ".join(d["tags"]), d["text"], d.get("text_hi", ""), d.get("text_bn", "")])
                  for d in self.kb]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.mat = self.vec.fit_transform(corpus)
        self.word_vec = TfidfVectorizer(stop_words="english").fit(corpus)
        self.word_mat = self.word_vec.transform(corpus)
        self.kb_hash = hashlib.sha256(KB_PATH.read_bytes()).hexdigest()[:12]

    # ------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: int = 2) -> list[tuple[dict, float]]:
        s1 = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        s2 = cosine_similarity(self.word_vec.transform([query]), self.word_mat)[0]
        s = 0.6 * s1 + 0.4 * s2
        idx = np.argsort(-s)[:k]
        return [(self.kb[i], float(s[i])) for i in idx]

    @staticmethod
    def _doc_text(doc: dict, lang: str) -> str:
        return doc.get(f"text_{lang}") or doc["text"]

    def _extract(self, doc_text: str, query: str, max_sent: int = 3) -> str:
        sents = [s for s in re.split(r"(?<=[.!?।])\s+", doc_text) if s.strip()]
        if len(sents) <= max_sent:
            return doc_text
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(sents + [query])
        sc = cosine_similarity(v.transform([query]), v.transform(sents))[0]
        keep = sorted(np.argsort(-sc)[:max_sent])
        return " ".join(sents[i] for i in keep)

    # ---------------------------------------------------------------- intents
    def intent(self, text: str) -> str:
        for name, pat in INTENTS:
            if pat.search(text):
                return name
        return "policy_question"

    def _tools(self, intent: str, text: str, lang: str) -> tuple[str, list[str], bool]:
        """Returns (draft answer, evidence strings, resolved)."""
        L = T[lang]
        pnr_m = re.search(r"\b(\d{10})\b", text)
        if intent == "greeting":
            return L["greet"], [L["greet"]], True
        if intent in ("pnr_status", "refund_status"):
            if not pnr_m:
                return L["ask_pnr"], [L["ask_pnr"]], False
            b = inventory.get_booking(pnr_m.group(1))
            if not b:
                msg = L["pnr_missing"].format(pnr=pnr_m.group(1))
                return msg, [msg], False
            ctx = {**b, "cls": b.get("class"), "status_detail": b.get("status_detail", b.get("status")),
                   "train_name": b.get("train_name", ""), "src": b.get("src", ""), "dst": b.get("dst", "")}
            if intent == "refund_status":
                key = "refund" if b.get("status") == "CAN" else "refund_none"
                ctx["amount"] = int(float(ctx.get("amount", 0)))
                msg = L[key].format(**ctx)
            else:
                msg = L["pnr"].format(**ctx)
            return msg, [json.dumps(b, default=str), msg], True
        if intent == "delay":
            tn = re.search(r"\b([12]\d{4})\b", text).group(1)
            try:
                from .delay import DelayPredictor
                p = DelayPredictor().predict(tn, date.today(), log=True, explain=False)
                t = data.train(tn)
                msg = L["delay"].format(train_no=tn, train_name=t["train_name"], d=date.today().isoformat(),
                                        delay=int(p["delay_min"]), lo=int(p["interval_80"][0]),
                                        hi=int(p["interval_80"][1]), eta=p["expected_arrival"].replace("T", " "))
                return msg, [msg, json.dumps(p["record"])], True
            except KeyError:
                msg = L["pnr_missing"].format(pnr=tn)
                return msg, [msg], False
        if intent == "journey_planning":
            from .planner import PlannerAgent
            r = PlannerAgent().plan(text)
            if r["outcome"] in ("plan_ready", "infeasible", "no_trains"):
                ev = [r["message"]] + [json.dumps(l, default=str) for l in (r["best"] or {}).get("legs", [])]
                return r["message"], ev, r["outcome"] == "plan_ready"
        # knowledge-base answer
        hits = self.retrieve(text)
        doc, score = hits[0]
        body = self._extract(self._doc_text(doc, lang), text)
        rail_words = re.search(r"train|rail|ticket|pnr|berth|seat|refund|cancel|book|tatkal|station|coach|fare|"
                               r"journey|travel|luggage|food|complaint|otp|chart|waiting|rac|ट्रेन|टिकट|रेल|यात्रा|"
                               r"রেল|টিকিট|ট্রেন|যাত্রা|bogie|safar|तत्काल|बुकिंग|रिफंड|किराया|स्टेशन|सीट|बर्थ|वेटिंग|"
                               r"তৎকাল|বুকিং|রিফান্ড|ভাড়া|ওয়েটিং|স্টেশন", text, re.I)
        oos_cut = 0.25 if lang == "en" else 0.10
        if not rail_words and score < oos_cut:
            return L["out_of_scope"], [L["out_of_scope"]], False
        if score < 0.08:
            msg = L["fallback"].format(text=body)
            return msg, [self._doc_text(doc, lang), doc["text"], msg], False
        return f"{body}\n\n{L['source']}: {doc['title']}", [self._doc_text(doc, lang), doc["text"], doc["title"]], True

    # ------------------------------------------------------------------- main
    def answer(self, text: str, session: str | None = None, channel: str = "web") -> dict:
        t0 = time.perf_counter()
        session = session or uuid.uuid4().hex[:8]
        lang = detect_language(text)
        intent = self.intent(text)
        draft, evidence, resolved = self._tools(intent, text, lang)
        final, used_llm = draft, False
        polished = llm.generate(
            "You are a railway customer-support assistant. Rewrite the DRAFT answer to be clear and friendly in "
            f"the language code '{lang}'. Do not add any facts, numbers, or promises that are not in the DRAFT.",
            f"User question: {text}\n\nDRAFT:\n{draft}")
        if polished:
            final, used_llm = polished, True
        if faults.active("llm_hallucination"):
            final = self._hallucinate(final)
        guard = hg.check(final, evidence)
        escalated = False
        if guard.hallucination:
            # never show an ungrounded answer: fall back to the verified draft (or escalate)
            g2 = hg.check(draft, evidence)
            final = draft if not g2.hallucination else T[lang]["escalate"]
            escalated = g2.hallucination
        latency = (time.perf_counter() - t0) * 1000
        lang_out = detect_language(final)
        quality = hg.response_quality(text, final, lang, lang_out, latency, guard, resolved)
        csat = hg.predicted_csat(quality, resolved, escalated)
        store = Store.get()
        cid = store.log_conversation(session, channel, lang, intent, text, final, guard.grounding,
                                     guard.hallucination, quality, csat, [e[:80] for e in evidence[:3]])
        store.audit(NAME, "conversation.turn", session, {"conversation_id": cid, "intent": intent, "lang": lang,
                                                          "hallucination_blocked": guard.hallucination,
                                                          "reasons": guard.reasons()}, "agent")
        version = registry.production_version(NAME) or "v1"
        store.log_inference(NAME, version, latency, "ok", 1.0, {"intent": intent, "lang": lang},
                            {"grounding": guard.grounding, "hallucination": guard.hallucination},
                            prediction=quality, segment=lang)
        return {"answer": final, "lang": lang, "intent": intent, "grounding": guard.grounding,
                "hallucination_blocked": guard.hallucination, "guard_reasons": guard.reasons(),
                "quality": quality, "csat_pred": csat, "escalated": escalated, "used_llm": used_llm,
                "latency_ms": round(latency, 1), "conversation_id": cid, "session": session}

    @staticmethod
    def _hallucinate(text: str) -> str:
        """Simulated faulty generator (fault injection) — invents numbers and guarantees."""
        rng = random.Random(len(text))
        out = re.sub(r"\b(\d{2,4})\b", lambda m: str(int(m.group(1)) + rng.randint(7, 90)), text, count=2)
        return out + " Your berth is guaranteed to confirm, and you will get a 100% refund within 2 hours."


def record_feedback(conversation_id: int, rating: int) -> None:
    Store.get().execute("UPDATE conversations SET csat=? WHERE id=?", (float(rating), conversation_id))
    Store.get().audit("passenger", "conversation.feedback", NAME, {"conversation_id": conversation_id,
                                                                 "rating": rating}, "user")


def register_assistant() -> str:
    a = SupportAssistant()
    cfg = {"type": "RAG + tools", "kb_hash": a.kb_hash, "kb_docs": len(a.kb), "retriever": "tfidf char(3-5)+word",
           "languages": ["en", "hi", "bn"], "llm_provider": llm.provider() or "none (grounded templates)",
           "guard": "grounding+numeric+entity+policy"}
    current = registry.info(NAME)
    if current and current.get("params", {}).get("kb_hash") == a.kb_hash:
        return current["version"]
    return registry.register(NAME, cfg, {"grounding_min": 0.35}, params=cfg, stage="production",
                             reason="knowledge base update" if current else "initial assistant configuration",
                             card={"description": "Multilingual RAG support assistant with hallucination guard",
                                   "risk": "medium", "owner": "Customer Experience AI"})

"""Optional LLM connector. The platform works fully offline; if an API key is configured,
the support assistant uses the LLM only to *rephrase* grounded answers (facts still come
from the knowledge base / tools, and the hallucination guard checks every output).

Configure with environment variables (or a .env file):
    RAILMIND_LLM=gemini | openai | anthropic | ollama
    GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
    RAILMIND_LLM_MODEL   (optional model override)
    OLLAMA_HOST          (default http://localhost:11434)
"""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_MODELS = {"gemini": "gemini-2.0-flash", "openai": "gpt-4o-mini",
                  "anthropic": "claude-haiku-4-5", "ollama": "llama3.1"}


def provider() -> str | None:
    p = os.getenv("RAILMIND_LLM", "").strip().lower()
    return p if p in DEFAULT_MODELS else None


def _post(url: str, payload: dict, headers: dict, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
                                                                                   **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed provider URLs
        return json.loads(r.read().decode())


def generate(system: str, prompt: str, max_tokens: int = 400) -> str | None:
    p = provider()
    if not p:
        return None
    model = os.getenv("RAILMIND_LLM_MODEL", DEFAULT_MODELS[p])
    try:
        if p == "gemini":
            key = os.environ["GEMINI_API_KEY"]
            r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                      {"system_instruction": {"parts": [{"text": system}]},
                       "contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}}, {})
            return r["candidates"][0]["content"]["parts"][0]["text"].strip()
        if p == "openai":
            r = _post("https://api.openai.com/v1/chat/completions",
                      {"model": model, "max_tokens": max_tokens, "temperature": 0.2,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
                      {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
            return r["choices"][0]["message"]["content"].strip()
        if p == "anthropic":
            r = _post("https://api.anthropic.com/v1/messages",
                      {"model": model, "max_tokens": max_tokens, "system": system,
                       "messages": [{"role": "user", "content": prompt}]},
                      {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
            return "".join(b.get("text", "") for b in r["content"]).strip()
        if p == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            r = _post(f"{host}/api/chat", {"model": model, "stream": False,
                                           "messages": [{"role": "system", "content": system},
                                                        {"role": "user", "content": prompt}]}, {}, timeout=60)
            return r["message"]["content"].strip()
    except Exception:  # noqa: BLE001 — any failure falls back to the grounded template answer
        return None
    return None

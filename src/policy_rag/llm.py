"""Model access: Ollama's local HTTP API, plus a deterministic fake backend for tests and offline dry runs.

Each chat call leaves its token counts and Ollama's timings in `last_usage`, so the caller can put them on the trace
without changing what `chat` returns.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import requests

from .config import Config

NS_PER_MS = 1e6


class Ollama:
    def __init__(self, cfg: Config):
        self.url = cfg["ollama_url"].rstrip("/")
        self.embed_model, self.temperature = cfg["embed_model"], cfg["temperature"]
        self.last_usage: dict = {}

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        # nomic-embed-text expects task prefixes ("raw" = none, for the ablation); other models ignore them
        prefix = {"query": "search_query: ", "document": "search_document: "}.get(kind, "")
        out = []
        for i in range(0, len(texts), 32):
            r = requests.post(f"{self.url}/api/embed", timeout=600,
                              json={"model": self.embed_model, "input": [prefix + t for t in texts[i:i + 32]]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def chat(self, model: str, system: str, user: str, schema: dict | None = None) -> str:
        body = {"model": model, "stream": False, "options": {"temperature": self.temperature, "num_ctx": 8192},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema:
            body["format"] = schema                          # Ollama structured outputs (JSON schema)
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=900)
        r.raise_for_status()
        j = r.json()
        self.last_usage = usage(j)
        return j["message"]["content"]


def usage(reply: dict) -> dict:
    """Token counts and timings from an Ollama /api/chat reply (durations arrive in nanoseconds)."""
    pt, ct = int(reply.get("prompt_eval_count") or 0), int(reply.get("eval_count") or 0)
    ms = {k: float(reply.get(f"{k}_duration") or 0) / NS_PER_MS for k in ("total", "load", "prompt_eval", "eval")}
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
            "ollama_total_ms": ms["total"], "load_ms": ms["load"], "prompt_eval_ms": ms["prompt_eval"],
            "eval_ms": ms["eval"], "tokens_per_s": ct / (ms["eval"] / 1000) if ms["eval"] else 0.0}


class Fake:
    """Hash-based embeddings and a canned chat reply; lets the pipeline and tests run without a model server."""

    dim = 256

    def __init__(self):
        self.last_usage: dict = {}

    def embed(self, texts, kind):
        v = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def chat(self, model, system, user, schema=None):
        if schema:
            reply = json.dumps({"correct": 1, "faithful": 1, "reason": "fake judge"})
        else:
            m = re.search(r"\[(\d+)\]", user)
            reply = f"See the cited policy [{m.group(1) if m else 1}]."
        # word counts stand in for tokens so the tracing path runs end to end in tests
        self.last_usage = usage({"prompt_eval_count": len((system + " " + user).split()),
                                 "eval_count": len(reply.split())})
        return reply


def client(cfg: Config, fake: bool = False):
    return Fake() if fake else Ollama(cfg)

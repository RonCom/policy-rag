"""Model access: Ollama's local HTTP API, plus a deterministic fake backend for tests and offline dry runs."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import requests

from .config import Config


class Ollama:
    def __init__(self, cfg: Config):
        self.url = cfg["ollama_url"].rstrip("/")
        self.embed_model, self.temperature = cfg["embed_model"], cfg["temperature"]

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        # nomic-embed-text expects task prefixes; other models ignore them harmlessly
        prefix = "search_query: " if kind == "query" else "search_document: "
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
        return r.json()["message"]["content"]


class Fake:
    """Hash-based embeddings and a canned chat reply; lets the pipeline and tests run without a model server."""

    dim = 256

    def embed(self, texts, kind):
        v = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def chat(self, model, system, user, schema=None):
        if schema:
            return json.dumps({"correct": 1, "faithful": 1, "reason": "fake judge"})
        m = re.search(r"\[(\d+)\]", user)
        return f"See the cited policy [{m.group(1) if m else 1}]."


def client(cfg: Config, fake: bool = False):
    return Fake() if fake else Ollama(cfg)

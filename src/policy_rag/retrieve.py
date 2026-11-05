"""Four retrievers over the same chunks: BM25 (lexical), dense (embeddings), hybrid (reciprocal rank fusion), and
rerank (hybrid candidates re-scored by a cross-encoder).

Billing policy is full of exact tokens (CPT codes, "modifier 59", "8-minute", section numbers) that lexical search
matches well and embeddings can blur; paraphrased questions go the other way. Hybrid fuses the two rankings.
BM25 and the embedding model score the query and each chunk separately. A cross-encoder reads the query and one chunk
together, so it can tell "stops improving" from "stops paying"; it's too slow for 851 chunks per query, so it re-scores
only hybrid's top candidates.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import duckdb
import numpy as np
import pandas as pd

from .config import Config

_CROSS_ENCODERS: dict = {}                                   # one load per model per process (the sweep builds many Index)


class CrossEncoderReranker:
    def __init__(self, model: str, max_length: int = 512):
        if model not in _CROSS_ENCODERS:
            from sentence_transformers import CrossEncoder  # imported here: torch loads only when rerank is used
            _CROSS_ENCODERS[model] = CrossEncoder(model, max_length=max_length)
        self.model = _CROSS_ENCODERS[model]

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False),
                          dtype=float)


class FakeReranker:
    """Share of query tokens found in the chunk; stands in for the cross-encoder in tests and --fake runs."""

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) / (len(q) or 1) for t in texts], dtype=float)

TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_STOP_WORDS = ("a an and are as at be by for from has have if in into is it its of on or that the their this to was "
               "were which with not no may can will shall should must")
STOP = set(_STOP_WORDS.split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.len = np.array([len(t) for t in self.toks], float)
        self.avg = self.len.mean()
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.toks))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return s


class Index:
    def __init__(self, cfg: Config, llm, db=None, reranker=None):
        con = duckdb.connect(str(db or cfg.db_path), read_only=True)
        self.chunks = con.execute("SELECT * FROM chunks ORDER BY chunk_id").df()
        emb = con.execute("SELECT embedding FROM chunks ORDER BY chunk_id").fetchnumpy()["embedding"]
        con.close()
        self.emb = np.stack([np.asarray(e, np.float32) for e in emb])
        self.docs = (self.chunks["heading"] + " " + self.chunks["text"]).tolist()
        self.bm25 = BM25(self.docs)
        self.llm = llm
        rr = cfg.get("rerank") or {}
        self.rerank_model = rr.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rerank_candidates = int(rr.get("candidates", 20))
        self._reranker = reranker
        self._fake = type(llm).__name__ == "Fake"

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = FakeReranker() if self._fake else CrossEncoderReranker(self.rerank_model)
        return self._reranker

    def rank(self, query: str, method: str, k: int) -> pd.DataFrame:
        if method == "bm25":
            s = self.bm25.scores(query)
        elif method == "dense":
            s = self.emb @ self.llm.embed([query], "query")[0]
        elif method == "hybrid":                           # reciprocal rank fusion (k = 60)
            s = self.rrf(query)
        elif method == "rerank":                           # cross-encoder over hybrid's top candidates
            cand = np.argsort(-self.rrf(query), kind="stable")[:max(self.rerank_candidates, k)]
            s = np.full(len(self.chunks), -np.inf)
            s[cand] = self.reranker.scores(query, [self.docs[i] for i in cand])
        else:
            raise ValueError(method)
        top = np.argsort(-s, kind="stable")[:k]
        return self.chunks.iloc[top].assign(score=s[top], rank=np.arange(1, len(top) + 1))

    def rrf(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.chunks))
        for m in ("bm25", "dense"):
            order = np.argsort(-self.rank_scores(query, m), kind="stable")
            s[order] += 1.0 / (60 + np.arange(1, len(order) + 1))
        return s

    def rank_scores(self, query: str, method: str) -> np.ndarray:
        if method == "bm25":
            return self.bm25.scores(query)
        return self.emb @ self.llm.embed([query], "query")[0]

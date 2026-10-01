"""Three retrievers over the same chunks: BM25 (lexical), dense (embeddings), and hybrid (reciprocal rank fusion).

Billing policy is full of exact tokens (CPT codes, "modifier 59", "8-minute", section numbers) that lexical search
matches well and embeddings can blur; paraphrased questions go the other way. Hybrid fuses the two rankings.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import duckdb
import numpy as np
import pandas as pd

from .config import Config

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
    def __init__(self, cfg: Config, llm):
        con = duckdb.connect(str(cfg.db_path), read_only=True)
        self.chunks = con.execute("SELECT * FROM chunks ORDER BY chunk_id").df()
        emb = con.execute("SELECT embedding FROM chunks ORDER BY chunk_id").fetchnumpy()["embedding"]
        con.close()
        self.emb = np.stack([np.asarray(e, np.float32) for e in emb])
        self.bm25 = BM25((self.chunks["heading"] + " " + self.chunks["text"]).tolist())
        self.llm = llm

    def rank(self, query: str, method: str, k: int) -> pd.DataFrame:
        if method == "bm25":
            s = self.bm25.scores(query)
        elif method == "dense":
            s = self.emb @ self.llm.embed([query], "query")[0]
        elif method == "hybrid":                           # reciprocal rank fusion (k = 60)
            s = np.zeros(len(self.chunks))
            for m in ("bm25", "dense"):
                order = np.argsort(-self.rank_scores(query, m), kind="stable")
                s[order] += 1.0 / (60 + np.arange(1, len(order) + 1))
        else:
            raise ValueError(method)
        top = np.argsort(-s, kind="stable")[:k]
        return self.chunks.iloc[top].assign(score=s[top], rank=np.arange(1, len(top) + 1))

    def rank_scores(self, query: str, method: str) -> np.ndarray:
        if method == "bm25":
            return self.bm25.scores(query)
        return self.emb @ self.llm.embed([query], "query")[0]

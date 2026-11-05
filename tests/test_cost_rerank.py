import duckdb
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracing.constant import SpanAttributeKey

from policy_rag import answer, evaluate
from policy_rag.config import Config
from policy_rag.llm import Fake, usage
from policy_rag.retrieve import FakeReranker, Index

TEXTS = ["Advance beneficiary notice before services Medicare may not cover",
         "Maintenance therapy coverage does not depend on improvement potential",
         "Timed codes bill one unit for 8 to 22 minutes"]


def _index(tmp_path, reranker=None):
    llm = Fake()
    df = pd.DataFrame({"chunk_id": [0, 1, 2], "doc": ["BPM Ch. 15"] * 3, "section": ["§10", "§220.2", "§20.2"],
                       "heading": ["", "", ""], "page": [1, 2, 3], "part": [0, 0, 0], "text": TEXTS})
    df["embedding"] = list(llm.embed(TEXTS, "document"))
    db = tmp_path / "index.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE chunks AS SELECT * FROM df")
    con.close()
    cfg = Config({"rerank": {"candidates": 3}})
    return Index(cfg, llm, db, reranker=reranker), llm


def test_usage_converts_ollama_nanoseconds():
    u = usage({"prompt_eval_count": 1200, "eval_count": 80, "total_duration": 3_000_000_000,
               "load_duration": 500_000_000, "prompt_eval_duration": 1_000_000_000, "eval_duration": 2_000_000_000})
    assert u["total_tokens"] == 1280 and u["ollama_total_ms"] == 3000 and u["load_ms"] == 500
    assert u["tokens_per_s"] == 40


def test_rerank_reorders_hybrid_candidates(tmp_path):
    class Prefers(FakeReranker):                    # scores the maintenance chunk highest whatever hybrid says
        def scores(self, query, texts):
            return np.array([1.0 if "Maintenance" in t else 0.0 for t in texts])
    idx, _ = _index(tmp_path, Prefers())
    top = idx.rank("does medicare stop paying when the patient stops improving", "rerank", 2)
    assert top.iloc[0]["section"] == "§220.2" and len(top) == 2
    assert set(top["chunk_id"]) <= set(idx.rank("x", "hybrid", 3)["chunk_id"])   # only hybrid's candidates


def test_fake_index_uses_fake_reranker(tmp_path):
    idx, _ = _index(tmp_path)
    assert isinstance(idx.reranker, FakeReranker)
    assert idx.rank("timed codes minutes", "rerank", 1).iloc[0]["section"] == "§20.2"


def test_trace_carries_tokens_latency_and_cost(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("t")
    idx, llm = _index(tmp_path)
    pricing = {"m": {"input_per_mtok": 2.0, "output_per_mtok": 10.0}}
    out, _ = answer.rag(idx, llm, "m", "How many units for 8 minutes?", "hybrid", 2, pricing)
    assert out["prompt_tokens"] > 0 and out["completion_tokens"] > 0 and out["latency_ms"] > 0
    expected = (out["prompt_tokens"] * 2.0 + out["completion_tokens"] * 10.0) / 1e6
    assert np.isclose(out["cost_usd"], expected)
    mlflow.flush_trace_async_logging()             # traces are written in the background
    gen = next(s for s in mlflow.get_trace(out["trace_id"]).data.spans if s.name == "generate")
    assert gen.get_attribute(SpanAttributeKey.CHAT_USAGE)["input_tokens"] == out["prompt_tokens"]
    assert np.isclose(gen.get_attribute(SpanAttributeKey.LLM_COST)["total_cost"], expected)


def test_local_model_costs_nothing():
    assert answer.cost({}, "gemma4:e4b", {"prompt_tokens": 5000, "completion_tokens": 300})["total_cost"] == 0


def test_cost_summary_reports_load_time_apart():
    a = pd.DataFrame({"latency_ms": [9000.0, 2000.0, 2200.0], "load_ms": [7000.0, 0.0, 0.0],
                      "retrieve_ms": [30.0, 20.0, 25.0], "generate_ms": [8900.0, 1950.0, 2150.0],
                      "prompt_tokens": [1000, 1100, 900], "completion_tokens": [50, 60, 40],
                      "cost_usd": [0.0, 0.0, 0.0]})
    s = evaluate.cost_summary(a)
    assert s["latency_p95_s"] > 8 and s["latency_p95_excl_load_s"] < 2.3
    assert evaluate.cost_summary(pd.DataFrame({"x": [1]})) == {}


def test_runtime_lineage_without_server():
    from policy_rag import tracking
    cfg = Config({"ollama_url": "http://127.0.0.1:9", "chat_model": "m", "judge_model": "j", "embed_model": "e"})
    assert tracking.runtime(cfg) == {"ollama_version": "unavailable"}


def test_runtime_lineage_reads_digests(monkeypatch):
    import requests

    from policy_rag import tracking

    class R:
        def __init__(self, j):
            self.j = j

        def json(self):
            return self.j
    replies = {"/api/version": {"version": "0.35.1"},
               "/api/tags": {"models": [{"name": "gemma4:e4b", "digest": "dc35e8d9c606aaaa"},
                                        {"name": "nomic-embed-text:latest", "digest": "0a109f422b47bbbb"}]}}
    monkeypatch.setattr(requests, "get", lambda u, timeout: R(replies[u[u.index("/api"):]]))
    cfg = Config({"ollama_url": "http://x", "chat_model": "gemma4:e4b", "judge_model": "gemma4:26b",
                  "embed_model": "nomic-embed-text"})
    assert tracking.runtime(cfg) == {"ollama_version": "0.35.1", "chat_model_digest": "dc35e8d9c606",
                                     "judge_model_digest": "missing", "embed_model_digest": "0a109f422b47"}

"""Grounded answers with citations, and an explicit "not in these documents" when the context does not cover it."""
from __future__ import annotations

import hashlib
import re
import time

import mlflow
import pandas as pd
from mlflow.entities import Document, SpanType
from mlflow.tracing.constant import SpanAttributeKey

SYSTEM = """You answer questions about Medicare billing policy using ONLY the numbered excerpts provided.
Rules:
- Cite every claim with the excerpt number in square brackets, e.g. [2]. Cite only excerpts that support the claim.
- If the excerpts do not contain the answer, reply exactly: NOT IN DOCUMENTS
- Be concise: 1-4 sentences. Quote numbers, codes and thresholds exactly as written.
- Do not use outside knowledge, even if you think you know the answer."""

ABSTAIN = "NOT IN DOCUMENTS"

# A decline the model wrote as prose instead of the exact reply. Matched in the first sentence only, so an answer that
# notes one missing detail after answering ("... [2]. The excerpts do not specify X.") still counts as an answer.
_DOC = r"(?:the\s+)?(?:provided\s+|given\s+|available\s+)?(?:excerpts?|documents?|context|text|sources?)"
_PROSE_DECLINE = re.compile("|".join([
    rf"\b{_DOC}\s+(?:do|does)\s+not\s+(?:provide|contain|include|address|specify|mention|state|cover|say|answer|discuss)",
    rf"\bnot\s+(?:addressed|covered|specified|mentioned|found|stated|included|provided|contained)\s+in\s+{_DOC}",
    (r"\b(?:there\s+is\s+)?no\s+(?:information|mention|specific\s+(?:answer|information|guidance))\s+"
     r"(?:in|within|about|regarding|on)\b"),
    rf"\b(?:cannot|can't|unable\s+to)\s+(?:be\s+)?(?:answer|determine|find|tell)\w*\s+(?:from|based\s+on|using)\s+{_DOC}",
    r"\bI\s+(?:do\s+not|don't)\s+(?:know|have\s+(?:enough|specific|access|the\s+information))",
]), re.IGNORECASE)


def declined(reply: str) -> str:
    """'exact' for the NOT IN DOCUMENTS reply, 'prose' for a decline written out in the first sentence, '' otherwise."""
    text = str(reply).strip()
    if text.upper().startswith(ABSTAIN):
        return "exact"
    first = re.split(r"(?<=[.!?])\s+|\n", text, maxsplit=1)[0]
    return "prose" if _PROSE_DECLINE.search(first) else ""


def context(hits: pd.DataFrame) -> str:
    return "\n\n".join(f"[{i}] ({r.doc}, {r.section}{', ' + r.heading if r.heading else ''}; p. {r.page})\n{r.text}"
                       for i, r in enumerate(hits.itertuples(), start=1))


def answer(llm, model: str, question: str, hits: pd.DataFrame) -> dict:
    reply = llm.chat(model, SYSTEM, f"Excerpts:\n\n{context(hits)}\n\nQuestion: {question}").strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply) if 1 <= int(n) <= len(hits)})
    kind = declined(reply)
    return {"answer": reply, "abstained": bool(kind), "abstain_kind": kind,
            "cited_chunks": [int(hits.iloc[n - 1]["chunk_id"]) for n in cited],
            "cited_sections": [f"{hits.iloc[n - 1]['doc']} {hits.iloc[n - 1]['section']}" for n in cited]}


PROMPT_SHA = hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]


def cost(pricing: dict | None, model: str, u: dict) -> dict:
    """USD from a per-model price table (USD per million tokens); local models are absent from it and cost 0."""
    p = (pricing or {}).get(model) or {}
    i = u.get("prompt_tokens", 0) * float(p.get("input_per_mtok", 0)) / 1e6
    o = u.get("completion_tokens", 0) * float(p.get("output_per_mtok", 0)) / 1e6
    return {"input_cost": i, "output_cost": o, "total_cost": i + o}


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def rag(idx, llm, model: str, question: str, retriever: str, k: int,
        pricing: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Retrieve and answer one question, recorded as an MLflow trace (question -> retrieve -> generate).

    A trace keeps what the model actually saw: the five excerpts, their sections and scores, the prompt version and
    the reply. When an answer is wrong, the trace shows whether retrieval missed or generation failed. Each trace
    also carries what the answer cost: wall-clock time per step, tokens in and out, Ollama's own timings (model load,
    prompt processing, generation) and USD from the `pricing` table. MLflow sums the token counts per trace.
    """
    t_all = time.perf_counter()
    with mlflow.start_span(name="rag", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"question": question})
        root.set_attributes({"retriever": retriever, "top_k": k, "chat_model": model, "prompt_sha": PROMPT_SHA})
        t0 = time.perf_counter()
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as s:
            s.set_inputs({"query": question, "method": retriever, "k": k})
            hits = idx.rank(question, retriever, k)
            s.set_outputs([Document(page_content=r.text[:1500], id=str(r.chunk_id),
                                    metadata={"doc": r.doc, "section": r.section, "heading": r.heading,
                                              "page": int(r.page), "rank": int(r.rank), "score": float(r.score)})
                           for r in hits.itertuples()])
            retrieve_ms = _ms(t0)
            s.set_attribute("latency_ms", retrieve_ms)
        t0 = time.perf_counter()
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as s:
            s.set_inputs({"system_prompt_sha": PROMPT_SHA, "question": question, "excerpts": len(hits)})
            out = answer(llm, model, question, hits)
            generate_ms = _ms(t0)
            u = dict(getattr(llm, "last_usage", {}) or {})
            c = cost(pricing, model, u)
            s.set_outputs({"reply": out["answer"]})
            s.set_attributes({SpanAttributeKey.MODEL: model, "latency_ms": generate_ms,
                              SpanAttributeKey.CHAT_USAGE: {"input_tokens": u.get("prompt_tokens", 0),
                                                            "output_tokens": u.get("completion_tokens", 0),
                                                            "total_tokens": u.get("total_tokens", 0)},
                              SpanAttributeKey.LLM_COST: c}
                             | {f"ollama.{key}": v for key, v in u.items() if key.endswith(("_ms", "_per_s"))})
        total_ms = _ms(t_all)
        root.set_attributes({"latency_ms": total_ms, "cost_usd": c["total_cost"]})
        root.set_outputs({"answer": out["answer"], "abstained": out["abstained"],
                          "cited_sections": out["cited_sections"]})
        out["trace_id"] = root.trace_id
    out |= {"latency_ms": total_ms, "retrieve_ms": retrieve_ms, "generate_ms": generate_ms,
            "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
            "load_ms": u.get("load_ms", 0.0), "cost_usd": c["total_cost"]}
    return out, hits

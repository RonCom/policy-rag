"""Grounded answers with citations, and an explicit "not in these documents" when the context does not cover it."""
from __future__ import annotations

import hashlib
import re

import mlflow
import pandas as pd
from mlflow.entities import Document, SpanType

SYSTEM = """You answer questions about Medicare billing policy using ONLY the numbered excerpts provided.
Rules:
- Cite every claim with the excerpt number in square brackets, e.g. [2]. Cite only excerpts that support the claim.
- If the excerpts do not contain the answer, reply exactly: NOT IN DOCUMENTS
- Be concise: 1-4 sentences. Quote numbers, codes and thresholds exactly as written.
- Do not use outside knowledge, even if you think you know the answer."""

ABSTAIN = "NOT IN DOCUMENTS"


def context(hits: pd.DataFrame) -> str:
    return "\n\n".join(f"[{i}] ({r.doc}, {r.section}{', ' + r.heading if r.heading else ''}; p. {r.page})\n{r.text}"
                       for i, r in enumerate(hits.itertuples(), start=1))


def answer(llm, model: str, question: str, hits: pd.DataFrame) -> dict:
    reply = llm.chat(model, SYSTEM, f"Excerpts:\n\n{context(hits)}\n\nQuestion: {question}").strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply) if 1 <= int(n) <= len(hits)})
    return {"answer": reply, "abstained": reply.upper().startswith(ABSTAIN),
            "cited_chunks": [int(hits.iloc[n - 1]["chunk_id"]) for n in cited],
            "cited_sections": [f"{hits.iloc[n - 1]['doc']} {hits.iloc[n - 1]['section']}" for n in cited]}


PROMPT_SHA = hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]


def rag(idx, llm, model: str, question: str, retriever: str, k: int) -> tuple[dict, pd.DataFrame]:
    """Retrieve and answer one question, recorded as an MLflow trace (question -> retrieve -> generate).

    A trace keeps what the model actually saw: the five excerpts, their sections and scores, the prompt version and
    the reply. When an answer is wrong, the trace shows whether retrieval missed or generation failed.
    """
    with mlflow.start_span(name="rag", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"question": question})
        root.set_attributes({"retriever": retriever, "top_k": k, "chat_model": model, "prompt_sha": PROMPT_SHA})
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as s:
            s.set_inputs({"query": question, "method": retriever, "k": k})
            hits = idx.rank(question, retriever, k)
            s.set_outputs([Document(page_content=r.text[:1500], id=str(r.chunk_id),
                                    metadata={"doc": r.doc, "section": r.section, "heading": r.heading,
                                              "page": int(r.page), "rank": int(r.rank), "score": float(r.score)})
                           for r in hits.itertuples()])
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as s:
            s.set_inputs({"system_prompt_sha": PROMPT_SHA, "question": question, "excerpts": len(hits)})
            out = answer(llm, model, question, hits)
            s.set_outputs({"reply": out["answer"]})
        root.set_outputs({"answer": out["answer"], "abstained": out["abstained"],
                          "cited_sections": out["cited_sections"]})
        out["trace_id"] = root.trace_id
    return out, hits

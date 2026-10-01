"""Grounded answers with citations, and an explicit "not in these documents" when the context does not cover it."""
from __future__ import annotations

import re

import pandas as pd

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

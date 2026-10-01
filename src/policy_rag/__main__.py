"""CLI:
  policy-rag index                 parse the PDFs, chunk by section, embed, store in DuckDB
  policy-rag ask "question"        answer one question with citations
  policy-rag eval [--no-judge | --judge-only] [--retrieval-only]
  policy-rag calibrate             judge vs your labels in eval/human_labels.csv
Add --fake to any command to run without Ollama (hash embeddings, canned replies) for testing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="policy-rag")
    ap.add_argument("step", choices=["index", "ask", "eval", "calibrate"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--fake", action="store_true", help="no model server: hash embeddings and canned replies")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--judge-only", action="store_true", help="grade the answers from the last --no-judge run")
    ap.add_argument("--retriever", help="override answer_retriever for ask")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    from . import answer, evaluate, ingest
    from .config import Config
    from .llm import client
    from .retrieve import Index
    cfg = Config.load()
    llm = client(cfg, a.fake)
    if a.step == "index":
        ingest.build(cfg, llm)
    elif a.step == "ask":
        idx = Index(cfg, llm)
        hits = idx.rank(a.question, a.retriever or cfg["answer_retriever"], cfg["top_k"])
        r = answer.answer(llm, cfg["chat_model"], a.question, hits)
        print("\n" + r["answer"] + "\n")
        for i, h in enumerate(hits.itertuples(), start=1):
            print(f"[{i}] {h.doc} {h.section} {h.heading} (p. {h.page})")
    elif a.step == "eval":
        print(json.dumps(evaluate.run(cfg, llm, not a.no_judge, a.retrieval_only, a.judge_only), indent=2, default=float))
    elif a.step == "calibrate":
        print(json.dumps(evaluate.calibrate(cfg), indent=2, default=float))


if __name__ == "__main__":
    main()

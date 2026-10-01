"""Feature ablation: does each retrieval feature earn its place? `policy-rag ablate [--bm25-only]`.

Each variant changes one thing from the configured pipeline and is scored on the same 50 answerable questions
(recall@k by question style, MRR, and a paired-bootstrap interval against the baseline). Gold labels are sections,
so chunk text can change without changing the answer key.

Keyword (BM25) variants, no model needed:
  fixed_windows    ignore the manuals' sections: one word stream per manual cut into 300-word windows (a window
                   counts as a hit if it overlaps a gold section at all, which favours this variant)
  no_toc_check     accept any line that looks like a numbered heading, without checking it against the
                   table of contents (cross-references then split sections and mislabel them)
  no_heading       index the chunk text without its section heading
  simple_tokens    split on every non-alphanumeric character, so "220.2" and "g-codes" break apart
  no_stopwords     keep "the", "of", "may"... in the index
Embedding variants (need Ollama; one re-embedding pass each):
  no_heading       embed the chunk text without its section heading
  no_prefixes      drop nomic-embed-text's search_query: / search_document: task prefixes
"""
from __future__ import annotations

import json
import logging
import re

import mlflow
import numpy as np
import pandas as pd

from . import ingest, retrieve, tracking
from .config import Config
from .evaluate import KS, load_questions
from .sweep import _boot_ci

log = logging.getLogger(__name__)
SIMPLE = re.compile(r"[a-z0-9]+")


def _ranks(scores: np.ndarray, keys: list[set], gold: list[str]) -> float:
    top = np.argsort(-scores, kind="stable")[:max(KS)]
    g = set(gold)
    hit = [i for i, c in enumerate(top) if keys[c] & g]
    return hit[0] + 1 if hit else np.inf


def _keys(chunks: pd.DataFrame) -> list[set]:
    return [{k} for k in chunks["doc"] + " " + chunks["section"]]


def fixed_windows(chunks: pd.DataFrame, words: int, overlap: int) -> tuple[pd.DataFrame, list[set]]:
    """Section-blind chunking of the same cleaned text; each window is labeled with every section it overlaps."""
    rows, keys = [], []
    for doc in chunks["doc"].unique():
        w, lab = [], []
        for r in chunks[(chunks["doc"] == doc)].itertuples():
            toks = r.text.split()
            if r.part > 0:                                           # drop the overlap repeated between parts
                toks = toks[overlap:]
            w += toks
            lab += [f"{doc} {r.section}"] * len(toks)
        for i in range(0, max(1, len(w) - overlap), words - overlap):
            rows.append({"doc": doc, "section": "", "heading": "", "text": " ".join(w[i:i + words])})
            keys.append(set(lab[i:i + words]))
    return pd.DataFrame(rows), keys


def _bm25(docs: list[str], tok) -> retrieve.BM25:
    orig = retrieve.tokenize
    retrieve.tokenize = tok                        # BM25 calls the module-level tokenizer
    try:
        b = retrieve.BM25(docs)
    finally:
        retrieve.tokenize = orig
    b._tok = tok
    return b


def _bm25_scores(b: retrieve.BM25, q: str) -> np.ndarray:
    orig = retrieve.tokenize
    retrieve.tokenize = b._tok
    try:
        return b.scores(q)
    finally:
        retrieve.tokenize = orig


def run(cfg: Config, llm, bm25_only: bool = False) -> dict:
    chunks = ingest.chunk_docs(cfg)
    keys = _keys(chunks)
    qs = load_questions(cfg)
    qa = qs[qs["answerable"]].reset_index(drop=True)
    styles = qa["style"].to_numpy()
    with_head = (chunks["heading"] + " " + chunks["text"]).tolist()
    text_only = chunks["text"].tolist()
    base_tok = retrieve.tokenize

    def simple(t):
        return [w for w in SIMPLE.findall(t.lower()) if w not in retrieve.STOP]

    def keep_stop(t):
        return retrieve.TOKEN.findall(t.lower())

    fw, fw_keys = fixed_windows(chunks, cfg["chunk"]["max_words"], cfg["chunk"]["overlap_words"])
    nt = ingest.chunk_docs(cfg, toc_check=False)
    variants = {("bm25", "baseline"): (with_head, base_tok, keys),
                ("bm25", "fixed_windows"): (fw["text"].tolist(), base_tok, fw_keys),
                ("bm25", "no_toc_check"): ((nt["heading"] + " " + nt["text"]).tolist(), base_tok, _keys(nt)),
                ("bm25", "no_heading"): (text_only, base_tok, keys),
                ("bm25", "simple_tokens"): (with_head, simple, keys),
                ("bm25", "no_stopwords"): (with_head, keep_stop, keys)}
    sizes = {"fixed_windows": len(fw), "no_toc_check": len(nt)}
    ranks = {}
    for (fam, name), (docs, tok, kk) in variants.items():
        b = _bm25(docs, tok)
        ranks[(fam, name)] = np.array([_ranks(_bm25_scores(b, q.question), kk, q.gold) for q in qa.itertuples()])
        log.info("%s %s recall@5 %.2f", fam, name, np.mean(ranks[(fam, name)] <= 5))
    if not bm25_only:
        dense = {"baseline": ((chunks["heading"] + ". " + chunks["text"]).tolist(), "document", "query"),
                 "no_heading": (text_only, "document", "query"),
                 "no_prefixes": ((chunks["heading"] + ". " + chunks["text"]).tolist(), "raw", "raw")}
        for name, (docs, dkind, qkind) in dense.items():
            emb = llm.embed(docs, dkind)
            qe = llm.embed(qa["question"].tolist(), qkind)
            ranks[("dense", name)] = np.array([_ranks(emb @ qe[i], keys, q.gold) for i, q in enumerate(qa.itertuples())])
            log.info("dense %s recall@5 %.2f", name, np.mean(ranks[("dense", name)] <= 5))

    rows = []
    for (fam, name), r in ranks.items():
        row = {"retriever": fam, "variant": name, "chunks": sizes.get(name, len(chunks)), "mrr": float(np.mean(1 / r))}
        row |= {f"recall_at_{k}": float(np.mean(r <= k)) for k in KS}
        row |= {f"recall_at_5_{s}": float(np.mean(r[styles == s] <= 5)) for s in ("exact", "paraphrase", "plain")}
        base = ranks[(fam, "baseline")]
        row["delta_recall_at_5"] = row["recall_at_5"] - float(np.mean(base <= 5))
        row["delta_ci95"] = _boot_ci(r, base) if name != "baseline" else [0.0, 0.0]
        row["questions_lost"] = int(((base <= 5) & (r > 5)).sum())
        row["questions_gained"] = int(((base > 5) & (r <= 5)).sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(cfg.reports / "ablation.csv", index=False)
    (cfg.reports / "ablation.json").write_text(json.dumps(rows, indent=2, default=float))
    multi = float(np.mean([len(k) > 1 for k in fw_keys]))
    _chart(t, cfg, multi)

    tracking.setup(cfg, "policy-rag-ablation")
    with mlflow.start_run(run_name="retrieval feature ablation") as parent:
        mlflow.log_params({"chunks": len(chunks), "embed_model": cfg["embed_model"], "bm25_only": bm25_only}
                          | cfg["chunk"] | tracking.lineage(cfg))
        for row in rows:
            with mlflow.start_run(run_name=f"{row['retriever']}_{row['variant']}", nested=True):
                mlflow.log_params({"retriever": row["retriever"], "variant": row["variant"]})
                mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})
        for f in ("ablation.csv", "ablation.json"):
            mlflow.log_artifact(str(cfg.reports / f))
        mlflow.log_artifact(str(cfg.reports / "figures" / "ablation.png"), "figures")
        run_id = parent.info.run_id
    return {"rows": rows, "fixed_windows_spanning_2plus_sections": multi, "mlflow_run_id": run_id}


def _chart(t: pd.DataFrame, cfg: Config, multi: float) -> None:
    from .figures import C, _style
    plt = _style()
    t = t.reset_index(drop=True)
    names = {"baseline": "baseline (as configured)", "fixed_windows": "fixed windows, no sections*",
             "no_toc_check": "no table-of-contents check", "no_heading": "no section heading",
             "simple_tokens": "simple tokenizer", "no_stopwords": "keep stopwords", "no_prefixes": "no task prefixes"}
    fam = {"bm25": "keyword", "dense": "embedding"}
    labels = [f"{fam[r.retriever]}: {names.get(r.variant, r.variant)}" for r in t.itertuples()]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(t) + 1.2))
    w = 0.38
    ax.barh(y - w / 2, 100 * t["recall_at_5"], w, color=[C[r] for r in t["retriever"]], label="all 50")
    ax.barh(y + w / 2, 100 * t["recall_at_5_plain"], w, color=C["muted"], label="plain language (10)")
    for i, r in t.iterrows():
        ax.text(100 * r["recall_at_5"] + 1, i - w / 2, f"{100 * r['recall_at_5']:.0f}", va="center", fontsize=7)
        ax.text(100 * r["recall_at_5_plain"] + 1, i + w / 2, f"{100 * r['recall_at_5_plain']:.0f}", va="center",
                fontsize=7, color="#52514e")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("% with gold section in top 5\n\n* A window counts as a hit if it overlaps a gold section at all; "
                  f"{100 * multi:.0f}% of windows span two or more sections,\nso they can't be cited to one section.",
                  fontsize=8)
    ax.set_title("Retrieval recall with one feature removed at a time")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="y", visible=False)
    fig.savefig(cfg.reports / "figures" / "ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

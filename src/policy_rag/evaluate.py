"""Measure the system instead of eyeballing it.

Retrieval (answerable questions): recall@k (a chunk from a gold section is in the top k) and MRR for BM25, dense and
hybrid retrieval, with paired bootstrap intervals for the differences.

Answers (all questions, using the configured retriever):
  - abstention: unanswerable questions should get "NOT IN DOCUMENTS"; answerable ones should not
  - citations: does the answer cite a chunk from a gold section (deterministic, no model needed)
  - LLM-as-a-judge: correctness against the reference answer's key facts, and faithfulness to the retrieved excerpts
  - closed-book baseline: the same model with no excerpts, to show what retrieval adds and how often it makes things up
Judge scores are only trusted after calibration: `policy-rag calibrate` compares them with human labels you add to
eval/human_labels.csv (agreement and Cohen's kappa).
"""
from __future__ import annotations

import json
import logging
import re

import mlflow
import numpy as np
import pandas as pd

from . import answer as ans
from . import tracking
from .config import Config
from .retrieve import Index

log = logging.getLogger(__name__)
KS = (1, 3, 5, 10)

JUDGE_SYSTEM = """You grade answers to Medicare billing-policy questions. Be strict and literal.
correct: 1 if the answer states the reference's key facts without contradicting them, 0.5 if it states some but
not all, 0 if it misses or contradicts them. If the reference is NOT IN DOCUMENTS, correct is 1 only when the answer
declines (says the documents do not contain it) and 0 if it gives a substantive answer.
faithful: 1 if every claim in the answer is supported by the excerpts shown, 0 if any claim is not supported.
For a declined answer, faithful is 1."""

JUDGE_SCHEMA = {"type": "object", "properties": {"correct": {"type": "number", "enum": [0, 0.5, 1]},
                                                  "faithful": {"type": "integer", "enum": [0, 1]},
                                                  "reason": {"type": "string"}},
                "required": ["correct", "faithful", "reason"]}

CLOSED_BOOK = """You answer questions about Medicare billing policy. If you do not know, reply exactly: NOT IN DOCUMENTS
Be concise: 1-4 sentences."""


def load_questions(cfg: Config) -> pd.DataFrame:
    return pd.read_json(cfg.path("eval_questions"), lines=True)


def first_hits(idx: Index, qa: pd.DataFrame, methods) -> dict[str, np.ndarray]:
    """Rank of the first chunk from a gold section, per answerable question (inf if not in the top 10)."""
    keys = (idx.chunks["doc"] + " " + idx.chunks["section"]).to_numpy()
    out = {}
    for m in methods:
        ranks = []
        for q in qa.itertuples():
            top = idx.rank(q.question, m, max(KS))
            hit = np.where(np.isin(keys[top["chunk_id"].to_numpy()], q.gold))[0]
            ranks.append(hit[0] + 1 if len(hit) else np.inf)
        out[m] = np.array(ranks)
    return out


def retrieval(cfg: Config, idx: Index, qs: pd.DataFrame, seed: int = 0) -> dict:
    qa = qs[qs["answerable"]].reset_index(drop=True)
    first_hit = first_hits(idx, qa, cfg["retrievers"])
    out = {m: {f"recall@{k}": float(np.mean(r <= k)) for k in KS} | {"mrr": float(np.mean(1 / r))}
           for m, r in first_hit.items()}
    rng = np.random.default_rng(seed)
    n = len(qa)
    for a, b in (("hybrid", "bm25"), ("hybrid", "dense"), ("dense", "bm25")):
        if a in first_hit and b in first_hit:
            d = [np.mean(first_hit[a][i] <= 5) - np.mean(first_hit[b][i] <= 5)
                 for i in (rng.integers(0, n, n) for _ in range(2000))]
            out[f"{a}_minus_{b}_recall@5"] = {"est": out[a]["recall@5"] - out[b]["recall@5"],
                                             "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
    out["by_style"] = {m: {s: float(np.mean(first_hit[m][qa["style"].to_numpy() == s] <= 5))
                           for s in qa["style"].unique()} for m in first_hit}
    return out


def judge(llm, model: str, q, reply: str, context: str) -> dict:
    user = (f"Question: {q.question}\nReference answer: {q.reference}\nKey facts: {'; '.join(q.key_facts) or '-'}\n\n"
            f"Excerpts shown to the answerer:\n{context or '(none)'}\n\nAnswer to grade:\n{reply}")
    raw = llm.chat(model, JUDGE_SYSTEM, user, schema=JUDGE_SCHEMA)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        j = json.loads(m.group(0)) if m else {"correct": np.nan, "faithful": np.nan, "reason": raw[:200]}
    return {"judge_correct": float(j.get("correct", np.nan)), "judge_faithful": float(j.get("faithful", np.nan)),
            "judge_reason": j.get("reason", "")}


def answers(cfg: Config, idx: Index, qs: pd.DataFrame, llm, use_judge: bool = True) -> pd.DataFrame:
    rows = []
    for q in qs.itertuples():
        a, hits = ans.rag(idx, llm, cfg["chat_model"], q.question, cfg["answer_retriever"], cfg["top_k"])
        ctx = ans.context(hits)
        retrieved_gold = bool(set(hits["doc"] + " " + hits["section"]) & set(q.gold))
        cb = llm.chat(cfg["chat_model"], CLOSED_BOOK, f"Question: {q.question}").strip()
        rows.append({"id": q.id, "question": q.question, "answerable": q.answerable, "style": q.style,
                     "gold": "; ".join(q.gold), "reference": q.reference, "key_facts": "; ".join(q.key_facts),
                     "answer": a["answer"], "abstained": a["abstained"], "context": ctx,
                     "cited_sections": "; ".join(a["cited_sections"]),
                     "cites_gold": bool(set(a["cited_sections"]) & set(q.gold)),
                     "retrieved_gold": retrieved_gold, "trace_id": a["trace_id"],
                     "closed_book_answer": cb, "closed_book_abstained": cb.upper().startswith(ans.ABSTAIN)})
        if q.answerable:                                   # deterministic checks, attached to the trace
            tracking.feedback(a["trace_id"], "retrieved_gold_section", retrieved_gold, "CODE", "gold_sections")
        else:
            tracking.feedback(a["trace_id"], "declined_out_of_scope", a["abstained"], "CODE", "abstain_check")
        log.info("%s %s | %s", q.id, "abstain" if a["abstained"] else "answer", a["answer"][:90].replace("\n", " "))
    out = pd.DataFrame(rows)
    return judge_all(cfg, llm, out) if use_judge else out


def judge_all(cfg: Config, llm, a: pd.DataFrame) -> pd.DataFrame:
    """Grade after all answers exist, so the (large) judge model is loaded once instead of swapping per question."""
    graded = []
    for r in a.itertuples():
        q = type("Q", (), {"question": r.question, "reference": r.reference,
                           "key_facts": [k for k in str(r.key_facts).split("; ") if k and k != "nan"]})
        g = judge(llm, cfg["judge_model"], q, r.answer, r.context)
        g |= {f"closed_book_{k}": v for k, v in judge(llm, cfg["judge_model"], q, r.closed_book_answer, "").items()}
        graded.append(g)
        for name in ("correct", "faithful"):
            tracking.feedback(getattr(r, "trace_id", None), f"judge_{name}", g[f"judge_{name}"], "LLM_JUDGE",
                              cfg["judge_model"], g["judge_reason"])
        log.info("judged %s: correct %s, faithful %s", r.id, g["judge_correct"], g["judge_faithful"])
    return pd.concat([a.reset_index(drop=True), pd.DataFrame(graded)], axis=1)


def summarize(a: pd.DataFrame) -> dict:
    A, U = a[a["answerable"]], a[~a["answerable"]]
    given = A[~A["abstained"]]
    out = {"questions": len(a), "answerable": len(A), "unanswerable": len(U),
           "abstained_on_unanswerable": float(U["abstained"].mean()),
           "abstained_on_answerable": float(A["abstained"].mean()),
           "retrieved_gold_section": float(A["retrieved_gold"].mean()),
           "cites_gold_when_answering": float(given["cites_gold"].mean()) if len(given) else np.nan,
           "closed_book_abstained_on_unanswerable": float(U["closed_book_abstained"].mean())}
    if "judge_correct" in a:
        out |= {"judge_correct_answerable": float(A["judge_correct"].mean()),
                "judge_faithful_when_answering": float(given["judge_faithful"].mean()) if len(given) else np.nan,
                "closed_book_judge_correct_answerable": float(A["closed_book_judge_correct"].mean()),
                "judge_correct_all": float(a["judge_correct"].mean()),
                "closed_book_judge_correct_all": float(a["closed_book_judge_correct"].mean())}
    return out


def run(cfg: Config, llm, use_judge: bool = True, retrieval_only: bool = False, judge_only: bool = False) -> dict:
    tracking.setup(cfg, "policy-rag")
    with mlflow.start_run(run_name=f"{cfg['chat_model']}|{cfg['answer_retriever']}") as run:
        mlflow.log_params({k: cfg[k] for k in ("embed_model", "chat_model", "judge_model", "top_k",
                                                "answer_retriever")} | cfg["chunk"] | tracking.lineage(cfg))
        mlflow.set_tags({"step": "judge-only" if judge_only else "retrieval-only" if retrieval_only
                         else "answers+judge" if use_judge else "answers"})
        mlflow.log_artifact(str(cfg.path("eval_questions")), "eval")
        mlflow.log_text(ans.SYSTEM, "prompts/answer_system.txt")
        mlflow.log_text(JUDGE_SYSTEM, "prompts/judge_system.txt")
        res = _run(cfg, llm, use_judge, retrieval_only, judge_only)
        for m in cfg["retrievers"]:
            mlflow.log_metrics({f"{m}_{k.replace('@', '_at_')}": v for k, v in res["retrieval"][m].items()})
        if "answers" in res:
            mlflow.log_metrics({k: v for k, v in res["answers"].items() if isinstance(v, float) and not np.isnan(v)})
            mlflow.log_artifact(str(cfg.reports / "answers.csv"))
        _chart(res, cfg)
        mlflow.log_artifact(str(cfg.reports / "eval.json"))
        mlflow.log_artifact(str(cfg.reports / "figures" / "retrieval_recall.png"), "figures")
        res["mlflow_run_id"] = run.info.run_id
    return res


def _run(cfg: Config, llm, use_judge: bool, retrieval_only: bool, judge_only: bool) -> dict:
    idx = Index(cfg, llm)
    qs = load_questions(cfg)
    res = {"retrieval": retrieval(cfg, idx, qs)}
    log.info("retrieval: %s", {m: res["retrieval"][m]["recall@5"] for m in cfg["retrievers"]})
    if judge_only:                                         # grade the answers saved by an earlier --no-judge run
        a = pd.read_csv(cfg.reports / "answers.csv")
        a = a.drop(columns=[c for c in a.columns if c.startswith(("judge_", "closed_book_judge_"))])
        a = judge_all(cfg, llm, a.fillna({"context": "", "key_facts": ""}))
    elif not retrieval_only:
        a = answers(cfg, idx, qs, llm, use_judge)
    if judge_only or not retrieval_only:
        a.to_csv(cfg.reports / "answers.csv", index=False)
        res["answers"] = summarize(a)
        human = cfg.path("human_labels")
        if not human.exists():                               # template for calibration: you fill the two columns
            a[["id", "question", "reference", "key_facts", "answer", "context"]].assign(
                human_correct="", human_faithful="").to_csv(human, index=False)
    (cfg.reports / "eval.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def calibrate(cfg: Config) -> dict:
    """Agreement between the LLM judge and your labels, on the rows you labeled."""
    h = pd.read_csv(cfg.path("human_labels"))
    a = pd.read_csv(cfg.reports / "answers.csv")
    h = h.dropna(subset=["human_correct"])
    # labels only count for the exact answer that was labeled; a later run may have produced a different answer
    d = a.merge(h[["id", "answer", "human_correct", "human_faithful"]], on=["id", "answer"])
    out = {"labeled": len(h), "matched_to_current_answers": len(d)}
    for j, hcol in (("judge_correct", "human_correct"), ("judge_faithful", "human_faithful")):
        x = d.dropna(subset=[hcol])
        if len(x):
            jb, hb = (x[j] >= 0.5).astype(int), (x[hcol].astype(float) >= 0.5).astype(int)
            po = float((jb == hb).mean())
            pe = float(jb.mean() * hb.mean() + (1 - jb.mean()) * (1 - hb.mean()))
            out[j] = {"n": len(x), "agreement": po, "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
                      "judge_rate": float(jb.mean()), "human_rate": float(hb.mean())}
    (cfg.reports / "judge_calibration.json").write_text(json.dumps(out, indent=2, default=float))
    if "trace_id" in d:                                  # attach your labels to the traces they describe
        tracking.setup(cfg, "policy-rag")
        for r in d.itertuples():
            for name in ("correct", "faithful"):
                v = getattr(r, f"human_{name}")
                if pd.notna(v):
                    tracking.feedback(r.trace_id, f"human_{name}", float(v), "HUMAN", "reviewer")
        out["labels_attached_to_traces"] = int(d["trace_id"].notna().sum())
    return out


def _chart(res: dict, cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e4e3df",
                         "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    cols = {"bm25": "#eb6834", "dense": "#52514e", "hybrid": "#2a78d6"}
    r = res["retrieval"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    w = 0.25
    for i, m in enumerate(cfg["retrievers"]):
        ax.bar(np.arange(len(KS)) + (i - 1) * w, [100 * r[m][f"recall@{k}"] for k in KS], w, label=m,
               color=cols.get(m))
    ax.set_xticks(range(len(KS)), [f"top {k}" for k in KS])
    ax.set_ylabel("% of questions with a gold section retrieved")
    ax.set_title("Retrieval recall by method")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(cfg.reports / "figures" / "retrieval_recall.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

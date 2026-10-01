"""Retrieval sweep: chunk size x overlap x retriever, one MLflow parent run with a nested child run per setting.

Why retrieval only: it is deterministic and costs one embedding pass per chunking, so every setting can be tried on
all 50 answerable questions. Answer-level evaluation needs the chat model and the judge for every question and
setting (hours on a laptop). On the eval set, a retrieval miss became a wrong answer 4 times out of 4.

Gold labels are sections, not chunks, so the same questions score every chunking.

Selection rule, fixed before running (config.yaml, sweep.rule):
  1. Keep settings whose top-k context fits the budget (sweep.max_context_words; larger chunks raise recall partly
     by showing the model more text, which costs latency and invites unsupported claims).
  2. Rank by recall@5, then plain-language recall@5, then MRR.
  3. Recommend changing config.yaml only if the paired-bootstrap 95% interval for (best - current) recall@5 is above
     zero. With 50 questions one question is 2 points, so most differences will be noise, and the rule says so.
"""
from __future__ import annotations

import json
import logging
import time

import mlflow
import numpy as np
import pandas as pd

from . import ingest, tracking
from .config import Config
from .evaluate import KS, first_hits, load_questions
from .retrieve import Index

log = logging.getLogger(__name__)


def _metrics(r: np.ndarray, plain: np.ndarray) -> dict:
    m = {f"recall_at_{k}": float(np.mean(r <= k)) for k in KS} | {"mrr": float(np.mean(1 / r))}
    if plain.any():
        m["plain_recall_at_5"] = float(np.mean(r[plain] <= 5))
        m["manual_wording_recall_at_5"] = float(np.mean(r[~plain] <= 5))
    return m


def _boot_ci(a: np.ndarray, b: np.ndarray, seed: int = 0, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = (a[idx] <= 5).mean(1) - (b[idx] <= 5).mean(1)
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def run(cfg: Config, llm) -> dict:
    sw = cfg["sweep"]
    qs = load_questions(cfg)
    qa = qs[qs["answerable"]].reset_index(drop=True)
    plain = (qa["style"] == "plain").to_numpy()
    current = (cfg["chunk"]["max_words"], cfg["chunk"]["overlap_words"], cfg["answer_retriever"])
    tracking.setup(cfg, "policy-rag-sweep")
    rows, ranks = [], {}
    with mlflow.start_run(run_name="chunking x retriever sweep") as parent:
        mlflow.log_params({"grid": json.dumps(sw["grid"]), "retrievers": ",".join(cfg["retrievers"]),
                           "top_k": cfg["top_k"], "max_context_words": sw["max_context_words"],
                           "embed_model": cfg["embed_model"], "selection_rule": sw["rule"]} | tracking.lineage(cfg))
        for words, overlap in sw["grid"]:
            db = cfg.path("data_dir") / f"index_w{words}_o{overlap}.duckdb"
            t0 = time.time()
            if (words, overlap) == current[:2] and cfg.db_path.exists():
                db = cfg.db_path                               # the current index: reuse it
            elif not db.exists() or sw.get("rebuild", False):
                ingest.build(cfg, llm, words, overlap, db)
            build_s = time.time() - t0
            idx = Index(cfg, llm, db)
            n_words = idx.chunks["text"].str.split().str.len()
            t0 = time.time()
            fh = first_hits(idx, qa, cfg["retrievers"])
            search_s = (time.time() - t0) / (len(qa) * len(cfg["retrievers"]))
            for m, r in fh.items():
                ctx = float(np.mean([n_words.iloc[idx.rank(q, m, cfg["top_k"])["chunk_id"]].sum()
                                     for q in qa["question"]]))
                with mlflow.start_run(run_name=f"w{words}_o{overlap}_{m}", nested=True):
                    mlflow.log_params({"max_words": words, "overlap_words": overlap, "retriever": m})
                    met = _metrics(r, plain) | {"chunks": len(idx.chunks), "context_words_top_k": ctx,
                                                "index_build_seconds": build_s, "search_seconds_per_query": search_s}
                    mlflow.log_metrics(met)
                    mlflow.log_text(pd.DataFrame({"id": qa["id"], "style": qa["style"], "first_gold_rank": r})
                                    .to_csv(index=False), "per_question_ranks.csv")
                rows.append({"max_words": words, "overlap_words": overlap, "retriever": m} | met)
                ranks[(words, overlap, m)] = r
            log.info("w%s o%s: %s", words, overlap,
                     {m: round(float(np.mean(r <= 5)), 3) for m, r in fh.items()})
        res = _select(pd.DataFrame(rows), ranks, current, sw["max_context_words"])
        table = pd.DataFrame(rows)
        table.to_csv(cfg.reports / "sweep.csv", index=False)
        (cfg.reports / "sweep.json").write_text(json.dumps(res, indent=2, default=float))
        _chart(table, cfg, current)
        mlflow.log_artifact(str(cfg.reports / "sweep.csv"))
        mlflow.log_artifact(str(cfg.reports / "sweep.json"))
        mlflow.log_artifact(str(cfg.reports / "figures" / "sweep.png"), "figures")
        mlflow.set_tags({"selected": res["best"]["name"], "recommendation": res["recommendation"]})
        mlflow.log_metrics({"best_recall_at_5": res["best"]["recall_at_5"],
                            "current_recall_at_5": res["current"]["recall_at_5"]})
        res["mlflow_run_id"] = parent.info.run_id
    return res


def _select(t: pd.DataFrame, ranks: dict, current: tuple, budget: int) -> dict:
    t = t.assign(name=t["max_words"].astype(str).radd("w") + "_o" + t["overlap_words"].astype(str) + "_"
                 + t["retriever"])
    ok = t[t["context_words_top_k"] <= budget]
    sort = [c for c in ("recall_at_5", "plain_recall_at_5", "mrr") if c in ok]
    best = ok.sort_values(sort, ascending=False).iloc[0]
    cur = t[(t["max_words"] == current[0]) & (t["overlap_words"] == current[1]) & (t["retriever"] == current[2])]
    if cur.empty:
        raise ValueError(f"current setting {current} is not in the sweep grid")
    cur = cur.iloc[0]
    key_b = (best["max_words"], best["overlap_words"], best["retriever"])
    ci = _boot_ci(ranks[key_b], ranks[current])
    if best["name"] == cur["name"]:
        rec = "keep current setting (it ranks first)"
    elif ci[0] > 0:
        rec = f"switch to {best['name']} (gain interval above zero)"
    else:
        rec = f"keep current setting ({best['name']} ranks first, but the gain is within noise)"
    cols = ["name", "recall_at_1", "recall_at_5", "mrr", "plain_recall_at_5", "chunks", "context_words_top_k"]
    cols = [c for c in cols if c in t]
    return {"best": best[cols].to_dict(), "current": cur[cols].to_dict(),
            "best_minus_current_recall_at_5": {"est": float(best["recall_at_5"] - cur["recall_at_5"]), "ci95": ci},
            "excluded_over_budget": t.loc[t["context_words_top_k"] > budget, "name"].tolist(),
            "recommendation": rec}


def _chart(t: pd.DataFrame, cfg: Config, current: tuple) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e4e3df",
                         "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    cols = {"bm25": "#eb6834", "dense": "#52514e", "hybrid": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, metric, title in ((axes[0], "recall_at_5", "All answerable (top 5)"),
                              (axes[1], "plain_recall_at_5", "Plain-language questions (top 5)")):
        if metric not in t:
            continue
        for m, g in t.groupby("retriever"):
            g = g.sort_values(["max_words", "overlap_words"])
            x = g["max_words"].astype(str) + "/" + g["overlap_words"].astype(str)
            ax.plot(x, 100 * g[metric], marker="o", label=m, color=cols.get(m))
        ax.set_title(title)
        ax.set_xlabel("chunk words / overlap")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("% with a gold section retrieved")
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, -0.04, f"Current setting: {current[0]}/{current[1]} {current[2]}", fontsize=8, color="#52514e")
    fig.savefig(cfg.reports / "figures" / "sweep.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

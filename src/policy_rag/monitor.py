"""Monitoring without labels: Evidently drift on per-query retrieval signals, logged to MLflow.

In use, nobody grades each answer, so quality can't be measured directly. What can be measured for every query is
how the query looks and how confident retrieval is. The evaluation showed why that matters: questions in the
manuals' wording were retrieved 100% of the time, plain-language ones 40-70%. A shift in the query mix toward
everyday wording or toward topics the manuals don't cover should show up in these signals before anyone notices
wrong answers.

Signals per query (no labels, no chat model):
  n_words, has_code            what the query looks like (CPT/HCPCS codes, modifiers, section numbers)
  oov_share                    share of query terms that never appear in the indexed manuals
  bm25_top1, dense_top1        best keyword and embedding match
  dense_margin                 gap between the best and fifth-best embedding match (flat = no clear winner)
  agreement                    overlap of the keyword and embedding top 5 (they disagree when unsure)
  sections_top5                distinct sections in the hybrid top 5 (scattered = no clear topic)

Steps:
  1. Check the signals are worth watching: on the labeled eval set, does each one separate retrieval misses and
     out-of-scope questions from the rest (AUC)? Signals that don't are reported, not trusted.
  2. Control: split the reference in two at random and run the same drift test. That is the false-alarm level with
     these sample sizes.
  3. Drift: reference = the evaluated questions, current = a traffic sample. Evidently's DataDriftPreset (K-S test
     for numeric columns, chi-square/Z-test for has_code at these sizes) per column; dataset drift if at least half
     the columns drift, so one noisy signal doesn't raise an alarm.
  4. Flags: current queries that fall below the reference's 10th percentile on at least two useful signals go to a
     review list (one signal alone fires too often to be worth a reviewer's time). Those
     are the queries to label and add to the eval set, which is how the system adapts: measure, then retune with
     `policy-rag sweep` on the larger set.
"""
from __future__ import annotations

import json
import logging
import re

import mlflow
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from . import answer as ans
from . import tracking
from .config import ROOT, Config
from .evaluate import first_hits, load_questions
from .retrieve import Index, tokenize

log = logging.getLogger(__name__)
CODE = re.compile(r"\b\d{5}\b|\b[A-V]\d{4}\b|modifier|§|\b\d{2,3}\.\d{1,2}\b", re.IGNORECASE)
NUMERIC = ["n_words", "oov_share", "bm25_top1", "dense_top1", "dense_margin", "agreement", "sections_top5"]
# direction in which a signal means "less confident": -1 = low is bad, +1 = high is bad
BAD = {"oov_share": 1, "bm25_top1": -1, "dense_top1": -1, "dense_margin": -1, "agreement": -1, "sections_top5": 1}


def signals(idx: Index, questions: pd.Series, k: int = 5) -> pd.DataFrame:
    vocab = set().union(*map(set, idx.bm25.toks))
    rows = []
    for q in questions:
        toks = tokenize(q)
        b = idx.bm25.scores(q)
        d = idx.rank_scores(q, "dense")
        bo, do = np.argsort(-b, kind="stable")[:k], np.argsort(-d, kind="stable")[:k]
        hy = idx.rank(q, "hybrid", k)
        rows.append({"n_words": len(q.split()), "has_code": int(bool(CODE.search(q))),
                     "oov_share": float(np.mean([t not in vocab for t in toks])) if toks else 1.0,
                     "bm25_top1": float(b[bo[0]]), "dense_top1": float(d[do[0]]),
                     "dense_margin": float(d[do[0]] - d[do[-1]]),
                     "agreement": len(set(bo) & set(do)) / k,
                     "sections_top5": hy["section"].nunique()})
    return pd.DataFrame(rows)


def _auc(bad: np.ndarray, score: np.ndarray) -> float:
    """P(score of a bad case > score of a good case); ties count half. 0.5 = no signal."""
    pos, neg = score[bad], score[~bad]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def validity(ref: pd.DataFrame, qs: pd.DataFrame, idx: Index) -> dict:
    """AUC of each signal (oriented so higher = less confident) for two failure types on the labeled set."""
    qa = qs["answerable"].to_numpy()
    miss = np.zeros(len(qs), bool)
    miss[qa] = first_hits(idx, qs[qa].reset_index(drop=True), ["hybrid"])["hybrid"] > 5
    out = {"n_answerable": int(qa.sum()), "n_retrieval_misses": int(miss.sum()), "n_out_of_scope": int((~qa).sum())}
    for c, sign in BAD.items():
        s = sign * ref[c].to_numpy(float)
        out[c] = {"auc_retrieval_miss": _auc(miss[qa], s[qa]), "auc_out_of_scope": _auc(~qa, s)}
    return out


def drift(ref: pd.DataFrame, cur: pd.DataFrame) -> tuple[dict, object]:
    dd = DataDefinition(numerical_columns=NUMERIC, categorical_columns=["has_code"])
    snap = Report([DataDriftPreset()]).run(current_data=Dataset.from_pandas(cur, data_definition=dd),
                                           reference_data=Dataset.from_pandas(ref, data_definition=dd))
    out = {"columns": {}}
    for m in snap.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            out["drifted_columns"], out["drift_share"] = int(m["value"]["count"]), float(m["value"]["share"])
            out["dataset_drift"] = out["drift_share"] >= cfg["drift_share"]
        elif cfg["type"].endswith("ValueDrift"):
            out["columns"][cfg["column"]] = {"method": cfg["method"], "p_value": float(m["value"]),
                                             "drifted": float(m["value"]) < cfg["threshold"]}
    return out, snap


def run(cfg: Config, llm, with_answers: bool = False, seed: int = 0) -> dict:
    mc = cfg["monitor"]
    idx = Index(cfg, llm)
    qs = load_questions(cfg)
    cur_q = pd.read_json(ROOT / mc["current"], lines=True)
    ref = signals(idx, qs["question"])
    cur = signals(idx, cur_q["question"])
    out_dir = cfg.reports / "monitoring"
    out_dir.mkdir(exist_ok=True)

    res = {"reference_queries": len(ref), "current_queries": len(cur), "signal_validity": validity(ref, qs, idx)}
    perm = np.random.default_rng(seed).permutation(len(ref))      # control: two halves of the same distribution
    res["control"], _ = drift(ref.iloc[perm[: len(ref) // 2]], ref.iloc[perm[len(ref) // 2:]])
    res["drift"], snap = drift(ref, cur)
    snap.save_html(str(out_dir / "drift_report.html"))
    res["means"] = {c: {"reference": float(ref[c].mean()), "current": float(cur[c].mean())}
                    for c in NUMERIC + ["has_code"]}

    # review list: below the reference's low-confidence threshold on any signal that proved useful
    useful = [c for c, v in res["signal_validity"].items() if isinstance(v, dict)
              and max(v["auc_retrieval_miss"], v["auc_out_of_scope"]) >= 0.65]
    q = mc["flag_quantile"]
    reasons = pd.Series([""] * len(cur))
    for c in useful:
        thr = ref[c].quantile(q if BAD[c] < 0 else 1 - q)
        bad = cur[c] < thr if BAD[c] < 0 else cur[c] > thr
        reasons[bad.to_numpy()] += f"{c} "
    flags = cur_q.assign(**cur, reasons=reasons.str.strip())
    flags["n_flags"] = flags["reasons"].str.split().str.len()
    if with_answers:                                          # optional: run the chat model too (slow)
        tracking.setup(cfg, "policy-rag-monitoring")
        replies = [ans.rag(idx, llm, cfg["chat_model"], x, cfg["answer_retriever"], cfg["top_k"])[0]
                   for x in cur_q["question"]]
        flags["abstained"] = [r["abstained"] for r in replies]
        flags["answer"] = [r["answer"] for r in replies]
        flags["trace_id"] = [r["trace_id"] for r in replies]
        res["abstain_rate"] = {"current": float(flags["abstained"].mean())}
        ref_ans = cfg.reports / "answers.csv"
        if ref_ans.exists():
            res["abstain_rate"]["reference"] = float(pd.read_csv(ref_ans)["abstained"].mean())
    flags.to_csv(out_dir / "traffic_signals.csv", index=False)
    review = flags[flags["n_flags"] >= mc["min_flags"]].sort_values("n_flags", ascending=False)
    review.to_csv(out_dir / "review_queue.csv", index=False)
    res["useful_signals"] = useful
    res["flagged_for_review"] = len(review)
    (out_dir / "monitor.json").write_text(json.dumps(res, indent=2, default=float))
    _chart(ref, cur, res, cfg)
    _validity_chart(res, cfg)

    tracking.setup(cfg, "policy-rag-monitoring")
    with mlflow.start_run(run_name=f"traffic vs eval set ({len(cur)} queries)") as r:
        mlflow.log_params({"reference": mc["reference"], "current": mc["current"], "flag_quantile": q,
                           "embed_model": cfg["embed_model"], "chunk_words": cfg["chunk"]["max_words"]}
                          | {"reference_sha": tracking.sha(cfg.path("eval_questions").read_bytes()),
                             "current_sha": tracking.sha((ROOT / mc["current"]).read_bytes())})
        mlflow.log_metrics({"drift_share": res["drift"]["drift_share"], "drifted_columns": res["drift"]["drifted_columns"],
                            "control_drift_share": res["control"]["drift_share"],
                            "flagged_for_review": len(review), "flagged_share": len(review) / max(len(cur), 1)}
                           | {f"p_value_{c}": v["p_value"] for c, v in res["drift"]["columns"].items()}
                           | {f"mean_{c}": v["current"] for c, v in res["means"].items()}
                           | ({"abstain_rate": res["abstain_rate"]["current"]} if "abstain_rate" in res else {}))
        mlflow.set_tag("dataset_drift", str(res["drift"]["dataset_drift"]))
        for f in ("drift_report.html", "monitor.json", "review_queue.csv", "traffic_signals.csv"):
            mlflow.log_artifact(str(out_dir / f), "monitoring")
        mlflow.log_artifact(str(cfg.reports / "figures" / "monitoring.png"), "figures")
        mlflow.log_artifact(str(cfg.reports / "figures" / "monitoring_signals.png"), "figures")
        res["mlflow_run_id"] = r.info.run_id
    return res


def _chart(ref: pd.DataFrame, cur: pd.DataFrame, res: dict, cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e4e3df",
                         "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    cols = ["oov_share", "dense_top1", "agreement", "sections_top5"]
    fig, axes = plt.subplots(1, len(cols), figsize=(11, 2.8))
    for ax, c in zip(axes, cols):
        ax.boxplot([ref[c], cur[c]], tick_labels=["eval set", "traffic"], widths=0.5,
                   medianprops={"color": "#2a78d6"})
        p = res["drift"]["columns"][c]["p_value"]
        ax.set_title(f"{c}\np = {p:.3f}", fontsize=9)
    fig.suptitle("Per-query retrieval signals: evaluated questions vs traffic sample", x=0.01, ha="left",
                 fontweight="bold", y=1.08)
    fig.savefig(cfg.reports / "figures" / "monitoring.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _validity_chart(res: dict, cfg: Config) -> None:
    """Which signals earn a place in monitoring: AUC on the labeled questions, with the 0.65 cut-off."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = res["signal_validity"]
    names = list(BAD)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.set_facecolor("#fcfcfb")
    fig.set_facecolor("#fcfcfb")
    ax.barh(y - 0.2, [v[c]["auc_retrieval_miss"] for c in names], 0.38, color="#2a78d6",
            label=f"retrieval miss ({v['n_retrieval_misses']} of {v['n_answerable']})")
    ax.barh(y + 0.2, [v[c]["auc_out_of_scope"] for c in names], 0.38, color="#eb6834",
            label=f"out of scope ({v['n_out_of_scope']})")
    ax.axvline(0.5, color="#b9b8b3", lw=1)
    ax.axvline(0.65, color="#2b2a27", lw=1, ls="--")
    ax.set_yticks(y, names)
    ax.set_ylim(len(names) - 0.5, -1.0)
    ax.text(0.655, -0.75, "used for flags", fontsize=7)
    ax.text(0.505, -0.75, "chance", fontsize=7, color="#7a7974")
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUC on the labeled questions (higher = signal separates the failure)")
    ax.set_title("Which retrieval signals predict a failure", loc="left", fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(cfg.reports / "figures" / "monitoring_signals.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

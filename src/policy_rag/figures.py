"""Figures for the write-up, built from the saved reports (no model calls): `policy-rag figures`.

  retrieval_by_style.png   recall@5 by question wording and retriever       -> which retriever, what to monitor
  answer_outcomes.png      answer grade split by whether retrieval found     -> where to invest next
                           the gold section
  rag_vs_closed_book.png   same model with and without the manuals           -> is retrieval worth it
  judge_calibration.png    LLM judge vs human labels                          -> can the judge grade at scale

Answer grades use the human label where one exists and the judge's grade otherwise ("final grade"), except in the
with/without comparison, where both sides are judge grades so they are measured the same way.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config

STYLES = {"exact": "Manual wording", "paraphrase": "Paraphrased", "plain": "Plain language"}
C = {"bm25": "#eb6834", "dense": "#52514e", "hybrid": "#2a78d6", "good": "#2a78d6", "partial": "#9cc3ee",
     "bad": "#eb6834", "miss": "#8c2d0a", "muted": "#b9b8b3", "ink": "#2b2a27"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e4e3df",
                         "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "font.size": 9})
    return plt


def graded(cfg: Config) -> pd.DataFrame:
    a = pd.read_csv(cfg.reports / "answers.csv")
    hp = cfg.path("human_labels")
    if hp.exists():
        h = pd.read_csv(hp).dropna(subset=["human_correct"])
        a = a.merge(h[["id", "answer", "human_correct"]], on=["id", "answer"], how="left")
    else:
        a["human_correct"] = np.nan
    a["final_correct"] = a["human_correct"].fillna(a["judge_correct"])
    return a


def retrieval_by_style(cfg: Config, plt) -> None:
    r = json.loads((cfg.reports / "eval.json").read_text())["retrieval"]["by_style"]
    qs = pd.read_json(cfg.path("eval_questions"), lines=True)
    n = qs[qs["answerable"]]["style"].value_counts()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    w = 0.26
    for i, m in enumerate(cfg["retrievers"]):
        vals = [100 * r[m][s] for s in STYLES]
        bars = ax.bar(np.arange(3) + (i - 1) * w, vals, w, label=m, color=C[m])
        ax.bar_label(bars, [f"{v:.0f}" for v in vals], fontsize=7, padding=2)
    ax.set_xticks(range(3), [f"{v}\n(n = {n.get(k, 0)})" for k, v in STYLES.items()])
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% with gold section in top 5")
    ax.set_title("Retrieval holds up on the manuals' wording, not on everyday wording")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    fig.savefig(cfg.reports / "figures" / "retrieval_by_style.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def answer_outcomes(a: pd.DataFrame, cfg: Config, plt) -> None:
    A = a[a["answerable"]]
    cats = [("Retrieval missed: wrong", lambda d: ~d["retrieved_gold"] & (d["final_correct"] < 0.5), C["miss"]),
            ("Retrieval missed: correct", lambda d: ~d["retrieved_gold"] & (d["final_correct"] >= 0.5), C["muted"]),
            ("Retrieved: wrong", lambda d: d["retrieved_gold"] & (d["final_correct"] == 0), C["bad"]),
            ("Retrieved: partial", lambda d: d["retrieved_gold"] & (d["final_correct"] == 0.5), C["partial"]),
            ("Retrieved: correct", lambda d: d["retrieved_gold"] & (d["final_correct"] == 1), C["good"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    rows = list(STYLES)
    for j, s in enumerate(rows):
        d = A[A["style"] == s]
        left = 0
        for label, f, col in cats:
            k = int(f(d).sum())
            if k:
                ax.barh(j, k, left=left, color=col, label=label if label not in ax.get_legend_handles_labels()[1]
                        else None, edgecolor="#fcfcfb")
                ax.text(left + k / 2, j, str(k), ha="center", va="center", fontsize=8,
                        color="white" if col in (C["miss"], C["good"], C["bad"]) else C["ink"])
            left += k
        score = 100 * d["final_correct"].mean()
        ax.text(left + 0.4, j, f"{score:.0f}% correct", va="center", fontsize=8, color=C["ink"])
    ax.set_yticks(range(len(rows)), [STYLES[s] for s in rows])
    ax.invert_yaxis()
    ax.set_xlabel("answerable questions")
    ax.set_xlim(0, A["style"].value_counts().max() + 7)
    ax.grid(axis="y", visible=False)
    ax.set_title("A retrieval miss always ended in a wrong answer; partial answers happen even when it hits")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.22))
    fig.text(0.01, -0.2, "Grade: human label where one exists (37 answers), otherwise the LLM judge.", fontsize=7,
             color="#52514e")
    fig.savefig(cfg.reports / "figures" / "answer_outcomes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def rag_vs_closed_book(a: pd.DataFrame, cfg: Config, plt) -> None:
    A, U = a[a["answerable"]], a[~a["answerable"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [3, 1.6]})
    x = np.arange(3)
    for i, (col, label, c) in enumerate((("judge_correct", "with the manuals", C["hybrid"]),
                                        ("closed_book_judge_correct", "model alone", C["muted"]))):
        vals = [100 * A.loc[A["style"] == s, col].mean() for s in STYLES]
        bars = ax1.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=c, label=label)
        ax1.bar_label(bars, [f"{v:.0f}" for v in vals], fontsize=7, padding=2)
    ax1.set_xticks(x, list(STYLES.values()))
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("% judged correct")
    ax1.set_title("Answerable questions")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    declined = [int(U["abstained"].sum()), int(U["closed_book_abstained"].sum())]
    answered = [len(U) - k for k in declined]
    ax2.bar([0, 1], declined, 0.55, color=C["good"], label="declined")
    ax2.bar([0, 1], answered, 0.55, bottom=declined, color=C["bad"], label="answered without a source")
    for i in (0, 1):
        ax2.text(i, declined[i] / 2, str(declined[i]), ha="center", va="center", color="white", fontsize=8)
        if answered[i]:
            ax2.text(i, declined[i] + answered[i] / 2, str(answered[i]), ha="center", va="center", color="white",
                     fontsize=8)
    ax2.set_xticks([0, 1], ["with the\nmanuals", "model\nalone"])
    ax2.set_title(f"Out of scope ({len(U)})")
    ax2.set_ylim(0, len(U) + 4)
    ax2.legend(frameon=False, fontsize=7, loc="upper center", ncol=1)
    ax2.grid(axis="x", visible=False)
    fig.suptitle("Retrieval's gain disappears on plain-language questions; declining out-of-scope ones holds",
                 x=0.01, ha="left", fontweight="bold", y=1.04)
    fig.savefig(cfg.reports / "figures" / "rag_vs_closed_book.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def judge_calibration(a: pd.DataFrame, cfg: Config, plt) -> None:
    d = a.dropna(subset=["human_correct"])
    if d.empty:
        return
    lv = [0.0, 0.5, 1.0]
    m = pd.crosstab(d["judge_correct"], d["human_correct"]).reindex(index=lv, columns=lv, fill_value=0).to_numpy()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=max(m.max(), 1) * 1.3)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=10,
                    color="white" if m[i, j] > m.max() * 0.6 else C["ink"])
    lenient = int(((d["judge_correct"] >= 0.5) & (d["human_correct"] < 0.5) & d["abstained"]).sum())
    if m[2, 0]:
        ax.add_patch(plt.Rectangle((-0.5, 1.5), 1, 1, fill=False, ec=C["bad"], lw=2))
        fig.text(0.12, -0.04, f"Outlined: judge passed, human failed. {lenient} of {m[2, 0]} are \"NOT IN DOCUMENTS\" "
                 "replies after a retrieval miss.", fontsize=7, color=C["bad"])
    ax.set_xticks(range(3), ["0", "0.5", "1"])
    ax.set_yticks(range(3), ["0", "0.5", "1"])
    ax.set_xlabel("human grade")
    ax.set_ylabel("judge grade")
    ax.grid(False)
    agree = float(((d["judge_correct"] >= 0.5) == (d["human_correct"] >= 0.5)).mean())
    ax.set_title(f"Judge vs human, correctness (n = {len(d)})\npass/fail agreement {100 * agree:.0f}%", fontsize=9)
    fig.savefig(cfg.reports / "figures" / "judge_calibration.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(cfg: Config) -> list[str]:
    plt = _style()
    a = graded(cfg)
    retrieval_by_style(cfg, plt)
    answer_outcomes(a, cfg, plt)
    if "closed_book_judge_correct" in a:
        rag_vs_closed_book(a, cfg, plt)
    judge_calibration(a, cfg, plt)
    return sorted(p.name for p in (cfg.reports / "figures").glob("*.png"))

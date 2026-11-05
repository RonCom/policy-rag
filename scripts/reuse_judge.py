"""Grade a new answers.csv, reusing judge grades from a graded one where the graded text is identical.

`eval --no-judge` rewrites reports/answers.csv without grades. At temperature 0 most answers come out word for word
the same as in the last graded run, and the judge's grade for identical inputs is already on file. This copies those
grades and sends only the changed answers to the judge model.

    uv run python scripts/reuse_judge.py --dry-run        # count reusable vs. changed rows, no model calls
    uv run python scripts/reuse_judge.py                  # copy grades, judge the rest, rewrite answers.csv + eval.json
    uv run python scripts/reuse_judge.py --old path.csv   # graded file to reuse (default: main:reports/answers.csv, the last graded run)
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys

import pandas as pd

from policy_rag import evaluate, tracking
from policy_rag.config import ROOT, Config
from policy_rag.llm import client

ANS = ["judge_correct", "judge_faithful", "judge_reason"]
CB = [f"closed_book_{c}" for c in ANS]


def graded_old(path: str | None) -> pd.DataFrame:
    if path:
        return pd.read_csv(path)
    out = subprocess.run(["git", "show", "main:reports/answers.csv"], cwd=ROOT, capture_output=True, check=True)
    return pd.read_csv(io.BytesIO(out.stdout))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", help="graded answers.csv to reuse (default: main:reports/answers.csv)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake", action="store_true", help="fake judge, for testing")
    a = ap.parse_args()
    cfg = Config.load()
    new = pd.read_csv(cfg.reports / "answers.csv")
    if "judge_correct" in new:
        sys.exit("reports/answers.csv already has judge grades; nothing to do.")
    old = graded_old(a.old)
    key = ["id", "answer", "context"]
    m = new.merge(old[key + ANS], on=key, how="left")
    m = m.merge(old[["id", "closed_book_answer"] + CB], on=["id", "closed_book_answer"], how="left")
    todo_a, todo_cb = m["judge_correct"].isna(), m["closed_book_judge_correct"].isna()
    print(f"answers: {int((~todo_a).sum())} reused, {int(todo_a.sum())} to judge | "
          f"closed-book: {int((~todo_cb).sum())} reused, {int(todo_cb.sum())} to judge")
    if a.dry_run:
        print("changed answer ids:", ", ".join(m.loc[todo_a, "id"]) or "none")
        return
    llm = client(cfg, a.fake)
    for i, r in m.iterrows():
        if not (todo_a[i] or todo_cb[i]):
            continue
        q = type("Q", (), {"question": r.question, "reference": r.reference,
                           "key_facts": [k for k in str(r.key_facts).split("; ") if k and k != "nan"]})
        if todo_a[i]:
            g = evaluate.judge(llm, cfg["judge_model"], q, r.answer, "" if pd.isna(r.context) else r.context)
            m.loc[i, ANS] = [g["judge_correct"], g["judge_faithful"], g["judge_reason"]]
        if todo_cb[i]:
            g = evaluate.judge(llm, cfg["judge_model"], q, r.closed_book_answer, "")
            m.loc[i, CB] = [g["judge_correct"], g["judge_faithful"], g["judge_reason"]]
        print(f"judged {r.id}")
    tracking.setup(cfg, "policy-rag")                       # put the grades on this run's traces
    for r in m.itertuples():
        for name in ("correct", "faithful"):
            tracking.feedback(r.trace_id, f"judge_{name}", getattr(r, f"judge_{name}"), "LLM_JUDGE",
                              cfg["judge_model"], r.judge_reason)
    m = evaluate.code_rules(m)
    m.to_csv(cfg.reports / "answers.csv", index=False)
    ev = json.loads((cfg.reports / "eval.json").read_text())
    ev["answers"] = evaluate.summarize(m)
    (cfg.reports / "eval.json").write_text(json.dumps(ev, indent=2, default=float))
    print(json.dumps(ev["answers"], indent=2, default=float))


if __name__ == "__main__":
    main()

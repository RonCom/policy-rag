"""MLflow helpers: one local store, lineage parameters that make runs comparable, and trace feedback.

Decisions (see README, "MLOps"):
- Local SQLite store (mlflow.db), no tracking server: one person, one laptop; `uv run mlflow ui --backend-store-uri
  sqlite:///mlflow.db` reads it. A server adds nothing until several people share runs.
- No model registry: nothing is trained here. The "model" is a set of Ollama tags, a prompt and a chunking setting;
  those are logged as parameters, and the prompt and eval set as hashes, so two runs are only compared when they
  answered the same questions with the same prompt.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess

import mlflow
from mlflow.entities import AssessmentSource

from .config import ROOT, Config

log = logging.getLogger(__name__)


def setup(cfg: Config, experiment: str) -> None:
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(experiment)


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:12]


def lineage(cfg: Config) -> dict:
    """What a metric depends on besides the model: the question set and the prompts."""
    from . import answer, evaluate
    qfile = cfg.path("eval_questions")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
                                timeout=5).stdout.strip() or "none"
    except (OSError, subprocess.SubprocessError):
        commit = "none"
    return {"eval_set_sha": sha(qfile.read_bytes()), "eval_questions": sum(1 for _ in open(qfile, encoding="utf-8")),
            "answer_prompt_sha": answer.PROMPT_SHA, "judge_prompt_sha": sha(evaluate.JUDGE_SYSTEM),
            "git_commit": commit}


def feedback(trace_id, name: str, value, source: str, source_id: str, rationale: str | None = None) -> None:
    """Attach a score to a trace (source: HUMAN, LLM_JUDGE or CODE). Best effort: never fails an eval."""
    if not isinstance(trace_id, str) or not trace_id.startswith("tr-"):
        return
    try:
        mlflow.log_feedback(trace_id=trace_id, name=name, value=value, rationale=rationale,
                            source=AssessmentSource(source_type=source, source_id=source_id))
    except Exception as e:                                      # noqa: BLE001 - feedback is optional
        log.debug("feedback on %s failed: %s", trace_id, e)

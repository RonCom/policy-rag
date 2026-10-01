import numpy as np
import pandas as pd

from policy_rag import monitor, sweep


def test_auc_orientation():
    bad = np.array([True, True, False, False])
    assert monitor._auc(bad, np.array([0.9, 0.8, 0.1, 0.2])) == 1.0
    assert monitor._auc(bad, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_code_pattern():
    assert monitor.CODE.search("Can 97110 and 97140 be billed together?")
    assert monitor.CODE.search("when is modifier 59 used")
    assert not monitor.CODE.search("does medicare pay for acupuncture")


def _table(r5_best: float):
    rows = [{"max_words": 300, "overlap_words": 40, "retriever": "hybrid", "recall_at_1": 0.7, "recall_at_5": 0.9,
             "mrr": 0.8, "chunks": 851, "context_words_top_k": 1400},
            {"max_words": 150, "overlap_words": 30, "retriever": "hybrid", "recall_at_1": 0.7,
             "recall_at_5": r5_best, "mrr": 0.8, "chunks": 1600, "context_words_top_k": 700},
            {"max_words": 800, "overlap_words": 80, "retriever": "hybrid", "recall_at_1": 0.9, "recall_at_5": 1.0,
             "mrr": 0.95, "chunks": 444, "context_words_top_k": 3900}]
    return pd.DataFrame(rows)


def test_selection_respects_budget_and_noise():
    cur = np.array([1.0] * 45 + [np.inf] * 5)
    ranks = {(300, 40, "hybrid"): cur, (800, 80, "hybrid"): np.ones(50)}
    ranks[(150, 30, "hybrid")] = cur.copy()
    ranks[(150, 30, "hybrid")][45] = 1.0                       # one more question found: within noise
    res = sweep._select(_table(0.92), ranks, (300, 40, "hybrid"), budget=3000)
    assert res["best"]["name"] == "w150_o30_hybrid"            # w800 is over the context budget
    assert "w800_o80_hybrid" in res["excluded_over_budget"]
    assert res["recommendation"].startswith("keep current")


def test_drift_flags_a_shift_but_not_a_random_split():
    rng = np.random.default_rng(0)

    def frame(n, shift):
        d = {c: rng.normal(shift, 1, n) for c in monitor.NUMERIC}
        return pd.DataFrame(d | {"has_code": rng.integers(0, 2, n)})
    ref = frame(80, 0)
    assert monitor.drift(ref, frame(40, 1.5))[0]["dataset_drift"]
    assert not monitor.drift(ref.iloc[:40], ref.iloc[40:])[0]["dataset_drift"]


def test_fixed_windows_label_every_section_they_overlap():
    from policy_rag import ablation
    chunks = pd.DataFrame({"doc": ["D", "D"], "section": ["§1", "§2"], "heading": ["", ""], "part": [0, 0],
                           "text": [" ".join(["a"] * 8), " ".join(["b"] * 8)]})
    _, keys = ablation.fixed_windows(chunks, 6, 2)
    assert {"D §1", "D §2"} in keys and all(k for k in keys)

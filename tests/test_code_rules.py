import numpy as np
import pandas as pd

from policy_rag import evaluate
from policy_rag.answer import declined


def _graded():
    return pd.DataFrame({
        "id": ["a1", "a2", "a3", "u1", "a4"], "answerable": [True, True, True, False, True],
        "answer": ["Bill 3 units [1].", "NOT IN DOCUMENTS", "Use modifier 59 [2].", "NOT IN DOCUMENTS",
                   "The excerpts do not provide a specific answer. However, a related rule says X [1]."],
        "abstained": [False, True, False, True, False],             # as written before prose detection existed
        "judge_correct": [1.0, 1.0, 0.5, 1.0, 1.0], "judge_faithful": [1.0, 0.0, 0.0, 0.0, 0.0],
        "closed_book_answer": ["Wrong guess.", "Another guess.", "NOT IN DOCUMENTS", "I don't know.", "Guess."],
        "closed_book_abstained": [False, False, True, False, False],
        "closed_book_judge_correct": [0.0, 1.0, 1.0, 1.0, 0.0], "closed_book_judge_faithful": [1.0] * 5})


def test_declined_detects_exact_and_prose_in_first_sentence_only():
    assert declined("NOT IN DOCUMENTS") == "exact"
    assert declined("The provided excerpts do not contain the 85% rate. However, ...") == "prose"
    assert declined("This is not addressed in the documents.") == "prose"
    assert declined("I don't know the current threshold.") == "prose"
    assert declined("Bill 3 units [1]. The excerpts do not specify the 4-unit threshold.") == ""
    assert declined("The KX modifier attests medical necessity [2].") == ""


def test_code_rules_score_declines_zero_and_exact_declines_faithful():
    g = evaluate.code_rules(_graded())
    assert g["abstain_kind"].tolist() == ["", "exact", "", "exact", "prose"]
    assert g["abstained"].tolist() == [False, True, False, True, True]
    assert g["correct_strict"].tolist() == [1.0, 0.0, 0.5, 1.0, 0.0]   # a2 and the prose decline a4 score 0
    assert g["closed_book_correct_strict"].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert g["judge_faithful"].tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]   # only exact declines are forced faithful
    assert g["judge_faithful_raw"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert g["judge_correct"].tolist() == [1.0, 1.0, 0.5, 1.0, 1.0]    # the judge's grade is kept as graded
    assert evaluate.code_rules(g).equals(g)                             # idempotent


def test_summary_reports_strict_beside_judged():
    a = _graded().assign(retrieved_gold=True, cites_gold=True)
    s = evaluate.summarize(a)
    assert np.isclose(s["judge_correct_answerable"], 3.5 / 4) and np.isclose(s["correct_strict_answerable"], 1.5 / 4)
    assert s["judge_faithful_overridden_on_declines"] == 2 and s["prose_declines"] == 1
    assert np.isclose(s["abstained_on_answerable"], 2 / 4)

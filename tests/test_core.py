import pandas as pd

from policy_rag import answer, ingest
from policy_rag.llm import Fake
from policy_rag.retrieve import BM25, tokenize


def test_sections_use_contents_order_and_skip_cross_references():
    lines = [(1, "Table of Contents"), (1, "10 - General"), (1, "20 - Units"), (1, "20.1 - Timed Codes"),
             (1, "30 - Payment"), (1, "40 - Other"), (1, "50 - More"),
             (2, "10 - General"), (2, "word " * 20),
             (3, "20 - Units"), (3, "Units are counted per code. " * 3),
             (3, "20.1 - Timed Codes"), (3, "Bill one unit for 8 to 22 minutes. " * 3),
             (3, "10 - General"),                       # a cross-reference back to an earlier section
             (3, "still part of 20.1 " * 5),
             (4, "30 - Payment"), (4, "Payment text here. " * 5)]
    secs = ingest.sections(lines, "iom")
    assert [s["section"] for s in secs] == ["§10", "§20", "§20.1", "§30"]
    assert "still part of 20.1" in " ".join(secs[2]["lines"])


def test_split_words_overlap():
    parts = ingest.split_words(" ".join(str(i) for i in range(700)), 300, 40)
    assert len(parts) == 3 and parts[1].split()[0] == "260"


def test_bm25_prefers_exact_terms():
    docs = ["modifier 59 distinct procedural service", "plan of care certification 90 days", "timed codes units"]
    s = BM25(docs).scores("When is modifier 59 used?")
    assert s.argmax() == 0
    assert "the" not in tokenize("The 8-minute rule")


def test_answer_parses_citations_and_abstention():
    hits = pd.DataFrame({"chunk_id": [5, 9], "doc": ["NCCI Ch. 1", "NCCI Ch. 1"], "section": ["Sec. V", "Sec. E"],
                         "heading": ["MUEs", "Modifiers"], "page": [28, 14], "text": ["a", "b"]})

    class Stub(Fake):
        def chat(self, model, system, user, schema=None):
            return "An MUE is the maximum units [1]. Modifiers [2] [7]."
    r = answer.answer(Stub(), "m", "q", hits)
    assert r["cited_chunks"] == [5, 9] and not r["abstained"]

    class Abstain(Fake):
        def chat(self, model, system, user, schema=None):
            return "NOT IN DOCUMENTS"
    assert answer.answer(Abstain(), "m", "q", hits)["abstained"]

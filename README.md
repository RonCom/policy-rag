# Policy RAG: cited answers from CMS billing-policy manuals, with a measured evaluation

A reviewer working a payment-integrity flag ("units per patient-day above the MUE", "timed therapy units averaging
under 15 minutes") needs the policy that applies, with a citation they can check. This project answers billing-policy
questions from four CMS manuals using **local models (Ollama)**, cites the manual section for every claim, says
**"NOT IN DOCUMENTS"** when the manuals don't cover a question, and **measures** all of that rather than eyeballing
a few answers.

It complements the [Medicare FWA outlier project](https://github.com/RonCom/medicare-fwa): that project flags
providers whose billing is unusual; this one retrieves the rules a reviewer would check those flags against.

## Corpus

| Document | Why it is here | Link |
|---|---|---|
| NCCI Policy Manual, Ch. 1 (2026): general correct coding | Unbundling, modifiers 25/59/X{EPSU}, MUEs, add-on codes | [PDF](https://www.cms.gov/files/document/01-chapter1-ncci-medicare-policy-manual-2026-final.pdf) |
| NCCI Policy Manual, Ch. 11 (2026): medicine and E&M | Physical medicine and rehabilitation, chiropractic, E&M edits | [PDF](https://www.cms.gov/files/document/11-chapter11a-ncci-medicare-policy-manual-2026-final.pdf) |
| Medicare Benefit Policy Manual, Ch. 15 | Coverage of therapy (§220–230): plans of care, certification, documentation | [PDF](https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/bp102c15.pdf) |
| Medicare Claims Processing Manual, Ch. 5 | Outpatient rehab billing: timed units and the 8-minute rule (§20.2), KX modifier, MPPR | [PDF](https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/clm104c05.pdf) |

Save the four PDFs to `data/docs/` (git-ignored).

## How it works

1. **Section-aware chunking** (`ingest.py`). The manuals have their own structure: NCCI uses lettered sections
   ("V. Medically Unlikely Edits"), the CMS manuals numbered ones ("§20.2 Reporting of Service Units"). Chunks follow
   that structure (about 300 words, 40-word overlap), so every answer can cite a section a reviewer can look up.
   Running headers, page labels, tables of contents and revision history are removed. A line counts as a
   section heading only if it is one of the next few entries in the manual's table of contents; this stops
   cross-references such as "see 220.2 - Reasonable and Necessary ...)" from splitting a section. The result is
   851 chunks across 304 sections.
2. **Three retrievers** (`retrieve.py`): BM25 (exact tokens such as CPT codes, "modifier 59", "8 minutes"), dense
   embeddings (`nomic-embed-text` via Ollama, for paraphrased questions), and hybrid (reciprocal rank fusion).
   Vectors live in DuckDB.
3. **Grounded answers** (`answer.py`): a local model answers only from the numbered excerpts, cites each claim
   `[n]`, and replies `NOT IN DOCUMENTS` when the excerpts don't cover the question.
4. **Evaluation** (`evaluate.py`), on 52 questions in `eval/questions.jsonl`: 40 answerable, each with a reference
   answer, key facts and gold manual sections, and 12 that the corpus does not answer (Part D penalties, MS-DRG
   weights, MIPS thresholds…). For each question it records:
   - **retrieval:** recall@k and MRR for each retriever, with paired bootstrap intervals;
   - **abstention:** declines on unanswerable questions, and false declines on answerable ones;
   - **citations:** whether the answer cites a gold section (deterministic, no model involved);
   - **LLM-as-a-judge:** correctness against the key facts, and faithfulness to the excerpts, scored by a larger
     local model than the one answering;
   - **closed-book baseline:** the same model with no excerpts, to show what retrieval adds and how often the
     model invents an answer.

   Runs are logged to MLflow.
5. **Judge calibration** (`policy-rag calibrate`): an LLM judge is only useful if it agrees with a person. `eval`
   writes `eval/human_labels.csv`; label a sample yourself (`human_correct` 0/0.5/1, `human_faithful` 0/1) and
   `calibrate` reports agreement and Cohen's kappa.

The questions were drafted from the manual text (with Claude) and checked against the source sections; each
unanswerable topic was confirmed absent from the corpus by searching the chunk text.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
ollama pull nomic-embed-text
ollama pull gemma4:e4b            # answers (fits 8 GB of VRAM)
# judge: gemma4:26b (config.yaml); any larger local model works

uv sync
uv run policy-rag index           # parse, chunk, embed -> data/index.duckdb
uv run policy-rag ask "How many units can be billed for 40 minutes of 97110 and 97140?"
uv run policy-rag eval --no-judge # answers + retrieval, abstention, citations (no judge model)
uv run policy-rag eval --judge-only  # grade those saved answers with the judge model (slow on 8 GB VRAM)
uv run policy-rag calibrate       # after labeling eval/human_labels.csv
uv run pytest -q
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`--fake` runs any command without Ollama (hash embeddings, canned replies), for tests and dry runs.

## Results

Models: `nomic-embed-text` (embeddings), `gemma4:e4b` (answers), `gemma4:26b` (judge), all local through Ollama.
62 questions: 50 answerable (27 phrased like the manuals, 13 paraphrased, 10 in plain reviewer language) and 12 the
manuals do not answer. Retrieval returns the top 5 excerpts.

**Retrieval** (50 answerable questions; a hit = a chunk from a gold section in the top k)

| Retriever | Top 1 | Top 5 | MRR | Top 5, manual wording (40) | Top 5, plain language (10) |
|---|---|---|---|---|---|
| BM25 | 76% | 88% | 0.81 | 100% | 40% |
| Dense (`nomic-embed-text`) | 70% | 90% | 0.79 | 95% | **70%** |
| **Hybrid (RRF)** | **76%** | **92%** | **0.83** | 100% | 60% |

Questions written in the manuals' own words are easy for keyword search; plain-language questions are where it
breaks (40%) and embeddings earn their place (70%). Hybrid keeps BM25's precision on exact terms and most of the
dense gain: +4 points over BM25 at top 5 (95% CI 0 to +10) on 50 questions, so the ranking among methods is
suggestive rather than settled. Ten plain-language questions is the main gap in the test set.

**Answers** (hybrid retrieval)

| | With retrieval | Same model, no documents |
|---|---|---|
| Judged correct, answerable questions | **84%** | 50% |
| Declined the 12 questions the manuals do not answer | **12 / 12** | 9 / 12 (answered 3 from memory, unsourced) |
| Declined an answerable question | 3 / 50 | – |
| Cited a gold section when answering | 89% | – |
| Judged faithful to the excerpts | 100% | – |

Retrieval is what makes the model usable: it raises correctness from 50% to 84%, and every claim it makes is tied to
a cited section. Without the documents, the model answered 3 out-of-scope questions from general knowledge, with
nothing a reviewer could check.
Most remaining errors trace to retrieval misses on plain-language questions (e.g. "does Medicare stop paying once the
patient stops improving?" retrieved a notification section instead of §220.2's maintenance rule, and the model
answered "yes"), or to answers that leave out a secondary key fact (the 14-day rule for verbal certifications).

**Judge calibration** (37 answers labeled by hand, chosen to include errors, partial answers and declines)

| | Agreement with human | Cohen's kappa | Judge pass rate | Human pass rate |
|---|---|---|---|---|
| Correctness | 92% (34 / 37) | **0.68** | 86% | 84% |
| Faithfulness | 97% (36 / 37) | not informative* | 97% | 100% |

\* Every labeled answer was faithful by the human rubric, so kappa has no variance to measure; agreement is the
useful number. The correctness judge is trustworthy enough to report (kappa above 0.6), with one systematic
leniency worth knowing: it accepted two "NOT IN DOCUMENTS" replies because the retrieved excerpts lacked the answer,
although the manuals contain it. The judge grades against what was retrieved, so retrieval misses can look like
correct declines; the deterministic "retrieved a gold section" check catches those cases.

## Limits

- Four chapters, not the full manuals; a production version would index all relevant chapters, LCDs and articles,
  and refresh them when CMS revises a manual.
- 52 questions is a small test set; differences between retrievers come with wide intervals.
- An LLM judge can be wrong in systematic ways; that is why calibration against human labels is part of the run.
- Answers quote policy; they are not legal or billing advice.

## License

Code: MIT. CMS manuals are U.S. government works; CPT codes and descriptions are © American Medical Association.

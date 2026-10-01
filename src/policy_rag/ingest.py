"""PDF manuals -> section-aware chunks with page numbers, embedded and stored in DuckDB.

Chunks follow the manuals' own structure (NCCI: lettered sections "A. Introduction"; CMS Internet-Only Manuals:
numbered sections "220.2 - Reasonable and Necessary..."), so an answer can cite a section a reviewer can look up.
Long sections are split into ~300-word windows with overlap. Running headers, page labels and table-of-contents
lines are dropped.
"""
from __future__ import annotations

import logging
import re

import duckdb
import pandas as pd
import pymupdf

from .config import Config

log = logging.getLogger(__name__)

DROP = [re.compile(p) for p in (
    r"^\s*Revision Date", r"^\s*[IVXL]+-\d+\s*$", r"^\s*\d+\s*$", r"\.{4,}",
    r"^\s*Medicare (Benefit Policy|Claims Processing) Manual\s*$", r"^\s*Chapter \d+ - .{0,120}$",
    r"^\s*Table of Contents", r"^\s*\(Rev\. [^)]*\)\s*$")]
NCCI_HEAD = re.compile(r"^([A-Z])\.\s+([A-Z][^\n]{2,150})$")
EXHIBIT = re.compile(r"^(Exhibit \d+)\s*[-–]\s*(.{3,150})$")
IOM_HEAD = re.compile(r"^(\d{1,3}(?:\.\d{1,2}){0,4})\s*[-–]\s*([A-Z(][^\n]{2,200})$")


def page_lines(path) -> list[tuple[int, str]]:
    out = []
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            for line in page.get_text().splitlines():
                s = line.strip()
                if s and not any(p.search(s) for p in DROP):
                    out.append((pno, s))
    return out


def sections(lines: list[tuple[int, str]], style: str, toc_check: bool = True) -> list[dict]:
    """Split on section headings.

    CMS Internet-Only Manuals open with a table of contents (pages made mostly of heading lines). In the body a
    line counts as a heading only if it is one of the next few entries in that contents list, so cross-references
    ("see 220.2 - Reasonable and Necessary ...)") and lists of section titles inside a paragraph are not mistaken
    for headings. NCCI chapters use lettered headings; their dotted table of contents is dropped earlier, and a
    letter seen twice keeps the copy with more text.
    """
    head = NCCI_HEAD if style == "ncci" else IOM_HEAD
    by_page: dict[int, list[str]] = {}
    for pno, s in lines:
        by_page.setdefault(pno, []).append(s)
    toc_pages = set()                                     # the run of contents pages at the start of the manual
    for p in sorted(by_page):
        ls = by_page[p]
        if len(ls) > 5 and sum(bool(head.match(x)) for x in ls) >= 0.4 * len(ls):
            toc_pages.add(p)
        elif toc_pages:
            break
    order = []                                            # section numbers in table-of-contents order
    for p in sorted(toc_pages):
        for x in by_page[p]:
            m = head.match(x)
            if m and m.group(1) not in order:
                order.append(m.group(1))
    pos = {k: i for i, k in enumerate(order)}
    secs, cur, last = [], {"section": "front", "heading": "", "page": 1, "lines": []}, -1
    for pno, s in lines:
        if style == "iom" and pno in toc_pages:
            continue
        if s.startswith("Transmittals Issued for this Chapter"):   # revision history at the end: not policy
            break
        ex = EXHIBIT.match(s) if style == "iom" else None
        if ex:
            secs.append(cur)
            cur = {"section": ex.group(1), "heading": ex.group(2).strip(), "page": pno, "lines": []}
            continue
        m = head.match(s)
        ok = bool(m) and not s.endswith((",", ";")) and len(s.split()) <= 25
        if ok and style == "iom" and toc_check:           # must be one of the next few entries in the contents
            i = pos.get(m.group(1), -1)
            ok = last < i <= last + 4
            if ok:
                last = i
        if ok:
            secs.append(cur)
            cur = {"section": ("§" if style == "iom" else "Sec. ") + m.group(1), "heading": m.group(2).strip(),
                   "page": pno, "lines": []}
        else:
            cur["lines"].append(s)
    secs.append(cur)
    best = {}
    for sc in secs:
        k = sc["section"]
        if k not in best or len(sc["lines"]) > len(best[k]["lines"]):
            best[k] = sc
    return [sc for sc in secs if best[sc["section"]] is sc and len(" ".join(sc["lines"]).split()) >= 15]


def split_words(text: str, max_words: int, overlap: int) -> list[str]:
    w = text.split()
    if len(w) <= max_words:
        return [text]
    step = max_words - overlap
    return [" ".join(w[i:i + max_words]) for i in range(0, max(1, len(w) - overlap), step)]


def chunk_docs(cfg: Config, max_words: int | None = None, overlap: int | None = None,
               toc_check: bool = True) -> pd.DataFrame:
    max_words = max_words or cfg["chunk"]["max_words"]
    overlap = cfg["chunk"]["overlap_words"] if overlap is None else overlap
    rows = []
    for fname, label in cfg["docs"].items():
        path = cfg.path("docs_dir") / fname
        if not path.exists():
            log.warning("missing %s (see README for the download link)", path)
            continue
        style = "ncci" if "ncci" in fname.lower() else "iom"
        for sc in sections(page_lines(path), style, toc_check):
            text = re.sub(r"\s+", " ", " ".join(sc["lines"])).strip()
            for j, piece in enumerate(split_words(text, max_words, overlap)):
                rows.append({"doc": label, "file": fname, "section": sc["section"], "heading": sc["heading"],
                             "page": sc["page"], "part": j, "text": piece})
    df = pd.DataFrame(rows)
    df.insert(0, "chunk_id", range(len(df)))
    return df


def build(cfg: Config, llm, max_words: int | None = None, overlap: int | None = None, db=None) -> pd.DataFrame:
    df = chunk_docs(cfg, max_words, overlap)
    emb = llm.embed((df["heading"] + ". " + df["text"]).tolist(), "document")
    df["embedding"] = list(emb)
    con = duckdb.connect(str(db or cfg.db_path))
    con.execute("DROP TABLE IF EXISTS chunks")
    con.register("df", df)
    con.execute("CREATE TABLE chunks AS SELECT * FROM df")
    con.close()
    log.info("indexed %d chunks from %d documents (%s)", len(df), df["doc"].nunique(),
             df.groupby("doc").size().to_dict())
    return df

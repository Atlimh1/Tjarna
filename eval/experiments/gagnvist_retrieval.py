"""Table search on the organizers' 60 questions (gagnvist-starter questions/examples.jsonl), ours vs theirs.

    .venv/bin/python eval/experiments/gagnvist_retrieval.py [path/to/gagnvist-starter]

A question counts at top-k when every part's table (any table in its `accepted` list) is in the top k.
Hard questions (two tables) are split at ", og hv..." / ", and what/how" and each half searched; ours only.
The starter searches bge-m3 vectors of its cards (title + dimension names), filtered to the question's
language, and answers from the top-1 table.

Results 2026-10-30 (catalog crawled ~22 Oct; 10 questions per cell, so one question = 10 points):
                     top1          top3          top8
                     is    en      is    en      is    en
    starter easy      5     1       6     1       8     3
            medium    2     1       3     1       7     1
            hard      0     0       1     0       1     0
    checked easy      7     7       8     9      10     9
            medium    5     5       9     9       9     9
            hard      3     3       5     6       9     7      (split into halves)
    lemma / embed / fusion: easy top1 14 / 6 / 13 of 20, medium 8 / 3 / 9, hard 5 / 2 / 4.
Starter top1 6/60, ours 30/60. Starter English is weak (cards searched in English, 1/10 easy).
Our misses at #1 are look-alike tables, not wrong subjects: wage index LAU04000 loses to LAU04007/04800/04001
(ranks 8-13), GDP THJ01102 to THJ01000 (3-4), immigrants MAN43000 to MAN43006/43005 (2-3),
CPI VIS01000 to VIS02500 (HICP, 2), Reykjavík population MAN02005 to MAN030101/MAN03270 (2-3).
The set uses only ~10 distinct tables, so fixes tuned on it overfit easily: check them on dev too.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hagstofa.retrieve import build  # noqa: E402

STARTER = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/ext/gagnvist-starter")
SPLIT_RE = re.compile(r",? og (?=hv)|,? and (?=what|how)")
QUESTIONS = [json.loads(line) for line in open(STARTER / "questions/examples.jsonl", encoding="utf-8")]


def report(name, ranked):
    """ranked(question) -> one rank per part (None = not in the top 20)."""
    print(f"\n== {name}")
    for level in ["easy", "medium", "hard"]:
        for lang in ["is", "en"]:
            rows = [ranked(q) for q in QUESTIONS if q["level"] == level and q["language"] == lang]
            at = lambda k: sum(all(r and r <= k for r in ranks) for ranks in rows)  # noqa: E731
            print(f"  {level:6} {lang}: top1 {at(1)}/{len(rows)}  top3 {at(3)}  top8 {at(8)}  top20 {at(20)}")


def ours(retriever):
    def ranked(q):
        halves = SPLIT_RE.split(q["question"]) if q["level"] == "hard" else [q["question"]]
        lists = [[t["id"] for t, _ in retriever.search(h, k=20)] for h in halves]
        return [min((i + 1 for ids in lists for i, t in enumerate(ids) if t in p["accepted"]), default=None)
                for p in q["parts"]]
    return ranked


def starter():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    cards = [json.loads(line) for line in open(STARTER / "index/cards.jsonl", encoding="utf-8")]
    vectors = np.load(STARTER / "index/vectors.npy")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    model = SentenceTransformer("BAAI/bge-m3")
    qvec = dict(zip((q["id"] for q in QUESTIONS),
                    model.encode([q["question"] for q in QUESTIONS], normalize_embeddings=True)))

    def ranked(q):
        order = [i for i in (vectors @ qvec[q["id"]]).argsort()[::-1] if cards[i]["language"] == q["language"]][:20]
        ids = [cards[i]["folder"].rsplit("/", 1)[1] for i in order]
        return [next((i + 1 for i, t in enumerate(ids) if t in p["accepted"]), None) for p in q["parts"]]
    return ranked


if __name__ == "__main__":
    report("starter (bge-m3 cards)", starter())
    for name in ["lemma", "embed", "fusion", "checked"]:
        report(name, ours(build(name)))

"""Experiment (2026-10-24): boost tables whose VALUE labels contain the question's words.
Idea: "Kópavogi", "makríl", "Neskaupstað" are values inside tables, not title words, so a title-only
index can't see them. An inverted index (value lemma -> tables), weighted by rarity (IDF), could boost them.

Result on the 36 test questions (top1 / top8), k=8:
    lemma                         20 / 33     fusion                          24 / 34
    lemma+val (all words)         21 / 32     fusion+val (boost after fusion) 17 / 32   <- scale mismatch
    fusion(lemma+val, embed)      24 / 34     fusion(lemma+val, embed+val)    20 / 32
    lemma+places (Greynir only)   20 / 33     fusion(lemma+places, embed)     23 / 34
Verdict: rejected; no gain in any form. Why:
  - boosting after rank fusion: RRF scores sit in a narrow band, so any multiplier overrides the ranking
  - all-word boost: function words (var, er, úr, mikill...) happen to be value labels somewhere -> noise
  - places only: few questions involve places, and the hard part was the topic (population), not the town;
    173 tables contain "kópavogur", so the boost can't single one out
Better use of the signal: at table-picking time, tell the LLM which candidates contain the place.

    .venv/bin/python eval/experiments/value_boost.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from hagstofa import icelandic, retrieve as R  # noqa: E402
from run_eval import evaluate_retrieval, load_questions, summary  # noqa: E402


class ValueIndex:
    MAX_DF = 300  # values in more tables than this ("alls", "kona") say nothing about which table

    def __init__(self, tables):
        index = {}
        for i, t in enumerate(tables):
            for v in t["variables_is"]:
                for text in v["valueTexts"]:
                    for lemma in R.lemmas(text):
                        index.setdefault(lemma, set()).add(i)
        self.n = len(tables)
        self.tables_with = {w: np.fromiter(s, int) for w, s in index.items() if len(s) <= self.MAX_DF}

    def idf(self, word):
        return np.log(self.n / len(self.tables_with[word])) / np.log(self.n)


class ValueBoosted(R._Base):
    """score * (1 + alpha * sum of idf of the question's rare values that the table contains)"""

    def __init__(self, base, values, alpha=0.5, places_only=False):
        super().__init__(base.tables)
        self.base, self.values, self.alpha, self.places_only = base, values, alpha, places_only
        self.weights = np.ones(len(self.tables))

    def scores(self, question):
        words = ({p.lower() for p in icelandic.entities(question)["places"]} if self.places_only
                 else set(R.lemmas(question)))
        b = np.ones(len(self.tables))
        for w in words & set(self.values.tables_with):
            b[self.values.tables_with[w]] += self.alpha * self.values.idf(w)
        return self.base.scores(question) * self.base.weights * b


if __name__ == "__main__":
    qs = load_questions()
    lem, emb = R.build("lemma"), R.build("embed")
    vals = ValueIndex(lem.tables)
    variants = {
        "lemma": lem, "fusion": R.build("fusion"),
        "lemma+val": ValueBoosted(lem, vals),
        "fusion+val (after fusion)": ValueBoosted(R.build("fusion"), vals),
        "fusion(lemma+val, embed)": R.FusionRetriever([ValueBoosted(lem, vals), emb]),
        "lemma+places": ValueBoosted(lem, vals, places_only=True),
        "fusion(lemma+places, embed)": R.FusionRetriever([ValueBoosted(lem, vals, places_only=True), emb]),
    }
    for name, r in variants.items():
        print(f"{name:30s} {summary(evaluate_retrieval(r, qs, 8), 8)}")

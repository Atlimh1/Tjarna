"""Experiment (2026-10-31): reject off-topic input by the table search's best score (no model, ~40 ms).
Idea: a question about statistics matches some table well; "segðu mér brandara" matches none.

Result, lemma top-1 cosine (correct=False):
    real questions   hand min 0.145 | dev min 0.068, p5 0.145 | dev_typo min 0.037, p5 0.096 | muni min 0.214
    off-topic (16)   0.040 - 0.194  (joke 0.065, poem 0.040, "what time is it" 0.159, capital of France 0.194)
Verdict: rejected. The ranges overlap: a threshold that catches half the off-topic inputs (0.12) blocks
~3% of dev and ~10% of dev_typo. Off-topic is about meaning, not word overlap; the fusion/checked scores
are rank-based (RRF), so they say even less. Off-topic needs a model: guard.in_scope() (one LLM call).

    .venv/bin/python eval/experiments/guard_scope_score.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hagstofa.retrieve import build  # noqa: E402

EVAL = ROOT / "eval"


def main():
    r = build("lemma")
    top = lambda q: r.search(q, k=1, correct=False)[0][1]  # noqa: E731
    for name in ["questions.jsonl", "questions_dev.jsonl", "questions_dev_typo.jsonl", "questions_muni.jsonl"]:
        s = np.array([top(json.loads(line)["q"]) for line in open(EVAL / name, encoding="utf-8")])
        print(f"{name:28} min {s.min():.3f}  p5 {np.percentile(s, 5):.3f}  median {np.median(s):.3f}")
    for line in open(EVAL / "questions_guard.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row["label"] == "offtopic":
            print(f"  offtopic {top(row['q']):.3f}  {row['q']}")


if __name__ == "__main__":
    main()

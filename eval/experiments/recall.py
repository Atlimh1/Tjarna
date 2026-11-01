"""Shared by the language experiments (accents.py, query_rewrite.py, greynir_audit.py): the question sets and
recall @1/@3/@8/@12 for any search function. Not an experiment itself."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CUTS = (1, 3, 8, 12)


def load(name):
    return [json.loads(line) for line in open(ROOT / "eval" / f"questions_{name}.jsonl", encoding="utf-8") if line.strip()]


def question_sets():
    """dev (decide on it), the no-accent and the slip halves of dev_typo, and the user's real questions."""
    typo = load("dev_typo")
    return {"dev": load("dev"),
            "no accents": [q for q in typo if q.get("typo_style") == "noaccent"],
            "slips": [q for q in typo if q.get("typo_style") == "typo"],
            "real": load("real")}


def accepted(item):
    return item["table"] if isinstance(item["table"], list) else [item["table"]]


def rank(ids, item):
    return next((i + 1 for i, tid in enumerate(ids) if tid in accepted(item)), None)


def recall(ranks):
    """'@1/@3/@8/@12' as whole percentages."""
    n = max(len(ranks), 1)
    return "/".join(f"{100 * sum(r is not None and r <= c for r in ranks) / n:.0f}" for c in CUTS)


def fuse(rankings, c=60, k=12):
    """Reciprocal rank fusion of several result lists [(table, score), ...] (the FusionRetriever idea): a table
    gets sum(1 / (c + rank)) over the lists it is in."""
    scores, tables = {}, {}
    for ranking in rankings:
        for r, (t, _) in enumerate(ranking):
            key = (t["id"], t["path"])
            scores[key] = scores.get(key, 0.0) + 1 / (c + r)
            tables[key] = t
    return [(tables[key], scores[key]) for key in sorted(scores, key=scores.get, reverse=True)[:k]]

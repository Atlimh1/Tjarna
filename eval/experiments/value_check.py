"""Experiment (2026-10-25): does hagstofa/checker.py (places/countries/years must exist in the table)
improve the fusion ranking? Reorders the top-N fusion results, scores top1/top3/top8.

Results (dev): first version (all capitalised words, years only in time values) top1 42 -> 46%, but it
rejected the right table in 36/252 questions (person names, "Iceland's", years written in the title).
Final version (BÍN place-like domains only, title years and ranges): 6/252 false rejects.
    fusion       top1 42%  top3 66%  top8 82%
    check top8   top1 50%  top3 69%  top8 82%
    check top20  top1 51%  top3 71%  top8 87%   <- adopted as retriever "checked" (pool=20), now DEFAULT
Confirmed once on test: fusion 44/66/83 -> checked 50/70/85. Hand set: 44/67/83 -> 53/78/92.

    .venv/bin/python eval/experiments/value_check.py -q dev
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from hagstofa.checker import missing, rerank  # noqa: E402
from hagstofa.retrieve import build  # noqa: E402
from run_eval import accepted, load_questions  # noqa: E402


def rank_of(ids, accept):
    return next((i + 1 for i, t in enumerate(ids) if t in accept), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--questions", default="dev")
    args = ap.parse_args()
    r = build("fusion")
    qs = load_questions(args.questions)
    ranks = {"fusion": [], "check top8": [], "check top20": []}
    checked = rejected_right = 0
    for item in qs:
        acc = accepted(item)
        c20 = r.search(item["q"], k=20)
        ranks["fusion"].append(rank_of([t["id"] for t, _ in c20], acc))
        ranks["check top8"].append(rank_of([t["id"] for t, _ in rerank(item["q"], c20[:8]) + c20[8:]], acc))
        ranks["check top20"].append(rank_of([t["id"] for t, _ in rerank(item["q"], c20)], acc))
        right = [t for t, _ in c20 if t["id"] in acc]
        if any(missing(item["q"], t) is not None for t, _ in c20[:1]) and right:
            checked += 1
            m = missing(item["q"], right[0])
            if m:
                rejected_right += 1
                print(f'right table fails check: {item["q"]}\n   {right[0]["id"]} missing {m}')
    n = len(qs)
    for name, rk in ranks.items():
        top = lambda j: sum(x is not None and x <= j for x in rk) / n  # noqa: E731
        print(f"{name:12s} top1 {top(1):.0%}  top3 {top(3):.0%}  top8 {top(8):.0%}")
    print(f"right table in top20 but fails the check: {rejected_right}/{checked}")


if __name__ == "__main__":
    main()

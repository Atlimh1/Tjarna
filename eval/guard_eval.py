"""Score the input guard (hagstofa/guard.py). No LLM, no network, a few seconds.

    .venv/bin/python eval/guard_eval.py            # bad inputs + false rejects on the tuning sets
    .venv/bin/python eval/guard_eval.py --confirm  # also the held-back sets (test, real): run rarely
    LLM_BASE_URL=... LLM_MODEL=... .venv/bin/python eval/guard_eval.py --llm   # + the LLM scope check
                                                   # (one call per input that passes check(); ~1,000 calls)

Two numbers matter:
  false rejects  real questions the guard blocks (every question set we have; target 0: a guard that
                 turns away real users is worse than none)
  caught         bad inputs per category (eval/questions_guard.jsonl, label = the category we expect;
                 "ok" items are odd but valid questions and broken encodings that must be repaired)
Tune on the tuning sets; test and real only confirm.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hagstofa import guard, language  # noqa: E402

EVAL = Path(__file__).resolve().parent
TUNING = ["questions.jsonl", "questions_dev.jsonl", "questions_dev_typo.jsonl", "questions_muni.jsonl",
          "questions_e2e.jsonl", "questions_lang.jsonl"]
CONFIRM = ["questions_test.jsonl", "questions_real.jsonl"]


def load(name):
    return [json.loads(line) for line in open(EVAL / name, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="also the held-back sets (test, real)")
    ap.add_argument("--llm", action="store_true", help="also guard.classify() (LLM) on what passes check()")
    ap.add_argument("--limit", type=int, help="at most this many questions per real set (a quicker --llm run)")
    ap.add_argument("--show", type=int, default=10, help="examples to print per kind of error")
    args = ap.parse_args()
    guard.english_vocabulary()  # build once, outside the timing

    def verdict(q):
        v = guard.check(q)
        if v.ok and args.llm:  # the LLM classifier the front door runs after the rules (guard.classify)
            category = guard.classify(v.text)
            if category != "statistics":
                v.category = category if category in guard.MESSAGES else "offtopic"
        return v

    pool = ThreadPoolExecutor(16 if args.llm else 1)  # vLLM answers many requests at once

    def verdicts_of(texts):
        return list(pool.map(verdict, texts))

    print("False rejects on real questions (should all be 0)")
    for name in TUNING + (CONFIRM if args.confirm else []):
        rows = load(name)[:args.limit]
        bad = [(r["q"], v) for r, v in zip(rows, verdicts_of([r["q"] for r in rows])) if not v.ok]
        flagged = [r["q"] for r in rows if guard.check(r["q"]).flags]
        print(f"  {name:28} {len(bad):3} of {len(rows)}   (opinion note on {len(flagged)})")
        for q, v in bad[:args.show]:
            print(f"      {v.category:10} {q}")
        for q in flagged[:3]:
            print(f"      opinion    {q}")

    print("\nLanguage (guard.check(...).lang): every question's own language; Danish/Norwegian count as one")
    for name in TUNING + (CONFIRM if args.confirm else []):
        rows = [r for r in load(name) if r.get("lang")]
        wrong = [(r["lang"], v.lang, r["q"]) for r in rows if (v := guard.check(r["q"])).lang != r["lang"]
                 and {v.lang, r["lang"]} != {"da", "no"}]
        if rows:
            print(f"  {name:28} {len(rows) - len(wrong):3} of {len(rows)} right")
        for want, got, q in wrong[:args.show]:
            print(f"      {want} -> {got}  {q}")

    print("\nSeveral questions in one message (language.split_questions): parts and their languages")
    rows = load("questions_mixed.jsonl")
    wrong = [r["q"] for r in rows if [(q, language.detect(q)) for q in language.split_questions(r["q"])]
             != [(p["q"], p["lang"]) for p in r["parts"]]]
    print(f"  questions_mixed.jsonl        {len(rows) - len(wrong):3} of {len(rows)} right")
    for q in wrong[:args.show]:
        print(f"      {q!r}")
    split = sum(len(language.split_questions(r["q"])) != 1 for name in TUNING for r in load(name))
    print(f"  real questions split in two  {split:3} (should be 0)")

    rows = load("questions_guard.jsonl")
    started = time.perf_counter()
    verdicts = verdicts_of([r["q"] for r in rows])
    ms = (time.perf_counter() - started) * 1000 / len(rows)
    matrix = defaultdict(Counter)
    misses = []
    for r, v in zip(rows, verdicts):
        got = v.category
        if r["label"] == "ok" and "fixed" in r and v.text != r["fixed"]:
            got = "not repaired"
        if got == "ok" and r.get("flag") and r["flag"] not in v.flags:
            got = "no flag"
        if got == "ok" and v.flags and not r.get("flag"):
            got = "flagged " + ",".join(v.flags)
        matrix[r["label"]][got] += 1
        if got != r["label"]:
            misses.append((r["label"], got, r["q"]))
    print("\nBad inputs (questions_guard.jsonl): expected -> what the guard said")
    total = right = 0
    for label in sorted(matrix):
        c = matrix[label]
        n, ok = sum(c.values()), c[label]
        total, right = total + n, right + ok
        others = ", ".join(f"{k} {v}" for k, v in c.most_common() if k != label)
        print(f"  {label:10} {ok:3}/{n:<3} {others}")
    print(f"  {'all':10} {right:3}/{total:<3} ({ms:.2f} ms per input)")
    if misses:
        print("\nMisses:")
        for label, got, q in misses[:args.show * 3]:
            print(f"  expected {label:9} got {got:12} {q[:80]!r}")


if __name__ == "__main__":
    main()

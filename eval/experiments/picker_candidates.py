"""Two tests of the table pick (search -> the LLM picks among the top k), one change at a time.

    LLM_BASE_URL=... LLM_MODEL=gemma LLM_API_KEY=... .venv/bin/python eval/experiments/picker_candidates.py --test 1
    ... --test 2 [--rewrite]          # run test 2 after test 1, with the language settings test 1 chose
    ... --test 0                      # search only, no LLM: how often the right table is in the top 8 vs top 25

Test 1, the language changes of 31 Oct, with k = 8 options for the picker throughout:
    before        ACCENTS=0, no rewrite (the pipeline before the language experiments)
    accents       ACCENTS=1 (on by default since fa945ab)
    + rewrite     ACCENTS=1 and QUERY_REWRITE (the split call's clean version searched too, RRF)
Test 2, more options for the picker, with the language settings fixed (ACCENTS=1; --rewrite adds the rewrite if
test 1 showed it helps):
    k = 8         the picker sees the search's top 8 (today)
    k = 25        the picker sees the top 25
The pick runs exactly as in pipeline.answer(): fresh_first() drops stale tables when no period is asked, the
villages note, and the search's #1 when the model declines.

Per question set (dev = decide; no accents, slips, real = confirm): "in top k" = the right table is among the
options (the most the picker can get), "picked" = the model chose a right table, seconds per pick. 270 questions:
noise about +-3 points. Answers are listed where the two arms differ.
Not run yet (31 Oct): needs the LLM (JURECA).

RESULTS
  Test 0 (31 Oct, search only, ACCENTS=1, checked): the right table among the options, 8 vs 25
      dev 86 -> 94%, no accents 86 -> 94%, slips 86 -> 93%, real 64 -> 73% (7 of 11 -> 8 of 11)
    So 25 options could add up to 8 points; tests 1 and 2 (with the LLM) show how much of that the picker takes.
  Test 1, dev (31 Oct 19:50, Gemma 4 31B on JURECA, 270 questions, k = 8):
                  in top 8   picked
      before         86%     61% (165)
      accents        86%     61% (165)   dev is written with accents: restoration never runs, nothing breaks
      + rewrite      86%     61% (164)   4 picks changed, 1 right -> wrong (THJ05511 -> THJ05532), 3 wrong both ways
    Rewriting doesn't help on clean questions (it fixes spelling and abbreviations; dev has neither), so test 2
    runs without it. The big loss is the pick itself: right table among the 8 in 86%, picked in 61%.
    Confirmation sets (no accents, slips, real): (fill in)
  Test 2: (fill in from a JURECA run)
"""
import argparse
import os
import sys
import time

from recall import ROOT, question_sets, rank

sys.path.insert(0, str(ROOT))
from hagstofa import pipeline, retrieve  # noqa: E402
from hagstofa.retrieve import build, places_of  # noqa: E402


def options(retriever, question, k, rewrite):
    clean = None
    if rewrite:
        pairs = pipeline.decompose_rewrite(question)
        clean = next((c for _, c in pairs if c), None)
    return pipeline.search_with_rewrite(retriever, question, clean, k=k)


def pick(retriever, question, found):
    fresh = pipeline.fresh_first(question, found)
    places = places_of(retriever)
    villages = places.villages(question) if places is not None else []
    table = pipeline.pick_table(question, fresh, villages) if found else None
    return table or (fresh[0][0] if fresh else None)


def run_arm(name, items, retriever, k, accents, rewrite, llm=True):
    retrieve.ACCENTS = accents
    in_top = picked = 0
    seconds, results = [], []
    for item in items:
        found = options(retriever, item["q"], k, rewrite)
        ok_top = rank([t["id"] for t, _ in found], item) is not None
        in_top += ok_top
        got = None
        if llm:
            t = time.time()
            table = pick(retriever, item["q"], found)
            seconds.append(time.time() - t)
            got = table and table["id"]
            picked += got in (item["table"] if isinstance(item["table"], list) else [item["table"]])
        results.append(got)
    n = max(len(items), 1)
    line = f"  {name:<14} in top {k:<2} {100 * in_top / n:3.0f}%"
    if llm:
        line += f"   picked {100 * picked / n:3.0f}% ({picked}/{len(items)})   {sum(seconds) / n:.1f} s per pick"
    print(line, flush=True)
    return results


def differences(items, a, b, label_a, label_b):
    rows = [(i["q"], x, y, i["table"]) for i, x, y in zip(items, a, b) if x != y]
    for q, x, y, want in rows[:20]:
        print(f"      {label_a} {x}  {label_b} {y}  (right: {want})  {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--rewrite", action="store_true", help="test 2: with QUERY_REWRITE too")
    ap.add_argument("--sets", help="comma-separated: dev, no accents, slips, real (default: all)")
    ap.add_argument("--limit", type=int, help="first N questions per set (a quick check)")
    args = ap.parse_args()
    sets = question_sets()
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets.split(",")}
    retriever = build("checked")  # the default `bilingual` adds a translation call for English questions only
    print(f"test {args.test}; model {os.environ.get('LLM_MODEL', '(llm.py default)')}")
    for name, items in sets.items():
        items = items[:args.limit] if args.limit else items
        print(f"\n{name} (n={len(items)})")
        if args.test == 0:
            for k in (8, 25):
                run_arm(f"k = {k}", items, retriever, k, "1", False, llm=False)
        elif args.test == 1:
            before = run_arm("before", items, retriever, 8, "0", False)
            accents = run_arm("accents", items, retriever, 8, "1", False)
            rewrite = run_arm("+ rewrite", items, retriever, 8, "1", True)
            differences(items, before, accents, "before", "accents")
            differences(items, accents, rewrite, "accents", "+rewrite")
        else:
            k8 = run_arm("k = 8", items, retriever, 8, "1", args.rewrite)
            k25 = run_arm("k = 25", items, retriever, 25, "1", args.rewrite)
            differences(items, k8, k25, "k=8", "k=25")


if __name__ == "__main__":
    main()

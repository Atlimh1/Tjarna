"""Questions in other languages, end to end with a real LLM: does a foreign question get the same answer as its
English original? No answer key needed: the English original's answer is the reference.

    LLM_BASE_URL=... LLM_MODEL=... .venv/bin/python eval/lang_eval.py [--lang pl --lang lt] [--limit 2]

Per question (eval/questions_lang.jsonl: 6 questions in 15 languages + 2 in 4 more, each with its English original):
  table     the first table used is the same as for the English original
  numbers   the English answer to the translated question states the same numbers as the original's answer
  back      the answer in the user's language kept every number (else the user got English + a note)
Results: eval/results/lang-<time>.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import ask  # noqa: E402
from hagstofa import llm, pipeline, translate  # noqa: E402
from hagstofa.retrieve import DEFAULT, build  # noqa: E402


def first_table(r):
    parts = r.get("parts") or []
    return parts[0]["table"]["id"] if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", action="append", help="only these languages")
    ap.add_argument("--limit", type=int, help="questions per language")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(ROOT / "eval/questions_lang.jsonl", encoding="utf-8")]
    if args.lang:
        rows = [r for r in rows if r["lang"] in args.lang]
    if args.limit:
        seen = defaultdict(int)
        rows = [r for r in rows if (seen.__setitem__(r["lang"], seen[r["lang"]] + 1) or seen[r["lang"]]) <= args.limit]
    retriever = build(DEFAULT)

    reference = {}
    for en in dict.fromkeys(r["en"] for r in rows):
        reference[en] = pipeline.answer(en, retriever)
        print(f"  reference  {first_table(reference[en])}  {reference[en]['answer'][:100]}")

    results, totals = [], defaultdict(lambda: defaultdict(int))
    for r in rows:
        started = time.perf_counter()
        try:
            checked, got = ask(r["q"], retriever)
        except Exception as e:  # noqa: BLE001
            got, checked = {"answer": f"error {type(e).__name__}: {e}"}, None
        ref = reference[r["en"]]
        row = {**r, "detected": checked.lang if checked else None, "question_en": got.get("question_en"),
               "answer": got["answer"], "answer_en": got.get("answer_en"), "seconds": round(time.perf_counter() - started, 1),
               "table": first_table(got), "reference_table": first_table(ref),
               "same_table": first_table(got) == first_table(ref),
               "same_numbers": set(translate.numbers(ref["answer"])) <= set(translate.numbers(got.get("answer_en") or "")),
               "back": "changed a number" not in got["answer"]}
        results.append(row)
        for key in ("same_table", "same_numbers", "back"):
            totals[r["lang"]][key] += row[key]
        totals[r["lang"]]["n"] += 1
        mark = "ok  " if row["same_table"] and row["same_numbers"] and row["back"] else "DIFF"
        print(f"{mark} {r['lang']}  {row['seconds']:5}s  {r['q'][:50]!r} -> {row['question_en']!r}\n"
              f"          {row['answer'][:140]!r}")

    print("\nlanguage  table  numbers  back   (of n)")
    for lang, t in totals.items():
        print(f"  {lang:6}  {t['same_table']:5}  {t['same_numbers']:7}  {t['back']:4}   {t['n']}")
    n = len(results)
    print(f"  all     {sum(r['same_table'] for r in results):5}  {sum(r['same_numbers'] for r in results):7}  "
          f"{sum(r['back'] for r in results):4}   {n}")
    out = ROOT / "eval/results" / time.strftime("lang-%Y-%m-%d-%H%M%S.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": llm.MODEL, "results": results}, ensure_ascii=False, indent=1))
    print(f"\nEvery answer is saved in {out}")


if __name__ == "__main__":
    main()

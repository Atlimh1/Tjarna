"""Calibrates the confidence shown under every answer: how often each level (confidence.level) was right, scored
by the organizers' evaluate() on questions with known answers, through the front door as users get them.

    LLM_BASE_URL=... LLM_MODEL=... python eval/confidence_eval.py            # their 60 + our 41 new, prints the table
    python eval/confidence_eval.py --write                                   # also writes data/confidence.json

The chat then says e.g. "Confidence: high (right 85% of the time on 70 test questions)". The 60 are questions our
picker rules were shaped on, so the rates are reported per set too; the file holds the two sets together.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from gagnvist_eval import OurPipeline, load_gagnvist, starter_path  # noqa: E402

from hagstofa import confidence, llm  # noqa: E402
from hagstofa.retrieve import DEFAULT, build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write data/confidence.json")
    ap.add_argument("-r", "--retriever", default=DEFAULT)
    args = ap.parse_args()
    starter = starter_path()
    gv = load_gagnvist(starter)
    from gagnvist.answering.pipeline import AnswerError

    sets = {"organizers' 60": gv.read_questions(starter / "questions/examples.jsonl"),
            "our 41 new": gv.read_questions(ROOT / "eval/questions_new.jsonl")}
    pipe = OurPipeline(build(args.retriever), AnswerError, front_door=True)
    tally = {name: defaultdict(lambda: [0, 0]) for name in [*sets, "together"]}
    by_agreement = defaultdict(lambda: [0, 0])
    for name, questions in sets.items():
        print(f"== {name}: {len(questions)} questions", flush=True)
        for o in gv.evaluate(pipe, questions, lambda o: print(gv.outcome_line(o), flush=True)):
            result = pipe.results.get(o.question.question)
            lvl = confidence.level(result) if result else None
            for key in (name, "together"):
                tally[key][lvl or "none"][0] += o.correct
                tally[key][lvl or "none"][1] += 1
            if result:
                by_agreement[confidence.signals(result)["agreement"]][0] += o.correct
                by_agreement[confidence.signals(result)["agreement"]][1] += 1
    print("\nRight answers by confidence level (none = no table: the guard stopped it or no data)")
    for name, t in tally.items():
        cells = "  ".join(f"{lvl} {r}/{n} ({round(100 * r / n)}%)" for lvl in ("high", "medium", "low", "none")
                          if (r := t[lvl][0]) is not None and (n := t[lvl][1]))
        print(f"  {name:16s} {cells}")
    print("By search agreement (how many of the three searches put the table first):")
    for a in sorted(by_agreement, key=lambda x: -1 if x is None else x):
        r, n = by_agreement[a]
        print(f"  {a}: {r}/{n} ({round(100 * r / n)}%)")
    if args.write:
        out = {lvl: {"right": tally["together"][lvl][0], "n": tally["together"][lvl][1]}
               for lvl in ("high", "medium", "low") if tally["together"][lvl][1]}
        out["model"] = llm.MODEL  # the rates hold for this model only (confidence.describe)
        confidence.CALIBRATION.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"written {confidence.CALIBRATION}: {out}")


if __name__ == "__main__":
    main()

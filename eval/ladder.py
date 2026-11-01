"""Ablation ladder: the same questions answered by the pipeline built up one step at a time, so the write-up can
say what each design choice adds. Scored with the organizers' evaluate() (see gagnvist_eval.py), same model for all.

    .venv/bin/python eval/ladder.py                         # every rung on their 60 questions (~2 h with Ollama on a small GPU)
    .venv/bin/python eval/ladder.py --rungs 0,2,6 --limit 6 # a quick look: three rungs, first 6 questions

Rungs (each adds one thing to the one above it):
    0 no-data   the LLM alone, no table: what it remembers or invents
    1 embed     bge-m3 search only (close to the starter kit), first table, no calculation, no split
    2 checked   our search (fusion + places + checker + spelling), first table
    3 +pick     the LLM picks among the top 8
    4 +calc     the LLM names an operation, code computes      (their "medium" questions)
    5 +split    two sub-questions for two tables               (their "hard" questions)
    6 +english  answers in the question's language = the full pipeline

"right" is their verdict (every number stated AND from an accepted table). "numbers" counts answers that state
every expected number from any table or none: for rung 0 it shows how often memory alone gets it right.
60 questions: one question = 1.7 points, so differences of 2-3 questions between rungs can be chance.
Results: eval/results/ladder-<time>/ (each rung's answers, their format) and ladder.md there.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gagnvist_eval import ROOT, Answer, OurPipeline, load_gagnvist, starter_path  # noqa: E402

from hagstofa import pipeline  # noqa: E402
from hagstofa.retrieve import build  # noqa: E402

RUNGS = [  # (name, retriever, steps); retriever None = no data
    ("no-data", None, None),
    ("embed", "embed", frozenset()),
    ("checked", "checked", frozenset()),
    ("+pick", "checked", frozenset({"pick"})),
    ("+calc", "checked", frozenset({"pick", "calc"})),
    ("+split", "checked", frozenset({"pick", "calc", "split"})),
    ("+english", "checked", frozenset({"pick", "calc", "split", "english"})),
    ("+check", "checked", pipeline.STEPS),
]


class NoData:
    """The LLM alone, in the shape their evaluate() expects: no tables, so never "right", but "numbers" shows
    whether the remembered figure happened to be the expected one."""

    def __init__(self, error_type):
        self.error_type = error_type

    def answer(self, question):
        try:
            return Answer(question, pipeline.answer_without_data(question), (), [])
        except Exception as e:  # noqa: BLE001
            raise self.error_type(f"{type(e).__name__}: {e}") from e


def row(name, outcomes):
    """One line of the ladder table: right per level and language, numbers stated, errors, seconds."""
    def right(pred):
        return sum(o.correct for o in outcomes if pred(o.question))
    def n(pred):
        return sum(1 for o in outcomes if pred(o.question))
    cells = [f"{right(lambda q, lv=lv: q.level == lv)}/{n(lambda q, lv=lv: q.level == lv)}"
             for lv in ("easy", "medium", "hard")]
    cells += [f"{right(lambda q, lg=lg: q.language == lg)}/{n(lambda q, lg=lg: q.language == lg)}" for lg in ("is", "en")]
    total = right(lambda q: True)
    numbers = sum(o.numbers_right for o in outcomes)
    errors = sum(o.answer is None for o in outcomes)
    seconds = sum(o.seconds for o in outcomes) / max(len(outcomes), 1)
    return (f"| {name:<9} | " + " | ".join(f"{c:>6}" for c in cells)
            + f" | **{total}/{len(outcomes)}** | {numbers} | {errors} | {seconds:5.1f} |")


HEADER = ("| rung      |   easy | medium |   hard |     is |     en | right | numbers | errors | s/question |\n"
          "|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("questions", nargs="?", help="questions file (default: their questions/examples.jsonl)")
    ap.add_argument("--starter", help="path to the gagnvist-starter clone")
    ap.add_argument("--rungs", help="comma-separated rung numbers (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N questions (a quick check)")
    ap.add_argument("--language", choices=["is", "en"])
    ap.add_argument("--level", action="append", choices=["easy", "medium", "hard"])
    args = ap.parse_args()

    starter = starter_path(args.starter)
    gv = load_gagnvist(starter)
    from gagnvist.answering.pipeline import AnswerError

    path = Path(args.questions) if args.questions else starter / "questions/examples.jsonl"
    questions = [q for q in gv.read_questions(path)
                 if (not args.language or q.language == args.language) and (not args.level or q.level in args.level)]
    questions = questions[:args.limit] if args.limit else questions
    chosen = [int(i) for i in args.rungs.split(",")] if args.rungs else range(len(RUNGS))
    out = ROOT / "eval/results" / time.strftime("ladder-%Y-%m-%d-%H%M%S")
    out.mkdir(parents=True)
    settings = {k: os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_REASONING")}
    lines = [f"# Ablation ladder, {time.strftime('%Y-%m-%d %H:%M')}",
             f"{len(questions)} questions from {path}; settings {json.dumps(settings)}", "", HEADER]
    print(f"{len(questions)} questions, rungs {list(chosen)}, results in {out}\n{HEADER}", flush=True)
    for i in chosen:
        name, retriever, steps = RUNGS[i]
        runner = NoData(AnswerError) if retriever is None else OurPipeline(build(retriever), AnswerError, steps)
        outcomes = gv.evaluate(runner, questions)
        gv.save(outcomes, out / f"{i}-{name.lstrip('+')}.json")
        lines.append(row(f"{i} {name}", outcomes))
        print(lines[-1], flush=True)
        (out / "ladder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")  # after every rung: a crash keeps the rest


if __name__ == "__main__":
    main()

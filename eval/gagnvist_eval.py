"""Score our pipeline with the organizers' own `gagnvist evaluate` (their questions, their scoring code, unchanged).

    .venv/bin/python eval/gagnvist_eval.py                       # their 60 questions
    .venv/bin/python eval/gagnvist_eval.py --language is --level easy
    .venv/bin/python eval/gagnvist_eval.py --only reykjavik my-questions.jsonl

Needs a clone of github.com/SyedMuhammadAzhar/gagnvist-starter: --starter PATH, or GAGNVIST_STARTER, or
../gagnvist-starter next to this repo. Its data is not needed: we read numbers live from PX-Web (their expected
numbers come from a snapshot of 18 Oct 2026, so a revised figure counts as wrong). Needs the LLM (llm.py env vars).

Their evaluate() calls pipeline.answer(question) and reads .text, .tables[i].table_code and .facts[i].text, so
the adapter below wraps our pipeline.answer() in that shape. Only the model's answer goes in .text: the source
and the rows go in .facts, else the scorer would find the expected number in our raw data and count it as stated.
Every part's table goes in .tables, which is how their scorer checks the two tables of a hard question.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hagstofa import pipeline  # noqa: E402
from hagstofa.retrieve import DEFAULT, build  # noqa: E402


def starter_path(arg=None):
    return Path(arg or os.environ.get("GAGNVIST_STARTER") or ROOT.parent / "gagnvist-starter").resolve()


def load_gagnvist(starter):
    """Their package straight from the clone (needs only httpx and numpy, which we have)."""
    if not (starter / "src/gagnvist/evaluate.py").is_file():
        sys.exit(f"gagnvist-starter not found at {starter}: pass --starter or set GAGNVIST_STARTER")
    sys.path.insert(0, str(starter / "src"))
    import gagnvist.evaluate
    return gagnvist.evaluate


@dataclass(frozen=True)
class Table:
    table_code: str
    title: str


@dataclass(frozen=True)
class Fact:
    text: str


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    tables: tuple
    facts: list


def facts_of(part):
    """One line per row read, like their Fact.text: "Tafla: Kyn: Alls; Ár: 2026 = 394324"."""
    lines = [f'{part["table"]["id"]}: ' + "; ".join(f"{k}: {v}" for k, v in row.items() if k != "value")
             + f' = {row["value"]}' for row in part["rows"][:pipeline.MAX_ROWS]]
    if part.get("calculation"):
        lines.append(f'{part["table"]["id"]}: {part["calculation"]["text"]}')
    return [Fact(line) for line in lines]


class OurPipeline:
    """Our pipeline in the shape their evaluate() expects."""

    def __init__(self, retriever, error_type, steps=None, front_door=False):
        self.retriever, self.error_type, self.steps = retriever, error_type, steps  # steps: see pipeline.STEPS
        self.front_door = front_door  # app.ask: guard, LLM scope check, translation, number check, as users get it
        self.results = {}  # question -> our full result (confidence_eval.py reads the signals from it)

    def answer(self, question):
        try:
            if self.front_door:
                from app import ask
                checked, result = ask(question, self.retriever)
                if not checked.ok:
                    raise self.error_type(f"stopped by the guard ({checked.category}): {result['answer'][:80]}")
                if result.get("unsupported"):
                    print(f"  number check replaced the answer: {result['unsupported']}", flush=True)
            elif self.steps is None:
                result = pipeline.answer(question, self.retriever)
            else:
                result = pipeline.answer(question, self.retriever, steps=self.steps)
        except Exception as e:  # noqa: BLE001 - their evaluate() catches only its own errors; record ours as "error"
            raise self.error_type(f"{type(e).__name__}: {e}") from e
        self.results[question] = result
        if not result.get("parts"):
            raise self.error_type(f'no table: {result["answer"]}')
        parts = result["parts"]
        tables = tuple(dict.fromkeys(Table(p["table"]["id"], p["table"]["title_is"]) for p in parts))
        return Answer(question, result["answer"], tables, [f for p in parts for f in facts_of(p)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("questions", nargs="?", help="questions file (default: their questions/examples.jsonl)")
    ap.add_argument("--starter", help="path to the gagnvist-starter clone")
    ap.add_argument("--language", choices=["is", "en"])
    ap.add_argument("--level", action="append", choices=["easy", "medium", "hard"])
    ap.add_argument("--only", help="only questions whose id contains this text")
    ap.add_argument("-r", "--retriever", default=DEFAULT)
    ap.add_argument("--front-door", action="store_true",
                    help="through app.ask (guard, LLM scope check, number check) instead of pipeline.answer")
    args = ap.parse_args()

    starter = starter_path(args.starter)
    gv = load_gagnvist(starter)
    from gagnvist.answering.pipeline import AnswerError

    path = Path(args.questions) if args.questions else starter / "questions/examples.jsonl"
    questions = [q for q in gv.read_questions(path)
                 if (not args.only or args.only in q.id) and (not args.language or q.language == args.language)
                 and (not args.level or q.level in args.level)]
    print(f"Answering {len(questions)} questions from {path} (retriever {args.retriever}) ...")
    outcomes = gv.evaluate(OurPipeline(build(args.retriever), AnswerError, front_door=args.front_door), questions,
                           lambda o: print(gv.outcome_line(o), flush=True))
    print("\n" + "\n".join(gv.summary(outcomes)))
    wrong = gv.not_right(outcomes)
    if wrong:
        print("\nNot right:\n" + "\n".join(wrong))
    saved = ROOT / "eval/results" / time.strftime("gagnvist-%Y-%m-%d-%H%M%S.json")
    gv.save(outcomes, saved)
    print(f"\nEvery answer is saved in {saved}")
    print("Settings:", json.dumps({**{k: os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_REASONING")},
                                   "front_door": args.front_door}))


if __name__ == "__main__":
    main()

"""The adapter that lets the organizers' `gagnvist evaluate` score our pipeline. Runs their scoring code,
so it needs a gagnvist-starter clone (GAGNVIST_STARTER or ../gagnvist-starter); skipped without one."""
import sys
from pathlib import Path

import pytest

from hagstofa import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import gagnvist_eval  # noqa: E402

STARTER = gagnvist_eval.starter_path()
pytestmark = pytest.mark.skipif(not (STARTER / "src/gagnvist/evaluate.py").is_file(),
                                reason="no gagnvist-starter clone")


def part(table_id, rows):
    return {"table": {"id": table_id, "title_is": f"Tafla {table_id}"}, "rows": rows, "calculation": None}


@pytest.fixture
def gv():
    return gagnvist_eval.load_gagnvist(STARTER)


def run(gv, monkeypatch, result, qid):
    from gagnvist.answering.pipeline import AnswerError
    monkeypatch.setattr(pipeline, "answer", lambda q, r: result)
    questions = [q for q in gv.read_questions(STARTER / "questions/examples.jsonl") if q.id == qid]
    return gv.evaluate(gagnvist_eval.OurPipeline(None, AnswerError), questions)[0]


def test_right_answer_counts(gv, monkeypatch):
    rows = [{"Kyn": "Alls", "Ár": "2026", "value": 394324}]
    o = run(gv, monkeypatch, {"answer": "Á Íslandi bjuggu 394.324 manns 1. janúar 2026.",
                              "parts": [part("MAN00101", rows)]}, "population-iceland-2026-is")
    assert o.verdict == "right"
    assert o.answer.facts[0].text == "MAN00101: Kyn: Alls; Ár: 2026 = 394324"


def test_number_only_in_rows_is_not_stated(gv, monkeypatch):
    """The rows go to .facts, never .text: the scorer must not find the number in our raw data."""
    rows = [{"Ár": "2026", "value": 394324}]
    o = run(gv, monkeypatch, {"answer": "Ég veit það ekki.", "parts": [part("MAN00101", rows)]},
            "population-iceland-2026-is")
    assert o.verdict == "WRONG"


def test_hard_question_needs_both_tables(gv, monkeypatch):
    text = "394.324 manns og vísitalan var 697,3."
    one = run(gv, monkeypatch, {"answer": text, "parts": [part("MAN00101", [])]}, "population-and-cpi-2026-is")
    both = run(gv, monkeypatch, {"answer": text, "parts": [part("MAN00101", []), part("VIS01000", [])]},
               "population-and-cpi-2026-is")
    assert one.verdict == "other table" and both.verdict == "right"


def test_no_table_and_crash_are_errors(gv, monkeypatch):
    o = run(gv, monkeypatch, {"answer": "Ég fann enga töflu."}, "population-iceland-2026-is")
    assert o.verdict == "error" and "no table" in o.error

    def boom(q, r):
        raise ConnectionError("LLM down")
    monkeypatch.setattr(pipeline, "answer", boom)
    from gagnvist.answering.pipeline import AnswerError
    q = gv.read_questions(STARTER / "questions/examples.jsonl")[:1]
    o = gv.evaluate(gagnvist_eval.OurPipeline(None, AnswerError), q)[0]
    assert o.verdict == "error" and "ConnectionError" in o.error

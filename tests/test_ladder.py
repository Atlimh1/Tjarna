"""Ablation switches in pipeline.answer (steps=...) and the ladder runner, with a fake LLM and PX-Web."""
import sys
from pathlib import Path

import pytest

from hagstofa import pipeline
from test_calc import FakeRetriever, fake_llm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import gagnvist_eval  # noqa: E402
import ladder  # noqa: E402

Q = "Hvað búa margir á Íslandi og hvert er atvinnuleysið?"


def systems(prompts):
    return [s for s, _ in prompts]


def test_all_steps_off_is_search_first_table_values_answer(monkeypatch):
    prompts = fake_llm(monkeypatch, operation="difference")
    r = pipeline.answer(Q, FakeRetriever(), steps=frozenset())
    assert systems(prompts) == [pipeline.SELECT_SYSTEM, pipeline.ANSWER_SYSTEM]  # no split, pick or calc
    assert [p["table"]["id"] for p in r["parts"]] == ["ATV"]  # the retriever's #1 for the whole question
    assert r["parts"][0]["calculation"] is None
    assert not prompts[-1][1].endswith(pipeline.ANSWER_IN["is"])  # no language rule


def test_each_step_adds_its_llm_call(monkeypatch):
    prompts = fake_llm(monkeypatch, operation="difference")
    pipeline.answer(Q, FakeRetriever(), steps=frozenset({"pick", "calc"}))
    assert systems(prompts) == [pipeline.PICK_SYSTEM, pipeline.SELECT_SYSTEM, pipeline.CALC_SYSTEM,
                                pipeline.ANSWER_SYSTEM]
    prompts.clear()
    r = pipeline.answer(Q, FakeRetriever())  # default = every step = the normal pipeline
    assert systems(prompts)[0] == pipeline.DECOMPOSE_SYSTEM and len(r["parts"]) == 2
    assert prompts[-1][1].endswith(pipeline.ANSWER_IN["is"])


def test_no_data_rung_sends_only_the_question(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.llm, "chat", lambda system, user, **kw: seen.append((system, user)) or "um 400.000")
    assert pipeline.answer_without_data("How many people lived in Iceland on 1 January 2026?") == "um 400.000"
    assert seen[0][0] == pipeline.NO_DATA_SYSTEM and seen[0][1].endswith(pipeline.ANSWER_IN["en"])


def test_rungs_build_up_one_step_at_a_time():
    steps = [s for _, r, s in ladder.RUNGS if r == "checked"]
    assert steps[0] == frozenset() and steps[-1] == pipeline.STEPS
    assert all(len(b - a) == 1 and a < b for a, b in zip(steps, steps[1:]))


STARTER = gagnvist_eval.starter_path()


@pytest.mark.skipif(not (STARTER / "src/gagnvist/evaluate.py").is_file(), reason="no gagnvist-starter clone")
def test_ladder_row_scores_no_data_by_numbers_only(monkeypatch):
    gv = gagnvist_eval.load_gagnvist(STARTER)
    from gagnvist.answering.pipeline import AnswerError
    questions = [q for q in gv.read_questions(STARTER / "questions/examples.jsonl")
                 if q.id == "population-iceland-2026-is"]
    monkeypatch.setattr(pipeline, "answer_without_data", lambda q: "Íbúar voru 394.324.")
    outcomes = gv.evaluate(ladder.NoData(AnswerError), questions)
    line = ladder.row("0 no-data", outcomes)
    assert "**0/1**" in line and "| 1 | 0 |" in line  # not right (no table), but the number was stated

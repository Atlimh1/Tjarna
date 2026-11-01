"""Calculations and two-table questions, with a fake LLM and a fake PX-Web (no network)."""
import json
import sys
from pathlib import Path

import pytest

from hagstofa import llm, pipeline, px
from hagstofa.pipeline import calculate, compute, decompose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import answer_ok, expected_numbers  # noqa: E402

ROWS = [{"Ár": "2025", "value": 138772}, {"Ár": "2026", "value": 139804}, {"Ár": "Alls", "value": 394324}]


def test_compute_and_calculate_match_the_organizers_examples():
    assert compute("difference", [139804, 138772]) == 1032
    assert calculate({"operation": "share", "a": 1, "b": 2}, ROWS)["text"] == "139.804 / 394.324 × 100 = 35,5%"
    assert calculate({"operation": "difference", "a": "1", "b": 0}, ROWS, english=True)["text"] == "139,804 − 138,772 = 1,032"
    assert calculate({"operation": "sum", "rows": [0, 1]}, ROWS)["result"] == 278576


@pytest.mark.parametrize("plan", [
    {"operation": "none"}, {"operation": "difference", "a": 7, "b": 0}, {"operation": "difference", "a": "x", "b": 0},
    {"operation": "magic", "a": 0, "b": 1}, {"operation": "sum"},
])
def test_calculate_refuses_bad_plans(plan):
    assert calculate(plan, ROWS) is None


def test_calculate_refuses_division_by_zero_and_missing_values():
    rows = [{"value": 5}, {"value": 0}, {"value": None}]
    assert calculate({"operation": "share", "a": 0, "b": 1}, rows) is None
    assert calculate({"operation": "difference", "a": 0, "b": 2}, rows) is None


@pytest.mark.parametrize("reply, expected", [
    ({"questions": ["Hvað búa margir á Íslandi?", "Hvert er atvinnuleysið?"]}, 2),
    ({"questions": ["einungis ein"]}, "original"),
    ({"questions": ["a?", "b?", "c?"]}, 3),  # one per table, up to MAX_PARTS (4) since 1 Nov
    ({"questions": ["a?", "b?", "c?", "d?", "e?"]}, 5),  # kept: answer() names those past MAX_PARTS
    ({"nothing": 1}, "original"),
])
def test_decompose(monkeypatch, reply, expected):
    monkeypatch.setattr(llm, "chat_json", lambda system, user: reply)
    out = decompose("Hvað búa margir á Íslandi og hvert er atvinnuleysið?")
    if expected == "original":
        assert out == ["Hvað búa margir á Íslandi og hvert er atvinnuleysið?"]
    else:
        assert len(out) == expected


def table(tid):
    return {"id": tid, "title_is": f"Tafla {tid}", "path": f"X/{tid}.px",
            "variables_is": [{"code": "Ár", "text": "Ár", "values": ["2025", "2026"], "valueTexts": ["2025", "2026"],
                              "time": True}]}


class FakeRetriever:
    def search(self, question, k=8):
        return [(table("ATV" if "atvinnu" in question else "MAN"), 1.0)]


def fake_llm(monkeypatch, pick=0, operation="none"):
    prompts = []

    def chat_json(system, user):
        prompts.append((system, user))
        if system == pipeline.DECOMPOSE_SYSTEM:
            return {"questions": ["Hvað búa margir á Íslandi 2026?", "Hvert var atvinnuleysið 2026?"]}
        if system == pipeline.PICK_SYSTEM:
            return {"table": pick if "atvinnu" in user else 0}
        if system == pipeline.SELECT_SYSTEM:
            return {"selections": {"Ár": ["2025", "2026"]}}
        return {"operation": operation, "a": 1, "b": 0}

    def chat(system, user, **kw):
        prompts.append((system, user))
        return "svar"

    monkeypatch.setattr(llm, "chat_json", chat_json)
    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(px, "query", lambda path, sel, lang="is": {
        "id": ["Ár"], "dimension": {"Ár": {"label": "Ár", "category": {"index": {"2025": 0, "2026": 1},
                                                                       "label": {"2025": "2025", "2026": "2026"}}}},
        "value": [100, 110] if "MAN" in path else [5.0, 6.0]})
    return prompts


def test_two_table_question_runs_each_part_and_cites_both(monkeypatch):
    prompts = fake_llm(monkeypatch, operation="difference")
    r = pipeline.answer("Hvað búa margir á Íslandi og hvert er atvinnuleysið?", FakeRetriever())
    assert [p["table"]["id"] for p in r["parts"]] == ["MAN", "ATV"]
    assert r["table"]["id"] == "MAN" and r["url"]  # first part, for callers that show one table
    final = prompts[-1][1]
    assert "Tafla MAN" in final and "Tafla ATV" in final
    assert "110 − 100 = 10" in final  # computed in code and handed to the answer prompt


def test_refused_pick_falls_back_to_the_search_top(monkeypatch):
    """Models answer "no table fits" (-1) too often, mostly on English questions: the search's #1 is used."""
    fake_llm(monkeypatch, pick=-1)
    r = pipeline.answer("Hvað búa margir á Íslandi og hvert er atvinnuleysið?", FakeRetriever())
    assert [p["table"]["id"] for p in r["parts"]] == ["MAN", "ATV"]


def test_part_without_data_is_reported_not_dropped(monkeypatch):
    prompts = fake_llm(monkeypatch)
    query = px.query
    monkeypatch.setattr(px, "query", lambda path, sel, lang="is": (
        {**query(path, sel, lang), "value": [None, None]} if "ATV" in path else query(path, sel, lang)))
    r = pipeline.answer("Hvað búa margir á Íslandi og hvert er atvinnuleysið?", FakeRetriever())
    assert [p["table"]["id"] for p in r["parts"]] == ["MAN"]
    assert "Engin gögn fundust fyrir: Hvert var atvinnuleysið 2026?" in prompts[-1][1]


def test_calculation_across_two_tables(monkeypatch):
    """"Per inhabitant", "per 1,000", "A and B together": one number from each sub-question's table."""
    prompts = fake_llm(monkeypatch, operation="per_1000")
    r = pipeline.answer("Hvað búa margir á Íslandi og hvert er atvinnuleysið?", FakeRetriever())
    rows = [row for p in r["parts"] for row in p["rows"]]
    assert r["calculation"]["operation"] == "per_1000"
    assert r["calculation"]["result"] == rows[1]["value"] / rows[0]["value"] * 1000
    assert "Útreikningur úr fleiri en einni töflu" in prompts[-1][1]


def test_results_the_answer_lacks_are_added():
    calc = pipeline.calculate({"operation": "percent_change", "a": 0, "b": 1}, [{"value": 9850428}, {"value": 9312620}])
    assert calc["text"].endswith("= 5,78%")
    rounded_wrong = pipeline.with_calculations("Fjölgunin var 5,7 prósent.", [calc])
    assert rounded_wrong.endswith("Útreikningur: " + calc["text"])
    assert pipeline.with_calculations("Fjölgunin var 5,78%.", [calc, None]) == "Fjölgunin var 5,78%."
    assert pipeline.with_calculations("It grew by 5.78%.", [calc], "en") == "It grew by 5.78%."


def test_expected_numbers_for_calc_and_two_table_items():
    cells = {"A": 139804, "B": 138772, "C": 394324}
    fetch = lambda path, truth: cells[truth["k"]]  # noqa: E731
    path_of = lambda tid: tid  # noqa: E731
    calc = {"q": "?", "table": ["T"], "calc": {"op": "share", "args": [{"table": "T", "truth": {"k": "A"}},
                                                                       {"table": "T", "truth": {"k": "C"}}]}}
    both = {"q": "?", "table": ["T", "U"], "all": [{"table": "T", "truth": {"k": "A"}}, {"table": "U", "truth": {"k": "B"}}]}
    assert round(expected_numbers(calc, path_of, fetch)[0], 2) == 35.45
    assert expected_numbers(both, path_of, fetch) == [139804, 138772]
    assert answer_ok("Hlutfallið er 35,5%.", [35.45])
    assert answer_ok("139.804 íbúar og 138.772 íbúar", [139804, 138772])
    assert not answer_ok("139.804 íbúar", [139804, 138772])  # both numbers must be there
    assert not answer_ok("35,5%", [None])
    json.dumps(calc)  # items stay plain JSON

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from compare_models import number_ok, values_ok  # noqa: E402

TABLE = {"id": "T", "variables_is": [
    {"code": "Ár", "values": ["2024", "2025"], "valueTexts": ["2024", "2025"], "time": True},
    {"code": "Kyn", "values": ["0", "2"], "valueTexts": ["Alls", "Konur"], "elimination": True},
]}


def test_number_ok_reads_icelandic_formatting():
    assert number_ok("Íbúar voru 394.324 árið 2025.", 394324)
    assert number_ok("Ársbreytingin var 3,8%.", 3.8)
    assert number_ok("394 324 einstaklingar", 394324)  # narrow no-break space (gpt-oss)
    assert not number_ok("Íbúar voru 390.000.", 394324)
    assert not number_ok("Veit ekki.", None)


def test_values_ok_needs_truth_codes_and_accepts_eliminated_total():
    truth = {"Ár": ["2025"], "Kyn": ["0"]}
    assert values_ok(TABLE, {"Ár": ["2025"], "Kyn": ["0"]}, truth)
    assert values_ok(TABLE, {"Ár": ["2024", "2025"]}, truth)  # Kyn left out = total; extra year is fine
    assert not values_ok(TABLE, {"Ár": ["2024"], "Kyn": ["0"]}, truth)
    assert not values_ok(TABLE, {"Ár": ["2025"]}, {"Ár": ["2025"], "Kyn": ["2"]})  # left out, but truth is Konur


def test_expected_numbers_are_cached_between_runs(tmp_path, monkeypatch):
    import compare_models
    calls = []
    monkeypatch.setattr(compare_models, "RESULTS", tmp_path)
    monkeypatch.setattr(compare_models, "expected_value", lambda path, truth: calls.append(path) or 42.0)
    questions = [{"q": "Hvað?", "table": ["T"], "truth": {"Ár": ["2025"]}}]
    by_id = {"T": {"path": "a/T.px"}}
    assert compare_models.cached_expected(questions, by_id) == {"Hvað?": [42.0]}
    assert compare_models.cached_expected(questions, by_id) == {"Hvað?": [42.0]}
    assert calls == ["a/T.px"]  # fetched once
    questions[0]["truth"] = {"Ár": ["2024"]}  # a changed truth is refetched
    compare_models.cached_expected(questions, by_id)
    assert len(calls) == 2

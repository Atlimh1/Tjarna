"""Confidence from signals computed in code; the model badge; the second opinion (31 Oct)."""
import app
from hagstofa import confidence, guard, llm

PART = {"table": {"id": "MAN00101"}, "agreement": 3, "fallback": [], "assumed": []}


def test_level_from_code_signals():
    assert confidence.level({"parts": [PART]}) == "high"
    assert confidence.level({"parts": [{**PART, "agreement": 2}]}) == "high"
    assert confidence.level({"parts": [{**PART, "agreement": 1}]}) == "medium"
    assert confidence.level({"parts": [{**PART, "agreement": 0}]}) == "low"
    assert confidence.level({"parts": [PART], "unsupported": ["4.980"]}) == "low"  # the number check replaced it
    assert confidence.level({"parts": [{**PART, "fallback": ["tables tried first had no numbers: ['X']"]}]}) == "low"
    assert confidence.level({"answer": "no table"}) is None


def test_describe_shows_the_measured_rate_only_when_there_is_one(monkeypatch):
    monkeypatch.setattr(confidence, "calibration", lambda: {"high": {"right": 34, "n": 40}})
    assert confidence.describe("high", "en") == "Confidence: 85% (answers like this were right in 34 of 40 tests)"
    assert confidence.describe("high", "is") == "Vissa: 85% (svör af þessu tagi voru rétt í 34 af 40 prófum)"
    assert confidence.describe("medium", "is") == "Vissa: miðlungs"


def test_the_rates_are_shown_only_for_the_model_they_were_measured_with(monkeypatch):
    monkeypatch.setattr(confidence, "calibration", lambda: {"high": {"right": 34, "n": 40}})  # no "model": gemma
    assert confidence.describe("high", "en", "Gemma 4 31B").startswith("Confidence: 85%")
    assert confidence.describe("high", "en", "gemma4:26b") == ""  # another model: no confidence line
    monkeypatch.setattr(confidence, "calibration", lambda: {"high": {"right": 34, "n": 40}, "model": "qwen3-30b"})
    assert confidence.describe("high", "en", "Qwen3-30B-A3B").startswith("Confidence: 85%")


def test_badge_leaves_out_the_check_when_the_answer_states_no_number(monkeypatch):
    q = {"lang": "is", "model": "Gemma 4 31B", "confidence": None, "no_numbers": True}
    assert app.badge(q, 2.0) == "<sub>Gemma 4 31B · 2.0 s</sub>"


def test_badge_names_model_time_check_and_confidence(monkeypatch):
    monkeypatch.setattr(confidence, "calibration", lambda: {})
    q = {"lang": "en", "model": "Gemma 4 31B", "confidence": "high"}
    assert app.badge(q, 5.04) == "<sub>Gemma 4 31B · 5.0 s · numbers checked against the table · Confidence: high</sub>"


def test_agree_compares_the_main_numbers_not_years():
    assert app.agree("On 1 January 2026, 394,324 people.", "394.324 manns árið 2026.") is True
    # dry run: "Same result as the first answer: 1841", read from the title "Population ... 1841-2026"
    a = "According to table MAN00101 (Population by sex and age 1841-2026), the population in 2026 is 394,324."
    assert app.key_numbers(a)[0][1] == "394,324"
    assert app.agree("35.5% in 2026.", "36.1% in 2026.") is False
    assert app.agree("No data for 2035.", "394,324") is None


def test_second_opinion_without_a_second_model(monkeypatch):
    monkeypatch.setitem(app.SECOND, "base_url", None)
    assert "No second model" in app.second_opinion("How many people live in Iceland?", "394,324")


def test_second_opinion_runs_the_pipeline_on_the_other_model(monkeypatch):
    monkeypatch.setitem(app.SECOND, "base_url", "http://other:8000/v1")
    monkeypatch.setitem(app.SECOND, "model", "qwen3-30b")
    seen = {}

    def fake_answer_once(message):
        seen["model"] = llm.current()[1]
        return guard.Verdict("ok", message, lang="en"), {"answer": "394,324 people.", "parts": []}, 1.0

    monkeypatch.setattr(app, "answer_once", fake_answer_once)
    out = app.second_opinion("How many people live in Iceland?", "There were 394,324 people.")
    assert seen["model"] == "qwen3-30b" and out.startswith("**Second opinion**") and "Same result" in out
    assert llm.current()[1] == llm.MODEL  # back to the configured model afterwards

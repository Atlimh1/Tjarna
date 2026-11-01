"""hagstofa/language.py and hagstofa/translate.py (with a fake LLM)."""
import json
from pathlib import Path

import pytest

from hagstofa import language, llm, translate

EVAL = Path(__file__).resolve().parent.parent / "eval"


@pytest.mark.parametrize("text, lang", [
    ("Hvað búa margir á Íslandi?", "is"), ("hvad bua margir a islandi", "is"), ("Hvað búa margir í Garðabæ?", "is"),
    ("How many people live in Garðabær?", "en"), ("How many babies were born in Kópavogur in 2025?", "en"),
    ("Ilu Polaków mieszka na Islandii?", "pl"), ("Kiek žmonių gyvena Islandijoje?", "lt"),
    ("Скільки людей живе в Ісландії?", "uk"), ("Сколько людей живет в Исландии?", "ru"),
    ("冰岛有多少人口？", "zh"), ("كم عدد سكان أيسلندا؟", "ar"), ("¿Cuántas personas viven en Islandia?", "es"),
])
def test_detect(text, lang):
    assert language.detect(text) == lang


@pytest.mark.parametrize("name", ["questions_dev.jsonl", "questions_dev_typo.jsonl", "questions_lang.jsonl"])
def test_every_labelled_question_gets_its_language(name):
    rows = [json.loads(line) for line in open(EVAL / name, encoding="utf-8")]
    wrong = [r["q"] for r in rows if language.detect(r["q"]) != r["lang"]
             and {language.detect(r["q"]), r["lang"]} != {"da", "no"}]  # often word for word the same
    assert wrong == []


def test_numbers_ignore_separators():
    assert translate.numbers("394,324 people (6.5%) in 2026") == translate.numbers("394 324 osób (6,5%) w 2026")


def fake_llm(monkeypatch, back):
    calls = []

    def chat(system, user, **kw):
        calls.append(user)
        return "How many Poles live in Iceland?" if system == translate.TO_ENGLISH else back
    monkeypatch.setattr(llm, "chat", chat)
    return calls


def test_round_trip_answers_in_the_users_language(monkeypatch):
    calls = fake_llm(monkeypatch, "W Islandii mieszka 25 314 Polaków.")
    r = translate.answer("Ilu Polaków mieszka na Islandii?", "pl",
                         lambda q: {"answer": f"25,314 Poles live in Iceland. ({q})".split(" (")[0], "parts": []})
    assert r["answer"] == "W Islandii mieszka 25 314 Polaków."
    assert r["question_en"] == "How many Poles live in Iceland?" and r["answer_en"].startswith("25,314")
    assert calls[0] == "Ilu Polaków mieszka na Islandii?"


def test_a_changed_number_falls_back_to_english(monkeypatch):
    fake_llm(monkeypatch, "W Islandii mieszka 25 341 Polaków.")  # digits swapped by the translation
    r = translate.answer("Ilu Polaków mieszka na Islandii?", "pl", lambda q: {"answer": "25,314 Poles live in Iceland."})
    assert r["answer"].startswith("25,314 Poles live in Iceland.") and "Polish changed a number" in r["answer"]


def test_mixed_and_multi_question_set():
    rows = [json.loads(line) for line in open(EVAL / "questions_mixed.jsonl", encoding="utf-8")]
    for r in rows:
        got = [(q, language.detect(q)) for q in language.split_questions(r["q"])]
        assert got == [(p["q"], p["lang"]) for p in r["parts"]], r["q"]


@pytest.mark.parametrize("name", ["questions_dev.jsonl", "questions_dev_typo.jsonl", "questions_lang.jsonl",
                                  "questions_e2e.jsonl"])
def test_a_real_question_is_never_split(name):
    rows = [json.loads(line) for line in open(EVAL / name, encoding="utf-8")]
    assert [r["q"] for r in rows if len(language.split_questions(r["q"])) != 1] == []


@pytest.mark.parametrize("english, translated, same", [
    ("On January 1, 2026, the population was 394,324.", "1 stycznia 2026 r. liczba ludności wynosiła 394 324.", True),
    ("On January 1, 2026, the population was 394,324.", "Am 1. Januar 2026 betrug die Bevölkerung 394.324.", True),
    ("Unemployment was 6.5% in Q2 2026.", "Le chômage était de 6,5 % au T2 2026.", True),
    ("The population was 394,324.", "La population était de 394 325.", False),  # a changed number is still caught
])
def test_translation_number_check(english, translated, same):
    from hagstofa.translate import numbers
    assert (numbers(english) == numbers(translated)) is same

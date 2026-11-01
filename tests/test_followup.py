"""hagstofa/followup.py: follow-up questions and mixed-language messages in the chat (31 Oct)."""
import pytest

import app
from hagstofa import followup, guard, llm


@pytest.mark.parametrize("message, mixed", [
    ("How many people búa í Reykjavík?", True), ("Hvað many people bjuggu in Iceland?", True),
    ("hvad var inflation rate i september", False),  # an English noun in Icelandic is normal
    ("What was the population of Kópavogur?", False),  # an Icelandic place in English is normal
    ("Hvað búa margir á Íslandi?", False), ("How many people live in Hafnarfjörður?", False), ("En 2024?", False),
])
def test_mixed(message, mixed):
    assert followup.mixed(message) is mixed


def test_standalone_rewrites_only_follow_ups(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "chat_json", lambda s, u: calls.append(u) or {"question": "Hvað bjuggu margir í Kópavogi 2024?"})
    prev = "Hvað búa margir í Kópavogi?"
    assert followup.standalone("En 2024?", prev) == "Hvað bjuggu margir í Kópavogi 2024?"
    assert followup.standalone("What about men?", "How many people live in Iceland?") == "Hvað bjuggu margir í Kópavogi 2024?"
    n = len(calls)
    assert followup.standalone("Hvað fæddust mörg börn á Íslandi árið 2025?", prev) == "Hvað fæddust mörg börn á Íslandi árið 2025?"
    assert followup.standalone("En 2024?", None) == "En 2024?"  # nothing to continue
    assert len(calls) == n  # a whole question, or no earlier one: no LLM call
    monkeypatch.setattr(llm, "chat_json", lambda s, u: {})
    assert followup.standalone("En 2024?", prev) == "En 2024?"  # no usable reply: as written


def test_the_chat_asks_for_a_language_and_answers_follow_ups(monkeypatch):
    out = list(app._ask("How many people búa í Reykjavík?", [], {}))
    assert out[-1][1][-1]["content"] == followup.CHOOSE
    monkeypatch.setattr(followup, "standalone", lambda m, p: "Hvað bjuggu margir í Kópavogi 2024?")
    monkeypatch.setattr(app, "answer_once", lambda q: (guard.Verdict("ok", q, lang="is"), {"answer": "38.000", "parts": []}, 1.0))
    out = list(app._ask("En 2024?", [], {"question": "Hvað búa margir í Kópavogi?", "answer": "40.286"}))
    box, history, last = out[-1][:3]
    assert history[-1]["content"].startswith("_Skilið sem: Hvað bjuggu margir í Kópavogi 2024?_")
    assert last["question"] == "Hvað bjuggu margir í Kópavogi 2024?"


def test_chips_offer_suggestions_or_the_choices_a_vague_question_left():
    r = {"suggestions": ["Hvað bjuggu margir í Reykjavík 1. janúar 2026?", "Q2?"]}
    head, found = app.chips("Hvað búa margir á Mars?", r, "is")
    assert head.startswith("Smelltu") and found[0] == (r["suggestions"][0], r["suggestions"][0])
    r = {"clarify": [{"variable": "Ár", "question": "Fyrir hvaða tímabil? Ég notaði 2026, nýjustu tölurnar.",
                      "options": ["2026", "2025", "2024"], "default": "2026", "kind": "time"}]}
    head, found = app.chips("Hvað búa margir á Íslandi?", r, "is")
    assert found == [("2025", "Hvað búa margir á Íslandi? (Ár: 2025)"), ("2024", "Hvað búa margir á Íslandi? (Ár: 2024)")]
    assert "Ég notaði 2026" in head
    assert app.chips("q", {"answer": "x"}, "en") == ("", [])


def test_a_chip_asks_its_question(monkeypatch):
    asked = []
    monkeypatch.setattr(app, "answer_once", lambda q: asked.append(q) or (guard.Verdict("ok", q, lang="is"),
                                                                        {"answer": "389.444", "parts": []}, 1.0))
    monkeypatch.setattr(followup, "standalone", lambda m, p: m)
    last = {"question": "Hvað búa margir á Íslandi?", "chips": [("2025", "Hvað búa margir á Íslandi? (Ár: 2025)")]}
    out = list(app._chip(0)([], last))
    assert asked == ["Hvað búa margir á Íslandi? (Ár: 2025)"] and out[-1][2]["question"] == asked[0]
    assert list(app._chip(3)([], last))[-1][1] == []  # no such chip: nothing asked
    assert len(out[-1]) == 3 + 1 + app.CHIPS  # textbox, chat, state, head line, the buttons


def test_progress_lines():
    assert app.step_line("pick", {"table": "MAN00101", "title": "Mannfjöldi"}, "is") == "Valin tafla: **MAN00101** Mannfjöldi"
    assert app.step_line("search", {"tables": 2053}, "en") == "Searching 2,053 Statistics Iceland tables"
    assert app.step_line("split", {"parts": ["a", "b"]}, "is") == "Spurningunni skipt í 2 hluta"
    assert app.step_line("unknown", {}, "is") == ""


def test_the_chat_shows_each_step_while_it_works(monkeypatch):
    from hagstofa import pipeline

    def fake_answer_once(q):
        pipeline.report("search", tables=2053)
        pipeline.report("pick", table="MAN00101", title="Mannfjöldi")
        return guard.Verdict("ok", q, lang="is"), {"answer": "394.324", "parts": []}, 1.0

    monkeypatch.setattr(app, "answer_once", fake_answer_once)
    monkeypatch.setattr(followup, "standalone", lambda m, p: m)
    shown = [o[1][-1]["content"] for o in app._ask("Hvað búa margir á Íslandi?", [], {})]  # read as each arrives
    assert any("Valin tafla: **MAN00101**" in s for s in shown)  # a step appeared before the answer
    assert shown[-1] == "394.324"


def test_new_conversation_forgets_the_last_question():
    out = app._new_conversation()
    assert out[:3] == ("", [], {}) and len(out) == 3 + 1 + app.CHIPS

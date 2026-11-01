"""hagstofa/clarify.py: what a vague question left open, as options a client may offer (idea E, 31 Oct)."""
from hagstofa import clarify

TABLE = {"id": "MAN02005", "title_is": "Mannfjöldi", "variables_is": [
    {"code": "Sveitarfélag", "text": "Sveitarfélag", "values": ["-", "0000", "1000"], "valueTexts": ["Alls", "Reykjavíkurborg", "Kópavogsbær"]},
    {"code": "Kyn", "text": "Kyn", "values": ["0", "1"], "valueTexts": ["Alls", "Karlar"]},
    {"code": "Ár", "text": "Ár", "time": True, "values": ["2023", "2024", "2025", "2026"], "valueTexts": ["2023", "2024", "2025", "2026"]}]}


def test_period_first_then_place_with_the_default_first():
    part = {"table": TABLE, "edition": "is", "assumed": [("Sveitarfélag", "Alls"), ("Kyn", "Alls"), ("Ár", "2026")]}
    got = clarify.clarifications({"parts": [part]}, "is")
    assert [c["variable"] for c in got] == ["Ár", "Sveitarfélag"]  # sex is not offered: a total is a fine default
    assert got[0]["options"] == ["2026", "2025", "2024", "2023"] and got[0]["default"] == "2026"
    assert got[1]["options"][0] == "Alls" and "Ég notaði 2026" in got[0]["question"]


def test_nothing_to_ask_when_the_question_said_it():
    part = {"table": TABLE, "edition": "is", "assumed": [("Kyn", "Alls")]}
    assert clarify.clarifications({"parts": [part]}, "en") == []
    assert clarify.clarifications({"answer": "no data"}, "en") == []


def test_the_newest_period_is_offered_back_when_the_question_named_none():
    part = {"table": TABLE, "edition": "is", "question": "Hvað fæddust mörg börn?", "assumed": [],
            "selections": {"Ár": ["2026"]}}  # the model took the newest period itself
    got = clarify.clarifications({"parts": [part]}, "is")
    assert [c["variable"] for c in got] == ["Ár"] and got[0]["options"][:2] == ["2026", "2025"]
    for question in ["Hvað fæddust mörg börn árið 2026?", "Hvað fæddust mörg börn í fyrra?"]:  # a period asked
        assert clarify.clarifications({"parts": [{**part, "question": question}]}, "is") == []

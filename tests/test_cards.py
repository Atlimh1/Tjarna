"""hagstofa/cards.py: table cards and the code filter before the pick (idea D, 31 Oct)."""
from hagstofa import cards


def table(tid, years=None, title="T", place=None):
    vs = []
    if years:
        vs.append({"code": "Ár", "text": "Ár", "time": True, "values": [str(y) for y in years], "valueTexts": [str(y) for y in years]})
    if place:
        vs.append({"code": "S", "text": place, "values": ["0"], "valueTexts": ["Alls"]})
    return {"id": tid, "title_is": title, "variables_is": vs}


def test_period_and_level():
    assert cards.period(table("A", [1998, 2026])) == (1998, 2026)
    assert cards.period(table("B", title="Eiginnöfn karla 1. janúar 2023")) == (2023, 2023)  # shown, never filtered
    assert cards.time_period(table("B", title="Eiginnöfn karla 1. janúar 2023")) is None
    assert cards.level(table("D", place="Sveitarfélag")) == "municipality"
    assert cards.level(table("E", place="Byggðakjarnar")) == "village"
    assert cards.level(table("F")) == "country"
    assert cards.card(table("A", [1998, 2026], place="Sveitarfélag")) == " | tímabil: 1998-2026 | stig: sveitarfélög"


def test_the_year_filter_drops_only_what_is_certain():
    a, b, c = table("A", [2000, 2010]), table("B", [2015, 2026]), table("C", title="Nöfn 1. janúar 2023")
    ranked = [(a, 3), (b, 2), (c, 1)]
    assert [t["id"] for t, _ in cards.filter_candidates("2025?", ranked, frozenset({2025}))] == ["B", "C"]
    assert [t["id"] for t, _ in cards.filter_candidates("1800?", ranked, frozenset({1800}))] == ["C"]  # a snapshot
    assert len(cards.filter_candidates("x", [(a, 1)], frozenset({1800}))) == 1  # nothing covers it: all kept
    assert len(cards.filter_candidates("no year", ranked)) == 3


def test_release_fields_and_the_note_cut_to_the_question(monkeypatch):
    note = ("Aðferð við mat á mannfjölda var endurskoðuð í mars 2024. Miðað er við 1. janúar ár hvert. "
            "Innflytjendur eru fæddir erlendis. Sjá nánar í greinargerð.")
    extra = {"P": {"path": "P", "note_is": note, "note_en": "Population 1 January.", "frequency": ["annual"],
                   "source": "Ferðamálastofa.", "survey": "", "provisional": True}}
    monkeypatch.setattr(cards, "_release", lambda: extra)
    t = table("A", [1998, 2026]) | {"path": "P"}
    # housekeeping (revisions, "see the paper") out; the sentence that shares a word with the question first
    assert cards.note(t, "Hvað eru margir innflytjendur?") == ("Innflytjendur eru fæddir erlendis. "
                                                              "Miðað er við 1. janúar ár hvert.")
    assert cards.note(t, "", "en") == "Population 1 January."
    assert cards.note(t, "", chars=20) == "Miðað er við 1." + " ..."
    line = cards.card(t, "is", "Hvað búa margir?")
    assert " | tíðni: árleg | heimild: Ferðamálastofa | sumar tölur bráðabirgðatölur | athugasemd: Miðað" in line
    assert cards.card(table("B", [2000, 2001]) | {"path": "none"}) == " | tímabil: 2000-2001 | stig: landið"
    monkeypatch.setattr(cards, "CARD_NOTES", False)
    assert cards.card(t) == " | tímabil: 1998-2026 | stig: landið"


def test_the_release_file_matches_the_catalog():
    import json
    paths = {json.loads(line)["path"] for line in open(cards.CARD_FILE.parents[0] / "catalog.jsonl", encoding="utf-8")}
    release = cards._release()
    assert len(release) == 1886 and set(release) <= paths

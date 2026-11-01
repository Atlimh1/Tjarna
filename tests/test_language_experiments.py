"""The language experiments' switches (31 Oct): accent restoration (ACCENTS, on) and query rewriting (QUERY_REWRITE,
off until measured); these tests run them with the switch on, with a fake LLM."""
import pytest

from hagstofa import llm, pipeline, retrieve
from hagstofa.retrieve import restore_accents


@pytest.mark.parametrize("typed, mode, want", [
    ("hvad bua margir i Kopavogi", 1, "hvað búa margir í Kópavogi"),
    ("throun mannfjolda", 1, "þróun mannfjölda"),
    ("Hve hatt hlutfall", 1, "Hve hatt hlutfall"),     # "hatt" is a word (hattur): mode 1 leaves it
    ("Hve hatt hlutfall", 2, "Hve hátt hlutfall"),     # mode 2: "hátt" is ~90x as common
    ("faeddust arid 2025", 1, "fæddust arid 2025"),    # two candidates (arið, árið): mode 1 declines
    ("faeddust arid 2025", 2, "fæddust árið 2025"),    # mode 2: one dominates
    ("fjoldi i rvk", 1, "fjöldi í rvk"),               # catalog abbreviations stay
])
def test_restore_accents(typed, mode, want):
    assert restore_accents(typed, mode) == want


@pytest.mark.parametrize("question", ["How many people live in Reykjavik?", "Hvað búa margir á Íslandi?",
                                      "hverjir voru beinir skatar ríkissjóðs"])  # accents elsewhere: a typo, not "skátar"
def test_restore_accents_leaves_english_and_correct_icelandic_alone(question):
    assert restore_accents(question, 2) == question


def test_accents_on_and_rewrite_off_by_default():
    assert retrieve.ACCENTS == "1" and restore_accents("hvad bua margir") == "hvað búa margir"
    assert restore_accents("hvad bua margir", 0) == "hvad bua margir"
    assert pipeline.QUERY_REWRITE is False  # until eval/experiments/query_rewrite.py has been run


def test_rewrite_prompt_carries_todays_date():
    import datetime
    system = pipeline.rewrite_system(datetime.date(2026, 10, 31))
    assert "2026-10-31" in system and "árið 2025" in system and '"clean"' in system
    assert system.startswith(pipeline.DECOMPOSE_SYSTEM[:40])


@pytest.mark.parametrize("reply, want", [
    ({"questions": [{"question": "x", "clean": "Hver var mannfjöldi í Reykjavík árið 2025?"}]},
     [("hvad bua margir i rvk i fyrra", "Hver var mannfjöldi í Reykjavík árið 2025?")]),  # one part: user's wording
    ({"questions": [{"question": "a?", "clean": "A?"}, {"question": "b?"}]}, [("a?", "A?"), ("b?", None)]),
    ({"questions": ["a?", "b?"]}, [("a?", None), ("b?", None)]),  # the old shape still works
    ({"nothing": 1}, [("hvad bua margir i rvk i fyrra", None)]),
])
def test_decompose_rewrite(monkeypatch, reply, want):
    monkeypatch.setattr(llm, "chat_json", lambda system, user: reply)
    assert pipeline.decompose_rewrite("hvad bua margir i rvk i fyrra") == want


class Recorder:
    def __init__(self):
        self.asked = []

    def search(self, question, k=8):
        self.asked.append(question)
        tid = "CLEAN" if question.startswith("Hver") else "TYPED"
        return [({"id": tid, "path": tid}, 1.0), ({"id": "BOTH", "path": "BOTH"}, 0.5)]


def test_search_with_rewrite_fuses_both_and_skips_an_identical_clean():
    r = Recorder()
    ids = [t["id"] for t, _ in pipeline.search_with_rewrite(r, "hvad bua margir", "Hver var mannfjöldi?", k=3)]
    assert r.asked == ["hvad bua margir", "Hver var mannfjöldi?"]
    assert ids[0] == "BOTH" and set(ids) == {"BOTH", "TYPED", "CLEAN"}  # in both lists: ranked first
    r.asked.clear()
    pipeline.search_with_rewrite(r, "Hver var mannfjöldi?", "hver var mannfjöldi?")
    assert r.asked == ["Hver var mannfjöldi?"]


def test_answer_with_rewrite_searches_clean_but_keeps_the_users_wording(monkeypatch):
    from test_calc import FakeRetriever, fake_llm
    prompts = fake_llm(monkeypatch)
    monkeypatch.setattr(pipeline, "QUERY_REWRITE", True)
    monkeypatch.setattr(pipeline, "decompose_rewrite", lambda q: [(q, "Hvert var atvinnuleysið árið 2025?")])
    r = pipeline.answer("hvad var atvinnuleysid i fyrra", FakeRetriever())
    assert r["parts"][0]["search_question"] == "Hvert var atvinnuleysið árið 2025?"
    select = next(user for system, user in prompts if system == pipeline.SELECT_SYSTEM)
    assert "hvad var atvinnuleysid i fyrra" in select  # values are chosen from the question as typed

"""The fallback chain for numbers: per sub-question, tables (picked, then by rank) x editions (the question's
language first) x sources (data release, then PX-Web); a result needs at least one number; else "no data"."""
import pytest

from hagstofa import llm, local_data, pipeline, px


def table(tid, english=True):
    var = lambda code, text, vals: {"code": code, "text": text, "values": vals, "valueTexts": vals}  # noqa: E731
    t = {"id": tid, "path": f"X/{tid}.px", "title_is": f"Tafla {tid}", "title_en": f"Table {tid}",
         "variables_is": [var("Ár", "Ár", ["2025"])]}
    if english:
        t["variables_en"] = [var("Year", "Year", ["2025"])]
    return t


def jsonstat(code, value):
    return {"id": [code], "dimension": {code: {"label": code, "category": {"index": {"2025": 0},
                                                                           "label": {"2025": "2025"}}}},
            "value": [value]}


@pytest.fixture
def calls(monkeypatch):
    """px.query fake: answers[(table id, edition)] is a number, None (".."), or an exception to raise."""
    log, answers = [], {}

    def query(path, selections, lang="is"):
        tid = path.split("/")[-1][:-3]
        log.append((tid, lang))
        result = answers.get((tid, lang), 7.0)
        if isinstance(result, Exception):
            raise result
        return jsonstat(next(iter(selections)), result)
    monkeypatch.setattr(px, "query", query)
    monkeypatch.setattr(llm, "chat_json", lambda system, user: {"selections": {}, "table": 0, "operation": "none"})
    monkeypatch.setattr(llm, "chat", lambda system, user: "svar")
    monkeypatch.setattr(pipeline, "decompose", lambda q: [q])
    return log, answers


EN, IS = "How many people lived in Iceland in 2025?", "Hve margir bjuggu á Íslandi 2025?"


def test_edition_order():
    assert pipeline.editions(table("A"), "en") == ["en", "is"]
    assert pipeline.editions(table("A"), "is") == ["is", "en"]
    assert pipeline.editions(table("A", english=False), "en") == ["is"]  # no English edition: Icelandic


def test_english_question_reads_english_edition_with_its_codes(calls):
    log, _ = calls
    part = pipeline.fetch_part(EN, table("A"), calc=False, lang="en")
    assert part["edition"] == "en" and part["selections"] == {"Year": ["2025"]} and log == [("A", "en")]


def test_failed_edition_falls_back_to_the_other(calls):
    log, answers = calls
    answers[("A", "en")] = ConnectionError("429")
    part = pipeline.fetch_part(EN, table("A"), calc=False, lang="en")
    assert part["edition"] == "is" and part["selections"] == {"Ár": ["2025"]}
    assert log == [("A", "en"), ("A", "is")] and "429" in part["fallback"][0]


def test_icelandic_question_falls_back_to_english(calls):
    _, answers = calls
    answers[("A", "is")] = KeyError("Ár")
    assert pipeline.fetch_part(IS, table("A"), calc=False, lang="is")["edition"] == "en"


def test_all_missing_numbers_is_no_data(calls):
    _, answers = calls
    answers[("A", "is")] = answers[("A", "en")] = None  # ".." in both editions
    with pytest.raises(pipeline.NoData):
        pipeline.fetch_part(IS, table("A"), calc=False, lang="is")


class Ranked:
    def __init__(self, *tables):
        self.found = [(t, 1.0) for t in tables]

    def search(self, q, k=8):
        return self.found


def test_next_table_by_rank_when_the_picked_one_has_no_numbers(calls):
    log, answers = calls
    answers[("A", "is")] = answers[("A", "en")] = None
    r = pipeline.answer(IS, Ranked(table("A"), table("B"), table("C")))
    assert r["table"]["id"] == "B" and "['A']" in r["parts"][0]["fallback"][0]


def test_no_data_anywhere_says_so(calls, monkeypatch):
    monkeypatch.setattr(pipeline, "answerable", lambda found, r, s: found)  # suggestions are tested on their own
    _, answers = calls
    for tid in "ABCD":
        answers[(tid, "is")] = answers[(tid, "en")] = RuntimeError("down")
    r = pipeline.answer(EN, Ranked(table("A"), table("B"), table("C"), table("D")))
    assert r["answer"].startswith(pipeline.NO_TABLE["en"]) and "parts" not in r
    assert len(r["suggestions"]) == 3 and all(q in r["answer"] for q in r["suggestions"])  # 3 related tables


def test_english_edition_has_its_own_address(monkeypatch):
    """English PX-Web paths can differ from the Icelandic ones (here "gisting" vs "Gisting")."""
    fixtures = local_data.starter() / "tests/fixtures/data"
    if not (fixtures / "catalogue.csv").is_file():
        pytest.skip("no gagnvist-starter clone")
    monkeypatch.setenv("GAGNVIST_DATA", str(fixtures))
    path = "Atvinnuvegir/ferdathjonusta/Gisting/4_eldraefni/SAM91452.px"
    assert local_data.api_path(path, "en") == "Atvinnuvegir/ferdathjonusta/gisting/4_eldraefni/SAM91452.px"
    assert local_data.query(path, {}, "en")["source"] == "local"
    assert "__gisting__" in px.web_url(path, "en") and px.web_url(path, "en").startswith(px.WEB_BASES["en"])

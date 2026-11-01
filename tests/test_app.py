"""app.py: what the chat shows under an answer (the demo's "retrieved values" and response time)."""
import app
from hagstofa import guard

TABLE = {"id": "MAN03270", "title_is": "Mannfjöldi í Reykjavík", "title_en": "Population in Reykjavík", "updated": "2026-03-20"}
PART = {"table": TABLE, "url": "https://px.hagstofa.is/x/MAN03270.px",
        "rows": [{"Hverfi": "Alls", "Ár": "2026", "value": 139804.0}],
        "calculation": None}
CROSS = {"text": "139,804 / 394,324 × 100 = 35.5%"}


def test_data_used_shows_table_values_calculation_and_time():
    block = app.data_used([PART], "en", CROSS, 1.234)
    assert block.startswith("<details><summary>The data behind the answer · Response time 1.2 s</summary>")
    assert "[MAN03270](https://px.hagstofa.is/x/MAN03270.px)" in block and "Population in Reykjavík" in block
    assert "| Hverfi | Ár | Value |" in block and "| Alls | 2026 | 139,804 |" in block
    assert "Calculated in code: 139,804 / 394,324 × 100 = 35.5%" in block and block.endswith("</details>")


def test_data_used_in_icelandic_and_long_tables_are_cut():
    rows = [{"Ár": str(y), "value": float(y)} for y in range(2000, 2015)]
    block = app.data_used([{**PART, "rows": rows}], "is")
    assert "Gögnin á bak við svarið" in block and "| Ár | Gildi |" in block
    assert block.count("\n| 20") == app.MAX_ROWS_SHOWN and "(+5 fleiri raðir)" in block


def test_respond_puts_sources_and_data_under_the_answer(monkeypatch):
    result = {"answer": "35.5% lived in Reykjavík.", "parts": [PART], "calculation": CROSS}
    monkeypatch.setattr(app, "ask", lambda message, retriever: (guard.Verdict("ok", message, lang="en"), result))
    monkeypatch.setattr(app, "retriever", lambda: None)
    out = app.respond("What share of Iceland lived in Reykjavík?", [])
    assert out.startswith("35.5% lived in Reykjavík.") and "**Source:**" in out and "<details>" in out


def test_data_used_shows_the_tables_unit():
    block = app.data_used([{**PART, "unit": "Milljónir króna"}], "is")
    assert "| Hverfi | Ár | Gildi (Milljónir króna) |" in block and "| Alls | 2026 | 139.804 |" in block


def test_data_used_shows_what_the_tables_own_header_says():
    population = {"id": "MAN00101", "path": "Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00101.px",
                  "title_is": "Mannfjöldi eftir kyni og aldri 1841-2026", "title_en": "Population by sex and age 1841-2026"}
    block = app.data_used([{**PART, "table": population}], "is")
    assert "<small>Tíðni: árleg · Um töfluna: Miðað er við 1. janúar ár hvert.</small>" in block
    assert "Frequency: annual · About the table: Population according to 1 January each year." in app.data_used(
        [{**PART, "table": population}], "en")
    assert app.table_facts(TABLE, "is") == ""  # no path: nothing from the release


def test_data_used_marks_the_runner_up_table():
    block = app.data_used([PART, {**PART, "also": True}], "is")
    assert block.count("**[MAN03270]") == 2 and "Önnur tafla sem telur öðruvísi: **[MAN03270]" in block
    assert "Another table that counts differently: **[" in app.data_used([PART, {**PART, "also": True}], "en")


POP = {"id": "MAN00101", "path": "Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00101.px", "updated": "2026-03-12",
       "title_is": "Mannfjöldi eftir kyni og aldri 1841-2026", "title_en": "Population by sex and age 1841-2026"}
SERIES = {"table": POP, "url": "u", "edition": "is", "selections": {"Kyn": ["Total"], "Aldur": ["-1"],
                                                                   "Ár": ["2023", "2024", "2025", "2026"]},
          "rows": [{"Kyn": "Alls", "Ár": str(y), "value": v} for y, v in
                   [(2023, 387758.0), (2024, 389444.0), (2025, 391810.0), (2026, 394324.0)]], "calculation": None}


def test_citation_in_both_languages():
    import datetime
    day = datetime.date(2026, 11, 1)
    assert app.citation(POP, "is", day) == ("Hagstofa Íslands (2026). Mannfjöldi eftir kyni og aldri 1841-2026, "
                                            "tafla MAN00101. Sótt 1. nóvember 2026 af px.hagstofa.is.")
    assert app.citation(POP, "en", day).startswith("Statistics Iceland (2026). Population by sex and age 1841-2026, "
                                                   "table MAN00101. Retrieved 1 November 2026")


def test_reproduce_gives_the_exact_px_api_query_as_curl():
    block = app.reproduce(SERIES, "is")
    assert "curl -X POST 'https://px.hagstofa.is/pxis/api/v1/is/Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00101.px'" in block
    assert '"code": "Ár", "selection": {"filter": "item", "values": ["2023", "2024", "2025", "2026"]}' in block
    assert '"response": {"format": "json-stat2"}' in block and block.startswith("<details>")


def test_understood_line_chart_and_three_layers():
    line = app.understood([SERIES], "is")
    assert line == ("<small>Svona skildi ég spurninguna: Mannfjöldi eftir kyni og aldri 1841-2026 · Kyn: Alls · "
                    "Ár: 2023 … 2026 (4)</small>")
    chart = app.chart([SERIES], "is")
    assert chart.startswith("```text\nMannfjöldi") and chart.count("█") > 24 and "2026 " in chart
    assert app.chart([{**SERIES, "rows": SERIES["rows"][:3]}], "is") == ""  # three periods: no chart
    r = {"answer": "Árið 2026 bjuggu 394.324 á Íslandi.", "parts": [SERIES], "model": "Gemma 4 31B",
         "confidence": "high", "unsupported": []}
    page = app.render(guard.check("Hvað búa margir á Íslandi?"), r, 1.0)
    for piece in ("Svona skildi ég spurninguna", "```text", "Gögnin á bak við svarið", "Tilvitnun", "curl -X POST",
                  "Fyrir forritara", "Vissa"):
        assert piece in page, piece


def test_an_answer_without_numbers_shows_no_rows_check_or_confidence():
    # the Pi test: "Reykjavík 1950" got 'Vissa: 95%', a check mark and another table's rows under "no data"
    r = {"answer": "Gögnin svara ekki spurningunni.", "parts": [SERIES], "model": "Gemma 4 31B", "confidence": "high",
         "unsupported": []}
    page = app.render(guard.check("Hvað bjuggu margir í Reykjavík 1950?"), r, 1.0)
    assert "Gögnin á bak við svarið" not in page and "Vissa" not in page and "bornar saman" not in page
    assert "Fyrir forritara" in page and "Gemma 4 31B" in page
    other = {**r, "answer": "Árið 2026 bjuggu 394.324 á Íslandi.", "model": "gemma4:e2b"}
    assert "Vissa" not in app.render(guard.check("Hvað búa margir á Íslandi?"), other, 1.0)  # not calibrated


def test_still_current_reads_the_same_cells_live(monkeypatch):
    from hagstofa import px
    parts = [{"id": "MAN00101", "path": POP["path"], "edition": "is", "selections": {"Ár": ["2026"]},
              "values": [394324.0]}]
    calls = []

    def live(method, url, attempts=5, **kw):
        calls.append((url, kw["json"]))
        return {"id": ["Ár"], "size": [1], "dimension": {"Ár": {"label": "Ár", "category": {"index": {"2026": 0},
                "label": {"2026": "2026"}}}}, "value": [394324.0]}
    monkeypatch.setattr(px, "_request", live)
    assert app.still_current(parts, "is") == "Athugað beint á PX-Web núna: tölurnar í MAN00101 eru óbreyttar."
    assert calls[0][0].endswith("/MAN00101.px") and calls[0][1]["query"][0]["selection"]["values"] == ["2026"]
    monkeypatch.setattr(px, "_request", lambda *a, **k: {**live(*a, **k), "value": [394500.0]})
    assert "PX-Web now shows other numbers in MAN00101: 394,500 (the answer used 394,324)" in app.still_current(parts, "en")

    def down(*a, **k):
        raise TimeoutError("slow")
    monkeypatch.setattr(px, "_request", down)
    assert app.still_current(parts, "is").startswith("Náði ekki sambandi við PX-Web núna (TimeoutError)")


def test_the_still_current_chip_checks_instead_of_asking(monkeypatch):
    monkeypatch.setattr(app, "still_current", lambda parts, lang: f"checked {len(parts)}")
    last = {"question": "Hvað búa margir á Íslandi?", "lang": "is", "parts": [{"id": "MAN00101"}],
            "chips": [(app.STILL_CURRENT_LABEL["is"], app.STILL_CURRENT)]}
    *_, final = list(app._chip(0)([], last))
    history = final[1]
    assert history[-2]["content"] == app.STILL_CURRENT_LABEL["is"] and history[-1]["content"] == "checked 1"
    assert final[2]["chips"] == []  # the check is used up

import pytest

from hagstofa import llm, pipeline, px
from hagstofa.pipeline import flatten, pick_table, validate_selections

TABLE = {
    "variables_is": [
        {"code": "Kyn", "text": "Kyn", "values": ["0", "1", "2"], "valueTexts": ["Alls", "Karlar", "Konur"]},
        {"code": "Ár", "text": "Ár", "time": True, "values": ["2024", "2025"], "valueTexts": ["2024", "2025"]},
        {"code": "Aldur", "text": "Aldur", "elimination": True, "values": ["a", "b"], "valueTexts": ["0-9", "10-19"]},
    ]
}


def test_validate_drops_unknown_codes_and_fills_defaults():
    sel = validate_selections(TABLE, {"Kyn": ["2", "bogus"], "Nope": ["x"]})
    assert sel == {"Kyn": ["2"], "Ár": ["2025"]}  # time -> latest; eliminable var omitted


def test_validate_reads_variables_and_values_named_as_the_prompt_shows_them():
    # run 6, Gemma 4: {"tími": ["2026M08"], "Rate": ["Atvinnuleysi..."]} lost the month and the measure
    table = {"variables_en": [
        {"code": "Mánuður", "text": "Month", "time": True, "values": ["2026M07", "2026M08", "2026M09"],
         "valueTexts": ["2026M07", "2026M08", "2026M09"]},
        {"code": "Hlutfall, %", "text": "Rate", "values": ["Atvinnuþátttaka", "Atvinnuleysi"],
         "valueTexts": ["Activity rate", "Unemployment rate"]},
        {"code": "Kyn", "text": "Sex", "values": ["0", "1"], "valueTexts": ["Total", "Males"]},
    ]}
    raw = {"tími": ["2026M08"], "Rate": ["Atvinnuleysi"], "Sex": ["Males"]}
    assert validate_selections(table, raw, "en") == {
        "Mánuður": ["2026M08"], "Hlutfall, %": ["Atvinnuleysi"], "Kyn": ["1"]}
    assert pipeline.assumptions(table, raw, "en") == []
    assert validate_selections(table, {"Month": "2026M07", "rate": ["unemployment rate"]}, "en") == {
        "Mánuður": ["2026M07"], "Hlutfall, %": ["Atvinnuleysi"], "Kyn": ["0"]}
    # unknown variables and values are still dropped; the code's own key wins over a label
    assert validate_selections(table, {"Mánuður": ["2026M07"], "tími": ["2026M08"], "Rate": ["x"], "Foo": ["1"]},
                               "en") == {"Mánuður": ["2026M07"], "Hlutfall, %": ["Atvinnuþátttaka"], "Kyn": ["0"]}


def test_validate_prefers_total_value():
    assert validate_selections(TABLE, {})["Kyn"] == ["0"]


def test_flatten_row_major():
    ds = {
        "id": ["Kyn", "Ár"],
        "dimension": {
            "Kyn": {"label": "Kyn", "category": {"index": {"1": 0, "2": 1}, "label": {"1": "Karlar", "2": "Konur"}}},
            "Ár": {"label": "Ár", "category": {"index": ["2024", "2025"], "label": {"2024": "2024", "2025": "2025"}}},
        },
        "value": [1, 2, 3, 4],
    }
    rows = flatten(ds)
    assert rows[1] == {"Kyn": "Karlar", "Ár": "2025", "value": 2}
    assert rows[2] == {"Kyn": "Konur", "Ár": "2024", "value": 3}


def test_pick_table_accepts_quoted_index(monkeypatch):
    cands = [({"id": t, "title_is": t, "variables_is": []}, 1.0) for t in ("A", "B")]
    for reply, want in [({"table": "1"}, "B"), ({"table": 0}, "A"), ({"table": -1}, None), ({"table": "x"}, None)]:
        monkeypatch.setattr(llm, "chat_json", lambda s, u, r=reply: r)
        got = pick_table("q", cands)
        assert (got and got["id"]) == want


def test_fresh_first_leaves_stale_tables_out_when_no_period_is_asked():
    cands = [({"id": i, "updated": u}, 1.0) for i, u in
             [("HEI11100", "2019-03-08T08:59"), ("MAN00101", "2026-03-12T08:57"), ("X", None), ("Y", "2024-01-01")]]
    ids = lambda c: [t["id"] for t, _ in c]  # noqa: E731
    assert ids(pipeline.fresh_first("Hvað búa margir á Íslandi?", cands)) == ["MAN00101", "X", "Y"]
    assert ids(pipeline.fresh_first("Hvað bjuggu margir á Íslandi árið 2017?", cands)) == ["HEI11100", "MAN00101",
                                                                                         "X", "Y"]
    old = [({"id": "A", "updated": "2010-01-01"}, 1.0), ({"id": "B", "updated": "2009-01-01"}, 1.0)]
    assert ids(pipeline.fresh_first("Hvað búa margir?", old)) == ["A", "B"]  # all old: none dropped
    # the data's own newest year counts, not the update stamp: FEL02001 updated 2024 but ends in 2022 (dev3, 1 Nov)
    year = lambda *ys: [{"code": "Ár", "text": "Ár", "time": True, "values": list(ys), "valueTexts": list(ys)}]  # noqa: E731
    pensions = [({"id": "FEL02001", "updated": "2024-02-22", "variables_is": year("2021", "2022")}, 1.0),
                ({"id": "TEK02014", "updated": "2026-10-07", "variables_is": year("2026M07", "2026M08")}, 1.0)]
    assert ids(pipeline.fresh_first("Hvað fá lífeyrisþegar á mánuði?", pensions)) == ["TEK02014"]


def test_no_assumptions_line_under_an_answer_without_numbers():
    parts = [{"assumed": [("Year", "2024"), ("Sex", "Total")]}]
    assert pipeline.with_assumptions("The data does not answer the question.", parts, "en") == \
        "The data does not answer the question."
    assert pipeline.with_assumptions("In 2024 there were 4,387.", parts, "en").endswith("Assuming: Year 2024, Sex Total.")


def test_pick_table_shows_each_tables_unit(monkeypatch):
    cands = [({"id": t, "title_is": "Landsframleiðsla", "path": f"{t}.px", "variables_is": []}, 1.0)
             for t in ("THJ01000", "THJ01102", "MAN00101")]
    units = {"THJ01000.px": "Vísitala/hlutfall", "THJ01102.px": "Milljónir króna"}
    monkeypatch.setattr(pipeline.local_data, "unit", lambda path, lang: units.get(path))
    seen = {}
    monkeypatch.setattr(llm, "chat_json", lambda s, u: seen.setdefault("listing", u) and {"table": 1})
    assert pick_table("Hver var landsframleiðslan 2025?", cands)["id"] == "THJ01102"
    lines = seen["listing"].splitlines()
    assert lines[-3].endswith("| eining: Vísitala/hlutfall") and lines[-2].endswith("| eining: Milljónir króna")
    assert "eining" not in lines[-1]  # no unit known: nothing shown


def test_web_url():
    assert px.web_url("Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00101.px") == (
        "https://px.hagstofa.is/pxis/pxweb/is/Ibuar/"
        "Ibuar__mannfjoldi__1_yfirlit__Yfirlit_mannfjolda/MAN00101.px"
    )


def test_parse_json_takes_first_object():
    assert llm.parse_json('{"a": 1}') == {"a": 1}
    assert llm.parse_json('{"a": 1} {"a": 2}') == {"a": 1}  # "Extra data" from qwen3:4b
    assert llm.parse_json('Svar:\n```json\n{"a": {"b": [1]}}\n```') == {"a": {"b": [1]}}


def test_px_retries_network_errors(monkeypatch):
    import requests

    class Ok:
        status_code, encoding = 200, None

        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    calls = []

    def flaky(method, url, timeout, **kw):
        calls.append(url)
        if len(calls) < 3:
            raise requests.ReadTimeout("slow server")
        return Ok()

    monkeypatch.setattr(px._session, "request", flaky)
    monkeypatch.setattr(px.time, "sleep", lambda s: None)
    assert px._request("GET", "u") == {"ok": True} and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(px._session, "request", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    try:
        px._request("GET", "u", attempts=2)
        raise AssertionError("should raise after the last attempt")
    except requests.ConnectionError:
        pass


def test_llm_reasoning_effort_only_when_set(monkeypatch):
    from types import SimpleNamespace

    from hagstofa import llm
    seen = []
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="já"))])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: seen.append(kw) or reply)))
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(llm, "REASONING", None)
    llm.chat("s", "u")
    monkeypatch.setattr(llm, "REASONING", "none")
    llm.chat("s", "u")
    assert "reasoning_effort" not in seen[0] and seen[1]["reasoning_effort"] == "none"


def test_answer_language_is_decided_in_code_and_comes_last(monkeypatch):
    """English questions get an English instruction as the prompt's last line and the English table title;
    the numbers still come from the Icelandic table."""
    from hagstofa import pipeline
    seen = {}
    monkeypatch.setattr(llm, "chat", lambda system, user: seen.setdefault("user", user))
    part = {"table": {"id": "MAN00101", "title_is": "Mannfjöldi", "title_en": "Population"},
            "rows": [{"Ár": "2026", "value": 394324}]}
    pipeline.write_answer("How many people lived in Iceland on 1 January 2026?", [part])
    assert seen["user"].endswith(pipeline.ANSWER_IN["en"]) and "MAN00101: Population" in seen["user"]
    seen.clear()
    pipeline.write_answer("Hve margir bjuggu á Íslandi 1. janúar 2026?", [part])
    assert seen["user"].endswith(pipeline.ANSWER_IN["is"]) and "MAN00101: Mannfjöldi" in seen["user"]


def test_no_table_message_in_the_question_language(monkeypatch):
    from hagstofa import pipeline

    class Nothing:
        def search(self, q, k=8):
            return []
    monkeypatch.setattr(pipeline, "decompose", lambda q: [q])
    monkeypatch.setattr(pipeline, "pick_tables", lambda q, c, v=(), second=False: [])
    assert pipeline.answer("How many trolls live in Iceland?", Nothing())["answer"] == pipeline.NO_TABLE["en"]
    assert pipeline.answer("Hve mörg tröll búa á Íslandi?", Nothing())["answer"] == pipeline.NO_TABLE["is"]


# Questions about a year a table doesn't have (31 Oct): the latest year must not stand in for it.
PERIOD_TABLE = {"id": "MAN03270", "variables_is": [
    {"code": "Ár", "time": True, "values": [str(y) for y in range(1998, 2027)],
     "valueTexts": [str(y) for y in range(1998, 2027)]}]}


@pytest.mark.parametrize("question, years", [
    ("What will the population of Reykjavík be in 2035?", {2035}), ("Hve margir bjuggu hér árið 1200?", {1200}),
    ("Hve mörg börn fæddust á hverja 1000 íbúa árið 2025?", {2025}), ("births per 1000 inhabitants", set()),
    ("frá 1999 til 2019", {1999, 2019}), ("Hvað kostaði 2500 kr?", set()),
])
def test_years_in(question, years):
    assert pipeline.years_in(question) == years


def test_period_gap_only_when_no_asked_year_is_in_the_table():
    assert pipeline.period_gap("population in 2035", PERIOD_TABLE) == (1998, 2026)
    assert pipeline.period_gap("population from 2020 to 2035", PERIOD_TABLE) is None  # 2026 data exists, 2020 not: fine
    assert pipeline.period_gap("population now", PERIOD_TABLE) is None
    numbered = {"id": "X", "variables_is": [{"code": "Ár", "time": True, "values": ["0", "1"], "valueTexts": ["2014", "2015"]}]}
    assert pipeline.table_years(numbered) == {2014, 2015}


# Run 4: the model named no operation for share and per-1,000 questions; code then builds the plan itself.
CROSS = [{"tafla": "MAN03270", "Ár": "2026", "value": 139804.0}, {"tafla": "MAN00000", "Ár": "2026", "value": 394324.0}]
BIRTHS = [{"tafla": "MAN05101", "Ár": "2025", "value": 4387.0}, {"tafla": "MAN03250", "Ár": "2025", "value": 389444.0}]
STAYS = [{"tafla": "SAM01601", "Ár": "2025", "value": 8_700_000.0}, {"tafla": "MAN03250", "Ár": "2025", "value": 389444.0}]


@pytest.mark.parametrize("question, rows, op, a, b", [
    ("What percentage of Iceland's population lived in Reykjavík?", CROSS, "share", 0, 1),
    ("Hve stór hluti landsmanna bjó í Reykjavík?", CROSS, "share", 0, 1),
    ("How many children were born per 1,000 inhabitants?", BIRTHS, "per_1000", 0, 1),
    ("Hve margar gistinætur voru á hvern íbúa?", STAYS, "ratio", 0, 1),  # bigger number, population still the base
    ("What share were women?", [{"Kyn": "Konur", "value": 191927.0}, {"Kyn": "Alls", "value": 394324.0}], "share", 0, 1),
])
def test_backup_plan(question, rows, op, a, b):
    assert pipeline.backup_plan(question, rows) == {"operation": op, "a": a, "b": b}


@pytest.mark.parametrize("question, rows", [
    ("Hvert var hlutfall atvinnulausra 2025 og 2026?", [{"Ár": "2025", "value": 3.9}, {"Ár": "2026", "value": 4.4}]),
    ("How many lived in Reykjavík and in Iceland?", CROSS),  # two numbers side by side: no calculation
])
def test_backup_plan_leaves_other_questions_alone(question, rows):
    assert pipeline.backup_plan(question, rows) is None


def test_plan_calculation_uses_the_backup_when_the_model_says_none(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda system, user: {"operation": "none"})
    c = pipeline.plan_calculation("What percentage of Iceland's population lived in Reykjavík?", CROSS)
    assert c["operation"] == "share" and round(c["result"], 2) == 35.45 and c["text"].endswith("35.5%")


# Stated assumptions (31 Oct): what code chose because the question didn't, said in the answer.
ASSUME_TABLE = {"id": "MAN00101", "variables_is": [
    {"code": "Ár", "text": "Ár", "time": True, "values": ["2025", "2026"], "valueTexts": ["2025", "2026"]},
    {"code": "Kyn", "text": "Kyn", "values": ["0", "1", "2"], "valueTexts": ["Alls", "Karlar", "Konur"]},
    {"code": "Aldur", "text": "Aldur", "elimination": True, "values": ["-1", "0"], "valueTexts": ["Alls", "0 ára"]}]}


def test_assumptions_name_the_defaults_code_filled():
    assert pipeline.assumptions(ASSUME_TABLE, {}) == [("Ár", "2026"), ("Kyn", "Alls")]  # Aldur: eliminated, not chosen
    assert pipeline.assumptions(ASSUME_TABLE, {"Ár": ["2025"], "Kyn": ["2"]}) == []
    assert pipeline.assumptions(ASSUME_TABLE, {"Ár": ["1900"]}) == [("Ár", "2026"), ("Kyn", "Alls")]  # invalid code


def test_with_assumptions_adds_one_line_in_the_answers_language():
    parts = [{"assumed": [("Year", "2026"), ("Sex", "Total")]}, {"assumed": [("Year", "2026")]}]
    assert pipeline.with_assumptions("394,324 people.", parts, "en") == "394,324 people.\n\nAssuming: Year 2026, Sex Total."
    assert pipeline.with_assumptions("Svar.", [{"assumed": []}], "is") == "Svar."


# Relative dates (31 Oct): made explicit in code.
@pytest.mark.parametrize("question, resolved", [
    ("Hvað fæddust mörg börn í fyrra?", "Hvað fæddust mörg börn í fyrra (2025)?"),
    ("How many tourists came last year?", "How many tourists came last year (2025)?"),
    ("Hver var verðbólgan í síðasta mánuði?", "Hver var verðbólgan í síðasta mánuði (nýjasti mánuður í gögnunum)?"),
    ("What was unemployment last quarter?", "What was unemployment last quarter (the newest quarter in the data)?"),
    ("Hvað búa margir á Íslandi?", "Hvað búa margir á Íslandi?"),
    # without accents (dev3, 1 Nov: "i fyrra" was answered for 2024)
    ("hvad fengu margir rikisborgararett i fyrra", "hvad fengu margir rikisborgararett i fyrra (2025)"),
    ("hver var verdbolgan a sidasta arsfjordungi", "hver var verdbolgan a sidasta arsfjordungi (nýjasti ársfjórðungur í gögnunum)"),
    ("Hver var velta gististaða árið sem leið?", "Hver var velta gististaða árið sem leið (2025)?"),
    ("hvad faeddust morg born a thessu ari", "hvad faeddust morg born a thessu ari (2026)"),
    ("Hvar bý ég?", "Hvar bý ég?"),
])
def test_resolve_dates(question, resolved):
    import datetime
    assert pipeline.resolve_dates(question, today=datetime.date(2026, 10, 31)) == resolved


def test_resolve_dates_in_january_looks_back_a_year():
    import datetime
    january = datetime.date(2027, 1, 15)
    assert pipeline.resolve_dates("How many were born last year?", january) == "How many were born last year (2026)?"


def test_measures_lists_what_a_table_measures():
    t = {"variables_is": [
        {"code": "Ár", "text": "Ár", "time": True, "values": ["2025"], "valueTexts": ["2025"]},
        {"code": "Kyn", "text": "Kyn", "values": ["0", "1", "2"], "valueTexts": ["Alls", "Karlar", "Konur"]},
        {"code": "H", "text": "Hlutfall, %", "values": ["a", "b"], "valueTexts": ["Atvinnuþátttaka", "Atvinnuleysi"]}]}
    assert pipeline.measures(t) == "Hlutfall, %: Atvinnuþátttaka, Atvinnuleysi"


@pytest.mark.parametrize("reply, ok", [({"answers": True}, True), ({"answers": False, "why": "x"}, False),
                                       ({"answers": "false"}, False), ({}, True)])
def test_rows_answer_reads_the_models_verdict(monkeypatch, reply, ok):
    monkeypatch.setattr(llm, "chat_json", lambda s, u: reply)
    part = {"table": {"id": "T", "title_is": "T"}, "rows": [{"value": 1}]}
    assert pipeline.rows_answer("q", part)[0] is ok


def test_the_check_reads_the_tables_note_when_switched_on(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "chat_json", lambda s, u: seen.append(u) or {"answers": True})
    population = {"id": "MAN00101", "title_is": "Mannfjöldi",
                  "path": "Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00101.px"}
    part = {"table": population, "rows": [{"value": 1}]}
    pipeline.rows_answer("Hvað búa margir á Íslandi?", part)
    monkeypatch.setattr(pipeline, "CHECK_NOTES", True)
    pipeline.rows_answer("Hvað búa margir á Íslandi?", part)
    assert "Athugasemd" not in seen[0] and "Athugasemd töflunnar: Miðað er við 1. janúar ár hvert." in seen[1]


def test_a_table_that_fails_the_check_gives_way_to_the_next(monkeypatch):
    monkeypatch.setattr(pipeline, "answerable", lambda found, r, s: found)  # suggestions are tested on their own
    t1, t2 = ({"id": i, "title_is": i, "path": f"{i}.px", "variables_is": []} for i in ("T1", "T2"))

    class R:
        def search(self, q, k=8):
            return [(t1, 1.0), (t2, 0.9)]

    monkeypatch.setattr(pipeline, "decompose", lambda q: [q])
    monkeypatch.setattr(pipeline, "pick_tables", lambda q, c, v=(), second=False: [t1])
    monkeypatch.setattr(pipeline, "fetch_part", lambda q, t, calc=True, lang="is": {
        "question": q, "table": t, "url": "", "selections": {}, "rows": [{"value": 5}], "fallback": [],
        "calculation": None, "assumed": []})
    monkeypatch.setattr(pipeline, "agreement", lambda r, q, t: 1)
    monkeypatch.setattr(pipeline, "write_answer", lambda *a, **k: "It was 5.")
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (p["table"]["id"] == "T2", "wrong measure"))
    r = pipeline.answer("How many?", R())
    assert r["table"]["id"] == "T2" and "did not answer" in r["parts"][0]["fallback"][0]
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (False, "wrong measure"))
    r = pipeline.answer("How many?", R())  # none passes: declined, no table cited
    assert "parts" not in r and r["answer"].startswith(pipeline.NO_TABLE["en"]) and r["suggestions"]
    monkeypatch.setattr(pipeline, "DECLINE_UNCHECKED", False)
    r = pipeline.answer("How many?", R())  # switched off: the picked table is kept, and says so
    assert r["table"]["id"] == "T1" and "did not pass the check" in r["parts"][0]["fallback"][0]


def test_validate_reads_the_prompts_own_formats():
    # Gemma, 31 Oct: the variable as "code (label)", a value as "code=label" (the National Church, Muslims)
    table = {"variables_en": [
        {"code": "Trú- og lífsskoðunarfélög", "text": "Religious", "values": ["-", "1", "J", "/"],
         "valueTexts": ["Total", "The National Church", "Muslim Association", "Islamic Foundation"]},
        {"code": "Ár", "text": "Year", "time": True, "values": ["2025", "2026"], "valueTexts": ["2025", "2026"]}]}
    raw = {"Trú- og lífsskoðunarfélög (Religious)": ["1"], "Ár [tími]": ["2025"]}
    assert validate_selections(table, raw, "en") == {"Trú- og lífsskoðunarfélög": ["1"], "Ár": ["2025"]}
    raw = {"Trú- og lífsskoðunarfélög": ["J", "/=Islamic Foundation", "x=nothing"]}
    assert validate_selections(table, raw, "en")["Trú- og lífsskoðunarfélög"] == ["J", "/"]


def test_measures_puts_the_questions_values_first():
    t = {"variables_is": [{"code": "B", "text": "Búpeningur", "values": [str(i) for i in range(9)],
                           "valueTexts": ["Nautgripir alls", "Mjólkurkýr", "Holdakýr", "Kvígur", "Geldneyti", "Kálfar",
                                          "Sauðfé alls", "Ær", "Hross"]}]}
    assert pipeline.measures(t, "Hversu margt sauðfé var á landinu árið 2024?").startswith("Búpeningur: Sauðfé alls")


def test_a_per_head_question_with_a_direct_table_is_not_split(monkeypatch):
    direct = {"id": "THJ01401", "title_is": "Landsframleiðslan á mann 1980-2024", "path": "x.px", "variables_is": []}
    other = {"id": "THJ01102", "title_is": "Landsframleiðsla og þjóðartekjur", "path": "y.px", "variables_is": []}

    class R:
        tables = []

        def search(self, q, k=8):
            return [(direct, 1.0), (other, 0.9)]

    seen = []
    monkeypatch.setattr(pipeline, "decompose", lambda q: ["Landsframleiðsla 2024?", "Mannfjöldi 2024?"])
    monkeypatch.setattr(pipeline, "period_gap", lambda q, t: None)
    monkeypatch.setattr(pipeline, "pick_tables", lambda q, c, v=(), second=False: seen.append(q) or [direct])
    monkeypatch.setattr(pipeline, "fetch_part", lambda q, t, calc=True, lang="is": {
        "question": q, "table": t, "url": "", "selections": {}, "rows": [{"value": 11927}], "fallback": [],
        "calculation": None, "assumed": []})
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (True, ""))
    monkeypatch.setattr(pipeline, "agreement", lambda r, q, t: 1)
    monkeypatch.setattr(pipeline, "write_answer", lambda *a, **k: "11.927 þúsund krónur.")
    pipeline.answer("Hver var landsframleiðsla á mann árið 2024?", R())
    assert seen == ["Hver var landsframleiðsla á mann árið 2024?"]  # one part, the question as asked


def test_short_names_and_missing_accents_keep_their_value_in_the_list():
    # 127 towns are trimmed to the ones the question names: "Vik" must keep "Vík í Mýrdal"
    towns = [f"Bær {i}" for i in range(60)] + ["Vík í Mýrdal"] + [f"Bær {i}" for i in range(60, 120)]  # mid-list
    table = {"variables_en": [{"code": "B", "text": "Towns", "values": [str(i) for i in range(len(towns))],
                               "valueTexts": towns}]}
    assert "Vík í Mýrdal" in pipeline.describe_variables(table, "How many people live in Vik?", "en")
    assert "Vík í Mýrdal" not in pipeline.describe_variables(table, "How many people live in the country?", "en")


def test_the_check_is_told_todays_date(monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "chat_json", lambda s, u: seen.setdefault("u", u) and {"answers": True})
    pipeline.rows_answer("Hvað búa margir á Húsavík núna?", {"table": {"id": "T", "title_is": "T"}, "rows": [{"value": 1}]})
    import datetime
    assert seen["u"].startswith(f"Í dag er {datetime.date.today().isoformat()}")


def test_the_pickers_second_choice_comes_before_the_search_rank(monkeypatch):
    tables = [{"id": i, "title_is": i, "path": f"{i}.px", "variables_is": []} for i in ("A", "B", "C")]
    monkeypatch.setattr(llm, "chat_json", lambda s, u: {"table": 0, "second": 2})
    assert [t["id"] for t in pipeline.pick_tables("q", [(t, 1.0) for t in tables], second=True)] == ["A", "C"]
    assert [t["id"] for t in pipeline.pick_tables("q", [(t, 1.0) for t in tables])] == ["A"]  # off: first only
    monkeypatch.setattr(llm, "chat_json", lambda s, u: {"table": -1, "second": 1})
    assert pipeline.pick_tables("q", [(t, 1.0) for t in tables], second=True) == []  # declined: no first choice


def test_a_look_alike_with_another_number_is_named_under_the_answer(monkeypatch):
    t1, t2 = ({"id": i, "title_is": i, "title_en": i, "path": f"{i}.px", "variables_is": []} for i in ("T1", "T2"))
    values = {"T1": 20068.4, "T2": 19136.2}

    class R:
        tables = []

        def search(self, q, k=8):
            return [(t1, 1.0), (t2, 0.9)]

    monkeypatch.setattr(pipeline, "ALSO_TABLE", True)
    monkeypatch.setattr(pipeline, "decompose", lambda q: [q])
    monkeypatch.setattr(pipeline, "pick_tables", lambda q, c, v=(), second=False: [t1, t2] if second else [t1])
    monkeypatch.setattr(pipeline, "fetch_part", lambda q, t, calc=True, lang="is": {
        "question": q, "table": t, "url": "", "selections": {}, "rows": [{"Ár": "2025", "value": values[t["id"]]}],
        "fallback": [], "calculation": None, "assumed": [], "unit": "Milljónir króna"})
    monkeypatch.setattr(pipeline, "agreement", lambda r, q, t: 1)
    monkeypatch.setattr(pipeline, "write_answer", lambda *a, **k: "It was 20,068.4 million.")
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (True, ""))
    r = pipeline.answer("How much did households pay for dental care in 2025?", R())
    assert r["table"]["id"] == "T1" and [p["table"]["id"] for p in r["parts"]] == ["T1", "T2"]
    assert r["answer"].endswith("Another Statistics Iceland table, T2 (T2), shows 19,136.2 Milljónir króna (2025).")
    assert r["parts"][1]["also"] and not r["parts"][0].get("also")
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (p["table"]["id"] == "T1", "other measure"))
    assert len(pipeline.answer("How much did households pay for dental care in 2025?", R())["parts"]) == 1
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (True, ""))
    values["T2"] = 20068.4  # the same number: nothing to add
    assert "Another" not in pipeline.answer("How much did households pay for dental care in 2025?", R())["answer"]
    monkeypatch.setattr(pipeline, "ALSO_TABLE", False)
    values["T2"] = 1.0
    assert len(pipeline.answer("How much did households pay for dental care in 2025?", R())["parts"]) == 1


def test_known_slips_in_icelandic_answers_are_fixed():
    assert pipeline.fix_slips("Árið 2026 býa 394.324 manns á Íslandi.") == "Árið 2026 búa 394.324 manns á Íslandi."
    assert pipeline.fix_slips("Fjöldi þingmenna var 63.") == "Fjöldi þingmanna var 63."
    assert pipeline.fix_slips("Ég bý í Reykjavík; þeir búa þar.") == "Ég bý í Reykjavík; þeir búa þar."


def test_the_also_line_leaves_out_mixed_units_and_writes_percent():
    part = {"table": {"id": "SJA09042", "title_is": "Afli", "title_en": "Catch"}, "unit": "Tonnes/1000 ISK",
            "rows": [{"Place": "Vestmannaeyjar", "Year": "2024", "value": 122487.785}]}
    assert pipeline.also_line(part, "en") == ("Another Statistics Iceland table, SJA09042 (Catch), shows 122,487.8 "
                                              "(Vestmannaeyjar, 2024).")
    part.update(unit="Hlutfall", rows=[{"Ár": "2025", "value": 50.0}])
    assert pipeline.also_line(part, "is") == "Önnur tafla Hagstofunnar, SJA09042 (Afli), sýnir 50% (2025)."


def test_the_also_line_reads_share_rows_and_skips_a_whole():
    # dev3, 1 Nov: THJ05611's "Innbyrðis hlutdeild" = 100 was shown as "100 Milljónir króna"
    main = {"table": {"id": "THJ05631"}, "rows": [{"Ár": "2025", "value": 15.23}], "unit": "Hlutfall"}
    whole = {"table": {"id": "THJ05611", "title_is": "Fræðsluútgjöld"}, "unit": "Milljónir króna",
             "rows": [{"Eining": "Innbyrðis hlutdeild", "Ár": "2025", "value": 100.0}]}
    assert not pipeline.differs(main, whole)
    whole["rows"][0]["value"] = 37.2
    assert pipeline.differs(main, whole)
    assert pipeline.also_line(whole, "is").endswith("sýnir 37,2% (Innbyrðis hlutdeild, 2025).")


def test_a_year_not_published_yet_gets_the_newest_figures(monkeypatch):
    import datetime
    today = datetime.date(2026, 11, 1)
    years = {"id": "MAN05100", "title_is": "Lifandi fæddir", "variables_is": [
        {"code": "Ár", "text": "Ár", "time": True, "values": [str(y) for y in range(2010, 2026)],
         "valueTexts": [str(y) for y in range(2010, 2026)]}]}
    names = {"id": "MAN11101", "title_is": "Eiginnöfn karla 1. janúar 2023", "variables_is": []}
    old = {"id": "SOG09002", "title_is": "Hitastig 1949-2017", "variables_is": []}
    monkeypatch.setattr(pipeline, "NOT_YET", False)
    assert pipeline.not_yet_published("Hvað fæddust mörg börn árið 2026?", years, today) is None  # switched off
    monkeypatch.setattr(pipeline, "NOT_YET", True)
    assert pipeline.not_yet_published("Hvað fæddust mörg börn árið 2026?", years, today) == ([2026], 2025)
    assert pipeline.not_yet_published("Hvað fæðast mörg börn árið 2035?", years, today) is None  # the future
    assert pipeline.not_yet_published("Hvað fæddust mörg börn árið 1200?", years, today) is None
    assert pipeline.not_yet_published("Hvað fæddust mörg börn árið 2024?", years, today) is None  # it has 2024
    assert pipeline.not_yet_published("Hve margir heita Jón 1. janúar 2025?", names, today) == ([2025], 2023)
    assert pipeline.not_yet_published("Hver var meðalhitinn í júlí 2025?", old, today) is None  # 2017: too old
    part = {"not_yet": ([2025], 2023)}
    assert pipeline.not_yet_note(part, "is") == ("Tölur fyrir 2025 eru ekki komnar út hjá Hagstofunni; þetta eru "
                                                 "nýjustu tölurnar, frá 2023.")
    seen = []
    monkeypatch.setattr(llm, "chat_json", lambda s, u: seen.append(u) or {"answers": True})
    pipeline.rows_answer("Hve margir heita Jón 1. janúar 2025?", {"table": names, "rows": [{"value": 5052}],
                                                                  "not_yet": ([2025], 2023)})
    assert "Tölur fyrir 2025 eru ekki komnar út; nýjustu tölur töflunnar eru frá 2023" in seen[0]
    assert "Spurning: Hve margir heita Jón 1. janúar 2023?" in seen[0]  # judged as about the newest year


def test_english_labels_for_tables_with_no_english_edition(monkeypatch, tmp_path):
    import json
    from hagstofa import retrieve
    monkeypatch.setattr(retrieve, "DATA", tmp_path)
    english = {"Kyn": "Sex", "Alls": "Total", "Ár": "Year"}
    calls = []
    monkeypatch.setattr(llm, "chat_json", lambda s, u: calls.append(u) or {"labels": [english[x] for x in json.loads(u)]})
    part = lambda edition: {"edition": edition, "assumed": [("Kyn", "Alls")],  # noqa: E731
                            "rows": [{"Ár": "2025", "Kyn": "Alls", "value": 5.0}]}
    p = part("is")
    pipeline.english_labels([p])
    assert p["assumed"] == [("Sex", "Total")] and p["rows"] == [{"Year": "2025", "Sex": "Total", "value": 5.0}]
    pipeline.english_labels([part("is")])
    assert len(calls) == 1  # cached
    p = part("en")
    pipeline.english_labels([p])
    assert p["assumed"] == [("Kyn", "Alls")]  # an English edition is left alone
    monkeypatch.setattr(llm, "chat_json", lambda s, u: (_ for _ in ()).throw(ValueError("down")))
    p = {"edition": "is", "assumed": [("Aldur", "Alls")], "rows": []}
    pipeline.english_labels([p])
    assert p["assumed"] == [("Aldur", "Total")]  # "Alls" from the cache, "Aldur" kept when the model fails


def test_a_comparison_of_two_years_states_the_change(monkeypatch):
    # set 1, 1 Nov: "var hún meiri eða minni en 2021?" gave 80,22% and 80,09% and "meiri", not by how much
    rows = [{"Ár": "2021, 25. september", "Skipting": "Kosningaþátttaka, %", "value": 80.090028516886},
            {"Ár": "2024, 30. nóvember", "Skipting": "Kosningaþátttaka, %", "value": 80.224589370181}]
    q = "Hver var kjörsóknin í alþingiskosningunum 2024 og var hún meiri eða minni en 2021?"
    monkeypatch.setattr(pipeline, "COMPARE", False)
    assert pipeline.backup_plan(q, rows) is None  # switched off
    monkeypatch.setattr(pipeline, "COMPARE", True)
    assert pipeline.backup_plan(q, rows) == {"operation": "difference", "a": 1, "b": 0}
    assert round(pipeline.calculate(pipeline.backup_plan(q, rows), rows)["result"], 4) == 0.1346
    assert pipeline.backup_plan("Hver var kjörsóknin 2024 og 2021?", rows) is None  # no comparison asked
    two_places = [{"Svæði": "Reykjavík", "value": 1.0}, {"Svæði": "Akureyri", "value": 2.0}]
    assert pipeline.backup_plan("Er meiri eða minni kjörsókn í Reykjavík en á Akureyri?", two_places) is None


def test_the_search_first_is_read_alongside_when_the_picker_leaves_it(monkeypatch):
    t1, t2, t3 = ({"id": i, "title_is": i, "title_en": i, "path": f"{i}.px", "variables_is": []} for i in ("T1", "T2", "T3"))
    values = {"T1": 10.0, "T2": 20.0, "T3": 30.0}

    class R:
        tables = []

        def search(self, q, k=8):
            return [(t1, 1.0), (t2, 0.9), (t3, 0.8)]

    monkeypatch.setattr(pipeline, "ALSO_TABLE", True)
    monkeypatch.setattr(pipeline, "ALSO_FROM_SEARCH", True)
    monkeypatch.setattr(pipeline, "decompose", lambda q: [q])
    monkeypatch.setattr(pipeline, "pick_tables", lambda q, c, v=(), second=False: [t2, t3])
    monkeypatch.setattr(pipeline, "fetch_part", lambda q, t, calc=True, lang="is": {
        "question": q, "table": t, "url": "", "selections": {}, "rows": [{"Ár": "2025", "value": values[t["id"]]}],
        "fallback": [], "calculation": None, "assumed": []})
    monkeypatch.setattr(pipeline, "agreement", lambda r, q, t: 1)
    monkeypatch.setattr(pipeline, "write_answer", lambda *a, **k: "It was 20.")
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (True, ""))
    r = pipeline.answer("How many in 2025?", R())
    assert [p["table"]["id"] for p in r["parts"]] == ["T2", "T1"]  # the search's #1, not the picker's second (T3)


def test_a_faroese_slip_is_fixed():
    assert pipeline.fix_slips("Árið 2000 føddust 4.315 börn.") == "Árið 2000 fæddust 4.315 börn."
    assert pipeline.fix_slips("Føddir alls") == "Fæddir alls"
    assert pipeline.fix_slips("5.052 mannlásir heita Jón") == "5.052 manns heita Jón"
    assert pipeline.fix_slips("- 2020: 354.042stk.") == "- 2020: 354.042"


def test_slips_fix_gemmas_bjuggu_and_whole_numbers_with_a_point_zero():
    assert pipeline.fix_slips("Árið 2020 býðu 37.036 manns.") == "Árið 2020 bjuggu 37.036 manns."
    assert pipeline.fix_slips("- 2015: 4107.0\n- 2016: 4026.0") == "- 2015: 4107\n- 2016: 4026"
    assert pipeline.fix_slips("Atvinnuleysi var 6.0% og 4,0%; 1.234.0") == "Atvinnuleysi var 6.0% og 4,0%; 1.234.0"


def test_three_tables_are_read_in_parallel_and_all_reach_the_answer(monkeypatch):
    # 1 Nov: "GDP per head, inflation and unemployment in 2025" lost two of its topics when only 2 parts were read
    import threading
    tables = {n: {"id": n, "title_is": n, "path": f"{n}.px", "variables_is": []} for n in ("GDP", "CPI", "UNEMP")}
    seen_threads, steps = set(), []

    class R:
        tables = []

        def search(self, q, k=8):
            return [(tables[q.split()[0]], 1.0)]

    monkeypatch.setattr(pipeline, "decompose", lambda q: ["GDP 2025?", "CPI 2025?", "UNEMP 2025?"])
    monkeypatch.setattr(pipeline, "pick_tables", lambda q, c, v=(), second=False: [c[0][0]])

    def fetch(q, t, calc=True, lang="is"):
        import time
        time.sleep(0.2)  # long enough that the three overlap
        seen_threads.add(threading.get_ident())
        pipeline.report("read", table=t["id"])  # progress from a worker thread still reaches the caller
        return {"question": q, "table": t, "url": "", "selections": {}, "rows": [{"value": len(t["id"])}],
                "fallback": [], "calculation": None, "assumed": []}
    monkeypatch.setattr(pipeline, "fetch_part", fetch)
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (True, ""))
    monkeypatch.setattr(pipeline, "agreement", lambda r, q, t: 1)
    monkeypatch.setattr(pipeline, "plan_calculation", lambda q, rows, scale=None: None)
    monkeypatch.setattr(pipeline, "write_answer", lambda q, parts, missing, english=True, cross=None:
                        " / ".join(p["table"]["id"] for p in parts))
    token = pipeline.PROGRESS.set(lambda step, info: steps.append((step, info.get("table"))))
    try:
        r = pipeline.answer("GDP, CPI and unemployment in 2025?", R())
    finally:
        pipeline.PROGRESS.reset(token)
    assert [p["table"]["id"] for p in r["parts"]] == ["GDP", "CPI", "UNEMP"]  # all three, in the question's order
    assert r["answer"].startswith("GDP / CPI / UNEMP")
    assert len(seen_threads) >= 2 and {t for s, t in steps if s == "read"} == {"GDP", "CPI", "UNEMP"}


def test_parts_past_the_limit_are_named_not_dropped(monkeypatch):
    tables = {n: {"id": n, "title_is": n, "path": f"{n}.px", "variables_is": []} for n in "ABCDEF"}

    class R:
        tables = []

        def search(self, q, k=8):
            return [(tables[q[0]], 1.0)]

    monkeypatch.setattr(pipeline, "decompose", lambda q: [f"{n} 2025?" for n in "ABCDEF"])
    monkeypatch.setattr(pipeline, "pick_tables", lambda q, c, v=(), second=False: [c[0][0]])
    monkeypatch.setattr(pipeline, "fetch_part", lambda q, t, calc=True, lang="is": {
        "question": q, "table": t, "url": "", "selections": {}, "rows": [{"value": 1.0}], "fallback": [],
        "calculation": None, "assumed": []})
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (True, ""))
    monkeypatch.setattr(pipeline, "agreement", lambda r, q, t: 1)
    monkeypatch.setattr(pipeline, "plan_calculation", lambda q, rows, scale=None: None)
    monkeypatch.setattr(pipeline, "write_answer", lambda q, parts, missing, english=True, cross=None: "Svar.")
    r = pipeline.answer("A, B, C, D, E og F árið 2025?", R())
    assert [p["table"]["id"] for p in r["parts"]] == ["A", "B", "C", "D"]
    assert r["answer"].endswith("Ég svara mest 4 atriðum í einni spurningu. Þessu svaraði ég ekki, spurðu um það sér:\n"
                                "- E 2025?\n- F 2025?")


def test_only_suggestions_we_can_answer_are_offered(monkeypatch):
    # 1 Nov: "apple production in 2025?" -> "production in agriculture in 2025?", which led to 'no data' again
    good = {"id": "G", "title_is": "G", "path": "G.px", "variables_is": []}

    class R:
        tables = []

        def search(self, q, k=8):
            return [(good, 1.0)] if q.startswith("good") else []

    monkeypatch.setattr(pipeline, "decompose", lambda q: [q])
    monkeypatch.setattr(pipeline, "pick_tables", lambda q, c, v=(), second=False: [c[0][0]] if c else [])
    monkeypatch.setattr(pipeline, "fetch_part", lambda q, t, calc=True, lang="is": {
        "question": q, "table": t, "url": "", "selections": {}, "rows": [{"value": 7.0}], "fallback": [],
        "calculation": None, "assumed": []})
    monkeypatch.setattr(pipeline, "rows_answer", lambda q, p: (True, ""))
    monkeypatch.setattr(pipeline, "agreement", lambda r, q, t: 1)
    monkeypatch.setattr(pipeline, "write_answer", lambda *a, **k: "It was 7.")
    monkeypatch.setattr(pipeline.suggest, "suggestions", lambda q, c, lang="is", **k: ["good question?", "dead end?"])
    r = pipeline.answer("apples in 2025?", R())
    assert r["suggestions"] == ["good question?"]  # the dead end is not offered
    assert "dead end?" not in r["answer"] and "good question?" in r["answer"]
    nested = pipeline.answer("dead end?", R(), suggestions_wanted=False)
    assert nested["suggestions"] == [] and nested["answer"] in pipeline.NO_TABLE.values()

    # 1 Nov, on the Pi: the model wrote only "apple production in 2025?", which led nowhere, so nothing was offered.
    # It is asked again, told which ones failed, and at most MOST are shown.
    asked = []

    def written(q, c, lang="is", most=3, avoid=()):
        asked.append((most, list(avoid)))
        return ["dead end?"] if not avoid else [f"good {i}?" for i in range(5)]
    monkeypatch.setattr(pipeline.suggest, "suggestions", written)
    r = pipeline.answer("apples in 2025?", R())
    assert asked == [(6, []), (6, ["dead end?"])]
    assert r["suggestions"] == ["good 0?", "good 1?", "good 2?"]

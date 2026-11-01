"""hagstofa/suggest.py: related, answerable questions after "no data" (idea B, 31 Oct)."""
from hagstofa import suggest


def t(tid, title, title_en=None):
    return {"id": tid, "title_is": title, "title_en": title_en}


def test_suggestions_use_the_tables_own_questions_in_the_users_language(monkeypatch):
    monkeypatch.setattr(suggest, "samples", lambda: {
        "SAM03105": [{"q": "Hvað voru margar fólksbifreiðar á Íslandi í árslok 2024?", "lang": "is"},
                     {"q": "Hversu mikið eldsneyti seldist 2010?", "lang": "is"},
                     {"q": "How many passenger cars were registered at the end of 2024?", "lang": "en"}]})
    cands = [(t("SAM03105", "Ökutæki"), 1.0), (t("SAM03105", "Ökutæki"), 0.9), (t("X1", "Mannfjöldi", "Population"), 0.8)]
    got = suggest.suggestions("hvað eru margir bílar á íslandi?", cands, "is")
    assert got == ["Hvað voru margar fólksbifreiðar á Íslandi í árslok 2024?",  # the closest of the table's questions
                   "Hvað segja nýjustu tölurnar í töflunni „Mannfjöldi“?"]  # no questions: the title; no duplicates
    assert suggest.suggestions("how many cars", cands, "en")[0].startswith("How many passenger cars")
    assert suggest.suggestions("x", cands, "en")[1] == "What are the latest figures in the table “Population”?"


def test_with_suggestions():
    assert suggest.with_suggestions("Engin gögn.", [], "is") == "Engin gögn."
    out = suggest.with_suggestions("No data.", ["A?", "B?"], "en")
    assert out.startswith("No data.\n\nStatistics Iceland does not publish") and out.endswith("- A?\n- B?")


def test_the_sample_file_ships_with_the_repo():
    assert sum(len(v) for v in suggest.samples().values()) == 4580


def test_the_model_keeps_only_related_tables_and_writes_their_questions(monkeypatch):
    seen = {}

    def chat(system, user, json_mode=False):
        seen.update(system=system, user=user, json=json_mode)
        return ('{"questions": [{"table": 2, "question": "Hve margir bjuggu í Laugardal 1. janúar 2026?"},'
                ' {"table": 9, "question": "A table that is not there?"}, {"table": 1, "question": ""}]}')
    monkeypatch.setattr(suggest, "WRITE", True)
    monkeypatch.setattr(suggest.llm, "chat", chat)
    hverfi = {**t("MAN03270", "Mannfjöldi eftir hverfum í Reykjavík"), "variables_is": [
        {"code": "Hverfi", "text": "Hverfi", "valueTexts": ["Vesturbær", "Laugardalur"]},
        {"code": "Ár", "text": "Ár", "time": True, "valueTexts": [str(y) for y in range(1998, 2027)]}]}
    cands = [(t("UMH11165", "Sólskinsstundir"), 1.0), (hverfi, 0.9)]
    got = suggest.suggestions("Hve margir búa í Laugardal árið 1950?", cands, "is")
    assert got == ["Hve margir bjuggu í Laugardal 1. janúar 2026?"]  # only the valid choice; no invented table
    assert seen["json"] and "Icelandic" in seen["system"] and "at most 3" in seen["system"]
    assert "Laugardalur" in seen["user"] and "2026" in seen["user"] and "1998," not in seen["user"]  # newest periods


def test_several_questions_per_table_and_the_failed_ones_are_named(monkeypatch):
    seen = {}

    def chat(system, user, json_mode=False):
        seen.update(system=system, user=user)
        return ('{"questions": [{"table": 1, "question": "How many tonnes of potatoes were harvested in 2025?"},'
                ' {"table": 1, "question": "How much grain was harvested in 2025?"},'
                ' {"table": 1, "question": "How many apples were produced in 2025?"}]}')
    monkeypatch.setattr(suggest, "WRITE", True)
    monkeypatch.setattr(suggest.llm, "chat", chat)
    got = suggest.suggestions("how many apples are eaten?", [(t("LAN10103", "Harvest"), 1.0)], "en", most=6,
                              avoid=["How many apples were produced in 2025?"])
    assert got == ["How many tonnes of potatoes were harvested in 2025?", "How much grain was harvested in 2025?"]
    assert "at most 6" in seen["system"] and "never about the missing thing" in seen["system"]
    assert "had no numbers for them" in seen["user"] and "- How many apples were produced in 2025?" in seen["user"]


def test_no_related_table_means_no_suggestions(monkeypatch):
    monkeypatch.setattr(suggest, "WRITE", True)
    monkeypatch.setattr(suggest.llm, "chat", lambda *a, **k: '{"questions": []}')
    cands = [(t("THJ01701", "Landsframleiðsla á mann"), 1.0), (t("UMH31220", "Losun loftmengandi efna"), 0.9)]
    assert suggest.suggestions("hvað eru framleidd mörg epli á íslandi?", cands, "is") == []


def test_the_model_chooses_among_the_top_ten(monkeypatch):
    seen = {}
    monkeypatch.setattr(suggest, "WRITE", True)
    monkeypatch.setattr(suggest.llm, "chat", lambda s, u, json_mode=False: seen.setdefault("user", u) and '{"questions": []}')
    suggest.suggestions("x", [(t(f"T{i}", f"Tafla {i}"), 1.0) for i in range(15)], "is")
    assert "10. Tafla 9" in seen["user"] and "Tafla 10" not in seen["user"]


def test_a_failing_model_still_gives_suggestions(monkeypatch):
    monkeypatch.setattr(suggest, "WRITE", True)
    monkeypatch.setattr(suggest.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("no model")))
    monkeypatch.setattr(suggest, "samples", lambda: {})
    assert suggest.suggestions("x", [(t("X1", "Mannfjöldi", "Population"), 1.0)], "en") == \
        ["What are the latest figures in the table “Population”?"]


def test_subject_folders_find_tables_the_search_missed(monkeypatch):
    monkeypatch.setattr(suggest, "SUBJECTS", suggest.DATA / "no-such-file.jsonl")  # load the rows below instead
    rows = [("Landbúnaður › Búfé og uppskera", suggest._keys("epli kartöflur uppskera", False),
             suggest._keys("apples potatoes harvest", True), ["LAN10103"]),
            ("Vísindi › Nýjungar", suggest._keys("vor nýsköpun", False), set(), ["FYR05200"])]
    monkeypatch.setattr(suggest, "subjects", lambda: rows)
    monkeypatch.setattr(suggest, "_tables_by_id", lambda: {"LAN10103": t("LAN10103", "Uppskera"), "FYR05200": t("FYR05200", "Nýjungar")})
    monkeypatch.setattr(suggest, "correct_question", lambda q: q.replace("epi ", "epli "))
    assert [x["id"] for x in suggest.subject_tables("hversu mörg epi voru framleidd á íslandi?", "is")] == ["LAN10103"]
    assert [x["id"] for x in suggest.subject_tables("How many apples are grown?", "en")] == ["LAN10103"]
    assert suggest.subject_tables("hvað kostar bensín?", "is") == []  # no subject word: nothing added


def test_the_subject_tables_join_the_models_list(monkeypatch):
    seen = {}
    monkeypatch.setattr(suggest, "WRITE", True)
    monkeypatch.setattr(suggest, "subject_tables", lambda q, lang: [t("LAN10103", "Uppskera og afurðir")])
    monkeypatch.setattr(suggest.llm, "chat", lambda s, u, json_mode=False: seen.setdefault("user", u) and '{"questions": []}')
    suggest.suggestions("epli?", [(t("THJ01701", "Landsframleiðsla á mann"), 1.0)], "is")
    assert "1. Landsframleiðsla á mann" in seen["user"] and "2. Uppskera og afurðir" in seen["user"]


def test_imports_are_trade_and_compounds_match(monkeypatch):
    rows = [("Utanríkisverslun › Vöruviðskipti", suggest._keys("kaffi innflutningur", False), set(), ["UTA06001"]),
            ("Mannfjöldi › Yfirlit", suggest._keys("flytja fluttur íbúi", False), set(), ["MAN00101"]),
            ("Verðlag › Vísitala neysluverðs", suggest._keys("mjólk kosta", False), set(), ["VIS01000"])]
    monkeypatch.setattr(suggest, "subjects", lambda: rows)
    monkeypatch.setattr(suggest, "_tables_by_id", lambda: {i: t(i, i) for i in ("UTA06001", "MAN00101", "VIS01000")})
    monkeypatch.setattr(suggest, "correct_question", lambda q: q)
    assert [x["id"] for x in suggest.subject_tables("Hvað var flutt inn mikið af kaffi?", "is")] == ["UTA06001"]
    assert [x["id"] for x in suggest.subject_tables("Hvað kostar mjólkurlítrinn?", "is")] == ["VIS01000"]

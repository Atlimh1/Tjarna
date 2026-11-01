from hagstofa.checker import missing, names, rerank

FISH = {"title_is": "Útfluttur ferskur fiskur eftir flutningsmáta 1988-2025", "title_en": "",
        "variables_is": [{"code": "Ár", "time": True, "values": ["1988", "2025"], "valueTexts": ["1988", "2025"]},
                         {"code": "Flutningsmáti", "values": ["0"], "valueTexts": ["Alls"]}]}
BY_COUNTRY = {"title_is": "Útflutningur eftir löndum 2015-2026", "title_en": "",
              "variables_is": [{"code": "Land", "values": ["FR"], "valueTexts": ["Frakkland"]},
                               {"code": "Mánuður", "time": True, "values": ["2025M03"], "valueTexts": ["2025M03"]}]}


def test_names_keeps_places_drops_people_and_iceland():
    assert names("Hvað bjuggu margir í Kópavogi?") == [{"kópavo"}]
    assert names("Hversu margar konur hétu Guðrún á Íslandi?") == []


def test_missing_place_and_year():
    q = "Hvað var flutt út af ferskum fiski til Frakklands árið 2025?"
    assert missing(q, FISH) == ["frakkl"]
    assert missing(q, BY_COUNTRY) == []
    assert missing("Útflutningur 2010", BY_COUNTRY) == ["2010"]  # before the title range 2015-2026
    assert missing("Útflutningur 2020", BY_COUNTRY) == []  # inside it


def test_rerank_moves_failing_tables_down():
    q = "fiskur til Frakklands 2025"
    assert [t for t, _ in rerank(q, [(FISH, 2), (BY_COUNTRY, 1)])] == [BY_COUNTRY, FISH]
    assert [t for t, _ in rerank("fiskur til Kína", [(FISH, 2), (BY_COUNTRY, 1)])] == [FISH, BY_COUNTRY]  # none pass

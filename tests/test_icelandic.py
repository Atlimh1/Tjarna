from hagstofa import icelandic
from hagstofa.retrieve import lemmas


def test_correct_fixes_missing_accent():
    fixed, notes = icelandic.correct("Hvað bjuggu margar konur í Kopavogi?")
    assert "Kópavogi" in fixed
    assert notes


def test_lemmas_normalize_inflection():
    assert "kópavogur" in lemmas("í Kópavogi")
    assert "kona" in lemmas("margar konur")
    assert "hve" not in lemmas("Hve mörg börn")  # stopword


def test_entities_find_place_and_time():
    found = icelandic.entities("Hve mörg börn fæddust á Akureyri í fyrra?")
    assert "Akureyri" in found["places"]
    assert any("fyrra" in t for t in found["times"])


def test_inflect_to_genitive():
    assert "margra kvenna" in icelandic.inflect("margar konur", "ef")


def test_query_expansion_adds_official_terms():
    from hagstofa.retrieve import expand_query
    assert expand_query("Hvað búa margir á Íslandi?").endswith("mannfjöldi")
    assert "dánir" in expand_query("Hve margir létust 2025?")
    assert expand_query("How many people live here?").count("mannfjöldi") == 1
    assert expand_query("Hve mörg ökutæki eru skráð?") == "Hve mörg ökutæki eru skráð?"  # nothing to add

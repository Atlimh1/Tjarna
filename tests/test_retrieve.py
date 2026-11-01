import numpy as np
import pytest

from hagstofa import retrieve
from hagstofa.retrieve import Places, correct_question, fold, is_english, lemma_table_text


def test_fold_removes_icelandic_letters():
    assert fold("Hvað búa margir á Þórshöfn? Æðey") == "Hvad bua margir a Thorshofn? Aedey"


def test_is_english_also_for_icelandic_without_accents():
    assert is_english("How many people live in Iceland?")
    assert not is_english("hvad bua margir a islandi")
    assert not is_english("Hvað búa margir í Kópavogi?")


def test_correct_question_fixes_icelandic_and_leaves_english(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve, "CORRECTIONS", tmp_path / "c.sqlite")
    assert "þjóðkirkjunni" in correct_question("hvað eru margir í þjóðkyrkjunni?")
    assert correct_question("How many cars are in Reykjavik?") == "How many cars are in Reykjavik?"
    assert "þjóðkirkjunni" in correct_question("hvað eru margir í þjóðkyrkjunni?")  # from the cache


def table(tid, title, var, codes, texts):
    return {"id": tid, "title_is": title, "breadcrumbs_is": [], "variables_is": [{"code": "S", "text": var, "values": codes, "valueTexts": texts}]}


TABLES = [
    table("OFFICIAL", "Mannfjöldi eftir sveitarfélögum", "Sveitarfélag", ["1000", "1400", "7502"], ["Kópavogsbær", "Hafnarfjarðarkaupstaður", "Fljótsdalshreppur"]),
    table("PLAIN", "Tekjur eftir sveitarfélögum", "Sveitarfélag", ["1000", "7502"], ["Kópavogur", "Vopnafjarðarhreppur"]),
    table("PARTY", "Úrslit kosninga", "Sveitarfélag - listi", ["0", "1"], ["Kópavogur", "Kópavogur B Framsóknarflokkur"]),
    table("TITLE", "Úrslit borgarstjórnakosninga í Kópavogur 1930-2022", "Stjórnmálaflokkur", ["0"], ["Alls"]),
    table("NONE", "Lifandi fæddir", "Kyn", ["0"], ["Alls"]),
]


def test_aliases_come_from_shared_codes_but_not_reused_codes():
    p = Places(TABLES)
    assert p.aliases["Kópavogsbær"] == ["Kópavogur"]
    assert p.aliases["Fljótsdalshreppur"] == []  # code 7502 reused for Vopnafjarðarhreppur: not an alias
    assert "Kópavogur B Framsóknarflokkur" not in p.names  # party list, no 4-digit code


def test_boost_hits_tables_with_the_town_in_values_lists_or_title():
    p = Places(TABLES)
    s = p.boost("Hvað búa margir í Kópavogi?", np.ones(len(TABLES)), 0.5)
    assert list(s) == [1.5, 1.5, 1.5, 1.5, 1.0]
    assert (p.boost("Hvað fæddust mörg börn?", np.ones(len(TABLES)), 0.5) == 1).all()


def test_lemma_text_drops_town_names_but_keeps_party_lists():
    p = Places(TABLES)
    assert "Kópavogsbær" not in lemma_table_text(TABLES[0], p.names)
    assert "Framsóknarflokkur" in lemma_table_text(TABLES[2], p.names)


# Spelling (31 Oct): GreynirCorrect's output is kept only where it repairs a word that isn't one.
@pytest.mark.parametrize("typed, greynir, kept", [
    ("Hvað bjuggu margir í Kopavogi 1. januar 2025?", "Hvað bjuggu margir í Kópavogi 1. janúar 2025?",
     "Hvað bjuggu margir í Kópavogi 1. janúar 2025?"),  # Icelandic letters restored
    ("hvað eru margir í þjóðkyrkjunni?", "Hvað eru margir í þjóðkirkjunni?", "Hvað eru margir í þjóðkirkjunni?"),
    ("Hve mörg bréf voru sent til útlanda?", "Hve mörg bréf voru send til útlanda?",
     "Hve mörg bréf voru sent til útlanda?"),  # "sent" is a word: left alone
    ("Eru margir bændur ennþá úti á landi?", "Eru margir bændur enn þá úti á landi?",
     "Eru margir bændur ennþá úti á landi?"),  # split into two words: as typed
    ("Hvað búa margir á Íslandi? En á höfuðborgarsvæðinu?", "Hvað búa margir á Íslandi?",
     "Hvað búa margir á Íslandi? En á höfuðborgarsvæðinu?"),  # a dropped question comes back
    ("Atvinnuþátttaka 16-24 ára 2025", "Atvinnuþátttaka 16 — 24 ára 2025", "Atvinnuþátttaka 16-24 ára 2025"),
])
def test_safe_correction(typed, greynir, kept):
    assert retrieve.safe_correction(typed, greynir) == kept


# Bilingual search (31 Oct): the question and its translation are both searched, the rankings fused.
class _FakeBase:
    tables = []

    def __init__(self, rankings):
        self.rankings = rankings

    def search(self, question, k=8, correct=None):
        return [({"id": t, "path": t}, 1.0) for t in self.rankings[question]][:k]


def test_bilingual_search_fuses_both_languages(monkeypatch):
    import hagstofa.translate as tr
    base = _FakeBase({"How many births?": ["A", "B", "C"], "Hve mörg fæddust?": ["B", "C", "A"]})
    monkeypatch.setattr(tr, "counterpart", lambda q: "Hve mörg fæddust?")
    found = [t["id"] for t, _ in retrieve.BilingualRetriever(base).search("How many births?", k=3)]
    assert found[0] == "B" and set(found) == {"A", "B", "C"}  # B is high in both languages


def test_bilingual_search_without_a_model_is_the_base_search(monkeypatch):
    import hagstofa.translate as tr
    base = _FakeBase({"How many births?": ["A", "B", "C"]})

    def broken(q):
        raise ConnectionError("no LLM")

    monkeypatch.setattr(tr, "counterpart", broken)
    assert [t["id"] for t, _ in retrieve.BilingualRetriever(base).search("How many births?", k=2)] == ["A", "B"]


def test_bilingual_search_leaves_icelandic_questions_alone(monkeypatch):
    import hagstofa.translate as tr

    def must_not_translate(q):
        raise AssertionError("Icelandic questions are searched as they are")

    monkeypatch.setattr(tr, "counterpart", must_not_translate)
    base = _FakeBase({"Hve mörg börn fæddust?": ["A", "B"]})
    assert [t["id"] for t, _ in retrieve.BilingualRetriever(base).search("Hve mörg börn fæddust?", k=2)] == ["A", "B"]


def test_places_know_localities_not_only_municipalities():
    tables = [{"id": "MAN030101", "title_is": "Mannfjöldi eftir byggðakjörnum", "variables_is": [
        {"code": "Byggðakjarnar", "text": "Byggðakjarnar", "values": ["0", "8716", "8717"],
         "valueTexts": ["Alls", "Hvolsvöllur", "Hella"]}]},
        {"id": "MAN02005", "title_is": "Mannfjöldi eftir sveitarfélögum", "variables_is": [
            {"code": "Sveitarfélag", "text": "Sveitarfélag", "values": ["8613"], "valueTexts": ["Rangárþing eystra"]}]}]
    p = Places(tables)
    assert {"Hvolsvöllur", "Hella"} <= p.names and "Alls" not in p.names
    scores = p.boost("Hve margir búa á Hvolsvelli?", np.ones(2), 0.5)
    assert scores[0] > scores[1]  # the locality table rises, the municipality table doesn't


def test_method_revision_tables_count_as_archive():
    # "Hvað búa margir á Íslandi?" ranked the 2024 revision tables (MAN90001-4) above the current population tables
    assert retrieve.is_archive({"id": "MAN90003", "path": "Ibuar/mannfjoldi/4_endurskodun/MAN90003.px"})
    assert not retrieve.is_archive({"id": "MAN00101", "path": "Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00101.px"})


def test_villages_are_told_apart_from_municipalities():
    p = retrieve.Places(retrieve.load_catalog())
    assert p.villages("How many people live in Vik?") == ["Vík í Mýrdal"]  # Mýrdalshreppur is the municipality
    for q in ["Hvað búa margir í Kópavogi?", "Hvað búa margir á Akureyri?", "Hvað eru margar búðir á Íslandi?"]:
        assert p.villages(q) == []  # municipalities, and a common word that is also a village (Búðir)


def test_village_tables_are_the_newest_listing_the_village():
    cat = retrieve.load_catalog()
    ids = [t["id"] for t in retrieve.Places(cat).village_tables({"Vík í Mýrdal"}, cat)]
    assert ids and ids[0].startswith("MAN0301")  # towns/villages population, not a municipality table


@pytest.mark.parametrize("question, extra", [
    ("hvað er flutt inn mikið af kaffi á ári?", "innflutningur vörur"),
    ("Hvað var flutt út mikið af fiski?", "útflutningur vörur"),
    ("hvað fluttu margir til landsins?", "búferlaflutningar"),  # people moving: migration, as before
])
def test_import_and_export_are_trade_not_migration(question, extra):
    expanded = retrieve.expand_query(question)
    assert expanded.endswith(extra) and (extra == "búferlaflutningar" or "búferlaflutningar" not in expanded)

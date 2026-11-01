"""hagstofa/verify.py: every number in an answer must come from the data."""
import pytest

from hagstofa import verify

TABLE = {"id": "MAN00101", "title_is": "Mannfjöldi 1841-2026"}
RESULT = {"parts": [{"table": TABLE, "rows": [{"Ár": "2026", "value": 394324}, {"Ár": "2025", "value": 389444}],
                     "calculation": {"values": [394324, 389444], "result": 1.25306, "text": "... = 1,25%"}}]}
QUESTION = "Hvað búa margir á Íslandi 1. janúar 2026?"


@pytest.mark.parametrize("answer", [
    "Á Íslandi bjuggu 394.324 manns 1. janúar 2026.", "On 1 January 2026, 394,324 people lived in Iceland.",
    "Um 394 þúsund manns.", "Fjölgunin var 1,3%.", "Um 0,39 milljónir.", "394 324 (tafla MAN00101)",
])
def test_honest_answers_and_roundings_pass(answer):
    assert verify.unsupported(answer, RESULT, QUESTION) == []


@pytest.mark.parametrize("answer, bad", [
    ("Á Íslandi búa 5 milljónir manna.", "5 milljónir"), ("Svarið er 9 milljónir.", "9 milljónir"),
    ("Verðbólgan er 0%.", "0%"), ("Fjölgunin var 4.980 manns.", "4.980"),  # the model's own maths, wrong
])
def test_numbers_not_in_the_data_are_caught(answer, bad):
    assert verify.unsupported(answer, RESULT, QUESTION) == [bad]


# Run 4 (31 Oct): correct arithmetic by the model was replaced by the rows, 5 of the 60 lost. It is checked instead.
SHARE = {"parts": [{"table": {"id": "MAN03270", "title_is": "Mannfjöldi í Reykjavík"},
                    "rows": [{"Ár": "2026", "value": 139804.0}]},
                   {"table": {"id": "MAN00000", "title_is": "Mannfjöldi"}, "rows": [{"Ár": "2026", "value": 394324.0}]}]}


@pytest.mark.parametrize("answer", [
    "(139,804 / 394,324) × 100 = 35.45%", "35,5% landsmanna bjó í Reykjavík", "About 35% lived in Reykjavík.",
    "254,520 lived outside Reykjavík.", "354.5 per 1,000 inhabitants",
])
def test_correct_arithmetic_on_the_rows_passes(answer):
    assert verify.unsupported(answer, SHARE, "What percentage lived in Reykjavík?") == []


def test_wrong_arithmetic_and_big_tables_are_still_caught():
    assert verify.unsupported("That is 36.2% of the population.", SHARE, "What percentage?") == ["36.2%"]
    many = {"parts": [{"table": TABLE, "rows": [{"Ár": str(y), "value": 1000.0 + y} for y in range(2000, 2020)]}]}
    assert verify.unsupported("The share was 99.9%.", many, "?") == ["99.9%"]  # 20 rows: no pairs derived


def test_fallback_is_built_from_the_rows():
    text = verify.fallback(RESULT)
    assert "MAN00101" in text and "394324" in text and "389444" in text
    # a row without a figure is left out, not shown as "None" (fruit imports, live check 1 Nov)
    empty = {"parts": [{"table": TABLE, "rows": [{"Ár": "2025", "value": 33337.185}, {"Ár": "2024", "value": None}]}]}
    assert "None" not in verify.fallback(empty, "en") and "33337.185" in verify.fallback(empty, "en")


# Run 6 (31 Oct): values copied with all their decimals were replaced; so were values the model scaled by the unit.
@pytest.mark.parametrize("answer", ["Meðalaldur var 38,823235 ár.", "The mean age was 38.823235 years.",
                                    "Kjörsókn var 63,83616384%.", "Hún var 63,8%."])
def test_full_precision_values_pass(answer):
    result = {"parts": [{"table": TABLE, "rows": [{"Ár": "2024", "value": 38.823235},
                                                  {"Ár": "2024", "value": 63.836163836163836}]}]}
    assert verify.unsupported(answer, result, "Meðalaldur 2024?") == []


@pytest.mark.parametrize("unit, answer, ok", [
    ("Þúsundir króna", "Hún var 16,0 milljónir króna á mann.", True),
    ("Thousand ISK", "It was ISK 15.98 million per person.", True),
    ("þús. kr.", "Um 16 milljónir.", True),
    ("Million ISK", "GDP was ISK 4,512 billion.", True),
    ("Fjöldi", "Hún var 16,0 milljónir króna á mann.", False),
    ("Tonnes/1000 ISK", "Hún var 16,0 milljónir króna á mann.", False),
    ("Þúsundir króna", "Hún var 17,0 milljónir króna á mann.", False),
])
def test_the_tables_unit_scales_its_values(unit, answer, ok):
    result = {"parts": [{"table": TABLE, "unit": unit, "rows": [{"Ár": "2024", "value": 15983.0},
                                                                {"Ár": "2025", "value": 4512345.0}]}]}
    assert (verify.unsupported(answer, result, "Landsframleiðsla á mann 2024?") == []) is ok


@pytest.mark.parametrize("text, tidy", [
    ("Meðalaldur var 38,823235 ár.", "Meðalaldur var 38,82 ár."),
    ("The rate was 1.55964413873723 (per woman).", "The rate was 1.56 (per woman)."),
    ("Kjörsókn var 63,83616384%.", "Kjörsókn var 63,84%."),
    ("index 120.669309808456", "index 120.67"),
    ("0.1346123 points", "0.135 points"), ("0.000123456", "0.0001"), ("0.99999999", "1.000"),
    ("GDP 4,941,210.916700 m", "GDP 4,941,210.92 m"), ("4.941.999,999999", "4.941.999,999999"),
    ("394.324 manns, 1,2345%, 2026M08, MAN00102, 2026", "394.324 manns, 1,2345%, 2026M08, MAN00102, 2026"),
    # a whole value copied with its zero decimal (1 Nov, on the Pi): 3+ digits only, so a rate keeps "6,0%"
    ("var 12348,0 tonn", "var 12348 tonn"), ("was 8,274.0 tonnes", "was 8,274 tonnes"), ("- 2015: 4107.0", "- 2015: 4107"),
    ("1.356,0 tonn, 6,0%, 12,5, 1.234,05 kr., 105,0 stig", "1.356 tonn, 6,0%, 12,5, 1.234,05 kr., 105 stig"),
])
def test_tidy_decimals(text, tidy):
    assert verify.tidy_decimals(text) == tidy


def test_a_whole_value_with_its_zero_decimal_is_grouped_and_still_stated():
    result = {"parts": [{"table": TABLE, "rows": [{"Ár": "2025", "value": 12348.0}]}]}
    text = verify.group_thousands(verify.tidy_decimals("Uppskera á korn alls árið 2025 var 12348,0 tonn."), "is")
    assert text == "Uppskera á korn alls árið 2025 var 12.348 tonn."
    assert verify.unsupported(text, result, "Uppskera á korni 2025?") == []


def test_tidied_numbers_are_still_supported_and_right_for_the_scorer():
    result = {"parts": [{"table": TABLE, "rows": [{"Ár": "2024", "value": 38.823235}]}]}
    tidy = verify.tidy_decimals("Meðalaldur var 38,823235 ár.")
    assert verify.unsupported(tidy, result, "Meðalaldur 2024?") == []
    # the organizers' rule (gagnvist.evaluate.states_number): within half a unit of the last digit shown
    assert abs(38.82 - 38.8232) <= 0.5 * 10 ** -2 and not abs(38.823235 - 38.8232) <= 0.5 * 10 ** -6


def test_unit_scale():
    assert [verify.unit_scale(u) for u in ["Million ISK", "Milljónir króna/hlutfall", "Þúsundir króna", "þús. kr.",
                                           "Fjöldi", "Tonnes/1000 ISK", None, "Thousands"]] ==         [1e6, 1e6, 1e3, 1e3, 1, 1, 1, 1e3]


@pytest.mark.parametrize("text, lang, out", [
    ("Árið 2025 voru 4387 lifandi fæddir.", "is", "Árið 2025 voru 4.387 lifandi fæddir."),
    ("In 2025 there were 4387 live births.", "en", "In 2025 there were 4,387 live births."),
    ("Árið 2025 býa 2441 (meðalmannfjöldi) á Húsavík.", "is", "Árið 2025 býa 2.441 (meðalmannfjöldi) á Húsavík."),
    ("Total 394324 people, 1 January 2026.", "en", "Total 394,324 people, 1 January 2026."),
    ("Tafla MAN00102, 2026M08, 1841-2026, 394.324, 38,82, 12345,6", "is",
     "Tafla MAN00102, 2026M08, 1841-2026, 394.324, 38,82, 12345,6"),
    ("Póstnúmer 0101 og árið 1999", "is", "Póstnúmer 0101 og árið 1999"),
])
def test_group_thousands(text, lang, out):
    assert verify.group_thousands(text, lang) == out


def test_grouped_numbers_still_pass_the_number_check():
    result = {"parts": [{"table": TABLE, "rows": [{"Ár": "2025", "value": 4387.0}]}]}
    assert verify.unsupported(verify.group_thousands("Það voru 4387 börn.", "is"), result, "?") == []


@pytest.mark.parametrize("unit, label, value, answer, ok", [
    # THJ01401: the unit line says "Krónur/USD/Evra", the measure's label says thousands (1 Nov, set 1)
    ("Krónur/USD/Evra", "Þúsundir króna", 11927.2011, "Landsframleiðsla á mann var 11.927,20 þúsundir króna.", True),
    ("Fob verð í milljónum króna og tonnum", "Alls", 1234.5, "Útflutningurinn var 1.234,5 milljónir króna.", True),
    ("Number/thous. ISK", "Total", 15983.0, "It was ISK 16.0 million.", True),
    ("Fjöldi á 100 þúsund íbúa", "Alls", 15983.0, "Hún var 16,0 milljónir.", False),  # a rate, not a scale
    ("Per thousand", "Total", 15983.0, "It was 16.0 million.", False),
    ("Krónur/USD/Evra", "USD, skráð gengi", 11927.2011, "Hún var 11.927,20 þúsundir króna.", False),
])
def test_scale_words_anywhere_in_the_unit_or_labels(unit, label, value, answer, ok):
    result = {"parts": [{"table": TABLE, "unit": unit, "rows": [{"Mælikvarði": label, "Ár": "2024", "value": value}]}]}
    assert (verify.unsupported(answer, result, "Hver var hún 2024?") == []) is ok


def test_scales():
    assert verify.scales("Fob verð í milljónum króna og tonnum") == {1e6}
    assert verify.scales("Number/thous. ISK | Total") == {1e3}
    assert verify.scales("Magnbreyting, % / Milljarðar króna | ISK billions | milljörðum") == {1e9}
    assert verify.scales("Fjöldi á 100 þúsund íbúa | Number per 100 thousand inhabitants | Per thousand") == set()
    assert verify.scales("Krónur/USD/Evra | Millimetrar") == set()

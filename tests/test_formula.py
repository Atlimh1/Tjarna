"""hagstofa/calc.py (formulas) and the formula step in pipeline.calculate: multi-step calculations, evaluated by code."""
import pytest

from hagstofa import calc, pipeline, verify

ROWS = [{"Landshlutar": "Alls", "Ár": "2025", "value": 389444.0},
        {"Landshlutar": "Höfuðborgarsvæði", "Ár": "2025", "value": 249049.0},
        {"Aldur": "68 ára", "value": 3000.0}, {"Aldur": "69 ára", "value": 2900.0}, {"Aldur": "70 ára", "value": 2800.0}]


@pytest.mark.parametrize("formula, names, want", [
    ("(total - capital) / total * 100", {"total": 389444.0, "capital": 249049.0}, (389444 - 249049) / 389444 * 100),
    ("sum(old) / total * 100", {"old": [3000.0, 2900.0], "total": 10000.0}, 59.0),
    ("mean(x)", {"x": [1.0, 2.0, 6.0]}, 3.0), ("median(x)", {"x": [1.0, 2.0, 6.0]}, 2.0),
    ("max(x) - min(x)", {"x": [1.0, 2.0, 6.0]}, 5.0), ("sum(a, b, c)", {"a": 1.0, "b": 2.0, "c": 3.0}, 6.0),
    ("cagr(a, b, 2)", {"a": 100.0, "b": 121.0}, 10.0), ("-a + 2 × 3 − 1", {"a": 1.0}, 4.0),
])
def test_formulas(formula, names, want):
    assert calc.evaluate(formula, names)[0] == pytest.approx(want)


@pytest.mark.parametrize("formula, names", [
    ("__import__('os').system('x')", {}), ("a.real", {"a": 1.0}), ("a ** 1000", {"a": 2.0}), ("b", {"a": 1.0}),
    ("a / 0", {"a": 1.0}), ("old * 2", {"old": [1.0, 2.0]}), ("sum(x=1)", {}), ("[1, 2]", {}), ("'text'", {}),
    ("lambda: 1", {}), ("", {}), ("a" * 300, {"a": 1.0}), ("cagr(a, b, 0)", {"a": 1.0, "b": 2.0}),
    ("open('f')", {}), ("1 if a else 2", {"a": 1.0}),
])
def test_anything_else_is_refused(formula, names):
    with pytest.raises(calc.FormulaError):
        calc.evaluate(formula, names)


def test_bind_checks_rows():
    assert calc.bind({"t": 0, "old": [2, 3]}, ROWS) == {"t": 389444.0, "old": [3000.0, 2900.0]}
    for bad in [{"t": 9}, {"t": -1}, {"t": "x"}, {"sum": 0}, {"1bad": 0}, {"t": []}, {}]:
        with pytest.raises(calc.FormulaError):
            calc.bind(bad, ROWS)


def test_formula_plan_through_calculate_and_the_number_check():
    plan = {"operation": "formula", "names": {"heild": 0, "hofudb": 1}, "formula": "(heild - hofudb) / heild * 100"}
    c = pipeline.calculate(plan, ROWS)
    assert c["result"] == pytest.approx(36.0503, abs=1e-3)
    assert c["text"] == "(389.444 − 249.049) / 389.444 × 100 = 36,1%"
    assert 140395.0 in c["values"]  # the step "heild - hofudb": the model may state it
    result = {"parts": [{"table": {"id": "MAN03250", "title_is": "Mannfjöldi"}, "rows": ROWS[:2], "calculation": c}]}
    assert verify.unsupported("Utan höfuðborgarsvæðisins bjuggu 140.395 manns, 36,1% landsmanna.", result, "?") == []


def test_formula_over_a_list_of_rows():
    plan = {"operation": "formula", "names": {"eldri": [2, 3, 4], "allir": 0}, "formula": "sum(eldri) / allir * 100"}
    c = pipeline.calculate(plan, ROWS, english=True)
    assert c["result"] == pytest.approx(8700 / 389444 * 100)
    assert c["text"].startswith("sum(eldri (3 rows)) / 389,444 × 100 = 2.23")


def test_a_bad_formula_gives_no_calculation():
    assert pipeline.calculate({"operation": "formula", "names": {"a": 0}, "formula": "a.__class__"}, ROWS) is None
    assert pipeline.calculate({"operation": "formula", "names": {"a": 99}, "formula": "a"}, ROWS) is None


def test_rows_in_different_scales_are_compared_in_whole_units():
    # dev3, 1 Nov: alcohol per inhabitant = 47,497.7 million krónur / 358,298 people; it came out as 0.13
    amount = {"unit": "Milljónir króna"}
    people = {"unit": "Fjöldi"}
    rows = [{"tafla": "THJ02103", "Liður": "02.1 Áfengi", "value": 47497.69},
            {"tafla": "MAN00000", "Eining": "Mannfjöldi 1. janúar", "value": 358298.0}]
    scale = [pipeline.row_scale(amount, rows[0]), pipeline.row_scale(people, rows[1])]
    assert scale == [1e6, 1.0]
    r = pipeline.calculate({"operation": "ratio", "a": 0, "b": 1}, rows, scale=scale)
    assert round(r["result"], 1) == 132564.8 and "47.497.690.000 / 358.298" in r["text"]
    f = pipeline.calculate({"operation": "formula", "formula": "a / b", "names": {"a": 0, "b": 1}}, rows, scale=scale)
    assert round(f["result"], 1) == 132564.8
    # one scale: the table's own unit is kept (a sum of two amounts in millions stays in millions)
    same = pipeline.calculate({"operation": "sum", "rows": [0, 0]}, rows, scale=[1e6, 1e6])
    assert same["result"] == 2 * 47497.69
    assert pipeline.calculate({"operation": "ratio", "a": 0, "b": 1}, rows)["result"] < 1  # no scale given: as before
    # a label's scale wins; a mixed unit line says nothing by itself
    assert pipeline.row_scale({"unit": "Fjöldi"}, {"Eining": "Million ISK", "value": 1.0}) == 1e6
    assert pipeline.row_scale({"unit": "Tonn/milljónir króna (FOB)"}, {"Land": "Kína", "value": 1.0}) == 1.0
    assert pipeline.row_scale({"unit": "Fjöldi á 100 þúsund íbúa"}, {"value": 1.0}) == 1.0
    # money besides percentages: the amount rows are in millions, the % rows are not (THJ02103, THJ05142)
    mixed = {"unit": "Milljónir króna/hlutfall"}
    assert pipeline.row_scale(mixed, {"Verðlag": "Verðlag hvers árs", "Liður": "02.1 Áfengi", "value": 1.0}) == 1e6
    assert pipeline.row_scale(mixed, {"Eining": "Hlutfall af VLF", "value": 1.0}) == 1.0
    # money besides a count: only the label tells (TEK02014)
    assert pipeline.row_scale({"unit": "Milljónir króna/fjöldi"}, {"Eining": "Lífeyrisþegar", "value": 1.0}) == 1.0
    assert pipeline.row_scale({"unit": "Milljónir króna/fjöldi"}, {"Eining": "Milljónir króna", "value": 1.0}) == 1e6


def test_a_formula_that_already_scales_is_not_scaled_again():
    # dev3, 1 Nov: the model wrote revenue * 1000000 / population; converting too gave 79,953 million krónur a head
    rows = [{"tafla": "MEN10001", "value": 29176.0}, {"tafla": "MAN00000", "value": 364917.0}]
    plan = {"operation": "formula", "formula": "a * 1000000 / b", "names": {"a": 0, "b": 1}}
    r = pipeline.calculate(plan, rows, scale=[1e6, 1.0])
    assert round(r["result"]) == round(29176.0 * 1e6 / 364917.0)
    plan["formula"] = "a / b * 1000"  # per 1,000 inhabitants: 1000 is no unit factor for millions
    assert round(pipeline.calculate(plan, rows, scale=[1e6, 1.0])["result"]) == round(29176.0 * 1e6 / 364917.0 * 1000)

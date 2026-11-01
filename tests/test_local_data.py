"""local_data.py and px.query's choice between it and PX-Web. Uses the organizers' six-table test copy of the
data release (gagnvist-starter tests/fixtures/data); skipped without a gagnvist-starter clone."""
import pytest

from hagstofa import local_data, px
from hagstofa.pipeline import flatten

FIXTURES = local_data.starter() / "tests/fixtures/data"
KOS02105 = "Ibuar/kosningar/althingi/althkjosendur/KOS02105.px"  # Skipting (0 = Alls, 1-3) x Kyn (0 = Alls, 1, 2)
pytestmark = pytest.mark.skipif(not (FIXTURES / "catalogue.csv").is_file(), reason="no gagnvist-starter clone")


@pytest.fixture(autouse=True)
def fixtures(monkeypatch):
    monkeypatch.setenv("GAGNVIST_DATA", str(FIXTURES))


def cell(skipting, kyn):
    return local_data.query(KOS02105, {"Skipting": [skipting], "Kyn": [kyn]})["value"][0]


def test_has_only_icelandic_tables_in_the_release():
    assert local_data.has(KOS02105)
    assert not local_data.has("Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00101.px")


def test_query_has_px_webs_shape():
    rows = flatten(local_data.query(KOS02105, {"Skipting": ["1", "2"], "Kyn": ["2", "1"]}))
    assert [(r["Skipting"], r["Kyn"]) for r in rows] == [  # table order, whatever order was asked for
        ("Vegna sjúkdóms", "Karlar"), ("Vegna sjúkdóms", "Konur"), ("Vegna fötlunar", "Karlar"),
        ("Vegna fötlunar", "Konur")]
    assert all(isinstance(r["value"], float) for r in rows)


def test_left_out_dimension_uses_its_total():
    rows = flatten(local_data.query(KOS02105, {"Skipting": ["1"]}))
    assert rows == [{"Skipting": "Vegna sjúkdóms", "value": cell("1", "0")}]


def test_unknown_code_raises():
    with pytest.raises(KeyError):
        local_data.query(KOS02105, {"Kyn": ["9"]})
    with pytest.raises(KeyError):
        local_data.query(KOS02105, {"Nope": ["0"]})


def test_px_query_reads_the_release_and_falls_back_to_px_web(monkeypatch):
    calls = []
    monkeypatch.setattr(px, "_request", lambda method, url, **kw: calls.append(url) or {"live": True})
    assert px.query(KOS02105, {"Kyn": ["0"]})["source"] == "local" and calls == []
    assert px.query(KOS02105, {"Kyn": ["9"]}) == {"live": True}  # code missing locally: ask PX-Web
    assert px.query("Ibuar/x/MAN00101.px", {}) == {"live": True}  # table not in the release
    monkeypatch.setattr(px, "SOURCE", "live")
    assert px.query(KOS02105, {"Kyn": ["0"]}) == {"live": True}
    assert len(calls) == 3


def test_unit_from_the_px_header_in_each_language():
    # Fixtures: VIS13401 is utf-8 with LANGUAGE="en" in its English file; KOS07112 is iso-8859-15.
    units = {path.split("/")[-1]: (local_data.unit(path, "is"), local_data.unit(path, "en"))
             for path in local_data._editions(str(FIXTURES))}
    assert units["KOS07112.px"] == ("Fjöldi/hlutfall, %", "Number /%")
    assert units["VIS13401.px"] == ("Ársmeðaltölur og ársbreytingar", "Annual average indices and yearly change in percentage")
    assert units["SAM91452.px"] == (None, None)  # "Tölur" / "Number": a plain count says nothing

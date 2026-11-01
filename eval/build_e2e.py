"""Write eval/questions_e2e.jsonl: questions with a known answer, for run_eval.py --e2e and compare_models.py.

Three levels, as in the organizers' 52 test questions: easy (one cell: "truth"), medium (a calculation over
cells of one table: "calc") and hard (numbers from two tables, all must be in the answer: "all").

Each spec gives the question, the acceptable tables (the first one is what `truth` refers to) and the values
that pin down the number, as value TEXTS ("Kópavogsbær", "2025"). This script turns them into value codes from
the catalog and fills every other variable with its total ("Alls", "Allir", ...), so the PX-Web query returns
exactly one number. It fails loudly on an unknown text or a variable without a total. Questions use explicit
years, so the expected number stays fixed as Hagstofa publishes new periods. No API calls: run_eval fetches
the expected number live.

    .venv/bin/python eval/build_e2e.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hagstofa.retrieve import load_catalog  # noqa: E402

TOTALS = {"alls", "allir", "allt landið", "alls lifandi fæddir"}

POP = ["MAN00101", "MAN02005"]
SPECS = [
    # population
    ("Hvað bjuggu margir á Íslandi 1. janúar 2026?", POP, {"Ár": "2026"}),
    ("Hvað voru margar konur á Íslandi árið 2000?", POP, {"Ár": "2000", "Kyn": "Konur"}),
    ("How many people lived in Iceland on 1 January 2026?", POP, {"Ár": "2026"}),
    ("Hvað bjuggu margir í Kópavogi árið 2025?", ["MAN02005"], {"Sveitarfélag": "Kópavogsbær", "Ár": "2025"}),
    ("Hvað voru margir karlar í Hafnarfirði 2026?", ["MAN02005"],
     {"Sveitarfélag": "Hafnarfjarðarkaupstaður", "Ár": "2026", "Kyn": "Karlar"}),
    ("Hver var meðalaldur kvenna á Íslandi árið 2026?", ["MAN00102"], {"Ár": "2026", "Kyn": "Konur"}),
    # births, deaths, migration
    ("Hve mörg börn fæddust á Akureyri árið 2024?", ["MAN05110"], {"Sveitarfélag": "Akureyrarbær", "Ár": "2024"}),
    ("Hvað fæddust mörg börn í Reykjavík 2020?", ["MAN05110"], {"Sveitarfélag": "Reykjavíkurborg", "Ár": "2020"}),
    ("How many babies were born in Kópavogur in 2025?", ["MAN05110"], {"Sveitarfélag": "Kópavogsbær", "Ár": "2025"}),
    ("hvað fæddust mörg börn árið 1950?", ["MAN05203", "MAN05204"], {"Ár": "1950"}),  # the user's own wording
    ("Hversu margir dóu í Vestmannaeyjum árið 2023?", ["MAN05221"],
     {"Sveitarfélag": "Vestmannaeyjabær", "Ár": "2023"}),
    ("Hvað fluttu margir til landsins frá útlöndum árið 2024?", ["MAN01201"],
     {"Ár": "2024", "Flutningssvæði": "Milli landa", "Flutningur": "Aðfluttir"}),
    ("Hvað voru margar einstæðar mæður í Reykjavík árið 2025?", ["MAN07108"],
     {"Sveitarfélag": "Reykjavíkurborg", "Ár": "2025", "Kjarnafjölskyldur": "Kjarnafjölskyldur alls",
      "Fjölskyldugerð": "Kona með börn"}),
    # citizenship, immigrants
    ("Hversu margir pólskir ríkisborgarar bjuggu á Íslandi 1. janúar 2026?", ["MAN04203"],
     {"Ríkisfang": "Pólland", "Ár": "2026"}),
    ("Hvað bjuggu margir innflytjendur í Reykjanesbæ árið 2026?", ["MAN43005"],
     {"Sveitarfélag": "Reykjanesbær", "Ár": "2026"}),
    # tourism
    ("Hversu margar gistinætur voru á Akureyri árið 2024?", ["SAM01603"],
     {"Sveitarfélag": "Akureyri", "Ár": "2024", "Eining": "Gistinætur"}),
    ("Hvað komu margir erlendir gestir á gististaði í Reykjavík í júlí 2025?", ["SAM01603"],
     {"Sveitarfélag": "Reykjavík", "Ár": "2025", "Mánuður": "Júlí", "Þjóðerni": "Útlendingar",
      "Eining": "Gestakomur"}),
    # income
    ("Hvert var meðaltal heildartekna þeirra sem höfðu tekjur í Garðabæ árið 2024?", ["TEK01002"],
     {"Sveitarfélag": "Garðabær", "Ár": "2024", "Tekjur og skattar": "Heildartekjur",
      "Eining": "Meðaltal - skilyrt"}),
    # fishing
    ("Hvað var heildarafli íslenskra fiskiskipa árið 2023?", ["SJA09005"], {"Ár": "2023", "Fisktegund": "Heildarafli"}),
    ("Hvað veiddist mikið af þorski árið 2020?", ["SJA09005"], {"Ár": "2020", "Fisktegund": "Þorskur"}),
    # projections, elections
    ("Hvað er spáð að mörg börn fæðist á Íslandi árið 2050?", ["MAN09012"],
     {"Ár": "2050", "Spáafbrigði": "Spá: miðgildi", "Helstu þættir": "Fæddir alls"}),
    ("Hvað fékk Sjálfstæðisflokkurinn mörg atkvæði í borgarstjórnarkosningunum 1958?", ["KOS03107"],
     {"Ár": "1958", "Einingar": "Gild atkvæði", "Stjórnmálaflokkur": "Sjálfstæðisflokkur"}),
    # prices
    ("Hver var ársbreyting vísitölu neysluverðs í september 2026?", ["VIS01000"],
     {"Mánuður": "2026M09", "Vísitala": "Vísitala neysluverðs", "Liður": "Ársbreyting, %"}),
    ("What was the annual inflation rate in Iceland in August 2026?", ["VIS01000"],
     {"Mánuður": "2026M08", "Vísitala": "Vísitala neysluverðs", "Liður": "Ársbreyting, %"}),
]


# "medium" (their word): one number computed from cells of one table, by pipeline.compute() (difference = a - b,
# percent_change = (a - b) / b * 100, share = a / b * 100, sum). (question, operation, [(table, values), ...])
REYKJAVIK, KOPAVOGUR = {"Sveitarfélag": "Reykjavíkurborg"}, {"Sveitarfélag": "Kópavogsbær"}
CALC_SPECS = [
    ("Hversu mikið fjölgaði íbúum Reykjavíkur frá 2025 til 2026?", "difference",
     [("MAN02005", {**REYKJAVIK, "Ár": "2026"}), ("MAN02005", {**REYKJAVIK, "Ár": "2025"})]),
    ("Hve hátt hlutfall landsmanna bjó í Reykjavík árið 2026?", "share",
     [("MAN02005", {**REYKJAVIK, "Ár": "2026"}), ("MAN02005", {"Ár": "2026"})]),
    ("Um hve mörg prósent fjölgaði íbúum Kópavogs frá 2020 til 2025?", "percent_change",
     [("MAN02005", {**KOPAVOGUR, "Ár": "2025"}), ("MAN02005", {**KOPAVOGUR, "Ár": "2020"})]),
    ("How much did the number of births in Iceland change from 2020 to 2024?", "difference",
     [("MAN05110", {"Ár": "2024"}), ("MAN05110", {"Ár": "2020"})]),
    ("Hvað fæddust mörg börn samtals á Akureyri og í Kópavogi árið 2024?", "sum",
     [("MAN05110", {"Sveitarfélag": "Akureyrarbær", "Ár": "2024"}), ("MAN05110", {**KOPAVOGUR, "Ár": "2024"})]),
    ("Hversu miklu meiri var þorskaflinn en ýsuaflinn árið 2023?", "difference",
     [("SJA09005", {"Ár": "2023", "Fisktegund": "Þorskur"}), ("SJA09005", {"Ár": "2023", "Fisktegund": "Ýsa"})]),
    ("What share of all overnight stays in Iceland in 2024 were in Reykjavík?", "share",
     [("SAM01603", {"Sveitarfélag": "Reykjavík", "Ár": "2024", "Eining": "Gistinætur"}),
      ("SAM01603", {"Ár": "2024", "Eining": "Gistinætur"})]),
]

# "hard": numbers from two tables, every one must be in the answer. (question, [(table, values), ...])
CPI_SEP_2026 = {"Mánuður": "2026M09", "Vísitala": "Vísitala neysluverðs", "Liður": "Ársbreyting, %"}
BOTH_SPECS = [
    ("Hvað bjuggu margir á Íslandi 1. janúar 2026 og hver var ársbreyting vísitölu neysluverðs í september 2026?",
     [("MAN00101", {"Ár": "2026"}), ("VIS01000", CPI_SEP_2026)]),
    ("Hversu margir fæddust og hversu margir dóu í Reykjavík árið 2023?",
     [("MAN05110", {**REYKJAVIK, "Ár": "2023"}), ("MAN05221", {**REYKJAVIK, "Ár": "2023"})]),
    ("How many people lived in Kópavogur in 2025, and how many babies were born there that year?",
     [("MAN02005", {**KOPAVOGUR, "Ár": "2025"}), ("MAN05110", {**KOPAVOGUR, "Ár": "2025"})]),
    ("Hver var heildarafli íslenskra fiskiskipa árið 2023 og hversu margar gistinætur voru á Íslandi það ár?",
     [("SJA09005", {"Ár": "2023", "Fisktegund": "Heildarafli"}), ("SAM01603", {"Ár": "2023", "Eining": "Gistinætur"})]),
    ("Hvað bjuggu margir pólskir ríkisborgarar á Íslandi 1. janúar 2026 og hvað fluttu margir til landsins frá "
     "útlöndum árið 2025?",
     [("MAN04203", {"Ríkisfang": "Pólland", "Ár": "2026"}),
      ("MAN01201", {"Ár": "2025", "Flutningssvæði": "Milli landa", "Flutningur": "Aðfluttir"})]),
]

def truth_for(table, wanted):
    out = {}
    for v in table["variables_is"]:
        texts = [t.strip() for t in v["valueTexts"]]
        if v["text"] in wanted:
            want = wanted[v["text"]]
            hits = [c for c, t in zip(v["values"], texts) if t == want]
        else:
            hits = [c for c, t in zip(v["values"], texts) if t.lower() in TOTALS]
        if len(hits) != 1:
            raise SystemExit(f'{table["id"]} {v["text"]!r}: {len(hits)} matches for {wanted.get(v["text"], "a total")!r}')
        out[v["code"]] = hits
    unknown = set(wanted) - {v["text"] for v in table["variables_is"]}
    if unknown:
        raise SystemExit(f'{table["id"]}: no variable {unknown}')
    return out


def unique(by_id, tid):
    if len(by_id[tid]) != 1:
        raise SystemExit(f"{tid}: id shared by several catalog tables; pick another")
    return by_id[tid][0]


def main():
    by_id = {}
    for t in load_catalog():
        by_id.setdefault(t["id"], []).append(t)
    cell = lambda tid, wanted: {"table": tid, "truth": truth_for(unique(by_id, tid), wanted)}  # noqa: E731
    items = [{"q": q, "table": tables, "truth": truth_for(unique(by_id, tables[0]), wanted), "level": "easy"}
             for q, tables, wanted in SPECS]
    items += [{"q": q, "table": list(dict.fromkeys(tid for tid, _ in cells)), "level": "medium",
               "calc": {"op": op, "args": [cell(tid, w) for tid, w in cells]}} for q, op, cells in CALC_SPECS]
    items += [{"q": q, "table": [tid for tid, _ in cells], "level": "hard", "all": [cell(tid, w) for tid, w in cells]}
              for q, cells in BOTH_SPECS]
    out = ROOT / "eval" / "questions_e2e.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    levels = {lv: sum(i["level"] == lv for i in items) for lv in ("easy", "medium", "hard")}
    print(f"wrote {len(items)} questions to {out}: {levels}")


if __name__ == "__main__":
    main()

"""Our own frozen question set, for measuring on questions nobody tuned on (31 Oct 2026).

    GAGNVIST_DATA=<release>/data .venv/bin/python eval/build_new_set.py

Questions: written by other LLMs from a prompt (asker personas, many topics, typos, relative dates, localities,
ambiguity, no-data cases) and by real people on Vísindavefurinn (`source`); de-duplicated and chosen by hand.
Four categories:
    A  answerable: one right number (or a few)                    -> eval/questions_new.jsonl
    B  answerable but ambiguous or partial: the number(s), with
       accepted alternatives where the question allows them        -> eval/questions_new.jsonl
    C  not published by Statistics Iceland (another agency has it) -> eval/questions_new_nodata.jsonl
    D  impossible, private, opinion or future                      -> eval/questions_new_nodata.jsonl
A/B are in the organizers' format, so eval/gagnvist_eval.py scores them as it scores theirs. Every expected
number is read here from the organizers' data release (snapshot of 18 Oct 2026, Icelandic edition): nobody types
a number in. `accepted` lists only the source table (and stated equivalents), so scores are pessimistic. Key fix 31 Oct
(evening): MAN43000 (teenagers) and MAN04001 (foreign citizens) added where they hold exactly the same number, as
the organizers' key accepts every such table; the pipeline was not changed for it.
C/D have no number: the right answer declines (C ideally names where the data is); they need their own check.

FROZEN: never change the pipeline because of these questions; run them rarely, to report.
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hagstofa import local_data  # noqa: E402
from hagstofa.retrieve import load_catalog  # noqa: E402

OUT, OUT_NODATA = ROOT / "eval/questions_new.jsonl", ROOT / "eval/questions_new_nodata.jsonl"
CATALOG = load_catalog()


def P(tid, cats, path="", alt=False, same=()):
    """One cell, or a sum of cells: cats {variable code: value code or [codes summed]}. `path` picks between
    tables that share an id (MAN10001 is three tables); alt=True: a cell only for an accepted alternative number,
    not a table the answer must use; `same`: other tables accepted for this part."""
    return {"table": tid, "cats": cats, "path": path, "alt": alt, "same": list(same)}


def ages(lo, hi):
    return [str(a) for a in range(lo, hi + 1)]


# id, language, level, category, source, question, parts, answer(values) -> [numbers], unit, how, also (alternatives)
SPECS = [
    # --- population: localities, municipalities, the country ---
    dict(id="husavik-now-is", lang="is", level="easy", cat="A", source="llm", q="hvað búa margir á Húsavík núna?",
         parts=[P("MAN030101", {"Byggðakjarnar": "710", "Aldur": "Alls", "Ár": "2026", "Kyn": "Alls"})], unit="people"),
    dict(id="vik-en", lang="en", level="easy", cat="A", source="llm", q="How many people live in Vik?",
         parts=[P("MAN030101", {"Byggðakjarnar": "1090", "Aldur": "Alls", "Ár": "2026", "Kyn": "Alls"})], unit="people"),
    dict(id="siglufjordur-vv-is", lang="is", level="easy", cat="A", source="visindavefur", q="Hvað búa margir á Siglufirði?",
         parts=[P("MAN030101", {"Byggðakjarnar": "580", "Aldur": "Alls", "Ár": "2026", "Kyn": "Alls"})], unit="people"),
    dict(id="kopavogur-2025-typo-is", lang="is", level="easy", cat="A", source="llm",
         q="Hvað bjuggu margir í Kopavogi 1. januar 2025?",
         parts=[P("MAN02005", {"Sveitarfélag": "1000", "Aldur": "-1", "Ár": "2025", "Kyn": "0"})], unit="people"),
    dict(id="iceland-and-capital-vv-is", lang="is", level="medium", cat="A", source="visindavefur",
         q="Hvað búa margir á Íslandi? En á höfuðborgarsvæðinu?",
         parts=[P("MAN03250", {"Landshlutar": "Total", "Aldur": "-1", "Ár": "2026", "Kyn": "0"}),
                P("MAN03250", {"Landshlutar": "Capital region", "Aldur": "-1", "Ár": "2026", "Kyn": "0"})],
         answer=lambda v: [v[0], v[1]], unit="people", how="both numbers, 1 January 2026"),
    dict(id="selfoss-growth-2020-2025-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað fjölgaði íbúum á Selfossi mikið frá 2020 til 2025?",
         parts=[P("MAN030101", {"Byggðakjarnar": "1150", "Aldur": "Alls", "Ár": "2025", "Kyn": "Alls"}),
                P("MAN030101", {"Byggðakjarnar": "1150", "Aldur": "Alls", "Ár": "2020", "Kyn": "Alls"})],
         answer=lambda v: [v[0] - v[1]], unit="people", how="1 January 2025 minus 1 January 2020"),
    dict(id="kopavogur-age-0-5-2025-is", lang="is", level="medium", cat="A", source="llm",
         q="Hversu mörg börn á aldrinum 0–5 ára voru búsett í Kópavogi 1. janúar 2025?",
         parts=[P("MAN02005", {"Sveitarfélag": "1000", "Aldur": ages(0, 5), "Ár": "2025", "Kyn": "0"})],
         unit="children", how="sum of ages 0 to 5"),
    dict(id="outside-capital-share-2025-en", lang="en", level="medium", cat="A", source="llm",
         q="What percentage of Iceland's population lived outside the capital region in 2025?",
         parts=[P("MAN03250", {"Landshlutar": "Total", "Aldur": "-1", "Ár": "2025", "Kyn": "0"}),
                P("MAN03250", {"Landshlutar": "Capital region", "Aldur": "-1", "Ár": "2025", "Kyn": "0"})],
         answer=lambda v: [(v[0] - v[1]) / v[0] * 100], unit="%", how="(all minus capital region) / all, 1 Jan 2025"),
    dict(id="reykjavik-over-67-share-2025-is", lang="is", level="medium", cat="B", source="llm",
         q="Hversu stór hluti íbúa Reykjavíkur var eldri en 67 ára í byrjun árs 2025?",
         parts=[P("MAN02005", {"Sveitarfélag": "0000", "Aldur": ages(68, 109), "Ár": "2025", "Kyn": "0"}),
                P("MAN02005", {"Sveitarfélag": "0000", "Aldur": "-1", "Ár": "2025", "Kyn": "0"}),
                P("MAN02005", {"Sveitarfélag": "0000", "Aldur": ages(67, 109), "Ár": "2025", "Kyn": "0"}, alt=True)],
         answer=lambda v: [v[0] / v[1] * 100], also=lambda v: [[v[2] / v[1] * 100]], unit="%",
         how="ages 68+ / all in Reykjavíkurborg (67+ also accepted: 'eldri en 67' is read both ways)"),
    dict(id="teenagers-vv-is", lang="is", level="medium", cat="B", source="visindavefur", q="Hvað búa margir unglingar á Íslandi?",
         parts=[P("MAN00101", {"Kyn": "Total", "Aldur": ages(13, 19), "Ár": "2026"}, same=["MAN43000"]),  # key fix 31 Oct: same number
                P("MAN00101", {"Kyn": "Total", "Aldur": ages(13, 17), "Ár": "2026"}, alt=True),
                P("MAN00101", {"Kyn": "Total", "Aldur": ages(10, 19), "Ár": "2026"}, alt=True)],
         answer=lambda v: [v[0]], also=lambda v: [[v[1], v[2]]], unit="people",
         how="ambiguous: 13-19 (also accepted 13-17 and 10-19); the answer should say which ages it took"),
    dict(id="population-1918-1944-vv-is", lang="is", level="medium", cat="B", source="visindavefur",
         q="Hvað bjuggu margir á Íslandi árin 1918 og 1944?",
         parts=[P("MAN00101", {"Kyn": "Total", "Aldur": "-1", "Ár": "1918"}, same=["MAN08000"]),
                P("MAN00101", {"Kyn": "Total", "Aldur": "-1", "Ár": "1944"}, same=["MAN08000"]),
                P("MAN08000", {"Kyn": "Total", "Aldur": "-1", "Ár": "1918"}, alt=True),
                P("MAN08000", {"Kyn": "Total", "Aldur": "-1", "Ár": "1944"}, alt=True)],
         answer=lambda v: [v[0], v[1]], also=lambda v: [[v[2]], [v[3]]], unit="people",
         how="1 January of each year (the yearly average, MAN08000, also accepted)"),
    # --- births, deaths, age ---
    dict(id="mean-age-2024-is", lang="is", level="easy", cat="A", source="llm", q="Hver var meðalaldur Íslendinga í byrjun árs 2024?",
         parts=[P("MAN00102", {"Ár": "2024", "Kyn": "Total"})], unit="years"),
    dict(id="boys-born-2023-is", lang="is", level="easy", cat="A", source="llm", q="Hvað fæddust margir drengir á Íslandi árið 2023?",
         parts=[P("MAN05100", {"Kyn": "1", "Ár": "2023", "Skipting": "0", "Fjöldi barna": "0"})], unit="boys"),
    dict(id="deaths-2023-en", lang="en", level="easy", cat="A", source="llm", q="How many deaths were registered in Iceland in 2023?",
         parts=[P("MAN05210", {"Ár": "2023", "Aldur": "0", "Eining": "0", "Kyn": "0"})], unit="deaths"),
    dict(id="fertility-2024-en", lang="en", level="easy", cat="A", source="llm",
         q="What was the total fertility rate per woman in Iceland in 2024?",
         parts=[P("MAN05202", {"Ár": "2024", "Eining": "0"})], unit="children per woman"),
    # --- citizenship, immigrants, migration, religion ---
    dict(id="foreign-citizens-2025-en", lang="en", level="easy", cat="A", source="llm",
         q="How many foreign citizens were living in Iceland on 1 January 2025?",
         parts=[P("MAN04208", {"Sveitarfélag": "IS", "Ríkisfang": "Erl", "Aldur": "999", "Ár": "2025", "Kyn": "0"},
                  same=["MAN04001"])], unit="people"),  # key fix 31 Oct: MAN04001 holds the same number
    dict(id="reykjanesbaer-foreign-share-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað er hátt hlutfall íbúa í Reykjanesbæ með erlent ríkisfang?",
         parts=[P("MAN04208", {"Sveitarfélag": "2000", "Ríkisfang": "Erl", "Aldur": "999", "Ár": "2026", "Kyn": "0"}),
                P("MAN04208", {"Sveitarfélag": "2000", "Ríkisfang": "Tot", "Aldur": "999", "Ár": "2026", "Kyn": "0"})],
         answer=lambda v: [v[0] / v[1] * 100], unit="%", how="foreign citizens / all, Reykjanesbær, 1 January 2026 (latest)"),
    dict(id="immigrants-vv-is", lang="is", level="easy", cat="B", source="visindavefur",
         q="Hversu margir innflytjendur búa á Íslandi og hvaðan koma þeir?",
         parts=[P("MAN43000", {"Ár": "2026", "Aldur": "-1", "Bakgrunnur": "124", "Kyn": "0"})], unit="immigrants",
         how="the count is scored; 'where from' is not (it needs another table)"),
    dict(id="akureyri-immigrant-share-2025-is", lang="is", level="hard", cat="A", source="llm",
         q="Hvert var hlutfall innflytjenda af heildaríbúafjölda Akureyrar 1. janúar 2025?",
         parts=[P("MAN43005", {"Sveitarfélag": "6000", "Aldur": "999", "Kyn": "0", "Ár": "2025"}),
                P("MAN02005", {"Sveitarfélag": "6000", "Aldur": "-1", "Ár": "2025", "Kyn": "0"})],
         answer=lambda v: [v[0] / v[1] * 100], unit="%", how="immigrants (MAN43005) / population (MAN02005), Akureyrarbær"),
    dict(id="net-migration-2025-en", lang="en", level="medium", cat="A", source="llm",
         q="What was net migration to Iceland in 2025 – how many more people moved in than moved out?",
         parts=[P("MAN01400", {"Kyn": "0", "Ár": "2025", "Ríkisfang": "1", "Tegund flutnings": "1"})], unit="people"),
    dict(id="national-church-2025-en", lang="en", level="easy", cat="A", source="llm",
         q="How many members did the National Church of Iceland have on January 1, 2025?",
         parts=[P("MAN10001", {"Ár": "2025", "Trú- og lífsskoðunarfélög": "1", "Skipting": "0"}, path="trufelog")], unit="members"),
    dict(id="national-church-now-is", lang="is", level="easy", cat="A", source="llm", q="Hvað eru margir í þjóðkirkjunni?",
         parts=[P("MAN10001", {"Ár": "2026", "Trú- og lífsskoðunarfélög": "1", "Skipting": "0"}, path="trufelog")], unit="members"),
    dict(id="muslims-vv-is", lang="is", level="medium", cat="B", source="visindavefur", q="Hvað búa margir múslimar á Íslandi?",
         parts=[P("MAN10001", {"Ár": "2026", "Trú- og lífsskoðunarfélög": ["J", "Æ", "/", "Ä"], "Skipting": "0"}, path="trufelog")],
         unit="people", how="members of the four Muslim associations together (registration, not belief)"),
    dict(id="name-jon-is", lang="is", level="easy", cat="B", source="llm",
         q="Hversu margir heita Jón á Íslandi í dag skv. tölum frá 1. janúar 2025?",
         parts=[P("MAN11101", {"Röð": "0", "Eining": "Fjoldi"})], unit="people",
         how="partial: the latest name data is 1 January 2023; the answer should say so"),
    # --- labour, wages, prices ---
    dict(id="unemployment-2026-08-typo-is", lang="is", level="easy", cat="A", source="llm",
         q="hvað var atvinuleisið mikið í ágúst 2026",
         parts=[P("VIN00001", {"Mánuður": "2026M08", "Kyn/aldur": "0", "Eining": "7"})], unit="%"),
    dict(id="weekly-hours-now-en", lang="en", level="easy", cat="B", source="llm",
         q="What is the average weekly working hours in Iceland right now?",
         parts=[P("VIN00941", {"Ár": "2025", "Landsvæði": "0", "Kyn": "0", "Aldur": "0", "Meðalfjöldi klst.": "0"}),
                P("VIN00941", {"Ár": "2025", "Landsvæði": "0", "Kyn": "0", "Aldur": "0", "Meðalfjöldi klst.": "3"}, alt=True)],
         answer=lambda v: [v[0]], also=lambda v: [[v[1]]], unit="hours", how="actual hours 2025 (usual hours also accepted)"),
    dict(id="gender-pay-gap-2024-en", lang="en", level="easy", cat="A", source="llm",
         q="What was the gender wage gap across all sectors in Iceland in 2024?",
         parts=[P("VIN02010", {"Ár": "2024", "Hópur": "0", "Mæling": "0"})], unit="%"),
    dict(id="wages-last-year-is", lang="is", level="medium", cat="A", source="llm", q="Hvað hækkuðu launin að meðaltali í fyrra?",
         parts=[P("LAU04200", {"Ár": "2025", "Eining": "change_A"})], unit="%", how="wage index, change of the 2025 annual average"),
    dict(id="inflation-2025-06-is", lang="is", level="easy", cat="A", source="llm", q="Hvað var verðbólgan mikil í júní 2025?",
         parts=[P("VIS01000", {"Mánuður": "2025M06", "Vísitala": "CPI", "Liður": "change_A"})], unit="%"),
    dict(id="inflation-12m-is", lang="is", level="easy", cat="A", source="llm", q="Hvað hefur verðbólgan verið mikil síðustu 12 mánuði?",
         parts=[P("VIS01000", {"Mánuður": "2026M09", "Vísitala": "CPI", "Liður": "change_A"})], unit="%",
         how="latest month in the release (September 2026)"),
    dict(id="unemployment-and-inflation-2026-03-is", lang="is", level="hard", cat="A", source="llm",
         q="Hvernig var atvinnuleysi og verðbólga á Íslandi í sama mánuði, til dæmis í mars 2026?",
         parts=[P("VIN00001", {"Mánuður": "2026M03", "Kyn/aldur": "0", "Eining": "7"}),
                P("VIS01000", {"Mánuður": "2026M03", "Vísitala": "CPI", "Liður": "change_A"})],
         answer=lambda v: [v[0], v[1]], unit="%", how="both numbers"),
    # --- economy, tourism, fisheries, agriculture, elections ---
    dict(id="gdp-per-capita-2024-is", lang="is", level="easy", cat="A", source="llm",
         q="Hver var landsframleiðsla á mann árið 2024 í krónum?",
         parts=[P("THJ01401", {"Ár": "2024", "Mælikvarði": "0"})], answer=lambda v: [v[0]], also=lambda v: [[v[0] * 1000]],
         unit="thousand ISK", how="table in thousands of krónur (the full krónur figure also accepted)"),
    dict(id="kef-passengers-2026-08-en", lang="en", level="easy", cat="A", source="llm",
         q="How many passengers flew through Keflavik airport in August 2026?",
         parts=[P("SAM02001", {"Ríkisfang": "0", "Mánuður": "2026M08"})], unit="passengers"),
    dict(id="westfjords-stays-per-inhabitant-2024-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað voru gistinætur á hvern íbúa á Vestfjörðum árið 2024?",
         parts=[P("SAM01601", {"Þjóðerni": "Total", "Ár": "2024", "Landshluti": "7", "Eining": "0", "Mánuður": "0"}),
                P("MAN03250", {"Landshlutar": "Westfjords", "Aldur": "-1", "Ár": "2024", "Kyn": "0"}),
                P("MAN03250", {"Landshlutar": "Westfjords", "Aldur": "-1", "Ár": "2025", "Kyn": "0"}, alt=True)],
         answer=lambda v: [v[0] / v[1]], also=lambda v: [[v[0] / v[2]]], unit="overnight stays per inhabitant",
         how="stays 2024 / population 1 Jan 2024 (1 Jan 2025 also accepted)"),
    dict(id="cod-catch-2024-is", lang="is", level="easy", cat="A", source="llm",
         q="Hversu mörg tonn af þorski veiddust af íslenskum skipum árið 2024?",
         parts=[P("SJA09005", {"Ár": "2024", "Fisktegund": "1"})], unit="tonnes"),
    dict(id="sheep-2024-is", lang="is", level="easy", cat="A", source="llm", q="Hversu margt sauðfé var á landinu árið 2024?",
         parts=[P("LAN10102", {"Ár": "2024", "Búpeningur": "6", "Landshluti": "0"})], unit="sheep"),
    dict(id="cattle-2015-2024-en", lang="en", level="medium", cat="A", source="llm",
         q="Did the number of cattle in Iceland go up or down between 2015 and 2024, and by how much?",
         parts=[P("LAN10102", {"Ár": "2024", "Búpeningur": "0", "Landshluti": "0"}),
                P("LAN10102", {"Ár": "2015", "Búpeningur": "0", "Landshluti": "0"})],
         answer=lambda v: [v[0] - v[1]], unit="cattle", how="2024 minus 2015"),
    dict(id="valid-votes-president-2024-en", lang="en", level="easy", cat="A", source="llm",
         q="How many valid votes were cast in the 2024 Icelandic presidential election?",
         parts=[P("KOS07100", {"Ár": "9", "Atkvæði": "1", "Eining": "0"})], unit="votes"),
    dict(id="votes-president-2024-vs-2020-en", lang="en", level="medium", cat="A", source="llm",
         q="How many more people voted in the 2024 presidential election than in the 2020 presidential election?",
         parts=[P("KOS07100", {"Ár": "9", "Atkvæði": "0", "Eining": "0"}), P("KOS07100", {"Ár": "8", "Atkvæði": "0", "Eining": "0"})],
         answer=lambda v: [v[0] - v[1]], unit="votes", how="votes cast 2024 minus 2020"),
    dict(id="turnout-althingi-2024-vs-2021-is", lang="is", level="medium", cat="A", source="llm",
         q="Hver var kjörsóknin í alþingiskosningunum 2024 og var hún meiri eða minni en 2021?",
         parts=[P("KOS02001", {"Ár": "46", "Skipting": "4", "Kyn": "0"}),
                P("KOS02001", {"Ár": "45", "Skipting": "4", "Kyn": "0"})],
         answer=lambda v: [v[0], v[0] - v[1]], unit="%", how="official turnout 2024, and 2024 minus 2021 in points"),
    dict(id="turnout-arborg-2022-is", lang="is", level="easy", cat="A", source="llm",
         q="Hversu hátt var kjörsókn í sveitarstjórnarkosningunum 2022 í Árborg?",
         parts=[P("KOS03190a", {"Sveitarfélag": "8200", "Aldur": "Alls", "Kyn": "0"})], unit="%"),
]

NODATA = [  # category, id, language, source, question, where the data is (C) or why not (D)
    ("C", "stocks-last-month-en", "en", "llm", "Which exact stocks on the Iceland Stock Exchange gained the most value last month?",
     "Nasdaq Iceland, not Statistics Iceland"),
    ("C", "new-cars-2025-01-is", "is", "llm", "Hversu margir nýir fólksbílar voru skráðir í janúar 2025?",
     "Samgöngustofa (vehicle registry); Statistics Iceland has year-end vehicle stock only"),
    ("C", "rent-reykjavik-en", "en", "llm", "What's the average monthly rent for apartments in Reykjavík these days?",
     "HMS (Housing and Construction Authority) publishes rents"),
    ("C", "rent-akureyri-en", "en", "llm", "How much did average rental prices in Akureyri change compared to two years ago?",
     "HMS; no current rent index at Statistics Iceland"),
    ("C", "registered-unemployed-ne-2025-12-is", "is", "llm",
     "Hvað voru margir skráðir atvinnulausir á Norðurlandi eystra í desember 2025?",
     "Vinnumálastofnun counts the registered unemployed; Statistics Iceland measures unemployment by survey"),
    ("C", "surname-2025-is", "is", "llm", "Hvaða eftirnafn var algengast á Íslandi árið 2025?",
     "no surname statistics (most Icelanders have patronymics); first names only, 1 Jan 2023"),
    ("C", "reykjavik-2035-en", "en", "llm", "What will the population of Reykjavík be in 2035, broken down by age and citizenship?",
     "partial: a projection exists for the whole country (MAN09010), not for Reykjavík or by citizenship"),
    ("C", "eu-unemployment-2025q4-en", "en", "llm",
     "What was the unemployment rate in Iceland compared with the EU average in the final quarter of 2025?",
     "partial: Iceland's rate is published; the EU average is Eurostat's"),
    ("D", "neighbour-salary-is", "is", "llm", "Hvað er nágranni minn í Hafnarfirði með í laun? Hann heitir Jón Pétursson.",
     "private: no individual's income is published"),
    ("D", "elves-en", "en", "llm", "What percentage of Icelanders believe in elves?", "opinion; not an official statistic"),
    ("D", "inflation-next-year-en", "en", "llm", "What will inflation in Iceland be next year?",
     "future; Statistics Iceland publishes no inflation forecast (the Central Bank does)"),
    ("D", "marriages-1240-is", "is", "llm", "Hvað voru margir búnir að giftast í þjóðkirkjunni árið 1240?",
     "no records; marriage statistics start in 1951 here"),
    ("D", "population-1200-en", "en", "llm",
     "How many people lived in Iceland in the year 1200, and how was the population distributed between towns and rural areas?",
     "no census before 1703; population series here starts in 1841"),
    ("D", "snorri-vv-is", "is", "visindavefur", "Hversu margir bjuggu á Íslandi þegar Snorri Sturluson var uppi?",
     "13th century: no data"),
    ("D", "friendliest-town-en", "en", "llm",
     "How happy are people in Iceland with their neighbours, and which town has the friendliest residents?",
     "opinion; not measured"),
    ("D", "disease-diagnosed-en", "en", "llm",
     "How many people in Iceland were diagnosed with a specific disease last year, and how many died from it?",
     "diagnoses are health-registry data; deaths by cause exist (MAN05302) but the disease is not named"),
]


def table(tid, path=""):
    return next(t for t in CATALOG if t["id"] == tid and path in t["path"])


def read(part):
    """The cell's number from the release, Icelandic edition; a list of codes is summed."""
    t = table(part["table"], part["path"])
    sel = {k: v if isinstance(v, list) else [v] for k, v in part["cats"].items()}
    values = local_data.query(t["path"], sel)["value"]
    if any(x is None for x in values):
        raise ValueError(f"{part['table']} {part['cats']}: missing number")
    return float(sum(values)), t


def labels(t, cats):
    out = {}
    for v in t["variables_is"]:
        if v["code"] in cats:
            texts = dict(zip(v["values"], v["valueTexts"]))
            c = cats[v["code"]]
            out[v["text"]] = " + ".join(texts[x] for x in c) if isinstance(c, list) and len(c) <= 4 else (
                f"{texts[c[0]]} ... {texts[c[-1]]}" if isinstance(c, list) else texts[c])
    return out


# Verified 31 Oct (evening): the answer read exactly this part's number from another table (its rows hold the same
# value, or values summing to it). The organizers' key accepts every table holding the same number; ours does too.
KEY_FIXES = {
    "siglufjordur-vv-is": {
        "MAN030101": [
            "MAN030102"
        ]
    },
    "reykjavik-over-67-share-2025-is": {
        "MAN02005": [
            "MAN03270"
        ]
    },
    "population-1918-1944-vv-is": {
        "MAN00101": [
            "MAN00000"
        ]
    },
    "reykjanesbaer-foreign-share-is": {
        "MAN04208": [
            "MAN04203"
        ]
    },
    "wages-last-year-is": {
        "LAU04200": [
            "LAU04201"
        ]
    }
}


def main():
    rows = []
    for s in SPECS:
        values, parts = [], []
        for p in s["parts"]:
            v, t = read(p)
            values.append(v)
            if not p["alt"]:  # cells the answer must come from; alternatives only add accepted numbers
                parts.append({"table": p["table"], "accepted": [p["table"], *p["same"], *KEY_FIXES.get(s["id"], {}).get(p["table"], [])], "categories": p["cats"],
                              "labels": labels(t, p["cats"]), "value": v})
        numbers = (s.get("answer") or (lambda v: [v[0]]))(values)
        also = (s.get("also") or (lambda v: [[] for _ in numbers]))(values)
        also += [[]] * (len(numbers) - len(also))
        answers = [{"value": round(n, 4), "unit": s["unit"], "also": [{"value": round(a, 4), "unit": s["unit"]} for a in alts]}
                   for n, alts in zip(numbers, also)]
        used = parts
        rows.append({"id": s["id"], "language": s["lang"], "level": s["level"], "question": s["q"], "answers": answers,
                     "how": s.get("how", ""), "parts": used, "category": s["cat"], "source": s["source"]})
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(OUT_NODATA, "w", encoding="utf-8") as f:
        for cat, qid, lang, source, q, note in NODATA:
            f.write(json.dumps({"id": qid, "language": lang, "category": cat, "source": source, "question": q,
                                "expect": "decline", "note": note}, ensure_ascii=False) + "\n")
    for r in rows:
        print(f'{r["category"]} {r["level"][:4]:4} {r["id"]:42} {" + ".join(str(a["value"]) for a in r["answers"]):28} '
              f'{"/".join(p["table"] for p in r["parts"])}')
    print(f"{len(rows)} answerable (A/B) -> {OUT.name}; {len(NODATA)} no-data (C/D) -> {OUT_NODATA.name}")


if __name__ == "__main__":
    main()

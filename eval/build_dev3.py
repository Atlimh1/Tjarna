"""dev3: a DEV set (not frozen): tune on it. Written 1 Nov 2026 by an LLM from the catalog alone (data/catalog.jsonl and
data/cards.jsonl), without looking at the pipeline or at any results.

    GAGNVIST_DATA=<release>/data .venv/bin/python eval/build_dev3.py

62 answerable questions (A/B: 27 easy, 20 medium, 15 hard; 37 Icelandic, 25 English) over tables that no other question
set uses (eval/questions*.jsonl, eval/heldout, set 1, set 2, the organizers' 60), except MAN00101 as the population base
of the per-inhabitant questions (1 January 2021, 2022, 2023 where possible: years no other question asks for). About a
third are typed the way people type (no accents, typos, lowercase, abbreviations), some use relative dates (today is
1 Nov 2026: "í fyrra" = 2025, "síðasta ársfjórðungi" = the table's newest quarter) or name no period (= the table's
newest), some name places, some have a look-alike table that does not fit. 12 more should be declined (C/D).
Same conventions as eval/build_new_set.py (P, answer, also, unit, how); every expected number is read from the
organizers' data release (Icelandic edition) by main(): nobody types a number in. Unlike set 1 and set 2 this set is
NOT frozen: use it to decide on changes, then confirm on a frozen set.
    -> eval/questions_dev3.jsonl and eval/questions_dev3_nodata.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_new_set import ROOT, P, ages, labels, read, table  # noqa: E402,F401

OUT, OUT_NODATA = ROOT / "eval/questions_dev3.jsonl", ROOT / "eval/questions_dev3_nodata.jsonl"


def months(year, first=1, last=12):
    return [f"{year}M{m:02d}" for m in range(first, last + 1)]


def pop(year):  # MAN00101: population on 1 January of `year`, both sexes, all ages
    return {"Kyn": "Total", "Aldur": "-1", "Ár": str(year)}


SOURCES = ["MALMAR", "HEIM", "BILAR", "VEIDI", "FLUG", "SJOFL", "BYGG", "LANDB", "ORKA", "SORP", "VERSL", "EFNI",
           "ANNAD"]  # UMH31220 has no total over sources
STUDENT_AGES = [str(a) for a in range(13)]  # SKO04108 has no total over ages: 19 and younger ... 40 and older

# id, language, level, category, source, question, parts, answer(values) -> [numbers], unit, how, also (alternatives)
SPECS3 = [
    # ===================== easy: one cell =====================
    dict(id="citizenship-granted-last-year-noaccent-is", lang="is", level="easy", cat="A", source="llm",
         q="hvað fengu margir islenskan rikisborgararett i fyrra",
         parts=[P("MAN04119", {"Fyrra ríkisfang": "01", "Aldur": "1", "Ár": "2025", "Kyn": "0"})],
         unit="people", how="í fyrra = 2025; all former citizenships, all ages, both sexes"),
    dict(id="old-age-dependency-nodate-en", lang="en", level="easy", cat="A", source="llm",
         q="What is the old-age dependency ratio in Iceland?",
         parts=[P("MAN00115", {"Ár": "2026", "Framfærsluhlutfall": "Old age dependancy ratio"})],
         unit="% (65+ per 100 aged 15-64)", how="no period: newest, 1 January 2026; Eurostat definition"),
    dict(id="egilsstadir-midyear-pop-2025-is", lang="is", level="easy", cat="A", source="llm",
         q="Hver var meðalmannfjöldi á Egilsstöðum árið 2025?",
         parts=[P("MAN030106", {"Þéttbýlisstaðir": "902", "Aldur": "-1", "Ár": "2025", "Kyn": "0"})], unit="people",
         how="mid-year (average) population of the locality Egilsstaðir in 2025 (MAN030106), not 1 January"),
    dict(id="employed-lfs-last-quarter-en", lang="en", level="easy", cat="A", source="llm",
         q="How many people were employed in Iceland last quarter according to the labour force survey?",
         parts=[P("VIN00930", {"Ársfjórðungur": "2026Q2", "Landsvæði": "Alls", "Kyn": "Karlar og konur",
                               "Aldur": "16-74 ára", "Eining": "Starfandi"})],
         unit="people", how="last quarter = newest quarter in the table, 2026Q2; 16-74, whole country"),
    dict(id="weekly-hours-last-quarter-noaccent-is", lang="is", level="easy", cat="A", source="llm",
         q="hvad unnu menn ad medaltali marga tima a viku a sidasta arsfjordungi",
         parts=[P("VIN01001", {"Ársfjórðungur": "2026Q2", "Landsvæði": "Alls", "Kyn": "Karlar og konur",
                               "Aldur": "16-74 ára", "Meðalfjöldi klst.": "Vinnustundir á viku"})],
         unit="hours per week",
         how="síðasta ársfjórðungi = newest quarter 2026Q2; actual hours in main and second job, people at work, 16-74"),
    dict(id="employed-registers-sudurnes-2026-08-abbrev-is", lang="is", level="easy", cat="A", source="llm",
         q="hvað eru margir starfandi á suðurnesjum skv skrám í ágúst 2026",
         parts=[P("VIN10012", {"Mánuður": "2026M08", "Búseta": "2", "Kyn": "0", "Aldursflokkar": "0",
                               "Bakgrunnur": "0"})],
         unit="people", how="register-based employment, residents of Suðurnes, August 2026, all ages and origins"),
    dict(id="labour-slack-sa-2026-08-en", lang="en", level="easy", cat="A", source="llm",
         q="What was the seasonally adjusted labour market slack in Iceland in August 2026?",
         parts=[P("VIN00004", {"Mánuður": "2026M08", "Slaki - ómætt þörf fyrir atvinnu": "1", "Kyn": "0",
                               "Eining": "2"})],
         unit="%", how="slack rate (Slaki, hlutfall), seasonally adjusted, both sexes, 16-74"),
    dict(id="mean-total-income-women-2025-is", lang="is", level="easy", cat="B", source="llm",
         q="Hverjar voru meðalheildartekjur kvenna á Íslandi árið 2025?",
         parts=[P("TEK01001", {"Tekjur og skattar": "0", "Eining": "0", "Kyn": "2", "Aldur": "0", "Ár": "2025"})],
         answer=lambda v: [v[0]], also=lambda v: [[v[0] * 1000]], unit="ISK per year (table unit)",
         how="mean total income (Heildartekjur), all women filing a tax return (Meðaltal - allir); if the table is in "
             "thousand krónur the full krónur figure is also accepted"),
    dict(id="public-sector-wage-index-2025-en", lang="en", level="easy", cat="A", source="llm",
         q="By how much did wages of public sector employees rise in 2025?",
         parts=[P("LAU04203", {"Ár": "2025", "Vísitala": "LVT", "Launþegahópur": "OPI", "Eining": "change_A"})],
         unit="%", how="wage index (Launavísitala), public sector employees total, change of the 2025 annual average"),
    dict(id="indexation-cpi-november-is", lang="is", level="easy", cat="A", source="llm",
         q="Hver er vísitala neysluverðs til verðtryggingar í nóvember?",
         parts=[P("VIS01004", {"Mánuður": "2026M11", "Vísitala": "financial_indexation"})], unit="index points",
         how="nóvember = November 2026, the current month and the newest value; the index for indexation (VIS01004), "
             "not the CPI measured in a month"),
    dict(id="hicp-norway-2026-08-en", lang="en", level="easy", cat="A", source="llm",
         q="What was inflation in Norway in August 2026 according to the harmonised index (HICP)?",
         parts=[P("VIS02502", {"Tegund": "changeA", "Mánuður": "2026M08", "Land": "NO"})], unit="%",
         how="12-month change of the HICP for Norway, August 2026"),
    dict(id="hotel-rooms-northeast-nodate-lowercase-en", lang="en", level="easy", cat="A", source="llm",
         q="how many hotel rooms are there in north east iceland",
         parts=[P("SAM01202", {"Mánuður": "2026M08", "Fjöldi": "1", "Landshluti": "3"})], unit="rooms",
         how="no period: newest month in the table, August 2026; region Norðurland eystra"),
    dict(id="hvalfjordur-tunnel-traffic-2025-07-is", lang="is", level="easy", cat="A", source="llm",
         q="Hvað fóru margir bílar að meðaltali á sólarhring um Hvalfjarðargöng í júlí 2025?",
         parts=[P("SAM31002", {"Mánuður": "2025M07", "Vegur": "1-f8"})], unit="vehicles per day",
         how="average daily traffic, Ring road through Hvalfjarðargöng, July 2025"),
    dict(id="fin-whales-2022-lowercase-is", lang="is", level="easy", cat="A", source="llm",
         q="hvað voru veiddar margar langreyðar 2022",
         parts=[P("SJA01110", {"Tegund": "Langreyður", "Ár": "2022"})], unit="whales"),
    dict(id="landed-vestmannaeyjar-2024-en", lang="en", level="easy", cat="A", source="llm",
         q="How much fish was landed in Vestmannaeyjar in 2024?",
         parts=[P("SJA09042", {"Fisktegund": "0", "Löndunarhöfn": "77", "Eining": "0", "Ár": "2024", "Mánuður": "0"})],
         unit="tonnes", how="all species, quantity, place of landing Vestmannaeyjar, whole year (by place of landing, "
                            "not by the vessels' home harbour)"),
    dict(id="fresh-cod-price-12m-nodate-is", lang="is", level="easy", cat="A", source="llm",
         q="Hvað hefur verð á ferskum þorski hækkað mikið síðasta árið samkvæmt verðvísitölu sjávarafurða?",
         parts=[P("SJA06100", {"Mánuður": "2026M08", "Liður": "change_A", "Flokkur": "FRSH_Cod"}, same=["VIS08003"])],
         unit="%", how="no period: newest month, August 2026, 12-month change of the subindex Ferskur þorskur "
                       "(SJA06100 and VIS08003 are the same table)"),
    dict(id="chicks-hatched-2026-08-typo-en", lang="en", level="easy", cat="A", source="llm",
         q="how many chiks were hatched in iceland in august 2026",
         parts=[P("LAN10204", {"Mánuður": "2026M08", "Egg eða ungar": "1"})], unit="chicks"),
    dict(id="turnout-reykjanesbaer-2024-is", lang="is", level="easy", cat="A", source="llm",
         q="Hver var kjörsóknin í Reykjanesbæ í alþingiskosningunum 2024?",
         parts=[P("KOS02101", {"Sveitarfélag": "87", "Ár": "7", "Skipting": "2", "Kyn": "0"})], unit="%"),
    dict(id="blank-ballots-last-election-lowercase-en", lang="en", level="easy", cat="B", source="llm",
         q="how many blank ballots were there in the last parliamentary election",
         parts=[P("KOS02123", {"Ár": "7", "Eining": "0", "Kjördæmi": "0", "Atkvæði/seðlar": "3"}),
                P("KOS02123", {"Ár": "7", "Eining": "0", "Kjördæmi": "0", "Atkvæði/seðlar": "2"}, alt=True)],
         answer=lambda v: [v[0]], also=lambda v: [[v[1]]], unit="ballots",
         how="last Althingi election = 30 November 2024, whole country; blank ballots (blank and void together also "
             "accepted)"),
    dict(id="ai-use-enterprises-nodate-is", lang="is", level="easy", cat="A", source="llm",
         q="Hve hátt hlutfall fyrirtækja á Íslandi notar gervigreind?",
         parts=[P("STI03266", {"Starfsemi/Stærð": "A12", "Ár": "2025", "Tækni": "A9"})], unit="% of enterprises",
         how="no period: the only year, 2025; enterprises using at least one of the AI technologies, all enterprises"),
    dict(id="school-cost-per-pupil-now-noaccent-is", lang="is", level="easy", cat="B", source="llm",
         q="hvad kostar ad reka hvern grunnskolanemanda a ari nuna",
         parts=[P("THJ05636B", {"Mánuður": "2026M10"})],
         answer=lambda v: [v[0]], also=lambda v: [[v[0] * 1000]], unit="ISK per pupil per year (table unit)",
         how="núna = newest month of the estimate table THJ05636B, October 2026 (THJ05636A has actual annual costs "
             "only up to 2025); if the table is in thousand krónur the full krónur figure is also accepted"),
    dict(id="gdp-growth-forecast-2027-en", lang="en", level="easy", cat="A", source="llm",
         q="What GDP growth does Statistics Iceland forecast for 2027?",
         parts=[P("THJ07000", {"Flokkun": "9", "Ár": "2027"})], unit="%",
         how="economic forecast (Þjóðhagsspá, May 2026), change in GDP in 2027"),
    dict(id="dental-out-of-pocket-2025-is", lang="is", level="easy", cat="A", source="llm",
         q="Hvað greiddu heimilin sjálf mikið fyrir tannlækningar árið 2025?",
         parts=[P("THJ05551", {"Fjármögnun (HF)": "HF.3", "Tegund þjónustu (HC)": "HC.132", "Eining": "Millions ISK",
                               "Ár": "2025"})],
         unit="million ISK", how="current health expenditure financed by households (HF.3), dental outpatient care"),
    dict(id="housing-investment-last-quarter-is", lang="is", level="easy", cat="A", source="llm",
         q="Hvað jókst fjárfesting í íbúðarhúsnæði mikið á síðasta ársfjórðungi miðað við sama tíma árið áður?",
         parts=[P("THJ03111", {"Mælikvarði": "1", "Skipting": "2", "Ársfjórðungur": "2026Q2"})], unit="%",
         how="síðasta ársfjórðungi = newest quarter 2026Q2; change from the same quarter a year earlier, residential "
             "buildings"),
    dict(id="renters-share-2022-en", lang="en", level="easy", cat="A", source="llm",
         q="What share of households in Iceland were renting their home in 2022?",
         parts=[P("LIF03217", {"Staða á húsnæðismarkaði": "6", "Búsetusvæði": "2", "Eining": "0", "Ár": "18"})],
         unit="%", how="all tenants (Leigjendur, alls), whole country, 2022 (EU-SILC)"),
    dict(id="negative-equity-families-nodate-lowercase-is", lang="is", level="easy", cat="A", source="llm",
         q="hvað eru margar fjölskyldur með neikvætt eigið fé í fasteigninni sinni",
         parts=[P("THJ09008", {"Ár": "2025", "Fjöldi fjölskyldna með neikvætt eigið fé í fasteign": "0"})],
         unit="families", how="no period: newest year, 2025 (tax returns)"),
    dict(id="household-heat-use-2024-en", lang="en", level="easy", cat="B", source="llm",
         q="How much heat did Icelandic households use in 2024?",
         parts=[P("UMH40220", {"Atvinnugrein": "HH", "Tafla (flæði)": "Table_B", "Ár": "2024"}),
                P("UMH40220", {"Atvinnugrein": "HH", "Tafla (flæði)": "Table_B.2", "Ár": "2024"}, alt=True)],
         answer=lambda v: [v[0]], also=lambda v: [[v[1]]], unit="TJ (energy accounts unit)",
         how="households, use of heat in total (final use also accepted)"),

    # ===================== medium: a calculation in one table =====================
    dict(id="foreign-net-migration-h1-2026-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað fluttu margir erlendir ríkisborgarar til landsins umfram brottflutta á fyrri helmingi ársins 2026?",
         parts=[P("MAN10002", {"Ársfjórðungur": ["2026Q1", "2026Q2"], "Atburður": "2", "Landsvæði": "IS",
                               "Kyn og ríkisfang": "4"})],
         unit="people", how="sum of 2026Q1 and 2026Q2, net migration of foreign citizens, whole country (rounded "
                            "quarterly figures)"),
    dict(id="capital-to-south-net-2025-lowercase-en", lang="en", level="medium", cat="A", source="llm",
         q="how many more people moved from the capital area to south iceland than the other way round in 2025",
         parts=[P("MAN01002", {"Kyn": "0", "Ár": "2025", "Brottflutningssvæði": "1", "Aldur": "-1",
                               "Aðflutningssvæði": "8"}),
                P("MAN01002", {"Kyn": "0", "Ár": "2025", "Brottflutningssvæði": "8", "Aldur": "-1",
                               "Aðflutningssvæði": "1"})],
         answer=lambda v: [v[0] - v[1]], unit="people",
         how="moves from the capital region to Suðurland minus moves from Suðurland to the capital region, 2025"),
    dict(id="polish-citizenship-2021-2025-is", lang="is", level="medium", cat="A", source="llm",
         q="Hversu margir Pólverjar fengu íslenskan ríkisborgararétt samtals á árunum 2021 til 2025?",
         parts=[P("MAN04119", {"Fyrra ríkisfang": "PL", "Aldur": "1", "Ár": [str(y) for y in range(2021, 2026)],
                               "Kyn": "0"})],
         unit="people", how="former citizenship Poland, sum of the single years 2021-2025"),
    dict(id="long-term-unemployed-change-2026q2-en", lang="en", level="medium", cat="A", source="llm",
         q="How did the number of people unemployed for a year or longer change between the second quarter of 2025 and "
           "the second quarter of 2026?",
         parts=[P("VIN01206", {"Ársfjórðungur": "2026Q2", "Lengd atvinnuleitar": "6", "Eining": "0"}),
                P("VIN01206", {"Ársfjórðungur": "2025Q2", "Lengd atvinnuleitar": "6", "Eining": "0"})],
         answer=lambda v: [v[0] - v[1]], unit="people", how="2026Q2 minus 2025Q2, job search of a year or more"),
    dict(id="real-wages-12m-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað jókst kaupmáttur launa mikið frá ágúst 2025 til ágúst 2026?",
         parts=[P("LAU04800", {"Mánuður": "2026M08"}), P("LAU04800", {"Mánuður": "2025M08"})],
         answer=lambda v: [(v[0] / v[1] - 1) * 100], unit="%",
         how="wage index at constant prices (LAU04800): August 2026 / August 2025 - 1 (1 Nov: the release ends in August,"
             " so the months are named rather than 'the last 12 months')"),
    dict(id="imported-food-prices-12m-abbrev-en", lang="en", level="medium", cat="A", source="llm",
         q="how much more expensive did imported food and drink get in the 12 months to sept 2026?",
         parts=[P("VIS01101", {"Mánuður": "2026M09", "Liður": "index", "Útgjaldaflokkur": "2aii"}),
                P("VIS01101", {"Mánuður": "2025M09", "Liður": "index", "Útgjaldaflokkur": "2aii"})],
         answer=lambda v: [(v[0] / v[1] - 1) * 100], unit="%",
         how="CPI by origin, '2b-i Innfluttur matur og drykkir' (code 2aii): September 2026 / September 2025 - 1"),
    dict(id="plumbing-cost-index-12m-abbrev-is", lang="is", level="medium", cat="A", source="llm",
         q="hvað hækkaði byggingarvísitalan fyrir pípulagnir frá sept 2025 til sept 2026",
         parts=[P("VIS13302", {"Mánuður": "2026M09", "Liður": "index", "Iðngrein": "Plumb"}),
                P("VIS13302", {"Mánuður": "2025M09", "Liður": "index", "Iðngrein": "Plumb"})],
         answer=lambda v: [(v[0] / v[1] - 1) * 100], unit="%",
         how="building cost index, trade Pípulögn: September 2026 / September 2025 - 1"),
    dict(id="accommodation-turnover-last-year-is", lang="is", level="medium", cat="A", source="llm",
         q="Hver var velta gististaða á Íslandi árið sem leið?",
         parts=[P("SAM08050", {"Ár": "2025", "Tímabil": ["0", "1", "2", "3", "4", "5"],
                               "Velta/Breyting milli ára": "Velta_mkr", "Atvinnugrein (ÍSAT2008)": "I55"})],
         unit="million ISK", how="árið sem leið = 2025; sum of the six two-month VAT periods, accommodation "
                                 "(551-553), without VAT"),
    dict(id="kef-aircraft-movements-2024-lowercase-is", lang="is", level="medium", cat="A", source="llm",
         q="hvað voru margar flughreyfingar á keflavíkurflugvelli árið 2024",
         parts=[P("SAM08048", {"Mánuður": months(2024), "Flughreyfingar": "Flughreyfingar"})], unit="movements",
         how="sum of the 12 months of 2024 (the table's 12-month total for December 2024 is the same number)"),
    dict(id="reykjavik-port-cargo-2024-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað fóru mörg tonn af vörum um Reykjavíkurhöfn árið 2024, bæði út og inn?",
         parts=[P("SAM05103", {"Vöruflutningar": ["0", "1"], "Hafnir": "1", "Ár": "2024", "Vörur": "0"})],
         unit="tonnes (table unit)", how="goods from port + goods to port, all goods, Reykjavík"),
    dict(id="cod-share-catch-value-2024-en", lang="en", level="medium", cat="A", source="llm",
         q="What share of the value of Iceland's total catch in 2024 came from cod?",
         parts=[P("SJA09003", {"Fisktegund": "1", "Veiðisvæði": "0", "Löndunartegund": "0", "Eining": "1",
                               "Ár": "2024", "Mánuður": "0"}),
                P("SJA09003", {"Fisktegund": "0", "Veiðisvæði": "0", "Löndunartegund": "0", "Eining": "1",
                               "Ár": "2024", "Mánuður": "0"})],
         answer=lambda v: [v[0] / v[1] * 100], unit="%", how="catch value of cod / catch value of all species, 2024"),
    dict(id="milk-share-agri-output-2025-en", lang="en", level="medium", cat="B", source="llm",
         q="What percentage of Iceland's agricultural output value came from milk in 2025?",
         parts=[P("LAN1101", {"Verð": "Grunnverð", "Mælikvarði": "Verðlag hvers árs", "Vöruflokkur": "12.1",
                              "Ár": "2025"}),
                P("LAN1101", {"Verð": "Grunnverð", "Mælikvarði": "Verðlag hvers árs", "Vöruflokkur": "18",
                              "Ár": "2025"}),
                P("LAN1101", {"Verð": "Grunnverð", "Mælikvarði": "Verðlag hvers árs", "Vöruflokkur": "16",
                              "Ár": "2025"}, alt=True),
                P("LAN1101", {"Verð": "Grunnverð", "Mælikvarði": "Hlutdeild í heildarframleiðsluvirði",
                              "Vöruflokkur": "12.1", "Ár": "2025"}, alt=True)],
         answer=lambda v: [v[0] / v[1] * 100], also=lambda v: [[v[0] / v[2] * 100, v[3]]],
         unit="%", how="milk / output of the agricultural industry (18), basic prices, current prices; milk / "
                       "agricultural output (16) and the table's own share also accepted"),
    dict(id="samfylking-vs-sjalfstaedis-2024-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað fékk Samfylkingin mörgum atkvæðum fleiri en Sjálfstæðisflokkurinn í alþingiskosningunum 2024?",
         parts=[P("KOS02018", {"Atriði": "1", "Flokkur": "S", "Kjördæmi": "AL"}, same=["KOS02121"]),
                P("KOS02018", {"Atriði": "1", "Flokkur": "D", "Kjördæmi": "AL"}, same=["KOS02121"])],
         answer=lambda v: [v[0] - v[1]], unit="votes", how="valid votes, whole country, S minus D"),
    dict(id="cultural-employment-share-2025-en", lang="en", level="medium", cat="A", source="llm",
         q="What percentage of employed people in Iceland worked in cultural jobs in 2025?",
         parts=[P("MEN10102", {"Ár": "2025", "Flokkar": "2", "Kyn": "0", "Ráðningarsamband": "0"}),
                P("MEN10102", {"Ár": "2025", "Flokkar": "0", "Kyn": "0", "Ráðningarsamband": "0"})],
         answer=lambda v: [v[0] / v[1] * 100], unit="%", how="employed in culture / all employed, 2025"),
    dict(id="gov-revenue-growth-2026q2-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað jukust heildartekjur hins opinbera mikið í prósentum frá 2. ársfjórðungi 2025 til 2. ársfjórðungs "
           "2026?",
         parts=[P("THJ05812", {"Aðili": "0", "Skipting": "0", "Ársfjórðungur": "2026Q2"}),
                P("THJ05812", {"Aðili": "0", "Skipting": "0", "Ársfjórðungur": "2025Q2"})],
         answer=lambda v: [(v[0] / v[1] - 1) * 100], unit="%",
         how="general government total revenue, 2026Q2 / 2025Q2 - 1"),
    dict(id="education-share-gov-spending-2025-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvert var hlutfall menntamála af heildarútgjöldum hins opinbera árið 2025?",
         parts=[P("THJ05142", {"Eining": "Million ISK", "Skipting": "09", "Ár": "2025"}),
                P("THJ05142", {"Eining": "Million ISK", "Skipting": "Total", "Ár": "2025"}),
                P("THJ05142", {"Eining": "Percentage breakdown", "Skipting": "09", "Ár": "2025"}, alt=True)],
         answer=lambda v: [v[0] / v[1] * 100], also=lambda v: [[v[2]]], unit="%",
         how="general government expenditure on education (09) / total expenditure, 2025 (the table's own share is "
             "the same number)"),
    dict(id="fish-exports-china-last-year-lowercase-is", lang="is", level="medium", cat="A", source="llm",
         q="hvað fluttum við út mikið af fiski til kína í fyrra",
         parts=[P("UTA06107", {"SITC": "03", "Land": "CN", "Mánuður": months(2025), "Flæði": "0"})],
         unit="million ISK (table unit)",
         how="í fyrra = 2025; SITC 03 fish and fish products, exports fob to China, sum of the months"),
    dict(id="marine-exports-jan-sep-2026-change-en", lang="en", level="medium", cat="A", source="llm",
         q="By what percentage did exports of marine products change in January to September 2026 compared with the "
           "same months of 2025?",
         parts=[P("UTA06001", {"Vara": "XFRETTVx1", "Mánuður": months(2026, 1, 9)}),
                P("UTA06001", {"Vara": "XFRETTVx1", "Mánuður": months(2025, 1, 9)})],
         answer=lambda v: [(v[0] / v[1] - 1) * 100], unit="%",
         how="exported marine products, sum of January-September 2026 / same sum 2025 - 1"),
    dict(id="sox-emissions-2024-lowercase-is", lang="is", level="medium", cat="A", source="llm",
         q="hver var heildarlosun brennisteinsoxíða (SOx) á íslandi 2024",
         parts=[P("UMH31220", {"Liður": [s for s in SOURCES if s != "VEIDI"], "Ár": "2024", "Efni": "SOX"})],
         unit="tonnes (table unit)",
         how="sum over the 12 sources with a number (the table has no total; fisheries and food, VEIDI, has none)"),
    dict(id="pension-per-recipient-nodate-lowercase-is", lang="is", level="medium", cat="A", source="llm",
         q="hvað fá lífeyrisþegar að meðaltali mikið greitt úr lífeyrissjóðum á mánuði",
         parts=[P("TEK02014", {"Mánuður": "2026M08", "Kyn": "0", "Búseta": "0",
                               "Tegundir staðgreiðsluskyldra greiðslna": "4", "Eining": "0"}),
                P("TEK02014", {"Mánuður": "2026M08", "Kyn": "0", "Búseta": "0",
                               "Tegundir staðgreiðsluskyldra greiðslna": "4", "Eining": "1"})],
         answer=lambda v: [v[0] * 1e6 / v[1]], unit="ISK per recipient per month",
         how="no period: newest month, August 2026; pension payments (Lífeyrisgreiðslur, million ISK) x 1e6 / number "
             "of recipients"),

    # ===================== hard: numbers from two tables =====================
    dict(id="new-citizens-per-1000-2023-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað fengu margir íslenskan ríkisborgararétt á hverja 1.000 íbúa árið 2023?",
         parts=[P("MAN04119", {"Fyrra ríkisfang": "01", "Aldur": "1", "Ár": "2023", "Kyn": "0"}),
                P("MAN00101", pop(2023)),
                P("MAN00101", pop(2024), alt=True)],
         answer=lambda v: [v[0] / v[1] * 1000], also=lambda v: [[v[0] / v[2] * 1000]], unit="per 1,000 inhabitants",
         how="citizenships granted 2023 (MAN04119) / population 1 January 2023 x 1,000 (1 January 2024 also accepted)"),
    dict(id="rd-spending-per-inhabitant-2023-en", lang="en", level="hard", cat="B", source="llm",
         q="How much did Iceland spend on research and development per inhabitant in 2023?",
         parts=[P("FYR05101", {"Ár": "2023", "Eining": "0", "Niðurstöður": "0"}),
                P("MAN00101", pop(2023)),
                P("MAN00101", pop(2024), alt=True)],
         answer=lambda v: [v[0] * 1e6 / v[1]], also=lambda v: [[v[0] * 1e6 / v[2]]], unit="ISK per inhabitant",
         how="R&D expenditure, all sectors, million ISK (FYR05101) / population 1 January 2023 (1 January 2024 also "
             "accepted)"),
    dict(id="museum-visits-per-inhabitant-2023-noaccent-is", lang="is", level="hard", cat="B", source="llm",
         q="hvad foru margir a sofn a hvern ibua arid 2023",
         parts=[P("MEN07002", {"Tegund safns": "0", "Ár": "2023", "Skráning": "0", "Skipting": "0",
                               "Landshluti": "IS"}),
                P("MAN00101", pop(2023)),
                P("MAN00101", pop(2024), alt=True)],
         answer=lambda v: [v[0] / v[1]], also=lambda v: [[v[0] / v[2]]], unit="visits per inhabitant",
         how="visits to museums, centres, exhibitions and gardens 2023 (MEN07002) / population 1 January 2023 "
             "(1 January 2024 also accepted)"),
    dict(id="media-revenue-per-inhabitant-2022-en", lang="en", level="hard", cat="B", source="llm",
         q="What was media revenue per inhabitant in Iceland in 2022?",
         parts=[P("MEN10001", {"Fjölmiðlar": "Mass media, total",
                               "Tekjur": "Media revenue in million ISK at current prices", "Ár": "2022"}),
                P("MAN00101", pop(2022)),
                P("MAN00101", pop(2023), alt=True)],
         answer=lambda v: [v[0] * 1e6 / v[1]], also=lambda v: [[v[0] * 1e6 / v[2]]], unit="ISK per inhabitant",
         how="media revenue, all mass media, million ISK (MEN10001) / population 1 January 2022 (1 January 2023 also "
             "accepted)"),
    dict(id="spending-per-foreign-visitor-2024-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað eyddi hver erlendur ferðamaður að meðaltali miklu á Íslandi árið 2024?",
         parts=[P("SAM08000", {"Ár": "2024", "Atvinnugrein": "0", "Ferðamenn": "2"}),
                P("SAM08005", {"Ár": "2024", "Fjöldi": "0", "Tegund": "0", "Ferðamenn": "0"}),
                P("SAM08000", {"Ár": "2024", "Atvinnugrein": "0", "Ferðamenn": "0"}, alt=True),
                P("SAM08005", {"Ár": "2024", "Fjöldi": "0", "Tegund": "0", "Ferðamenn": "2"}, alt=True)],
         answer=lambda v: [v[0] * 1e6 / v[1]], also=lambda v: [[v[2] * 1e6 / v[3]]], unit="ISK per visitor",
         how="inbound tourism expenditure, all visitors, million ISK (SAM08000) / trips of foreign visitors, all "
             "(SAM08005); overnight visitors only (both tables) also accepted"),
    dict(id="public-order-spending-per-inhabitant-2022-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað varði hið opinbera miklu til löggæslu og öryggismála á hvern íbúa árið 2022?",
         parts=[P("THJ05142", {"Eining": "Million ISK", "Skipting": "03", "Ár": "2022"}),
                P("MAN00101", pop(2022)),
                P("MAN00101", pop(2023), alt=True)],
         answer=lambda v: [v[0] * 1e6 / v[1]], also=lambda v: [[v[0] * 1e6 / v[2]]], unit="ISK per inhabitant",
         how="general government expenditure on public order and safety (03), million ISK (THJ05142) / population "
             "1 January 2022 (1 January 2023 also accepted)"),
    dict(id="household-cars-per-1000-2022-lowercase-en", lang="en", level="hard", cat="B", source="llm",
         q="how many cars did households own per 1000 people in 2022",
         parts=[P("SAM30111", {"Ár": "2022", "Atvinnugrein": "HEIM", "Orkugjafi": "SAMTALS", "Mæling": "FJOLDI"}),
                P("MAN00101", pop(2022)),
                P("MAN00101", pop(2023), alt=True)],
         answer=lambda v: [v[0] / v[1] * 1000], also=lambda v: [[v[0] / v[2] * 1000]],
         unit="vehicles per 1,000 inhabitants",
         how="vehicles of households, all energy sources, 2022 (SAM30111) / population 1 January 2022 x 1,000 "
             "(1 January 2023 also accepted)"),
    dict(id="class-size-and-foreign-tongue-pupils-2025-en", lang="en", level="hard", cat="A", source="llm",
         q="In 2025, how many pupils were there on average per class in Icelandic compulsory schools, and how many "
           "compulsory school pupils had a mother tongue other than Icelandic?",
         parts=[P("SKO02106", {"Bekkur": "0", "Ár": "2025", "Eining": "2"}),
                P("SKO02103", {"Tungumál": "Alls", "Ár": "2025", "Landsvæði": "0"})],
         answer=lambda v: [v[0], v[1]], unit="pupils",
         how="both numbers: pupils per class, all grades (SKO02106); pupils with another mother tongue, all "
             "languages, whole country (SKO02103)"),
    dict(id="inhabitants-per-mp-nodate-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað eru margir landsmenn á hvern alþingismann?",
         parts=[P("MAN00101", pop(2026)),
                P("KOS02121", {"Atriði": "2", "Flokkur": "0", "Ár": "18"}, same=["KOS02018", "KOS02102"]),
                P("MAN00101", pop(2025), alt=True)],
         answer=lambda v: [v[0] / v[1]], also=lambda v: [[v[2] / v[1]]], unit="inhabitants per member of parliament",
         how="no period: population 1 January 2026 (newest) / seats elected in 2024 (63); 1 January 2025 also "
             "accepted"),
    dict(id="hotel-rooms-per-1000-2023-01-en", lang="en", level="hard", cat="A", source="llm",
         q="How many hotel rooms were there per 1,000 inhabitants in Iceland in January 2023?",
         parts=[P("SAM01202", {"Mánuður": "2023M01", "Fjöldi": "1", "Landshluti": "0"}),
                P("MAN00101", pop(2023))],
         answer=lambda v: [v[0] / v[1] * 1000], unit="rooms per 1,000 inhabitants",
         how="hotel rooms, whole country, January 2023 (SAM01202) / population 1 January 2023 x 1,000"),
    dict(id="catch-per-inhabitant-2022-lowercase-is", lang="is", level="hard", cat="B", source="llm",
         q="hvað veiddu íslensk skip mörg tonn á hvern íbúa árið 2022",
         parts=[P("SJA09003", {"Fisktegund": "0", "Veiðisvæði": "0", "Löndunartegund": "0", "Eining": "0",
                               "Ár": "2022", "Mánuður": "0"}),
                P("MAN00101", pop(2022)),
                P("MAN00101", pop(2023), alt=True)],
         answer=lambda v: [v[0] / v[1]], also=lambda v: [[v[0] / v[2]]], unit="tonnes per inhabitant",
         how="total catch, all species, areas and landing types, 2022 (SJA09003) / population 1 January 2022 "
             "(1 January 2023 also accepted)"),
    dict(id="south-students-and-matriculations-is", lang="is", level="hard", cat="A", source="llm",
         q="Hvað voru margir háskólanemar með lögheimili á Suðurlandi haustið 2025, og hve margir af Suðurlandi luku "
           "stúdentsprófi skólaárið 2024-2025?",
         parts=[P("SKO04108", {"Ár": "2025", "Kyn": "0", "Aldur": STUDENT_AGES, "Eining": "0",
                               "Lögheimili nemenda": "9"}),
                P("SKO03201", {"Landsvæði": "9", "Aldursflokkur": "0", "Ár": "2024-2025", "Kyn": "0"})],
         answer=lambda v: [v[0], v[1]], unit="students",
         how="both numbers: tertiary students with domicile in Suðurland 2025, sum of all ages (SKO04108 has no age "
             "total); matriculation graduates from Suðurland 2024-2025 (SKO03201)"),
    dict(id="goods-services-balance-2026-06-lowercase-is", lang="is", level="hard", cat="A", source="llm",
         q="hver var vöru og þjónustujöfnuðurinn í júní 2026",
         parts=[P("UTA06001", {"Vara": "M_X_00", "Mánuður": "2026M06"}),
                P("UTA04008", {"Þjónustuviðskipti": "2", "Flokkur": "0", "Mánuður": "2026M06"})],
         answer=lambda v: [v[0] + v[1]], unit="million ISK",
         how="balance of goods trade, fob/cif (UTA06001) + balance of trade in services (UTA04008), June 2026; "
             "negative = deficit"),
    dict(id="alcohol-spending-per-inhabitant-2021-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað eyddi hver Íslendingur að meðaltali miklu í áfengi árið 2021?",
         parts=[P("THJ02103", {"Mælikvarði": "0", "Skipting": "5", "Ár": "2021"}),
                P("MAN00101", pop(2021)),
                P("MAN00101", pop(2022), alt=True)],
         answer=lambda v: [v[0] * 1e6 / v[1]], also=lambda v: [[v[0] * 1e6 / v[2]]], unit="ISK per inhabitant",
         how="household final consumption of alcohol (02.1), current prices, million ISK (THJ02103) / population "
             "1 January 2021 (1 January 2022 also accepted)"),
    dict(id="other-accommodation-nights-per-visitor-2024-en", lang="en", level="hard", cat="A", source="llm",
         q="In 2024, how many nights did foreign tourists spend in non-hotel accommodation (campsites, holiday homes "
           "and the like) per foreign overnight visitor?",
         parts=[P("SAM01502", {"Þjóðerni": "Foreigners", "Ár": "2024", "Landshluti": "IS", "Mánuður": "0"}),
                P("SAM08005", {"Ár": "2024", "Fjöldi": "0", "Tegund": "0", "Ferðamenn": "2"})],
         answer=lambda v: [v[0] / v[1]], unit="nights per visitor",
         how="overnight stays of foreigners in other types of accommodation 2024 (SAM01502) / trips of foreign "
             "overnight visitors 2024 (SAM08005)"),
]

NODATA3 = [  # category, id, language, source, question, where the data is (C) or why not (D)
    ("C", "temperature-reykjavik-2025-07-lowercase-is", "is", "llm", "hver var meðalhitinn í reykjavík í júlí 2025",
     "Veðurstofa Íslands; Statistics Iceland's temperature series (SOG09002) ends in 2017"),
    ("C", "earthquakes-reykjanes-2024-en", "en", "llm",
     "How many earthquakes were recorded on the Reykjanes peninsula in 2024?", "Veðurstofa Íslands (Met Office)"),
    ("C", "icelandair-profit-2025-en", "en", "llm", "What was Icelandair's net profit in 2025?",
     "the company's own accounts; Statistics Iceland publishes industry totals, not single companies"),
    ("C", "nursing-home-waitlist-is", "is", "llm", "Hvað bíða margir eftir hjúkrunarrými á höfuðborgarsvæðinu núna?",
     "Embætti landlæknis; the nursing home tables here (HEI03xxx) end in 2010 and have no waiting lists"),
    ("C", "flats-sold-hafnarfjordur-2026-09-is", "is", "llm",
     "Hvað seldust margar íbúðir í Hafnarfirði í september 2026?",
     "HMS (Húsnæðis- og mannvirkjastofnun) publishes purchase contracts"),
    ("C", "petrol-price-n1-today-lowercase-en", "en", "llm", "whats the price of a litre of petrol at N1 today",
     "a company's price; the CPI has a fuel subindex but no price per litre at a station"),
    ("C", "er-visits-landspitali-2025-en", "en", "llm",
     "How many patients came to the emergency department at Landspítali last year?",
     "Landspítali / Embætti landlæknis"),
    ("D", "house-prices-next-year-en", "en", "llm", "Will house prices in Reykjavík go up next year?",
     "future and opinion; the economic forecast (THJ07000) has no house price forecast"),
    ("D", "tourists-2030-is", "is", "llm", "Hvað verða margir erlendir ferðamenn á Íslandi árið 2030?",
     "future; no forecast of tourist numbers is published (THJ07000 forecasts exports, not visitors)"),
    ("D", "next-election-winner-lowercase-is", "is", "llm", "hvaða flokkur vinnur næstu alþingiskosningar",
     "future and opinion"),
    ("D", "vestmannaeyjar-1650-is", "is", "llm", "Hvað bjuggu margir í Vestmannaeyjum árið 1650?",
     "before records: population figures start with the 1703 census"),
    ("D", "ceo-tax-landsvirkjun-is", "is", "llm", "Hvað greiddi forstjóri Landsvirkjunar mikinn tekjuskatt í fyrra?",
     "private: a named person's taxes are not published by Statistics Iceland"),
]


# Verified later: the answer read exactly this part's number from another table (its rows hold the same value, or
# values summing to it). The organizers' key accepts every table holding the same number; ours does too. Empty: dev3
# is new, nothing verified yet.
# Verified 1 Nov 02:00 (first run, keyfix check on the saved answers): the answer read exactly this part's number
# from another table. Like the organizers' key, ours accepts every table holding the same number. Not added:
# foreign-net-migration (MAN10003 lacks the exact 1,150) and catch-per-inhabitant's catch (SJA09005).
KEY_FIXES = {
    "egilsstadir-midyear-pop-2025-is": {"MAN030106": ["MAN030104"]},
    "cod-share-catch-value-2024-en": {"SJA09003": ["SJA09001"]},
    "gov-revenue-growth-2026q2-is": {"THJ05812": ["THJ05811"]},
    "new-citizens-per-1000-2023-is": {"MAN00101": ["MAN00000"]},
    "media-revenue-per-inhabitant-2022-en": {"MAN00101": ["MAN00000"]},
    "public-order-spending-per-inhabitant-2022-is": {"MAN00101": ["MAN00000"]},
    "household-cars-per-1000-2022-lowercase-en": {"MAN00101": ["MAN00000"]},
    "inhabitants-per-mp-nodate-is": {"MAN00101": ["MAN00000"], "KOS02121": ["KOS02054"]},
    "catch-per-inhabitant-2022-lowercase-is": {"MAN00101": ["MAN00000"]},
    # 1 Nov 11:00, same check on the confirmation round (answers the whole-units fix and 'árið sem leið' made right)
    "accommodation-turnover-last-year-is": {"SAM08050": ["FYR04111"]},
    "rd-spending-per-inhabitant-2023-en": {"MAN00101": ["MAN00000"]},
    "alcohol-spending-per-inhabitant-2021-is": {"MAN00101": ["MAN00000"]},
}


def main():
    rows = []
    for s in SPECS3:
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
        rows.append({"id": s["id"], "language": s["lang"], "level": s["level"], "question": s["q"], "answers": answers,
                     "how": s.get("how", ""), "parts": parts, "category": s["cat"], "source": s["source"]})
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(OUT_NODATA, "w", encoding="utf-8") as f:
        for cat, qid, lang, source, q, note in NODATA3:
            f.write(json.dumps({"id": qid, "language": lang, "category": cat, "source": source, "question": q,
                                "expect": "decline", "note": note}, ensure_ascii=False) + "\n")
    for r in rows:
        print(f'{r["category"]} {r["level"][:4]:4} {r["id"]:52} {" + ".join(str(a["value"]) for a in r["answers"]):28} '
              f'{"/".join(p["table"] for p in r["parts"])}')
    print(f"{len(rows)} answerable (A/B) -> {OUT.name}; {len(NODATA3)} no-data (C/D) -> {OUT_NODATA.name}")


if __name__ == "__main__":
    main()

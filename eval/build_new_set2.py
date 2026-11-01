"""Our second frozen question set, written 31 Oct 2026 evening before any fix aimed at set 1's misses. FROZEN: never change the pipeline because of these questions; run them rarely, to report.

    GAGNVIST_DATA=<release>/data .venv/bin/python eval/build_new_set2.py

45 answerable questions (A/B) over tables that no other question set uses (eval/questions*.jsonl, eval/heldout,
the organizers' 60, set 1), except MAN00101 as the population base of four per-inhabitant questions (1 January 2024,
a year no other question asks for). Written by an LLM from the catalog alone, without looking at the pipeline.
Same conventions as eval/build_new_set.py (P, answer, also, unit, how); every expected number is read from the
organizers' data release (Icelandic edition) by main(): nobody types a number in.
    -> eval/questions_new2.jsonl (and eval/questions_new2_nodata.jsonl when NODATA2 is not empty)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_new_set import ROOT, P, ages, labels, read, table  # noqa: E402,F401

OUT, OUT_NODATA = ROOT / "eval/questions_new2.jsonl", ROOT / "eval/questions_new2_nodata.jsonl"


def months(year, first=1, last=12):
    return [f"{year}M{m:02d}" for m in range(first, last + 1)]


WEIGHTS = ["L500", "U500L1000", "U1000L1200", "U1200L1400", "U1400L1600", "U1600L2000", "U2000L3000",
           "U3000L4000", "U4000L5000", "U6000"]  # SAM30120 has no total over weight classes
POP_2024 = {"Kyn": "Total", "Aldur": "-1", "Ár": "2024"}  # MAN00101, 1 January 2024
POP_2025 = {"Kyn": "Total", "Aldur": "-1", "Ár": "2025"}

# id, language, level, category, source, question, parts, answer(values) -> [numbers], unit, how, also (alternatives)
SPECS2 = [
    # ===================== easy: one cell =====================
    dict(id="unemployed-count-2025-noaccent-is", lang="is", level="easy", cat="A", source="llm",
         q="hve margir voru atvinnulausir ad medaltali a islandi arid 2025",
         parts=[P("VIN00921", {"Ár": "2025", "Landsvæði": "0", "Kyn": "0", "Aldur": "0", "Eining": "1"})],
         unit="people", how="labour force survey estimate, 16-74, annual 2025"),
    dict(id="hotel-occupancy-capital-last-year-en", lang="en", level="easy", cat="A", source="llm",
         q="What was the average room occupancy rate of hotells in the capital region last year?",
         parts=[P("SAM01104", {"Mánuður": "0", "Landshluti": "2", "Ár": "2025", "Eining": "0"})],
         unit="%", how="last year = 2025, all months, room occupancy"),
    dict(id="aquaculture-total-2025-is", lang="is", level="easy", cat="A", source="llm",
         q="Hver var heildarframleiðsla í fiskeldi á Íslandi árið 2025?",
         parts=[P("SJA07001", {"Tegund": "Total", "Ár": "2025"})], unit="tonnes"),
    dict(id="road-deaths-2023-is", lang="is", level="easy", cat="A", source="llm",
         q="Hversu margir létust í umferðarslysum á Íslandi árið 2023?",
         parts=[P("SAM03201", {"Umferðaslys": "5", "Ár": "2023"})], unit="deaths"),
    dict(id="upper-secondary-students-2025-is", lang="is", level="easy", cat="B", source="llm",
         q="Hvað voru margir nemendur í framhaldsskólum haustið 2025?",
         parts=[P("SKO00000", {"Skólastig": "3", "Ár": "2025", "Kyn": "Alls"}),
                P("SKO00000", {"Skólastig": ["3", "4"], "Ár": "2025", "Kyn": "Alls"}, alt=True)],
         answer=lambda v: [v[0]], also=lambda v: [[v[1]]], unit="students",
         how="upper secondary level (Framhaldsskólastig); with post-secondary non-tertiary (Viðbótarstig) also accepted"),
    dict(id="life-expectancy-women-2025-en", lang="en", level="easy", cat="A", source="llm",
         q="What was life expectancy at birth for women in Iceland in 2025?",
         parts=[P("MAN05401", {"Kyn": "1", "Aldur": "0", "Skipting": "0", "Ár": "28"})], unit="years"),
    dict(id="women-elected-althingi-2024-is", lang="is", level="easy", cat="A", source="llm",
         q="Hvað náðu margar konur kjöri á Alþingi í kosningunum 2024?",
         parts=[P("KOS02055", {"Eining": "0", "Ár": "19", "Frambjóðendur/kjörnir þingm": "1", "Kyn": "2"})],
         unit="members of parliament"),
    dict(id="women-managers-2025-is", lang="is", level="easy", cat="A", source="llm",
         q="Hversu margar konur voru framkvæmdastjórar fyrirtækja á Íslandi árið 2025?",
         parts=[P("FYR06102", {"Stjórnendur": "0", "Stærð fyrirtækis": "0", "Ár": "2025", "Fyrirtæki": "0",
                               "Eining": "0", "Kyn": "2"})], unit="women", how="all enterprises, all sizes"),
    dict(id="median-pay-fulltime-2025-en", lang="en", level="easy", cat="A", source="llm",
         q="What were the median total monthly earnings of full-time employees in Iceland in 2025?",
         parts=[P("VIN02002", {"Ár": "2025", "Launþegahópur": "0", "Starfsstétt": "0", "Kyn": "0",
                               "Laun og vinnutími": "7", "Eining": "2"})],
         answer=lambda v: [v[0]], also=lambda v: [[v[0] * 1000]], unit="ISK per month (table unit)",
         how="median of total earnings (Heildarlaun), full-time, all groups; if the table is in thousand krónur "
             "the full krónur figure is also accepted"),
    dict(id="lumpfish-catch-2024-is", lang="is", level="easy", cat="A", source="llm",
         q="Hvað veiddist mikið af grásleppu árið 2024?",
         parts=[P("SJA09035", {"Fisktegund": "1", "Löndunarsvæði": "0", "Eining": "0", "Ár": "2024", "Mánuður": "0"})],
         unit="tonnes", how="Grásleppa, quantity, all landing areas and months"),
    dict(id="foreign-advertising-2024-is", lang="is", level="easy", cat="A", source="llm",
         q="Hvað greiddu Íslendingar erlendum aðilum mikið fyrir auglýsingar árið 2024?",
         parts=[P("MID00104", {"Skipting": "Million ISK", "Ár": "2024", "Tegund greiðslu": "Total"})],
         unit="million ISK"),
    dict(id="name-jon-thor-nodate-is", lang="is", level="easy", cat="A", source="llm",
         q="hvad heita margir jón þór",
         parts=[P("MAN11102", {"Röð": "0", "Eining": "0"})], unit="people",
         how="no date asked: the table's only date, 1 January 2023 (Jón Þór is the most common male double name)"),
    dict(id="services-exports-last-year-en", lang="en", level="easy", cat="A", source="llm",
         q="How much did Iceland earn from exports of services last year?",
         parts=[P("UTA04006", {"Þjónustuflokkur": "0", "Ár": "2025", "Þjónustuviðskipti": "0"})],
         unit="million ISK", how="last year = 2025, all services"),
    dict(id="mothers-first-child-age-2025-is", lang="is", level="easy", cat="A", source="llm",
         q="Hver var meðalaldur kvenna sem eignuðust sitt fyrsta barn árið 2025?",
         parts=[P("MAN05103", {"Ár": "2025", "Aldur foreldra": "0", "Börn": "1", "Eining": "0"})], unit="years"),
    dict(id="municipal-revenue-2025-is", lang="is", level="easy", cat="A", source="llm",
         q="Hverjar voru heildartekjur sveitarfélaganna árið 2025?",
         parts=[P("THJ05322", {"Skipting": "0", "Ár": "2025"}, same=["THJ05331"])], unit="million ISK",
         how="all municipalities together (local government accounts)"),
    dict(id="household-electricity-2024-en", lang="en", level="easy", cat="B", source="llm",
         q="How much electricity did Icelandic households use in 2024?",
         parts=[P("UMH40210", {"Atvinnugrein": "HH", "Tafla (flæði)": "Table_B", "Ár": "2024"}),
                P("UMH40210", {"Atvinnugrein": "HH", "Tafla (flæði)": "Table_B.2", "Ár": "2024"}, alt=True)],
         answer=lambda v: [v[0]], also=lambda v: [[v[1]]], unit="TJ (energy accounts unit)",
         how="households, use in total (final use also accepted)"),
    dict(id="ppi-latest-is", lang="is", level="easy", cat="A", source="llm",
         q="Hver er ársbreyting vísitölu framleiðsluverðs samkvæmt nýjustu tölum?",
         parts=[P("VIS08000", {"Mánuður": "2026M08", "Liður": "change_A", "Flokkur": "PPI"})], unit="%",
         how="newest month in the table (August 2026), 12-month change"),
    dict(id="ghg-monthly-2026-05-en", lang="en", level="easy", cat="A", source="llm",
         q="What were the greenhouse gas emissions from Iceland's economy in May 2026, according to the preliminary "
           "monthly figures?",
         parts=[P("UMH31109", {"Ár": "2026M05", "Atvinnugrein": "BRIDGING_ITEMS"})], unit="CO2 equivalents (table unit)",
         how="households and industries together"),

    # ===================== medium: a calculation in one table =====================
    dict(id="employed-registers-change-2025-2026-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað fjölgaði starfandi fólki mikið samkvæmt skrám frá ágúst 2025 til ágúst 2026?",
         parts=[P("VIN10001", {"Mánuður": "2026M08", "Kyn": "0", "Aldursflokkar": "Total", "Bakgrunnur": "0",
                               "Lögheimili": "0"}),
                P("VIN10001", {"Mánuður": "2025M08", "Kyn": "0", "Aldursflokkar": "Total", "Bakgrunnur": "0",
                               "Lögheimili": "0"})],
         answer=lambda v: [v[0] - v[1]], unit="people", how="August 2026 minus August 2025, register-based"),
    dict(id="work-from-home-share-latest-en", lang="en", level="medium", cat="A", source="llm",
         q="According to the latest figures, what share of employed people in Iceland work from home at least some "
           "of the time?",
         parts=[P("VIN01501", {"Ársfjórðungur": "2026Q1", "Kyn": "0", "Aldurshópar": "0", "Atvinnustétt": "0",
                               "Starf unnið heima að hluta eða öllu leyti": "4"}),
                P("VIN01501", {"Ársfjórðungur": "2026Q1", "Kyn": "0", "Aldurshópar": "0", "Atvinnustétt": "0",
                               "Starf unnið heima að hluta eða öllu leyti": "0"})],
         answer=lambda v: [v[0] / v[1] * 100], unit="%",
         how="newest quarter (2026 Q1), 16-74: 'Alls - vinna eitthvað heima' / all employed"),
    dict(id="gender-pay-gap-total-earnings-2025-is", lang="is", level="medium", cat="B", source="llm",
         q="Hversu miklu hærri voru meðalheildarlaun fullvinnandi karla en kvenna árið 2025, í prósentum?",
         parts=[P("VIN02002", {"Ár": "2025", "Launþegahópur": "0", "Starfsstétt": "0", "Kyn": "1",
                               "Laun og vinnutími": "7", "Eining": "0"}),
                P("VIN02002", {"Ár": "2025", "Launþegahópur": "0", "Starfsstétt": "0", "Kyn": "2",
                               "Laun og vinnutími": "7", "Eining": "0"})],
         answer=lambda v: [(v[0] - v[1]) / v[1] * 100], also=lambda v: [[(v[0] - v[1]) / v[0] * 100]], unit="%",
         how="mean total earnings, full-time: (men - women) / women; (men - women) / men also accepted"),
    dict(id="building-cost-2024-2026-en", lang="en", level="medium", cat="A", source="llm",
         q="By what percentage did the building cost index rise from September 2024 to September 2026?",
         parts=[P("VIS13000", {"Mánuður": "2026M09", "Liður": "BCI"}),
                P("VIS13000", {"Mánuður": "2024M09", "Liður": "BCI"})],
         answer=lambda v: [(v[0] / v[1] - 1) * 100], unit="%", how="index September 2026 / September 2024 - 1"),
    dict(id="flat-prices-capital-12m-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað hefur verð á íbúðum í fjölbýli á höfuðborgarsvæðinu hækkað mikið síðustu 12 mánuði?",
         parts=[P("VIS01106", {"Mánuður": "2026M09", "Vísitala": "RI_cap_mult"}),
                P("VIS01106", {"Mánuður": "2025M09", "Vísitala": "RI_cap_mult"})],
         answer=lambda v: [(v[0] / v[1] - 1) * 100], unit="%",
         how="newest month in the table (September 2026) vs September 2025, capital region multi-dwellings"),
    dict(id="cruise-share-arrivals-2025-en", lang="en", level="medium", cat="A", source="llm",
         q="What share of international arrivals to Iceland in 2025 came on cruise ships?",
         parts=[P("SAM08006", {"Ferðamáti": "6", "Ár": "2025"}), P("SAM08006", {"Ferðamáti": "0", "Ár": "2025"})],
         answer=lambda v: [v[0] / v[1] * 100], unit="%", how="cruise ships / all arrivals"),
    dict(id="salmon-export-price-per-kg-2025-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað fékkst að meðaltali mikið fyrir hvert kíló af útfluttum eldislaxi árið 2025?",
         parts=[P("SJA04905", {"Fisktegund": "1", "Land": "0", "Ár": "2025", "Eining": "1"}),
                P("SJA04905", {"Fisktegund": "1", "Land": "0", "Ár": "2025", "Eining": "0"})],
         answer=lambda v: [v[0] / v[1]], unit="ISK per kg",
         how="FOB value (thousand ISK) / tonnes = krónur per kg; salmon, all countries"),
    dict(id="electric-cars-2025-typo-en", lang="en", level="medium", cat="A", source="llm",
         q="how many electrc cars were there in iceland in 2025",
         parts=[P("SAM30120", {"Orkugjafi": "RAFMAGN", "Þyngdarflokkur": WEIGHTS, "Ár": "2025"})], unit="vehicles",
         how="battery-electric vehicles (Rafmagn), sum of all weight classes (the table has no total); "
             "plug-in hybrids not included"),
    dict(id="matriculation-women-minus-men-2025-is", lang="is", level="medium", cat="A", source="llm",
         q="Hversu mörgum fleiri konur en karlar luku stúdentsprófi skólaárið 2024-2025?",
         parts=[P("SKO03200", {"Ár": "2024-2025", "Skólar, nemendur": "3"}),
                P("SKO03200", {"Ár": "2024-2025", "Skólar, nemendur": "2"})],
         answer=lambda v: [v[0] - v[1]], unit="graduates", how="women minus men"),
    dict(id="avoidable-deaths-2019-2023-is", lang="is", level="medium", cat="A", source="llm",
         q="Hversu mörg ótímabær dauðsföll voru á Íslandi samtals á árunum 2019 til 2023?",
         parts=[P("MAN05311", {"Eining": "0", "Kyn": "0", "Dánarorsök": "0", "Ár": [str(y) for y in range(2019, 2024)]})],
         unit="deaths", how="sum of the single years 2019-2023, all causes in the table"),
    dict(id="halla-minus-katrin-votes-2024-en", lang="en", level="medium", cat="A", source="llm",
         q="How many more votes did Halla Tómasdóttir get than Katrín Jakobsdóttir in the 2024 presidential election?",
         parts=[P("KOS07110", {"Frambjóðendur": "31", "Gild atkvæði": "0", "Kjördæmi": "0"}),
                P("KOS07110", {"Frambjóðendur": "34", "Gild atkvæði": "0", "Kjördæmi": "0"})],
         answer=lambda v: [v[0] - v[1]], unit="votes", how="valid votes, whole country"),
    dict(id="bankruptcies-monthly-avg-h1-2026-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað urðu að meðaltali mörg fyrirtæki gjaldþrota á mánuði fyrstu sex mánuði ársins 2026?",
         parts=[P("FYR03001", {"Mánuður": months(2026, 1, 6), "Atvinnugreinar": "Alls", "Rekstrarform": "Alls",
                               "Breytur": "Fjöldi gjaldþrota"})],
         answer=lambda v: [v[0] / 6], unit="bankruptcies per month", how="sum of January-June 2026 / 6"),
    dict(id="goods-trade-deficit-jan-aug-2026-en", lang="en", level="medium", cat="B", source="llm",
         q="What was Iceland's trade deficit in goods for January to August 2026?",
         parts=[P("UTA06002", {"Mánuður": months(2026, 1, 8), "Flæði": "4"}),
                P("UTA06002", {"Mánuður": months(2026, 1, 8), "Flæði": "3"}, alt=True)],
         answer=lambda v: [v[0]], also=lambda v: [[v[1]]], unit="million ISK",
         how="sum of the monthly balances, fob-cif (fob-fob also accepted); negative = deficit"),
    dict(id="poultry-meat-last-year-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað var framleitt mikið af alifuglakjöti í fyrra?",
         parts=[P("LAN10205", {"Mánuður": months(2025), "Tegund": "3", "Eining": "0"})], unit="kg",
         how="í fyrra = 2025, sum of the months"),
    dict(id="road-co2-2019-2025-en", lang="en", level="medium", cat="A", source="llm",
         q="By how much did CO2 emissions from road transport in Iceland change between 2019 and 2025, in percent?",
         parts=[P("UMH31119", {"Atvinnugrein": "SAMTALS", "Lofttegund": "CO2", "Ár": "2025"}),
                P("UMH31119", {"Atvinnugrein": "SAMTALS", "Lofttegund": "CO2", "Ár": "2019"})],
         answer=lambda v: [(v[0] / v[1] - 1) * 100], unit="%", how="2025 / 2019 - 1, households and industries"),
    dict(id="municipal-debt-change-2024-2025-is", lang="is", level="medium", cat="A", source="llm",
         q="Hvað jukust skuldir sveitarfélaga um margar milljónir króna milli áranna 2024 og 2025?",
         parts=[P("THJ05381", {"Eining": "0", "Skipting": "5", "Ár": "2025"}),
                P("THJ05381", {"Eining": "0", "Skipting": "5", "Ár": "2024"})],
         answer=lambda v: [v[0] - v[1]], unit="million ISK", how="liabilities (33 Skuldir) 2025 minus 2024"),
    dict(id="married-share-16plus-2026-is", lang="is", level="medium", cat="B", source="llm",
         q="Hversu hátt hlutfall fólks 16 ára og eldri var gift 1. janúar 2026?",
         parts=[P("MAN07209", {"Aldur": "-1", "Hjúskaparstétt": ["3", "4"], "Ár": "2026", "Kyn": "0"}),
                P("MAN07209", {"Aldur": "-1", "Hjúskaparstétt": "0", "Ár": "2026", "Kyn": "0"}),
                P("MAN07209", {"Aldur": "-1", "Hjúskaparstétt": "3", "Ár": "2026", "Kyn": "0"}, alt=True)],
         answer=lambda v: [v[0] / v[1] * 100], also=lambda v: [[v[2] / v[1] * 100]], unit="%",
         how="married (living with spouse + not) / all 16+; married living with spouse only also accepted"),

    # ===================== hard: numbers from two tables =====================
    dict(id="preschool-children-per-staff-2024-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað voru mörg börn á hvern starfsmann í leikskólum landsins árið 2024?",
         parts=[P("SKO01000", {"Leikskóli": "Total", "Ár": "2024", "Kyn/aldur": "Total"}),
                P("SKO01300", {"Ár": "2024", "Starfssvið": "0", "Rekstrarform": "0", "Eining": "0", "Kyn": "0"}),
                P("SKO01300", {"Ár": "2024", "Starfssvið": "0", "Rekstrarform": "0", "Eining": "1", "Kyn": "0"},
                  alt=True)],
         answer=lambda v: [v[0] / v[1]], also=lambda v: [[v[0] / v[2]]], unit="children per staff member",
         how="children in preschools December 2024 (SKO01000) / staff headcount (SKO01300); "
             "per full-time equivalent also accepted"),
    dict(id="pupils-per-teacher-2024-en", lang="en", level="hard", cat="A", source="llm",
         q="How many compulsory school pupils were there per teaching staff member in Iceland in 2024?",
         parts=[P("SKO02202", {"Ár": "2024", "Tegund skóla": "0", "Fjöldi": "1"}),
                P("SKO02301", {"Ár": "2024", "Eining": "0", "Landsvæði": "0", "Kennsluréttindi": "0", "Kyn": "0"})],
         answer=lambda v: [v[0] / v[1]], unit="pupils per teacher",
         how="pupils in all compulsory schools (SKO02202) / teaching staff headcount (SKO02301)"),
    dict(id="electric-cars-per-1000-2024-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað voru margir rafbílar á hverja 1.000 íbúa á Íslandi árið 2024?",
         parts=[P("SAM30120", {"Orkugjafi": "RAFMAGN", "Þyngdarflokkur": WEIGHTS, "Ár": "2024"}),
                P("MAN00101", POP_2024),
                P("MAN00101", POP_2025, alt=True)],
         answer=lambda v: [v[0] / v[1] * 1000], also=lambda v: [[v[0] / v[2] * 1000]],
         unit="electric vehicles per 1,000 inhabitants",
         how="battery-electric vehicles 2024, all weight classes (SAM30120) / population 1 January 2024 x 1,000 "
             "(1 January 2025 also accepted)"),
    dict(id="uk-goods-and-services-exports-2025-en", lang="en", level="hard", cat="A", source="llm",
         q="How much did Iceland export to the United Kingdom in 2025, goods and services separately?",
         parts=[P("UTA06003", {"Land": "GB", "Flæði": "0", "Mánuður": months(2025)}),
                P("UTA04007", {"Land": "GB", "Ár": "2025", "Þjónustuviðskipti": "C"})],
         answer=lambda v: [v[0], v[1]], unit="million ISK",
         how="both numbers: goods exports fob, sum of 2025 months (UTA06003); services exports 2025 (UTA04007)"),
    dict(id="university-students-and-graduates-2025-is", lang="is", level="hard", cat="A", source="llm",
         q="Hvað voru margir nemendur á háskólastigi haustið 2025 og hve margir brautskráðust af háskólastigi "
           "skólaárið 2024-2025?",
         parts=[P("SKO00000", {"Skólastig": "5", "Ár": "2025", "Kyn": "Alls"}),
                P("SKO04203", {"Próftegund": "Alls", "Aldursflokkur": "Alls", "Ár": "2024-2025", "Kyn": "Alls"})],
         answer=lambda v: [v[0], v[1]], unit="people",
         how="both numbers: students at tertiary level autumn 2025 (SKO00000); graduations at tertiary and "
             "doctorate level 2024-2025 (SKO04203)"),
    dict(id="health-spending-per-capita-2024-en", lang="en", level="hard", cat="B", source="llm",
         q="What was total health care expenditure per inhabitant in Iceland in 2024?",
         parts=[P("THJ05532", {"Skipting": "0", "Heildarútgjöld": "10", "Ár": "2024"}),
                P("MAN00101", POP_2024),
                P("MAN00101", POP_2025, alt=True),
                P("THJ05532", {"Skipting": "3", "Heildarútgjöld": "10", "Ár": "2024"}, alt=True)],
         answer=lambda v: [v[0] * 1e6 / v[1]], also=lambda v: [[v[0] * 1e6 / v[2], v[3] * 1000]],
         unit="ISK per inhabitant",
         how="total health expenditure in million ISK (THJ05532) / population 1 January 2024 (1 January 2025 also "
             "accepted; so is the table's own per-capita figure at 2025 prices, in krónur)"),
    dict(id="municipal-tax-per-inhabitant-2024-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað voru skatttekjur sveitarfélaga miklar á hvern íbúa árið 2024?",
         parts=[P("THJ05331", {"Eining": "0", "Skipting": "1", "Ár": "2024"}),
                P("MAN00101", POP_2024),
                P("MAN00101", POP_2025, alt=True)],
         answer=lambda v: [v[0] * 1e6 / v[1]], also=lambda v: [[v[0] * 1e6 / v[2]]], unit="ISK per inhabitant",
         how="tax revenue (11 Skatttekjur, million ISK, THJ05331) / population 1 January 2024 "
             "(1 January 2025 also accepted)"),
    dict(id="salmon-produced-and-exported-2024-en", lang="en", level="hard", cat="A", source="llm",
         q="How many tonnes of farmed salmon did Iceland produce in 2024, and how many tonnes of farmed salmon were "
           "exported that year?",
         parts=[P("SJA07001", {"Tegund": "Salmo salar", "Ár": "2024"}),
                P("SJA04905", {"Fisktegund": "1", "Land": "0", "Ár": "2024", "Eining": "0"})],
         answer=lambda v: [v[0], v[1]], unit="tonnes",
         how="both numbers: production (SJA07001) and exports to all countries (SJA04905)"),
    dict(id="lamb-meat-per-inhabitant-2024-is", lang="is", level="hard", cat="B", source="llm",
         q="Hvað var framleitt mikið lambakjöt á hvern íbúa árið 2024?",
         parts=[P("LAN10205", {"Mánuður": months(2024), "Tegund": "0", "Eining": "0"}),
                P("MAN00101", POP_2024),
                P("MAN00101", POP_2025, alt=True)],
         answer=lambda v: [v[0] / v[1]], also=lambda v: [[v[0] / v[2]]], unit="kg per inhabitant",
         how="lamb meat (Dilkar) 2024, sum of months, kg (LAN10205) / population 1 January 2024 "
             "(1 January 2025 also accepted)"),
    dict(id="census-dwellings-minus-households-2021-en", lang="en", level="hard", cat="A", source="llm",
         q="In the 2021 census, how many more conventional dwellings were there in Iceland than private households?",
         parts=[P("CEN1709", {"Sveitarfélag": "IS", "Atriði": "DW", "Ár": "A2021"}),
                P("CEN1509", {"Sveitarfélag": "IS", "Gerð heimilis": "TOTAL", "Ár": "A2021"})],
         answer=lambda v: [v[0] - v[1]], unit="dwellings",
         how="conventional dwellings (CEN1709) minus private households (CEN1509), whole country, 2021"),
]

NODATA2 = []  # category, id, language, source, question, where the data is (C) or why not (D)


# Verified 31 Oct (evening): the answer read exactly this part's number from another table (its rows hold the same
# value, or values summing to it). The organizers' key accepts every table holding the same number; ours does too.
# 1 Nov 11:00, same check: the two per-inhabitant questions the whole-units fix (made on dev3) got right.
KEY_FIXES = {
    "health-spending-per-capita-2024-en": {"THJ05532": ["THJ05531"], "MAN00101": ["MAN00000"]},
    "municipal-tax-per-inhabitant-2024-is": {"THJ05331": ["THJ05321"], "MAN00101": ["MAN02005"]},
    "bankruptcies-monthly-avg-h1-2026-is": {
        "FYR03001": [
            "FYR03010"
        ]
    },
    "median-pay-fulltime-2025-en": {
        "VIN02002": [
            "VIN02004"
        ]
    },
    "preschool-children-per-staff-2024-is": {
        "SKO01000": [
            "SKO01101"
        ],
        "SKO01300": [
            "SKO01305"
        ]
    },
    "pupils-per-teacher-2024-en": {
        "SKO02301": [
            "SKO02305"
        ]
    },
    "uk-goods-and-services-exports-2025-en": {
        "UTA04007": [
            "UTA04009",
            "UTA04010"
        ]
    },
    "unemployed-count-2025-noaccent-is": {
        "VIN00921": [
            "VIN01002"
        ]
    },
    "women-elected-althingi-2024-is": {
        "KOS02055": [
            "KOS02054"
        ]
    },
    "women-managers-2025-is": {
        "FYR06102": [
            "FYR06101"
        ]
    }
}


def main():
    rows = []
    for s in SPECS2:
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
    if NODATA2:
        with open(OUT_NODATA, "w", encoding="utf-8") as f:
            for cat, qid, lang, source, q, note in NODATA2:
                f.write(json.dumps({"id": qid, "language": lang, "category": cat, "source": source, "question": q,
                                    "expect": "decline", "note": note}, ensure_ascii=False) + "\n")
    for r in rows:
        print(f'{r["category"]} {r["level"][:4]:4} {r["id"]:46} {" + ".join(str(a["value"]) for a in r["answers"]):28} '
              f'{"/".join(p["table"] for p in r["parts"])}')
    print(f"{len(rows)} answerable (A/B) -> {OUT.name}" + (f"; {len(NODATA2)} no-data (C/D) -> {OUT_NODATA.name}"
                                                         if NODATA2 else ""))


if __name__ == "__main__":
    main()

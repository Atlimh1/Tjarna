"""Question -> table -> PX-Web query -> numbers -> answer with citation."""
import contextvars
import datetime
import itertools
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from . import calc, cards, clarify, llm, local_data, px, suggest, translate, verify
from .retrieve import STOPWORDS, agreement, fold, is_english, places_of, rrf
from .verify import group_thousands, tidy_decimals, values_in

log = logging.getLogger(__name__)
PROGRESS = contextvars.ContextVar("pipeline_progress", default=None)


def report(step, **info):
    """Tell a listening client what the pipeline is doing now (the chat shows each step as it happens: the
    table chosen, the numbers read, a table refused); silent when nobody listens, never a reason to fail."""
    listener = PROGRESS.get()
    if listener:
        try:
            listener(step, info)
        except Exception:  # noqa: BLE001
            log.debug("progress listener failed", exc_info=True)

MAX_VALUES_SHOWN = 60  # variable values listed in the prompt before trimming
MAX_ROWS = 200  # data rows passed to the answer prompt
TOTAL_WORDS = {"alls", "samtals", "total", "allir", "allt landið", "bæði kyn", "landið allt", "heild"}

PICK_SYSTEM = (
    "Þú velur töflu úr gagnagrunni Hagstofu Íslands sem svarar spurningu notanda best. Margar töflur líkjast "
    "hver annarri; veldu þá eftir þessum reglum nema spurningin biðji sérstaklega um annað: "
    "(1) aðaltöfluna fyrir landið allt og heildina, ekki sundurliðun eftir svæðum, sveitarfélögum, hverfum, "
    "atvinnugreinum eða hópum, nema spurningin nefni slíkan stað eða hóp; "
    "(2) aðalvísitöluna eða aðalröðina, ekki undirvísitölur, sundurliðanir eða samræmdar (HICP) vísitölur; "
    "ef tvær töflur sýna sömu vísitölu, þá sem nær lengst aftur (hún er á upprunalegum grunni); "
    "(3) gildi á verðlagi hvers árs, ekki á föstu verðlagi eða raunvirði, nema spurt sé um raunbreytingu; "
    "(4) töflu sem nær yfir tímabilið sem spurt er um, og ef ekki er spurt um tímabil, nýlega uppfærða töflu með "
    "nýjustu tölunum, ekki töflu sem hætt er að uppfæra; "
    "(5) töflu með upphæðum eða fjölda (sjá eininguna, t.d. milljónir króna) frekar en vísitölum eða "
    "magnbreytingum, nema spurt sé um vísitölu, breytingu eða vöxt; "
    "(6) mannfjölda 1. janúar (töflur sem heita Mannfjöldi ...) frekar en meðalmannfjölda ársins (Meðalmannfjöldi "
    "...), líka þegar mannfjöldinn er grunnur fyrir tölu á íbúa, nema spurt sé um meðalmannfjölda. "
    'Svaraðu eingöngu með JSON: {"table": <númer>}. Notaðu -1 ef engin tafla á við.'
)
SELECT_SYSTEM = (
    "Þú býrð til fyrirspurn í PX-Web töflu Hagstofu Íslands. Fyrir hverja breytu velur þú "
    "kóða (code) gilda sem þarf til að svara spurningunni. Notaðu eingöngu kóða sem eru á listanum. "
    "Ef spurt er um nýjustu tölur, veldu síðasta tímabilið. Veldu sem fæst gildi, en ef spurningin ber saman "
    "eða kallar á útreikning (breyting milli tímabila, munur, hlutfall af heild, samtala) skaltu velja öll gildi "
    "sem þarf, t.d. bæði árin, eða hlutann og heildina (Alls). "
    + ("'Menn', 'fólk', 'landsmenn' og 'Íslendingar' þýða alla (Alls, bæði kyn) nema spurt sé um karla eða konur. "
       if os.environ.get("MENN_RULE", "1") == "1" else "")  # dev3, 1 Nov: "hvað unnu menn" was read as Karlar
    + 'Svaraðu eingöngu með JSON: {"selections": {"<breytukóði>": ["<gildiskóði>", ...]}}'
)
ANSWER_SYSTEM = (
    "Þú ert aðstoðarmaður sem svarar spurningum um hagtölur Hagstofu Íslands. Svaraðu á sama "
    "tungumáli og spurningin, stutt og skýrt, og notaðu EINGÖNGU tölurnar í gögnunum hér að neðan. "
    "Tilgreindu tímabil og einingu. Ef útreikningur fylgir gögnunum, notaðu niðurstöðu hans óbreytta og "
    "reiknaðu ekkert sjálfur. Ef gögnin koma úr fleiri en einni töflu, segðu úr hvaða töflu og frá hvaða "
    "tímabili hver tala er. Ef gögnin svara ekki spurningunni, segðu það hreinskilnislega. Ekki búa til tölur."
)
# Their test questions: "medium" needs a calculation in one table, "hard" numbers from two tables. The model only
# chooses (which operation, which rows, how to split the question); code does the arithmetic.
DECOMPOSE_SYSTEM = (
    "Þú undirbýrð spurningu fyrir leit í töflum Hagstofu Íslands. Ef spurningin spyr um fleira en eitt sem "
    "þarf gögn úr ólíkum töflum (t.d. mannfjölda OG atvinnuleysi, eða landsframleiðslu, verðbólgu OG "
    "atvinnuleysi), skiptu henni í sjálfstæðar spurningar, eina fyrir hverja töflu (mest fjórar), á sama "
    "tungumáli; endurtaktu stað og tíma í hverri. Ef ein tafla dugar, líka fyrir "
    "samanburð eða útreikning (breyting milli ára, hlutfall af heild, munur á svæðum), skilaðu "
    "spurningunni óbreyttri. En tala á hvern íbúa, á hverja 1.000 íbúa eða á mann þarf tvær töflur þegar "
    "talan sjálf er ekki mannfjöldi (gistinætur, fæðingar, afli o.s.frv.): skiptu þá í spurningu um töluna "
    "og spurningu um mannfjöldann á sama stað og tíma, t.d. 'Hve margar gistinætur voru á Íslandi árið 2025?' "
    "og 'Hve margir bjuggu á Íslandi 1. janúar 2025?'. "
    'Svaraðu eingöngu með JSON: {"questions": ["...", ...]}'
)
# Query rewriting in the same call (31 Oct, eval/experiments/query_rewrite.py). OFF unless QUERY_REWRITE=1.
# Each sub-question also gets a clean version for the search; the later steps keep the user's own wording.
QUERY_REWRITE = os.environ.get("QUERY_REWRITE", "0") == "1"
# tables the picker chooses among (eval/experiments/picker_candidates.py). 12, decided 31 Oct with Gemma 4 and the
# final code: the 60 58 -> 60, set 1 36 -> 37, set 2 36 -> 36, 4.0 -> 4.1 s (the 1989 wage index ranked 7th-12th)
PICK_K = int(os.environ.get("PICK_K", 12))
# When no tried table passes the self-check, decline ("I found no data ...") instead of answering from the closest
# one: 3 of 16 unpublished questions were declined before (eval/decline_eval.py). DECLINE_UNCHECKED=0 switches it off.
DECLINE_UNCHECKED = os.environ.get("DECLINE_UNCHECKED", "1") == "1"
# Table cards (cards.py): search WIDE, drop by code what is certain (year, place, index), richer lines for the
# picker. CARDS=1 switches it on. Off: the picker A/B (eval/experiments/pick_cards.py, 1 Nov) gained nothing beyond
# noise and lost 3-4 parts of the organizers' 60; end to end the 60 60 -> 54, set 2 35 -> 33.
CARDS = os.environ.get("CARDS", "0") == "1"
# The picker also names its second choice, tried right after the first when that one fails the self-check (before
# the search's next rank). PICK_SECOND=1 switches it on. Off: end to end (1 Nov 01:40, Gemma) it tied everywhere,
# the 60 60 -> 60, set 1 37 -> 37, set 2 35 -> 35, real2 33 -> 33, for a longer pick prompt.
PICK_SECOND = os.environ.get("PICK_SECOND", "0") == "1"
# The self-check also reads the table's own note (cards.note, from the release's px header). CHECK_NOTES=1 switches
# it on. Off: no question changed on any set (1 Nov 11:40, Gemma, front door).
CHECK_NOTES = os.environ.get("CHECK_NOTES", "0") == "1"
PICK_JSON_SECOND = ('Svaraðu eingöngu með JSON: {"table": <númer>, "second": <númer næstbestu töflunnar, eða -1>}. '
                    "Notaðu -1 ef engin tafla á við.")
REWRITE_RULES = (
    "Gefðu líka hverri spurningu hreina útgáfu fyrir leitina: stöðluð íslensk stafsetning með broddstöfum, "
    "heil orð í stað skammstafana (rvk -> Reykjavík, mfj -> mannfjöldi) og afstæðar dagsetningar gerðar nákvæmar "
    "miðað við að í dag er {today} (í fyrra -> árið {last_year}). Enskar spurningar haldast á ensku. Breyttu engu "
    "öðru: hvorki efni, stöðum né tölum. "
    'Svaraðu eingöngu með JSON: {{"questions": [{{"question": "...", "clean": "..."}}, ...]}}'
)
CALC_SYSTEM = (
    "Þú ákveður hvort svarið við spurningunni krefjist útreiknings úr röðunum hér að neðan. Reiknaðu EKKI "
    "sjálfur. Veldu eina aðgerð: none (talan stendur í röðunum), difference (a mínus b), percent_change "
    "(breyting frá b til a í %, b er upphafsgildið), share (a sem % af b, a er hlutinn og b heildin), "
    "ratio (a deilt með b), per_1000 (a á hverja 1.000 af b, t.d. fæðingar á hverja 1.000 íbúa), "
    "sum eða average (yfir raðirnar í rows). Veldu none ef spurt er um tvær tölur hlið við hlið (t.d. "
    "mannfjölda OG atvinnuleysi). En hlutfall, prósenta, tala á íbúa eða á hverja 1.000 íbúa er EINN útreikningur "
    "úr tveimur tölum, líka þegar þær koma úr tveimur töflum: 'Hve stór hluti landsmanna býr í Reykjavík?' = share "
    "(a = Reykjavík, b = allt landið); 'fæðingar á hverja 1.000 íbúa' = per_1000 (a = fæðingar, b = mannfjöldi). "
    "Ef útreikningurinn þarf fleiri en eitt skref eða margar raðir, notaðu formula: gefðu hverri tölu stutt nafn "
    "(númer raðar, eða listi raða sem á að leggja saman eða taka meðaltal af) og skrifaðu formúlu með + - * / ( ) "
    "og föllunum sum, mean, median, min, max, cagr(fyrsta, síðasta, fjöldi ára) (meðalvöxtur á ári í %). Dæmi: "
    "'hlutfall landsmanna utan höfuðborgarsvæðisins' = (heild - hofudb) / heild * 100; 'hlutfall íbúa eldri en "
    "67 ára' = sum(eldri) / allir * 100 þar sem eldri er listi aldursraðanna. "
    'Svaraðu eingöngu með JSON: {"operation": "difference", "a": <röð>, "b": <röð>}, '
    '{"operation": "sum", "rows": [<raðir>]}, '
    '{"operation": "formula", "names": {"heild": <röð>, "hofudb": <röð>}, "formula": "(heild - hofudb) / heild * 100"} '
    'eða {"operation": "none"}.'
)
ANSWER_IN = {  # the last line of the answer prompt
    "en": "Answer in English (the question is in English), with English number formatting (394,324). "
          "Translate the Icelandic labels in the data.",
    "is": "Svaraðu á íslensku.",
}
NO_TABLE = {
    "is": "Ég fann engin gögn hjá Hagstofunni sem svara þessu.",
    "en": "I found no data at Statistics Iceland that answers this.",
}
MAX_TABLES = 3  # per sub-question: the picked table, then the next by search rank when one has no numbers for it


class NoData(Exception):
    """No edition of a table gave numbers for the selection (each reason in the message)."""
NO_DATA_SYSTEM = (  # the ablation's bottom rung: the model alone, no table (shows what it makes up)
    "Þú svarar spurningum um hagtölur á Íslandi. Svaraðu stutt, með tölu, tímabili og einingu."
)
# Steps that can be switched off, for the ablation ladder (eval/ladder.py). All on = the normal pipeline.
#   split: decompose into <=2 sub-questions (two tables)    pick: the LLM picks among the top k (off: take #1)
#   calc:  the LLM names an operation, code computes        english: answer in the question's language
#   check: the LLM says whether the rows read answer the sub-question; if not, the next table is tried
STEPS = frozenset({"split", "pick", "calc", "english", "check"})
CHECK_SYSTEM = (
    "Þú athugar hvort gögn úr töflu Hagstofu Íslands svari spurningu. Svaraðu true ef raðirnar sýna það sem spurt "
    "er um: rétta stærð eða mælingu, réttan stað, hóp og tímabil (heildina ef ekki er spurt um hóp), eða tölurnar "
    "sem þarf til að reikna svarið. Svaraðu false ef þær sýna eitthvað annað, t.d. vísitölu þegar spurt er um "
    "fjölda eða upphæð, aðra mælingu (atvinnuþátttöku í stað atvinnuleysis), annan stað eða hóp, eða annað "
    "tímabil en spurt er um. Ef spurningin nefnir ekkert tímabil ('á ári', 'núna', 'hvað eru margir') eru "
    "nýjustu tölur töflunnar rétt svar, líka þótt þær séu nokkurra ára gamlar: svarið segir þá frá hvaða ári þær eru. "
    'Svaraðu eingöngu með JSON: {"answers": true} eða {"answers": false, "why": "<stutt>"}.'
)
# sub-questions per question: up to 4, one per table ("GDP per head, inflation and unemployment in 2025" lost two of
# its three topics at 2, 1 Nov); read in parallel. MAX_PARTS=2 restores the old limit.
MAX_PARTS = int(os.environ.get("MAX_PARTS", 4))
MAX_SPLIT = 12  # sub-questions decompose() keeps; past MAX_PARTS they are named in the answer, never dropped silently
LEFT_OUT = {"is": "Ég svara mest {n} atriðum í einni spurningu. Þessu svaraði ég ekki, spurðu um það sér:",
            "en": "I answer at most {n} things in one question. These were not answered; ask about them separately:"}
MAX_CALC_ROWS = 60  # more rows than this: a listing, not something to compute on
BINARY_OPS = {
    "difference": lambda a, b: a - b,
    "percent_change": lambda a, b: (a - b) / b * 100,
    "share": lambda a, b: a / b * 100,
    "ratio": lambda a, b: a / b,
    "per_1000": lambda a, b: a / b * 1000,
}
LIST_OPS = {"sum": sum, "average": lambda xs: sum(xs) / len(xs)}


def _words(text):
    """Accent-free words to match values with: 4+ letters, or a capitalised name of 3 ("Vik" must find "Vík í
    Mýrdal" in a list of 127 towns; it was trimmed away, so the model could not choose it)."""
    return {fold(w.lower()) for w in re.findall(r"\w+", text)
            if (len(w) >= 4 or (len(w) == 3 and w[0].isupper())) and w.lower() not in STOPWORDS}


def _matches(question_words, value_text):
    """Loose match that tolerates inflection and missing accents: shared 4-letter stem."""
    vt = fold(value_text.lower())
    return any(w[:4] in vt for w in question_words)


def editions(table, lang):
    """The editions of a table to try, the question's language first. An English question reads the English
    edition (English labels; its own codes, sometimes its own numbers: the organizers' English answer key is read
    from it) when the table has one; the Icelandic edition is the fallback, and the other way round."""
    has_en = bool(table.get("variables_en"))
    order = ["en", "is"] if lang == "en" else ["is", "en"]
    return [e for e in order if e == "is" or has_en]


def variables(table, edition="is"):
    return table["variables_en"] if edition == "en" else table["variables_is"]


def describe_variables(table, question, edition="is"):
    """Variables and their values as prompt text, trimming long value lists to what looks relevant."""
    qw = _words(question)
    lines = []
    for v in variables(table, edition):
        pairs = list(zip(v["values"], v["valueTexts"]))
        if len(pairs) > MAX_VALUES_SHOWN:
            keep = set(range(5)) | set(range(len(pairs) - 15, len(pairs)))
            keep |= {i for i, (_, t) in enumerate(pairs) if _matches(qw, t) or t.lower() in TOTAL_WORDS}
            shown = [pairs[i] for i in sorted(keep)][:MAX_VALUES_SHOWN]
            note = f" (sýni {len(shown)} af {len(pairs)})"
        else:
            shown, note = pairs, ""
        flag = " [tími]" if v.get("time") else ""
        vals = "; ".join(f"{c}={t}" for c, t in shown)
        lines.append(f'- {v["code"]} ({v["text"]}){flag}{note}: {vals}')
    return "\n".join(lines)


def default_value(v):
    """Value to use when the model didn't select one for a required variable."""
    if v.get("time"):
        return [v["values"][-1]]
    for code, text in zip(v["values"], v["valueTexts"]):
        if text.strip().lower() in TOTAL_WORDS:
            return [code]
    return [v["values"][0]]


def chosen(v, raw):
    """The real codes the model chose for variable v. It is asked for codes, but it sometimes names the variable
    as the prompt shows it (its label, the English label, the time marker "tími") or a value by its label; run 6
    lost "August 2026" ({"tími": ["2026M08"]}: the latest month was used) and "unemployment" ({"Rate": [...]}: the
    first measure was used). Also the prompt's own line formats: a variable as "code (label)" and a value as
    "code=label" ({"Trú- og lífsskoðunarfélög (Religious)": ["1"]} lost the National Church; "/=Stofnun múslima á
    Íslandi" lost one Muslim association). An unknown variable or value is still dropped."""
    vals = raw.get(v["code"])
    if vals is None:
        names = {v["code"].lower(), v["text"].lower(), f'{v["code"]} ({v["text"]})'.lower()}
        names |= {"tími", "[tími]", "time"} if v.get("time") else set()
        vals = next((val for key, val in raw.items()
                     if str(key).strip().lower() in names
                     or re.sub(r"\s*(\[tími\]|\([^()]*\))\s*$", "", str(key).strip()).lower() in names), None)
    if isinstance(vals, (str, int)):
        vals = [vals]
    by_text = {t.strip().lower(): c for c, t in zip(v["values"], v["valueTexts"])}
    out = []
    for c in vals if isinstance(vals, list) else []:
        c = str(c).strip()
        code = (c if c in v["values"]
                else c.split("=", 1)[0].strip() if "=" in c and c.split("=", 1)[0].strip() in v["values"]
                else by_text.get(c.lower()))
        if code and code not in out:
            out.append(code)
    return out


def assumptions(table, raw, edition="is"):
    """[(variable, value)] that code chose because the question didn't: the latest period, the total. Said in the
    answer ("Assuming: Year 2026, Sex Total"), so a vague question shows what it was answered as."""
    out = []
    for v in variables(table, edition):
        if not chosen(v, raw) and not v.get("elimination"):
            code = default_value(v)[0]
            out.append((v["text"], v["valueTexts"][v["values"].index(code)]))
    return out


ASSUMED = {"is": "Miðað við: {}.", "en": "Assuming: {}."}


def english_labels(parts):
    """In an English answer, the labels of parts read from a table with no English edition (281 tables) in English:
    rows and the "Assuming" line ("Kyn Alls" -> "Sex Total"), after the answer is written. Numbers stay as they are;
    translate.labels keeps a label it cannot translate."""
    for p in parts:
        if p.get("edition") != "is":
            continue
        rows = p.get("rows") or []
        texts = [x for pair in p.get("assumed") or [] for x in pair]
        texts += [x for r in rows for k, v in r.items() if k != "value" for x in (k, v) if isinstance(x, str)]
        if not texts:
            continue
        m = translate.labels(texts)
        p["assumed"] = [(m.get(a, a), m.get(b, b)) for a, b in p.get("assumed") or []]
        p["rows"] = [{(k if k == "value" else m.get(k, k)): (m.get(v, v) if k != "value" and isinstance(v, str) else v)
                      for k, v in r.items()} for r in rows]


def with_assumptions(text, parts, lang="is"):
    said = list(dict.fromkeys(f"{var} {val}" for p in parts for var, val in p.get("assumed") or []))
    states_a_number = any(not any(o.is_integer() and (o <= 31 or 1800 <= o <= 2100) for o in options)
                          for options, _ in values_in(text))
    if not states_a_number:  # "The data does not answer the question.": nothing to qualify
        return text
    return f'{text.rstrip()}\n\n{ASSUMED[lang].format(", ".join(said))}' if said else text


STALE_YEARS = 3


def fresh_first(question, candidates):
    """Without a period in the question, candidates last updated STALE_YEARS or more before the newest one are left
    out of the picker's list (they stay in the fallback chain): "Hvað búa margir á Íslandi?" otherwise took HEI11100,
    titled just "Mannfjöldi", last updated 2019, over the current population tables."""
    if years_in(question):
        return candidates
    newest = max((y for t, _ in candidates if (y := newest_year(t))), default=None)
    fresh = [(t, s) for t, s in candidates if not (y := newest_year(t)) or y > newest - STALE_YEARS]
    return fresh or candidates


def newest_year(t):
    """The newest year a table holds (its time variable), else the year it was last updated, else None. The data's
    own year: FEL02001 (pensions) was updated in 2024 but ends in 2022, and "how much do pensioners get a month?"
    was answered with December 2022 over TEK02014's August 2026 (dev3, 1 Nov)."""
    span = cards.time_period(t)
    if span:
        return span[1]
    return int(t["updated"][:4]) if (t.get("updated") or "")[:4].isdigit() else None


def validate_selections(table, raw, edition="is"):
    """Keep only real codes; fill required variables with sensible defaults."""
    out = {}
    for v in variables(table, edition):
        picked = chosen(v, raw)
        if picked:
            out[v["code"]] = picked
        elif not v.get("elimination"):
            out[v["code"]] = default_value(v)
    return out


def flatten(jsonstat):
    """json-stat2 dataset -> list of rows {dimension label: value label, ..., 'value': x}."""
    dims = []
    for dim_id in jsonstat["id"]:
        cat = jsonstat["dimension"][dim_id]["category"]
        index = cat["index"]
        codes = sorted(index, key=index.get) if isinstance(index, dict) else index
        dims.append((jsonstat["dimension"][dim_id]["label"], [cat["label"][c] for c in codes]))
    rows = []
    for labels, value in zip(itertools.product(*(d[1] for d in dims)), jsonstat["value"]):
        row = dict(zip((d[0] for d in dims), labels))
        row["value"] = value
        rows.append(row)
    return rows


NOT_MEASURES = {"kyn", "aldur", "sex", "age"}


def measures(t, question="", most=6):
    """What a table measures, from its short non-time variables ("Hlutfall, %: Atvinnuþátttaka, Starfandi, ...
    Atvinnuleysi"): titles alone don't tell look-alikes apart (VIN00910 holds four rates; the question wants one).
    Values that share a word stem with the question come first: LAN10102's first six are cattle, its sheep later."""
    qw = _words(question)
    out = []
    for v in t.get("variables_is") or []:
        if v.get("time") or v["text"].lower() in NOT_MEASURES or not 2 <= len(v["values"]) <= 15:
            continue
        texts = [x for x in v["valueTexts"] if x.strip().lower() not in TOTAL_WORDS]
        texts = sorted(texts, key=lambda x: not _matches(qw, x))[:most]
        out.append(f'{v["text"]}: {", ".join(texts)}' + (" ..." if len(v["values"]) > most + 1 else ""))
    return "; ".join(out[:2])


VILLAGE_NOTE = ("Athugið: {} er bær eða þorp (byggðakjarni), ekki sveitarfélag: veldu töflu um byggðakjarna eða "
                "þéttbýlisstaði ef hún er á listanum, ekki töflu eftir sveitarfélögum.")


def pick_tables(question, candidates, villages=(), second=False):
    """[first choice], or with second=True [first, second]; [] when the model declines."""
    # the unit and the last update tell look-alikes apart: "What was Iceland's GDP?" took THJ01000 (Vísitala/
    # hlutfall) over THJ01102 (Milljónir króna) in run 6; "Hvað búa margir á Íslandi?" took HEI11100 ("Mannfjöldi",
    # last updated 2019, mean population 2017) over MAN00101 in the dry run
    listing = "\n".join(
        f'[{i}] {t["id"]}: {t["title_is"]} / {t.get("title_en") or ""} | breytur: '
        f'{", ".join(v["text"] for v in t["variables_is"])}'
        + (f" | {m}" if (m := measures(t, question)) else "")
        + (f" | eining: {u}" if (u := table_unit(t, "is")) else "")
        + (f' | uppfærð {t["updated"][:4]}' if t.get("updated") else "")
        + (cards.card(t, language(question), question) if CARDS else "")
        for i, (t, _) in enumerate(candidates)
    )
    note = f"\n\n{VILLAGE_NOTE.format(', '.join(villages))}" if villages else ""
    system = PICK_SYSTEM[:PICK_SYSTEM.index("Svaraðu eingöngu")] + PICK_JSON_SECOND if second else PICK_SYSTEM
    reply = llm.chat_json(system, f"Spurning: {question}\n\nTöflur:\n{listing}{note}")
    picked = []
    for key in ("table", "second") if second else ("table",):
        try:
            idx = int(reply.get(key, 0 if key == "table" else -1))  # small models often quote it: {"table": "4"}
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(candidates) and candidates[idx][0] not in picked:
            picked.append(candidates[idx][0])
        elif key == "table":
            return []  # no first choice: the caller takes the search's #1
    return picked


def pick_table(question, candidates, villages=()):
    """The model's choice among the candidates, or None (it declined, or replied with no usable number)."""
    picked = pick_tables(question, candidates, villages)
    return picked[0] if picked else None


# The picker's runner-up, read while the answer's own table is: when it passes the self-check too and holds another
# single number, a line under the answer names it. Look-alike tables that count differently (dental care paid by
# households or by everyone private, renters by one survey or another) were most of dev3's misses (1 Nov), and the
# organizers ask two-table answers to name each table. ALSO_TABLE=0 switches it off. On since 1 Nov 10:45 (Gemma,
# front door): dev3 40 -> 42, the 60 + set 1 97 -> 98 (weekly hours), set 2 35, real2 33, declines unchanged;
# 3 of the gains are this line naming the accepted table, 0.6 s more per answer on average.
ALSO_TABLE = os.environ.get("ALSO_TABLE", "1") == "1"
ALSO_WAIT = 20  # seconds to wait for it once the answer's own table is read
# When the picker leaves the search's #1, read that #1 instead of the picker's second choice: the picker left a right
# #1 on 25 of 270 dev questions and 25 of the 60's 100 parts (eval/experiments/pick_ranks.py, 1 Nov).
# ALSO_FROM_SEARCH=1 switches it on. Off: dev3 46 -> 45 (fish landed in Vestmannaeyjar: the picker's own second
# choice held the accepted number), nothing gained elsewhere (1 Nov 12:10, 4-arm round).
ALSO_FROM_SEARCH = os.environ.get("ALSO_FROM_SEARCH", "0") == "1"
_ALSO_POOL = ThreadPoolExecutor(4)
ALSO_LINE = {"is": "Önnur tafla Hagstofunnar, {id} ({title}), sýnir {value}{unit} ({labels}).",
             "en": "Another Statistics Iceland table, {id} ({title}), shows {value}{unit} ({labels})."}


def _checked_part(question, table, calc, lang):
    """The table's part when it has numbers and passes the self-check, else None."""
    try:
        part = fetch_part(question, table, calc=calc, lang=lang)
    except NoData:
        return None
    return part if rows_answer(question, part)[0] else None


def _single(part):
    """The part's one number (no calculation, one row with a value), else None."""
    nums = [r["value"] for r in part.get("rows") or [] if isinstance(r.get("value"), (int, float))]
    return nums[0] if len(nums) == 1 and not part.get("calculation") else None


SHARE_LABEL = re.compile(r"hlutfall|hlutdeild|prósent|%|\bshare\b|percent", re.I)


def _labels(part):
    row = next((r for r in part.get("rows") or [] if isinstance(r.get("value"), (int, float))), {})
    return ", ".join(str(v) for k, v in row.items() if k != "value")


def _is_share(part):
    return bool(PERCENT_UNIT.match(part.get("unit") or "") or SHARE_LABEL.search(_labels(part)))


def differs(part, other):
    """other is another table holding another single number than the answer's own (over 0.5% apart), and not a
    whole of 100% ("Innbyrðis hlutdeild" = 100 said nothing about education's share of spending, dev3 1 Nov)."""
    if other["table"]["id"] == part["table"]["id"]:
        return False
    a, b = _single(part), _single(other)
    if a is None or b is None or (_is_share(other) and abs(b - 100) < 1e-9):
        return False
    return abs(a - b) > 0.005 * max(abs(a), abs(b))


PERCENT_UNIT = re.compile(r"^\s*(hlutfall|prósent\w*|percent\w*|percentage|%)\s*$", re.I)


def also_line(part, lang):
    """The runner-up's number in one sentence: table, value, unit (left out when the unit line names several
    measures, "Tonnes/1000 ISK": the labels say which), and the labels of its row."""
    row = next(r for r in part["rows"] if isinstance(r.get("value"), (int, float)))
    labels = _labels(part)
    raw = (part.get("unit") or "").strip()
    unit = "%" if _is_share(part) else "" if not raw or re.search(r"[/,;]", raw) else f" {raw}"
    return ALSO_LINE["en" if lang == "en" else "is"].format(
        id=part["table"]["id"], title=title(part["table"], lang), value=fmt(row["value"], english=lang == "en"),
        unit=unit, labels=labels)


def decompose(question):
    """The question as 1 to MAX_PARTS self-contained sub-questions (more than one only for different tables)."""
    try:
        parts = llm.chat_json(DECOMPOSE_SYSTEM, question).get("questions")
    except (ValueError, AttributeError):  # bad JSON: answer the question as asked
        return [question]
    parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()] if isinstance(parts, list) else []
    return parts[:MAX_SPLIT] if len(parts) > 1 else [question]  # one part: keep the user's own wording


def rewrite_system(today=None):
    """DECOMPOSE_SYSTEM with its JSON line replaced by the rewrite rules and today's date."""
    today = today or datetime.date.today()
    base = DECOMPOSE_SYSTEM[:DECOMPOSE_SYSTEM.index("Svaraðu eingöngu með JSON")]
    return base + REWRITE_RULES.format(today=today.isoformat(), last_year=today.year - 1)


def decompose_rewrite(question, today=None):
    """decompose() plus a clean version of each part, in the same LLM call: [(part, clean or None), ...].
    One part keeps the user's own wording, as in decompose(); a bad reply is the question as asked."""
    try:
        items = llm.chat_json(rewrite_system(today), question).get("questions")
    except (ValueError, AttributeError):
        return [(question, None)]
    pairs = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("question"), str) and item["question"].strip():
            clean = item.get("clean")
            pairs.append((item["question"].strip(), clean.strip() if isinstance(clean, str) and clean.strip() else None))
        elif isinstance(item, str) and item.strip():  # the old shape: no clean version
            pairs.append((item.strip(), None))
    if len(pairs) > 1:
        return pairs[:MAX_SPLIT]
    return [(question, pairs[0][1] if pairs else None)]


def search_with_rewrite(retriever, question, clean, k=8, pool=20):
    """Search the question as typed and its clean version, fused by reciprocal rank (as FusionRetriever does
    across retrievers); without a different clean version, the plain search."""
    if not clean or clean.strip().lower() == question.strip().lower():
        return retriever.search(question, k=k)
    return rrf([retriever.search(question, k=max(k, pool)), retriever.search(clean, k=max(k, pool))], k=k)


def compute(operation, values):
    """The arithmetic, in code: binary operations take [a, b], sum/average a list."""
    if operation in BINARY_OPS:
        return BINARY_OPS[operation](*values)
    return LIST_OPS[operation](values)


def fmt(x, english=False):
    """1234.5 -> '1.234,5' (Icelandic) or '1,234.5'; whole numbers without decimals, small ones with two."""
    decimals = 0 if abs(x - round(x)) < 1e-9 else 1 if abs(x) >= 10 else 2  # 35,5 / 0,74
    s = f"{x:,.{decimals}f}"
    return s if english else s.translate(str.maketrans(",.", ".,"))


def rows_answer(question, part):
    """(ok, why): do the rows read answer the (sub-)question? A second look by the model at what was actually read,
    before answering: the wrong measure (activity rate for unemployment), an index for an amount, another place."""
    data = "\n".join(json.dumps(r, ensure_ascii=False) for r in part["rows"][:30])
    unit = f'\nEining: {part["unit"]}' if part.get("unit") else ""
    if CHECK_NOTES and (note := cards.note(part["table"], question, "is", chars=300)):
        unit += f"\nAthugasemd töflunnar: {note}"  # who and when it counts: "16 ára og eldri", "1. janúar ár hvert"
    if part.get("not_yet"):  # the asked year isn't out yet: the newest figures, said to be the newest, are the answer
        unit += (f"\nTölur fyrir {', '.join(map(str, part['not_yet'][0]))} eru ekki komnar út; nýjustu tölur töflunnar "
                 f"eru frá {part['not_yet'][1]} og eru rétt svar ef svarið segir frá hvaða ári þær eru.")
        # the check judges the question as about the newest year: a note alone lost to its "another period" rule
        # ("Tölurnar eru frá 1. janúar 2023, en spurt er um 1. janúar 2025", set 1, 1 Nov)
        for y in part["not_yet"][0]:
            question = re.sub(rf"(?<!\d){y}(?!\d)", str(part["not_yet"][1]), question)
    # today's date: without it the model called 2026 "a forecast, not now" and refused the right table (Húsavík)
    today = f"Í dag er {datetime.date.today().isoformat()}; nýjustu tölurnar eru frá þessu ári eða því síðasta.\n\n"
    try:
        reply = llm.chat_json(CHECK_SYSTEM, f'{today}Spurning: {question}\n\nTafla {part["table"]["id"]}: '
                                            f'{part["table"]["title_is"]}{unit}\nGögn:\n{data}')
    except ValueError:  # an unreadable reply is no reason to drop the table
        return True, ""
    said = reply.get("answers")
    return not (said is False or str(said).strip().lower() in ("false", "nei", "no")), str(reply.get("why") or "")


def calculate_formula(plan, rows, english=False):
    """A formula over named rows (calc.py), for calculations of more than one step: '(total - capital) / total * 100',
    'sum(old) / all * 100'. The text shows the formula with each number in it; every row value, list sum, step and
    the result are kept in 'values', so the number check accepts any of them in the answer."""
    try:
        names = calc.bind(plan.get("names"), rows)
        result, steps = calc.evaluate(plan.get("formula"), names)
    except calc.FormulaError as e:
        log.info("formula refused: %s (%r)", e, plan)
        return None
    f = lambda x: fmt(x, english)  # noqa: E731
    rows_word = "rows" if english else "raðir"
    shown = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", lambda m: (
        f(names[m.group(0)]) if isinstance(names.get(m.group(0)), float)
        else f"{m.group(0)} ({len(names[m.group(0)])} {rows_word})" if m.group(0) in names else m.group(0)),
        plan["formula"]).replace("*", "×").replace(" - ", " − ")
    percent = plan.get("unit") == "%" or re.search(r"\*\s*100\s*\)?\s*$", plan["formula"]) or "cagr(" in plan["formula"]
    values = [v for x in names.values() for v in (x if isinstance(x, list) else [x])]
    values += [sum(x) for x in names.values() if isinstance(x, list)] + steps
    refs = [i for r in plan["names"].values() for i in (r if isinstance(r, list) else [r])]
    return {"operation": "formula", "rows": [int(i) for i in refs], "values": values, "result": result,
            "text": f"{shown} = {f(result)}{'%' if percent else ''}"}


# a row measuring this is not in the money scale of a mixed unit line ("Milljónir króna/hlutfall": its % rows)
UNSCALED_MEASURE = re.compile(r"%|hlutfall|prósent|breyting|vísitala|index|percent|share|ratio|fjöldi|number|count|"
                              r"tonn|kg|magn|quantity", re.I)
# a unit line naming another absolute measure besides its scale: only the row's own label can tell which it has
OTHER_MEASURE = re.compile(r"tonn|kg|lítr|fjöldi|fjóldi|number|count|quantity|magn|klst|hours", re.I)


def row_scale(part, row):
    """What one unit of a row's value is: a scale its own labels name ("Million ISK"); else the unit line's scale
    when the line has one besides percentages or an index ("Milljónir króna/hlutfall") and the row is not a percent,
    index or count row; else 1. A line with a second absolute measure ("Milljónir króna/fjöldi") leaves it to the label."""
    labels = " ".join(str(v) for k, v in row.items() if k not in ("value", "tafla"))
    own = verify.scales(labels)
    if not own:
        unit = part.get("unit") or ""
        if OTHER_MEASURE.search(unit) or UNSCALED_MEASURE.search(labels):
            return 1.0
        own = verify.scales(unit)
    return next(iter(own)) if len(own) == 1 else 1.0


CONSTANT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e\d+)?")


def _in_whole_units(refs, rows, scale, formula=""):
    """rows with values in whole units (krónur, people) when the rows used are counted in different scales, else
    rows: 47,497.7 million krónur / 358,298 people is 132,565 krónur a head, not 0.13 (dev3, 1 Nov: per-inhabitant
    spending on alcohol and on R&D). Rows in one scale keep the table's unit (a sum in millions stays in millions).
    A formula that already multiplies by the gap ("a * 1000000 / b") is left alone: converting too gave 79,953 million
    krónur of media revenue per head (dev3, 1 Nov)."""
    used = [scale[i] for i in refs if scale and 0 <= i < len(scale)]
    if len(set(used)) < 2:
        return rows
    gap = max(used) / min(used)
    if any(float(c) >= gap for c in CONSTANT.findall(formula or "") if float(c) >= 1000):
        return rows
    return [dict(r, value=r["value"] * k) if isinstance(r.get("value"), (int, float)) else r for r, k in zip(rows, scale)]


def calculate(plan, rows, english=False, scale=None):
    """Check the model's plan against the rows and compute it. None when there is nothing (valid) to compute.
    scale: each row's unit (row_scale), given for rows of two tables."""
    op = plan.get("operation")
    if op == "formula":
        try:
            refs = [int(i) for r in (plan.get("names") or {}).values() for i in (r if isinstance(r, list) else [r])]
        except (TypeError, ValueError, AttributeError):
            refs = []
        return calculate_formula(plan, _in_whole_units(refs, rows, scale, str(plan.get("formula") or "")), english)
    if op in BINARY_OPS:
        picks = [plan.get("a"), plan.get("b")]
    elif op in LIST_OPS:
        picks = plan.get("rows")
    else:
        return None
    try:
        picks = [int(i) for i in picks]
        rows = _in_whole_units(picks, rows, scale)
        values = [float(rows[i]["value"]) for i in picks]
        result = compute(op, values)
    except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError):  # bad index, missing value, b = 0
        return None
    f = lambda x: fmt(x, english)  # noqa: E731
    if op in BINARY_OPS:
        a, b = f(values[0]), f(values[1])
        text = {"difference": f"{a} − {b}", "ratio": f"{a} / {b}", "share": f"{a} / {b} × 100",
                "per_1000": f"{a} / {b} × 1000", "percent_change": f"({a} − {b}) / {b} × 100"}[op] + f" = {f(result)}"
        text += "%" if op in ("share", "percent_change") else ""
    else:
        total = " + ".join(f(v) for v in values)
        text = (f"({total}) / {len(values)}" if op == "average" else total) + f" = {f(result)}"
    return {"operation": op, "rows": picks, "values": values, "result": result, "text": text}


SHARE_Q = re.compile(r"hlutfall|prósent|hve stór hluti|percentage|percent\b|share of|what share|proportion", re.I)
PER_1000_Q = re.compile(r"(á hverja|per)\s+(1[.,]?000|þúsund|thousand)", re.I)
DIRECT_PER_HEAD = re.compile(r"á mann\b|á íbúa|á hvern íbúa|per capita|per head|per inhabitant", re.I)
PER_HEAD_Q = re.compile(r"á (hvern )?íbúa|á mann\b|per (inhabitant|capita|person|resident)", re.I)


def backup_plan(question, rows):
    """When the model names no operation although the question asks for a share or a rate of exactly two numbers
    (run 4: 5 of the 60 lost this way; the model then did the arithmetic itself). Two tables (rows carry "tafla"):
    the population table (MAN...) is the base. One table: only a part against its total row ("Alls", "Total"),
    so two years or two places are never divided by each other."""
    nums = [i for i, r in enumerate(rows) if isinstance(r.get("value"), (int, float))]
    if len(nums) != 2:
        return None
    small, big = sorted(nums, key=lambda k: rows[k]["value"])
    if "tafla" in rows[0]:  # across two tables
        pop = [k for k in nums if str(rows[k]["tafla"]).startswith("MAN")]
        base = pop[0] if len(pop) == 1 else big
        other = nums[1] if base == nums[0] else nums[0]
        if PER_1000_Q.search(question):
            return {"operation": "per_1000", "a": other, "b": base}
        if PER_HEAD_Q.search(question):
            return {"operation": "ratio", "a": other, "b": base}
        return {"operation": "share", "a": small, "b": big} if SHARE_Q.search(question) else None
    total = [k for k in nums if any(str(v).strip().lower() in TOTAL_WORDS for c, v in rows[k].items() if c != "value")]
    if SHARE_Q.search(question) and total == [big]:
        return {"operation": "share", "a": small, "b": big}
    if COMPARE and COMPARE_Q.search(question) and (order := _two_periods(rows[nums[0]], rows[nums[1]])):
        later, earlier = (nums[0], nums[1]) if order > 0 else (nums[1], nums[0])
        return {"operation": "difference", "a": later, "b": earlier}
    return None


# "Var hún meiri eða minni en 2021?": the answer gave both turnouts and "meiri" but not by how much, and the
# planner named no operation (set 1, 1 Nov). Two rows that differ only in their year: code states the change.
# COMPARE=0 switches it off. On since 1 Nov 12:15 (Gemma, front door, 4-arm round): the 60 + set 1 97 -> 98 (the
# turnout question), dev3 46 -> 46, set 2 37, declines unchanged.
COMPARE = os.environ.get("COMPARE", "1") == "1"
COMPARE_Q = re.compile(r"meiri eða minni|minni eða meiri|hærri eða lægri|lægri eða hærri|fleiri eða færri|"
                       r"færri eða fleiri|more or less|less or more|higher or lower|lower or higher|"
                       r"more or fewer|fewer or more|increase or decrease|went up or down", re.I)


def _two_periods(a, b):
    """+1 when row a is the later of two rows that differ only in a label with a year, -1 when b is, else 0."""
    keys = [k for k in a if k != "value"]
    differ = [k for k in keys if a.get(k) != b.get(k)]
    if len(differ) != 1 or set(a) != set(b):
        return 0
    ya, yb = (YEAR.findall(str(r[differ[0]])) for r in (a, b))
    if not ya or not yb or ya[0] == yb[0]:
        return 0
    return 1 if int(ya[0]) > int(yb[0]) else -1


def plan_calculation(question, rows, scale=None):
    """The model names the operation and rows; code computes. Only for 2+ rows (one row: nothing to compute)."""
    if not 2 <= len(rows) <= MAX_CALC_ROWS:
        return None
    listing = "\n".join(f"[{i}] {json.dumps(r, ensure_ascii=False)}" for i, r in enumerate(rows))
    try:
        plan = llm.chat_json(CALC_SYSTEM, f"Spurning: {question}\n\nRaðir:\n{listing}")
    except ValueError:
        plan = {}
    result = calculate(plan, rows, english=is_english(question), scale=scale)
    if result is None and (backup := backup_plan(question, rows)):
        result = calculate(backup, rows, english=is_english(question), scale=scale)
    return result


MONTHS = {"is": ["janúar", "febrúar", "mars", "apríl", "maí", "júní", "júlí", "ágúst", "september", "október",
                 "nóvember", "desember"],
          "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"]}
RELATIVE = [  # (pattern, period it means, given today)
    (r"\b(í fyrra|síðasta ár(i|s)?|árið sem leið|síðastliðið ár|á liðnu ári|last year|the previous year)\b",
     lambda d, lang: str(d.year - 1)),
    (r"\b(fyrir tveimur árum|two years ago)\b", lambda d, lang: str(d.year - 2)),
    (r"\b(á þessu ári|this year)\b", lambda d, lang: str(d.year)),
    # the last month or quarter = the newest one published, not the calendar's: on 1 Nov "last quarter" was Q3 2026,
    # the labour force survey's newest was Q2, and the self-check declined every table (dev3: 2 questions)
    (r"\b(í síðasta mánuði|síðasta mánuði|last month)\b",
     lambda d, lang: "nýjasti mánuður í gögnunum" if lang == "is" else "the newest month in the data"),
    (r"\b(á síðasta ársfjórðungi|síðasta ársfjórðungi|last quarter)\b",
     lambda d, lang: "nýjasti ársfjórðungur í gögnunum" if lang == "is" else "the newest quarter in the data"),
]


LOOSE = {"í": "[íi]", "ð": "[ðd]", "á": "[áa]", "ó": "[óo]", "ú": "[úu]", "é": "[ée]", "ý": "[ýy]", "ö": "[öo]",
         "æ": "(?:æ|ae)", "þ": "(?:þ|th)"}


def loose(pattern):
    """Accents optional: "í fyrra" also matches "i fyrra", "síðasta ársfjórðungi" "sidasta arsfjordungi". About a third
    of typed questions have none; "hvad fengu margir ... i fyrra" was answered for 2024, not 2025 (dev3, 1 Nov)."""
    return "".join(LOOSE.get(c, c) for c in pattern)


def resolve_dates(question, today=None):
    """Relative dates made explicit in code ("í fyrra" -> "í fyrra (2025)"), so value selection, the period check
    and search see a real period instead of falling back to the latest one."""
    today = today or datetime.date.today()
    lang = "en" if is_english(question) else "is"
    for pattern, period in RELATIVE:
        question = re.sub(loose(pattern), lambda m: f"{m.group(0)} ({period(today, lang)})", question, flags=re.I)
    return question


YEAR = re.compile(r"(?<![\d.,])(1\d{3}|20\d{2}|2100)(?![\d.,]\d)")
NOT_A_YEAR = re.compile(r"\b(per|hverj\w*|hvern|hvert|á)\s*$|^\s*(íbú\w*|inhabit\w*|manns|people|resident\w*|kr\b|isk\b)",
                        re.I)


def years_in(question):
    """Years the question names ("árið 1200", "in 2035"), not numbers like "per 1000 inhabitants"."""
    out = set()
    for m in YEAR.finditer(question):
        if not (NOT_A_YEAR.search(question[max(0, m.start() - 12):m.start()]) or NOT_A_YEAR.search(question[m.end():])):
            out.add(int(m.group(1)))
    return out


def table_years(table):
    """Years in the table's time variable, read from codes ("2025", "2026M07") and labels: some tables number
    their periods 0, 1, 2 ... and only the labels carry the year."""
    years = set()
    for v in table["variables_is"]:
        if v.get("time"):
            for s in list(v["values"]) + list(v["valueTexts"]):
                years |= {int(y) for y in re.findall(r"(?<!\d)(1\d{3}|20\d{2}|2100)(?!\d)", str(s))}
    return years


def period_gap(question, table):
    """(first, last) year of the table when it has none of the years the question names, else None: then the
    latest period must not stand in for "1200" or "2035", which the table doesn't have."""
    asked, have = years_in(question), table_years(table)
    if not asked or not have or asked & have:
        return None
    return min(have), max(have)


# A year that is not published yet is no reason to decline: "Hvað fæddust mörg börn árið 2026?" or "hversu margir
# heita Jón skv. tölum frá 1. janúar 2025" (the newest names are from 2023) get the newest figures, said to be the
# newest. Only for years after the table's newest, up to this year, from a table no more than NOT_YET_YEARS old:
# "árið 2035" and "1200" still decline. NOT_YET=0 switches it off. On since 1 Nov 12:30 (Gemma, front door): set 1
# 37 -> 39 (the Jón question; Reykjanesbær flips between runs), the 60 60/60, dev3 45, set 2 37, real2 33, both
# decline sets unchanged (12/12, 15/16).
NOT_YET = os.environ.get("NOT_YET", "1") == "1"
NOT_YET_YEARS = 3
NOT_YET_NOTE = {"is": "Tölur fyrir {asked} eru ekki komnar út hjá Hagstofunni; þetta eru nýjustu tölurnar, frá {newest}.",
                "en": "Figures for {asked} are not published yet; these are the newest, from {newest}."}
TITLE_YEAR = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")


def not_yet_published(question, table, today=None):
    """(asked years, the table's newest year) when the question asks only for years the table doesn't have yet
    (after its newest, not after this year) and its newest is recent; else None. A table without a time variable
    (a snapshot, "Eiginnöfn karla 1. janúar 2023") counts its title's year."""
    if not NOT_YET:
        return None
    asked = years_in(question)
    have = table_years(table) or {int(y) for y in TITLE_YEAR.findall(table.get("title_is") or "")}
    if not asked or not have or asked & have:
        return None
    this, newest = (today or datetime.date.today()).year, max(have)
    if all(newest < y <= this for y in asked) and newest >= this - NOT_YET_YEARS:
        return sorted(asked), newest
    return None


def not_yet_note(part, lang):
    asked, newest = part["not_yet"]
    return NOT_YET_NOTE["en" if lang == "en" else "is"].format(asked=", ".join(map(str, asked)), newest=newest)


def table_unit(table, edition):
    """The unit from the table itself (local_data.unit), in the edition's language; None when unknown. The model is
    told it rather than left to guess (it once wrote kg and kW for tonnes)."""
    try:
        return local_data.unit(table["path"], edition)
    except Exception:  # noqa: BLE001 - no release here, an unreadable header: the answer works without a unit
        return None


def fetch_part(question, table, calc=True, lang="is"):
    """For one (sub-)question and its table: choose values, fetch the numbers, compute if the question asks to.
    Tries the table's editions in turn (editions()); each reads the data release, else PX-Web (px.query). An
    edition that fails or gives no number at all ("..") is skipped; NoData when none is left. LLM errors propagate."""
    later = not_yet_published(question, table)
    gap = period_gap(question, table)
    if gap and not later:  # the question's year isn't in this table: no number at all beats the wrong year's
        raise NoData(f'{table["id"]} covers {gap[0]}-{gap[1]}, not {sorted(years_in(question))}')
    asked = f"{question} (þær tölur eru ekki komnar út: veldu nýjustu tölurnar, {later[1]})" if later else question
    skipped = []
    for edition in editions(table, lang):
        raw = llm.chat_json(
            SELECT_SYSTEM,
            f'Spurning: {asked}\n\nTafla {table["id"]}: {title(table, edition)}\nBreytur:\n'
            f"{describe_variables(table, question, edition)}",
        ).get("selections", {})
        selections = validate_selections(table, raw, edition)
        try:
            rows = flatten(px.query(table["path"], selections, lang=edition))
        except Exception as e:  # noqa: BLE001 - PX-Web down or rate-limited, a code the edition lacks: try the next
            skipped.append(f"{edition}: {type(e).__name__}: {e}")
            continue
        if all(r["value"] is None for r in rows):
            skipped.append(f"{edition}: no numbers for {selections}")
            continue
        return {"question": question, "table": table, "edition": edition, "url": px.web_url(table["path"], edition),
                "selections": selections, "rows": rows, "fallback": skipped, "unit": table_unit(table, edition),
                "assumed": assumptions(table, raw, edition), "not_yet": later,
                "calculation": plan_calculation(question, rows) if calc else None}
    raise NoData(f'{table["id"]}: ' + "; ".join(skipped))


def language(question):
    """"en" or "is": decided in code, not left to the model (models answered English questions in Icelandic)."""
    return "en" if is_english(question) else "is"


def title(table, lang):
    """The table's title in the answer's language (the Icelandic one if the table has no English edition)."""
    return (table.get("title_en") if lang == "en" else None) or table["title_is"]


def write_answer(question, parts, missing=(), english=True, cross=None):
    """One answer over all parts; each part brings its table, its rows and any calculation done in code.
    The numbers come from the Icelandic tables either way; the language rule goes last, where models heed it.
    english=False: no language rule and Icelandic titles (the pipeline before English answers, for the ablation)."""
    lang = language(question) if english else None
    blocks = []
    for p in parts:
        data = "\n".join(json.dumps(r, ensure_ascii=False) for r in p["rows"][:MAX_ROWS])
        unit = f'\nEining: {p["unit"]}' if p.get("unit") else ""
        block = f'Tafla {p["table"]["id"]}: {title(p["table"], lang or "is")}{unit}\nGögn:\n{data}'
        if p.get("calculation"):
            block += f'\nÚtreikningur (gerður í kóða, notaðu niðurstöðuna óbreytta): {p["calculation"]["text"]}'
        if p.get("not_yet"):
            block += (f"\nAthugið: tölur fyrir {', '.join(map(str, p['not_yet'][0]))} eru ekki komnar út; þetta eru "
                      f"nýjustu tölurnar, frá {p['not_yet'][1]}. Segðu frá hvaða ári þær eru.")
        blocks.append(block)
    if cross:
        blocks.append(f"Útreikningur úr fleiri en einni töflu (gerður í kóða, notaðu niðurstöðuna óbreytta): "
                      f"{cross['text']}")
    blocks += [f"Engin gögn fundust fyrir: {q}" for q in missing]
    rule = f"\n\n{ANSWER_IN[lang]}" if lang else ""
    return llm.chat(ANSWER_SYSTEM, f"Spurning: {question}\n\n" + "\n\n".join(blocks) + rule)


def with_calculations(text, calculations, lang="is"):
    """The answer with every result the code computed: models round them their own way ("5,7" for 5,77) or redo
    the sum. A result the answer doesn't state (in Icelandic or English form) is added, with how it was computed."""
    missing = [c["text"] for c in calculations if c and not any(
        fmt(c["result"], english) in text for english in (False, True))]
    if not missing:
        return text
    label = "Calculation" if lang == "en" else "Útreikningur"
    return text.rstrip() + "\n\n" + "\n".join(f"{label}: {m}" for m in missing)


SHOWN_CANDIDATES = 5


def candidate_list(found, chosen_table, n=SHOWN_CANDIDATES):
    """The search's top n for one sub-question, for display: id, titles, rank (1 = first) and whether it is the
    table the answer used; the used table is added at its own rank when it is further down (rank None: it came
    from outside the search, e.g. the villages tables). Only a record of what happened: nothing reads it back.
    Compared by identity: table ids are not unique in the catalog."""
    def entry(t, rank):
        return {"id": t["id"], "title_is": t.get("title_is"), "title_en": t.get("title_en"), "rank": rank,
                "chosen": t is chosen_table}
    shown = [entry(t, i + 1) for i, (t, _) in enumerate(found[:n])]
    if not any(c["chosen"] for c in shown):
        rank = next((i + 1 for i, (t, _) in enumerate(found) if t is chosen_table), None)
        shown.append(entry(chosen_table, rank))
    return shown


# Non-words Gemma wrote in Icelandic answers (1 Nov: "Árið 2026 býa 394.324 manns", "þingmenna" 3 times in 355
# answers). No Icelandic word has these forms, so replacing them cannot harm a right answer.
SLIPS = [(re.compile(r"\bbýa\b"), "búa"), (re.compile(r"\bBýa\b"), "Búa"), (re.compile(r"\bþingmenna\b"), "þingmanna"),
         (re.compile(r"\b([Ff])ødd"), r"\1ædd"),  # "Árið 2000 føddust 4.315" (a Faroese ø, real2 1 Nov)
         (re.compile(r"\bmannlásir\b"), "manns"),  # "5.052 mannlásir heita Jón" (concurrency test, 1 Nov)
         (re.compile(r"(?<=\d)stk\."), ""),  # "2020: 354.042stk." (live page check, 1 Nov)
         (re.compile(r"\bbýðu\b"), "bjuggu"), (re.compile(r"\bBýðu\b"), "Bjuggu"),  # "Árið 2020 býðu 37.036" (1 Nov)
         # "- 2015: 4107.0": a whole number copied from the rows' JSON (4107.0); 3+ digits, so "6,0%" keeps its decimal
         (re.compile(r"(?<![\d.,])(\d{3,})\.0(?![\d.,])"), r"\1")]


def fix_slips(text):
    for wrong, right in SLIPS:
        text = wrong.sub(right, text)
    return text


def _result(text, parts, **extra):
    """The answer plus provenance; table/url/selections/rows of the first part for callers that show one table."""
    first = parts[0]
    return {"answer": text, "table": first["table"], "url": first["url"], "selections": first["selections"],
            "rows": first["rows"], "parts": parts, **extra}


SUGGEST_CHECK_WAIT = 25  # seconds for the suggestions' own answers; one not answered in time is left out
SUGGEST_ROUNDS = 2  # when none of the written suggestions answers, the model writes others once more


def answerable(found, retriever, steps):
    """Only the suggestions this system really answers: each is asked for real, in parallel, and kept only when it
    comes back with numbers that passed the self-check. "Apple production in 2025?" suggested "production in
    agriculture in 2025", which led to 'no data' again (1 Nov): a suggestion must never lead nowhere."""
    if not found:
        return []

    def works(q):
        PROGRESS.set(None)  # its steps are not this question's: keep them off the step panel
        try:
            r = answer(q, retriever, steps=steps, suggestions_wanted=False, rows_only=True)
        except Exception:  # noqa: BLE001 - a suggestion that fails is simply not offered
            return False
        return any(isinstance(row.get("value"), (int, float)) for p in r.get("parts") or [] for row in p.get("rows") or [])

    pool = ThreadPoolExecutor(len(found))
    deadline = time.monotonic() + SUGGEST_CHECK_WAIT  # for all of them together
    try:
        jobs = [pool.submit(contextvars.copy_context().run, works, q) for q in found]
        keep = []
        for q, job in zip(found, jobs):  # best first: done once the first MOST that answer are known
            try:
                if job.result(timeout=max(0.0, deadline - time.monotonic())):
                    keep.append(q)
            except Exception:  # noqa: BLE001 - too slow: not offered
                log.info("suggestion not checked in time: %r", q)
            if len(keep) == suggest.MOST:
                break
        return keep
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def answer(question, retriever, k=None, steps=STEPS, suggestions_wanted=True, rows_only=False):
    """Full pipeline: split if it needs two tables, then per part retrieve -> pick -> values -> numbers (->
    calculation), then one answer. Returns the text plus everything needed to show provenance.
    `steps` switches parts off for the ablation ladder (see STEPS); the default runs them all.
    `rows_only`: one table, stop once its rows passed the check (answerable() asks suggestions this way)."""
    k = k or PICK_K
    question = resolve_dates(question)
    lang = language(question) if "english" in steps else None
    parts, missing, candidates, also = [], [], None, None
    if "split" not in steps or rows_only:
        pairs = [(question, None)]
    elif QUERY_REWRITE:
        pairs = decompose_rewrite(question)
    else:
        pairs = [(q, None) for q in decompose(question)]
    if len(pairs) == 2 and PER_HEAD_Q.search(question) and any(
            DIRECT_PER_HEAD.search(f'{t["title_is"]} {t.get("title_en") or ""}') and not period_gap(question, t)
            for t, _ in retriever.search(question, k=3)):
        pairs = [(question, None)]  # a table reports it per head already: "Landsframleiðslan á mann" (THJ01401)
    left_out = [q for q, _ in pairs[MAX_PARTS:]]  # named under the answer: ask about these separately
    pairs = pairs[:MAX_PARTS]
    if len(pairs) > 1:
        report("split", parts=[q for q, _ in pairs])

    def one(q, clean):
        """One sub-question: search, pick, read, check -> (part or None, the search's candidates, a look-alike)."""
        also = None
        report("search", question=q, tables=len(getattr(retriever, "tables", []) or []))
        found = search_with_rewrite(retriever, q, clean, k=max(k, cards.WIDE) if CARDS else k)
        if CARDS:
            found = cards.filter_candidates(q, found, frozenset(years_in(q)))[:k]
        fresh = fresh_first(q, found)
        places = places_of(retriever)
        villages = places.villages(q) if places is not None else []  # "Vík": the village, not Mýrdalshreppur
        if villages and "pick" in steps:  # the towns/villages tables, if search left them out
            listed = {t["id"] for t, _ in fresh}
            fresh = fresh + [(t, 0.0) for t in places.village_tables(set(villages), retriever.tables)
                             if t["id"] not in listed]
        also_wanted = ALSO_TABLE and len(pairs) == 1 and "check" in steps and not rows_only
        picks = pick_tables(q, fresh, villages, second=PICK_SECOND or also_wanted) if "pick" in steps and found else []
        table = picks[0] if picks else None
        also_job = None
        runner = picks[1] if len(picks) > 1 else None
        if ALSO_FROM_SEARCH and picks and fresh and picks[0] is not fresh[0][0]:
            runner = fresh[0][0]  # the picker left the search's #1: that one is the likeliest other answer
        if also_wanted and runner is not None:  # read the runner-up meanwhile: shown if it holds another number
            also_job = _ALSO_POOL.submit(_checked_part, q, runner, "calc" in steps, lang or "is")
        if table is None and found:  # the model refused ("no table fits") or picking is off: the search's #1
            table = fresh[0][0]
        if table is not None:
            report("pick", table=table["id"], title=title(table, lang or "is"), candidates=len(fresh))
        part, refused = None, []
        if table is not None:  # the picked table first, then the next ones by rank if it has no numbers
            first = [table] + [t for t in picks[1:] if t is not table]  # the model's choices, then the search's rank
            tried = first + [t for t, _ in found if all(t is not x for x in first)][:MAX_TABLES - len(first)]
            for i, t in enumerate(tried):
                report("read", table=t["id"])
                try:
                    part = fetch_part(q, t, calc="calc" in steps, lang=lang or "is")
                except NoData as e:
                    log.info("no data in table %d of %d for %r: %s", i + 1, len(tried), q, e)
                    report("nodata", table=t["id"])
                    continue
                ok, why = rows_answer(q, part) if "check" in steps else (True, "")
                if not ok:  # the rows measure something else: the next table, keeping this one in case none fits
                    log.info("check: %s does not answer %r: %s", t["id"], q, why)
                    report("refused", table=t["id"], why=why)
                    refused.append((part, why))
                    part = None
                    continue
                if i:
                    part["fallback"].insert(0, f"tables tried first had no numbers or did not answer: "
                                               f"{[x['id'] for x in tried[:i]]}")
                break
            if part is None and refused and not DECLINE_UNCHECKED:  # keep the best one, marked as a fallback
                part, why = refused[0]
                part["fallback"].insert(0, f"tables tried first did not pass the check: {why}")
            elif part is None and refused:  # no table answers it: say so rather than answer from the closest
                log.info("declined %r: no table passed the check (%s)", q, refused[0][1])
                report("declined", question=q)
        if part is not None and not rows_only:
            try:  # a confidence signal (confidence.py); never a reason to fail the answer
                part["agreement"] = agreement(retriever, q, part["table"])
            except Exception:  # noqa: BLE001
                part["agreement"] = None
            if part.get("calculation"):
                report("calc", text=part["calculation"]["text"])
            if clean:
                part["search_question"] = clean  # what QUERY_REWRITE searched besides the question
            part["candidates"] = candidate_list(found, part["table"])  # shown on the web page's route panel
        if also_job is not None:
            try:
                other = also_job.result(timeout=ALSO_WAIT)
            except Exception:  # noqa: BLE001 - slow or failed: the answer stands without it
                other = None
            if part is not None and other is not None and differs(part, other):
                also = {**other, "also": True}
        return part, found, also

    if len(pairs) > 1:  # the sub-questions are independent: read them at the same time, each in a copy of this
        contexts = [contextvars.copy_context() for _ in pairs]  # context, so progress and the trace still arrive
        with ThreadPoolExecutor(len(pairs)) as pool:
            done = list(pool.map(lambda job: job[0].run(one, *job[1]), zip(contexts, pairs)))
    else:
        done = [one(*pairs[0])]
    for (q, _), (part, found, also_found) in zip(pairs, done):  # in the question's own order
        candidates = candidates or found
        if part is None:
            missing.append(q)
        else:
            parts.append(part)
        also = also or also_found
    if rows_only:
        return {"answer": "" if parts else NO_TABLE[lang or "is"], "parts": parts, "candidates": candidates, "suggestions": []}
    if not parts:  # "no data", with 2-3 related questions the tables can answer (suggest.py)
        if not suggestions_wanted:  # checking a suggestion: its own suggestions are never shown
            return {"answer": NO_TABLE[lang or "is"], "candidates": candidates, "suggestions": []}
        nearby, found, tried = fresh_first(question, candidates or []), [], []
        for _ in range(SUGGEST_ROUNDS):  # none of them answered: written again, the model told which ones failed
            offered = suggest.suggestions(question, nearby, lang or "is", most=suggest.MOST + suggest.SPARE, avoid=tried)
            found = answerable([q for q in offered if q not in tried], retriever, steps)[:suggest.MOST]
            if found or not offered:
                break
            tried += offered
        return {"answer": suggest.with_suggestions(NO_TABLE[lang or "is"], found, lang or "is"),
                "candidates": candidates, "suggestions": found}
    # two sub-questions often need one number from both: "per inhabitant", "per 1,000", "A and B together"
    cross = None
    if "calc" in steps and len(parts) > 1:
        cross = plan_calculation(question, [{"tafla": p["table"]["id"], **r} for p in parts for r in p["rows"]],
                                 scale=[row_scale(p, r) for p in parts for r in p["rows"]])
    if cross:
        report("calc", text=cross["text"])
    report("write")
    text = tidy_decimals(write_answer(question, parts, missing, english="english" in steps, cross=cross))
    text = group_thousands(text, lang or "is")  # "4387" -> 4.387 / 4,387
    if (lang or "is") == "is":
        text = fix_slips(text)
    if "calc" in steps:
        text = with_calculations(text, [p["calculation"] for p in parts] + [cross], lang or "is")
    if lang == "en":
        english_labels(parts + ([also] if also else []))
    text = with_assumptions(text, parts, lang or "is")
    for p in parts:  # said by code too, so no answer passes the old year off as the asked one
        if p.get("not_yet"):
            text = f"{text}\n\n{not_yet_note(p, lang or 'is')}"
    if left_out:
        text = "\n".join([text, "", LEFT_OUT[lang or "is"].format(n=MAX_PARTS), *(f"- {q}" for q in left_out)])
    choices = clarify.clarifications({"parts": parts}, lang or "is")  # buttons a client may offer
    if also is not None:  # a look-alike table that counts differently, named in a line written by code
        text = f"{text}\n\n{also_line(also, lang or 'is')}"
        parts = parts + [also]
    return _result(text, parts, candidates=candidates, calculation=cross, clarify=choices)


def answer_without_data(question):
    """The model alone, no table: the bottom rung of the ablation ladder. Its numbers are whatever it remembers
    or invents, which is the point of measuring it."""
    return llm.chat(NO_DATA_SYSTEM, f"{question}\n\n{ANSWER_IN[language(question)]}")


def answer_from_table(question, table):
    """Steps after the table is chosen: values, numbers, calculation, answer. Separate so the LLM steps can be
    measured on the right table alone (eval/compare_models.py oracle mode)."""
    parts = [fetch_part(question, table, lang=language(question))]
    text = with_calculations(write_answer(question, parts), [parts[0]["calculation"]], language(question))
    return _result(text, parts)

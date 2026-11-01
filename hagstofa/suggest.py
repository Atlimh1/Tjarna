"""When the answer is "no data": 2-3 questions Statistics Iceland's tables can answer, close to what was asked
(idea B in "Hagstofan: improvements to consider", 31 Oct).

  suggestions(question, candidates, lang) -> ["Hvað bjuggu margir í Reykjavík 1. janúar 2026?", ...]

The tables are the search's candidates for the question, each one real but not always related: for "hvað eru
framleidd mörg epli á íslandi?" the search's top 15 were GDP, price indices and emissions (1 Nov). The model sees
the top SUGGEST_FROM, keeps the ones a person asking this would want, and writes concrete questions about what they
list, close to the user's place, group and period (SUGGEST_SYSTEM, one call; SUGGEST_LLM=0 turns it off). None
related: no suggestions, rather than three that lead nowhere. The pipeline asks each one before showing it
(pipeline.answerable): "apples in 2025?" got back only "apple production in 2025?", which had no data either, so the
model now writes MOST + SPARE, about what the tables list, and is asked again (told which failed) if none answers.
The search can miss the subject entirely ("epli" never reached the crop tables), so the tables of the subject
folders whose everyday words match the question are added to the model's list (data/subjects.jsonl, written by
scripts/build_subjects.py and corrected by hand). Only here, after "no data": the search itself is unchanged, since
richer table text in the search cost 6 of the organizers' 60 (1 Nov). Where that fails, the closest of the questions an LLM wrote
for the table for the fine-tuning (data/sample_questions.jsonl, 4,580 questions on 924 tables, one English per
table), or else a question built from the table's title ("Hvað segja nýjustu tölurnar í töflunni ...": the 1 Nov
dry run showed three of those in a row, which told the user nothing). The pipeline returns them in the result
("suggestions") and after the "no data" line; the web page shows them as buttons that ask them.
"""
import json
import logging
import os
from collections import defaultdict
from functools import cache

from . import llm
import tokenizer

from .retrieve import DATA, STOPWORDS, TRADE, _lemmas_of, correct_question, fold, is_english, load_catalog

log = logging.getLogger(__name__)
WRITE = os.environ.get("SUGGEST_LLM", "1") != "0"
SUGGEST_SYSTEM = (
    "A user asked Statistics Iceland's chatbot a question that no table could answer. Below are tables the search "
    "found, numbered, each with its variables and some of their values. Most may be unrelated. Choose the tables "
    "whose numbers a person asking this question would actually want: the same subject (crops for a "
    "question about apples, population for a question about inhabitants). A table on the same subject IS related "
    "even when it lacks the period or place asked: that is why the question got no answer, and the nearest period "
    "or place it has is the useful offer. A table that only shares a word (production, Reykjavík, a year) is "
    "unrelated. Write at most {most} short, concrete questions the chosen tables can answer (several from one "
    "table is fine), as close as possible to what the user wanted: keep the user's place, group and period when the "
    "table has them, otherwise use values the table has (its newest period). Ask only about what a table lists: "
    "when it lacks the thing the user asked about (apples, in a table of potatoes and grain), ask about the nearest "
    "things it does list, never about the missing thing. Use only values that are listed. Write the "
    "questions in {language}, as a person would ask them, without table ids or titles in quotes. "
    'Answer only with JSON: {{"questions": [{{"table": <number>, "question": "..."}}]}}, best first; '
    '{{"questions": []}} when no table is related.'
)
LANGUAGE = {"is": "Icelandic", "en": "English"}
VALUES_SHOWN = 8  # periods in the listing: the newest ones
ITEMS_SHOWN = 30  # other values: "Production in agriculture" lists hay and grain first, the vegetables after the 8th

SAMPLES = DATA / "sample_questions.jsonl"
MOST = 3
SPARE = 3  # written beyond MOST: the pipeline asks each one first and shows the first MOST that answer
SUGGEST_FROM = 10  # tables the model chooses from
SUBJECTS = DATA / "subjects.jsonl"
SUBJECT_FOLDERS = 2  # the best-matching subject folders whose tables are added
SUBJECT_TABLES = 8  # at most this many tables added from them
PER_FOLDER = 5  # so the best folder can't crowd out the second
TEMPLATE = {"is": "Hvað segja nýjustu tölurnar í töflunni „{title}“?",
            "en": "What are the latest figures in the table “{title}”?"}
INTRO = {"is": "Hagstofan birtir ekki tölur sem svara þessu beint. Kannski eitthvað af þessu?",
         "en": "Statistics Iceland does not publish figures that answer this directly. Did you mean one of these?"}


@cache
def samples():
    """{table id: [{"q", "lang", ...}]} from the fine-tuning questions; empty if the file is missing."""
    by_table = defaultdict(list)
    try:
        for line in open(SAMPLES, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                by_table[row["table"]].append(row)
    except OSError:
        pass
    return by_table


def _keys(text, english):
    """Words to match on: base forms without accents (Icelandic), or lower-case words without a plural s (English)."""
    if english:
        return {w.rstrip("s") for w in fold(text.lower()).replace("-", " ").split() if len(w) > 2}
    out = set()
    for tok in tokenizer.tokenize(text):
        if tok.kind == tokenizer.TOK.WORD:
            forms = _lemmas_of(tok.txt)
            if not STOPWORDS.intersection(forms):  # "voru" is also "vor" (spring): a function word is no subject
                out |= {fold(f) for f in forms}
    return out


@cache
def subjects():
    """[(folder name, Icelandic keys, English keys, table ids)] from data/subjects.jsonl; [] if it is missing."""
    try:
        rows = [json.loads(line) for line in open(SUBJECTS, encoding="utf-8") if line.strip()]
    except OSError:
        return []
    return [(r["name"], set().union(*[_keys(w, False) for w in r["words_is"]] or [set()]),
             set().union(*[_keys(w, True) for w in r["words_en"]] or [set()]), r["tables"]) for r in rows]


@cache
def _tables_by_id():
    out = {}
    for t in load_catalog():
        out.setdefault(t["id"], t)  # ids are not unique in the catalog: the first is enough for a suggestion
    return out


def subject_tables(question, lang):
    """Tables of the subject folders whose words the question uses, best folder first. A word found in many folders
    counts for little ("tekjur"), one found in a single folder for a lot ("epli")."""
    english = lang == "en" or is_english(question)
    text = question if english else correct_question(question)
    words = _keys(text, english)
    if not english and (trade := TRADE.search(text)):  # "flutt inn kaffi" is trade, not people moving (retrieve.TRADE)
        words = (words - {"flytja", "fluttur"}) | {"innflutningur" if fold(trade.group(2).lower()) == "inn" else "utflutningur"}
    pick = 2 if english else 1

    def hits(row):  # the list's words in the question, also inside a compound: "mjólkurlítrinn" has "mjólk"
        return {w for w in row[pick] if w in words or (len(w) >= 4 and any(k.startswith(w) for k in words))}
    df = {}
    for row in subjects():
        for k in hits(row):
            df[k] = df.get(k, 0) + 1
    scored = sorted(((sum(1 / df[k] for k in hits(row)), row) for row in subjects() if hits(row)), key=lambda x: -x[0])
    if not scored:
        return []
    best = scored[0][0]
    ids = [tid for score, row in scored[:SUBJECT_FOLDERS] if score >= best / 2 for tid in row[3][:PER_FOLDER]]
    found = [_tables_by_id()[tid] for tid in dict.fromkeys(ids) if tid in _tables_by_id()]
    return found[:SUBJECT_TABLES]


def _grams(text):
    t = f" {fold(text.lower())} "
    return {t[i:i + 3] for i in range(len(t) - 2)}


def _similarity(a, b):
    ga, gb = _grams(a), _grams(b)
    return len(ga & gb) / (len(ga | gb) or 1)


def _listing(tables, lang):
    """The tables as the model sees them: title, then each variable with its first values (the newest periods)."""
    lines = []
    for i, t in enumerate(tables, 1):
        title = (t.get("title_en") if lang == "en" else None) or t["title_is"]
        lines.append(f"{i}. {title}")
        for v in (t.get("variables_en") if lang == "en" else None) or t.get("variables_is") or []:
            texts = v.get("valueTexts") or []
            shown = texts[-VALUES_SHOWN:] if v.get("time") else texts[:ITEMS_SHOWN]
            more = f" (+{len(texts) - len(shown)} more)" if len(texts) > len(shown) else ""
            lines.append(f"   {v.get('text', v.get('code'))}: {', '.join(shown)}{more}")
    return "\n".join(lines)


def written(question, tables, lang, most=MOST, avoid=()):
    """The model's questions for the tables it judged related, best first (each on a real table): [] when none is.
    `avoid`: questions offered before that the tables had no numbers for; the model is told and writes others."""
    tried = ("\n\nThese were tried and the tables had no numbers for them; write different ones:\n"
             + "\n".join(f"- {q}" for q in avoid)) if avoid else ""
    reply = llm.chat(SUGGEST_SYSTEM.format(language=LANGUAGE[lang], most=most),
                     f"The user's question: {question}\n\nTables:\n{_listing(tables, lang)}{tried}", json_mode=True)
    found = llm.parse_json(reply).get("questions")
    if not isinstance(found, list):
        raise ValueError(f"no question list in {reply!r}")
    out = []
    for f in found:
        n, q = (f.get("table"), f.get("question")) if isinstance(f, dict) else (None, None)
        if isinstance(n, int) and 1 <= n <= len(tables) and isinstance(q, str) and 5 < len(q.strip()) < 200:
            out.append(q.strip())
    return out[:most]


def fallback(question, t, lang):
    """Without the model: the table's closest sample question, else one built from its title."""
    pool = [s["q"] for s in samples().get(t["id"], []) if s.get("lang", "is") == lang]
    if pool:
        return max(pool, key=lambda s: _similarity(question, s))
    title = (t.get("title_en") if lang == "en" else None) or t["title_is"]
    return TEMPLATE[lang].format(title=title)


def suggestions(question, candidates, lang="is", most=MOST, avoid=()):
    lang = "en" if lang == "en" else "is"
    tables, seen = [], set()
    for t, _ in candidates or []:
        if t["id"] not in seen:
            seen.add(t["id"])
            tables.append(t)
        if len(tables) == (SUGGEST_FROM if WRITE else most):
            break
    if WRITE:
        try:  # the subject's tables too: the search may have missed the subject (a missing file adds none)
            tables += [t for t in subject_tables(question, lang) if t["id"] not in seen]
        except Exception:  # noqa: BLE001 - suggestions never fail an answer
            log.warning("subject tables failed for %r", question, exc_info=True)
    if WRITE and tables:
        try:
            return [q for q in dict.fromkeys(written(question, tables, lang, most, avoid)) if q not in avoid]
        except Exception:  # noqa: BLE001 - no model or a bad reply: the tables' own questions, as before
            log.warning("suggestions by the model failed for %r", question, exc_info=True)
    return [q for q in dict.fromkeys(fallback(question, t, lang) for t in tables[:most]) if q not in avoid]


def with_suggestions(text, found, lang="is"):
    """The 'no data' text followed by the suggestions as a list."""
    if not found:
        return text
    lang = "en" if lang == "en" else "is"
    return f"{text}\n\n{INTRO[lang]}\n" + "\n".join(f"- {q}" for q in found)

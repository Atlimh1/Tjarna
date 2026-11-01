"""Questions in other languages: translated to English, answered by the normal pipeline, translated back.

  to_english(question, lang)          one LLM call; numbers, years and names kept
  from_english(answer, lang)          one LLM call; code checks that every number survived, else the
                                      English answer is returned with a note (a wrong number is worse
                                      than an answer in English)
  answer(question, lang, answer_fn)   the round trip around answer_fn (pipeline.answer with a retriever)

English is the pivot: tables have English editions (the Icelandic one is the fallback when there's none)
and models translate to and from English best. Icelandic and English questions never come here.
"""
import json
import re
import sqlite3
import threading

from . import llm
from .language import NAMES

TO_ENGLISH = """Translate the user's question into English. Keep every number, year, date and name exactly;
write Icelandic place names in their Icelandic form (Reykjavík, Kópavogur, Akureyri) and the country as Iceland.
Reply with the English question only, nothing else."""

FROM_ENGLISH = """Translate the text into {name}. Copy every number exactly as written, digit for digit, with the
same separators (394,324 stays 394,324; 6.5% stays 6.5%). Keep table codes, names of Icelandic places and links
unchanged. Reply with the translation only, nothing else."""

NOTE = "(Automatic translation into {name} changed a number, so the answer is shown in English.)"
NUMBER = re.compile(r"\d{1,3}(?:[.,    ]\d{3})+(?![\d])(?:[.,]\d+)?|\d+(?:[.,]\d+)?")


def numbers(text):
    """The digit sequences in a text, separators ignored: "394,324" and "394 324" both give "394324". A thousands
    separator is one character before exactly three digits: the old pattern read "January 1, 2026" as 12026, so
    every translation that writes "1 stycznia 2026" looked like a changed number (19 of 38 shown in English)."""
    return sorted(re.sub(r"\D", "", m) for m in NUMBER.findall(text))


def _one_line(text):
    return text.strip().strip('"').strip()


def to_english(question, lang):
    return _one_line(llm.chat(TO_ENGLISH, question))


def from_english(answer, lang):
    """The answer in the user's language, or the English answer with a note if any number changed."""
    name = NAMES.get(lang, lang)
    try:
        translated = llm.chat(FROM_ENGLISH.format(name=name), answer).strip()
    except Exception:  # noqa: BLE001 - an untranslated answer beats no answer
        return f"{answer}\n\n{NOTE.format(name=name)}"
    if numbers(translated) != numbers(answer):
        return f"{answer}\n\n{NOTE.format(name=name)}"
    return translated


TO_ICELANDIC = """Translate the user's question into natural Icelandic, the way an Icelander would ask Statistics
Iceland. Keep every number, year, date and name exactly; write place names in their Icelandic form (Reykjavík,
Kópavogur, Akureyri). Reply with the Icelandic question only, nothing else."""
_cache_lock = threading.Lock()


def counterpart(question):
    """The question in the other search language: English -> Icelandic, Icelandic -> English. Search runs on both
    (retrieve.BilingualRetriever), so each language gets the other's strengths: Icelandic search has BÍN, the
    synonyms and the place checks; English titles and labels match English words. Cached in data/translations.sqlite."""
    from .retrieve import DATA, is_english  # retrieve imports nothing from here at module level

    target = "is" if is_english(question) else "en"
    with _cache_lock, sqlite3.connect(DATA / "translations.sqlite", timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS tr (target TEXT, q TEXT, out TEXT, PRIMARY KEY (target, q))")
        row = db.execute("SELECT out FROM tr WHERE target=? AND q=?", (target, question)).fetchone()
    if row:
        return row[0]
    out = _one_line(llm.chat(TO_ICELANDIC if target == "is" else TO_ENGLISH, question)).splitlines()[0].strip()
    with _cache_lock, sqlite3.connect(DATA / "translations.sqlite", timeout=30) as db:
        db.execute("INSERT OR REPLACE INTO tr VALUES (?, ?, ?)", (target, question, out))
    return out


LABELS = """Translate these labels from Statistics Iceland's tables into short English labels, as Statistics Iceland
would write them ("Kyn" -> "Sex", "Alls" -> "Total", "Höfuðborgarsvæði" -> "Capital region"). Keep numbers, codes
and names of places and people as they are. Reply with JSON only: {"labels": ["<English>", ...]}, same order."""


def labels(texts):
    """{Icelandic label: English label} for English answers from the 281 tables with no English edition, whose
    labels are Icelandic ("Assuming: Kyn Alls"). One LLM call for the labels not cached yet (data/translations.sqlite,
    target "label-en"); numbers and years are left alone; on any failure a label stays as it is."""
    from .retrieve import DATA

    texts = list(dict.fromkeys(str(t) for t in texts if str(t).strip()))
    out = {}
    with _cache_lock, sqlite3.connect(DATA / "translations.sqlite", timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS tr (target TEXT, q TEXT, out TEXT, PRIMARY KEY (target, q))")
        for t in texts:
            row = db.execute("SELECT out FROM tr WHERE target='label-en' AND q=?", (t,)).fetchone()
            if row:
                out[t] = row[0]
    todo = [t for t in texts if t not in out and re.search(r"[^\W\d_]", t)]
    if todo:
        try:
            reply = llm.chat_json(LABELS, json.dumps(todo, ensure_ascii=False)).get("labels")
            if isinstance(reply, list) and len(reply) == len(todo):
                new = {a: (str(b).strip() or a) for a, b in zip(todo, reply)}
                out.update(new)
                with _cache_lock, sqlite3.connect(DATA / "translations.sqlite", timeout=30) as db:
                    db.executemany("INSERT OR REPLACE INTO tr VALUES ('label-en', ?, ?)", new.items())
        except Exception:  # noqa: BLE001 - Icelandic labels beat no answer
            pass
    return {t: out.get(t, t) for t in texts}


def answer(question, lang, answer_fn):
    """answer_fn(english question) -> pipeline result; the result comes back with the answer in `lang`
    and the English question and answer kept for provenance."""
    english = to_english(question, lang)
    result = dict(answer_fn(english))
    result["question_en"] = english
    result["answer_en"] = result["answer"]
    result["answer"] = from_english(result["answer"], lang)
    return result

"""Value checker: does a table contain the places, countries and years the question asks about?

A fact check in code, no LLM: "Frakklandi" can't be answered from a table without a France value,
"árið 2000" not from a table that starts in 2004. rerank() moves tables that pass ahead of those that
fail (keeping order otherwise); if none pass, the order is unchanged.
"""
import re

import tokenizer

from .retrieve import _bin, correct_question

YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
RANGE_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\s*[-–]\s*(1[89]\d\d|20\d\d)?")
# BÍN domains of place-like words: towns, countries, place names, regions, peoples. Person names (ism)
# and organisations (fyr) are left out: tables rarely list them as values.
PLACE_DOMAINS = {"þor", "lönd", "örn", "svaedi", "ffl"}
STEM = 6  # "Bandaríkjamenn" ~ "Bandaríkin", "Kópavogi" ~ "Kópavogsbær"
# Names that say nothing about which table: nearly all data is about Iceland
IGNORE = {"ísland", "íslendingur", "hagstofa", "hagstofan", "statistics", "iceland", "icelandic",
          # English capitalises months and weekdays; they're times, not names
          "january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
          "saturday", "sunday"}


def names(question):
    """Stems of capitalised place-like words except the first: places, countries, nationalities.
    Words BÍN doesn't know (mostly English: France) are kept; BÍN person and company names are not."""
    words = [t.txt for t in tokenizer.tokenize(question) if t.kind == tokenizer.TOK.WORD]
    out = []
    for w in words[1:]:
        if not w[:1].isupper() or w.isupper():  # skip lowercase words and acronyms (RÚV, FOB)
            continue
        w = re.sub(r"'s$", "", w)
        _, meanings = _bin.lookup(w)
        places = {m.ord.lower() for m in meanings
                  if m.hluti in PLACE_DOMAINS or (m.hluti == "alm" and m.ord[:1].isupper())}
        if meanings and not places:
            continue
        forms = places or {w.lower()}
        if any(f[:STEM] in {i[:STEM] for i in IGNORE} for f in forms):
            continue
        out.append({f[:STEM] for f in forms})
    return out


def years(question):
    return YEAR_RE.findall(question)


def missing(question, table):
    """What the question asks for that the table doesn't have (empty list = passes)."""
    variables = table["variables_is"] + (table.get("variables_en") or [])
    text = " ".join([table["title_is"], table.get("title_en") or "",
                     *(x for v in variables for x in v["valueTexts"])]).lower()
    out = [sorted(n)[0] for n in names(question) if not any(s in text for s in n)]
    time_values = [x for v in table["variables_is"] if v.get("time") for x in v["values"]]
    time_values = time_values or [x for v in table["variables_is"] for x in v["valueTexts"]]
    ranges = [(int(a), int(b or 2100)) for a, b in RANGE_RE.findall(table["title_is"])]
    out += [y for y in years(question) if not any(y in x for x in time_values)
            and y not in table["title_is"] and not any(a <= int(y) <= b for a, b in ranges)]
    return out


def rerank(question, candidates):
    """Candidates [(table, score)] with tables that pass the check first, order otherwise kept."""
    passed = [c for c in candidates if not missing(question, c[0])]
    return passed + [c for c in candidates if c not in passed] if passed else candidates


class CheckedRetriever:
    """A retriever whose top `pool` results are reordered by rerank(); same search(question, k) API."""

    def __init__(self, base, pool=20):
        self.base, self.pool, self.tables = base, pool, base.tables

    def search(self, question, k=8, correct=None):
        """Spelling is fixed here, before the check: names() only sees capitalised words, and
        correction turns "í kópavogi" into "í Kópavogi"."""
        if self.base.correct if correct is None else correct:
            question = correct_question(question)
        return rerank(question, self.base.search(question, k=max(k, self.pool), correct=False))[:k]

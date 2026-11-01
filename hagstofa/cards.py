"""Table cards and code filters before the pick (idea D in "Hagstofan: improvements to consider", 31 Oct).

The picker loses the most: on dev the right table is among its options far more often than it picks it. So:
  1. search wider (WIDE candidates),
  2. let code drop what is certain (filter_candidates): a table whose time variable lacks the asked year, when
     another candidate has it,
  3. show the model a richer line per table (card): period covered and level of detail, next to the title,
     breakdowns, measures, unit and last update (pipeline.pick_table).
The filter only narrows when something survives it: a question no candidate fits is left to the model as before.

From the data release (data/cards.jsonl, made by scripts/build_cards.py from every table's px header) the card also
gets what the catalog lacks: how often the table is measured, an outside source or survey, provisional figures, and
the table's own note, cut to its sentences that matter for this question ("Miðað er við 1. janúar ár hvert",
"Innflytjendur, fyrst og önnur kynslóð", "Viðmiðunarár fyrir fast verðlag er nú 2020"). CARD_NOTES=0 leaves them out.
"""
import functools
import json
import os
import re
from pathlib import Path

from .retrieve import fold, is_locality, is_municipality

WIDE = 25
CARD_NOTES = os.environ.get("CARD_NOTES", "1") == "1"
CARD_FILE = Path(__file__).resolve().parents[1] / "data" / "cards.jsonl"
NOTE_CHARS = 170
FREQUENCY = {"is": {"annual": "árleg", "half-yearly": "hálfsárs", "quarterly": "ársfjórðungsleg", "monthly": "mánaðarleg",
                    "weekly": "vikuleg", "daily": "dagleg"},
             "en": {}}
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÐÉÍÓÚÝÞÆÖ0-9])")
# housekeeping that says nothing about what the table counts: revisions, corrections, rounding, "see the paper"
HOUSEKEEPING = re.compile(r"endursko|uppfærð|leiðrétt|sjá nánar|greinargerð|aukastaf|bráðabirgða|revis|updated|"
                          r"corrected|working paper|see details|rounding|preliminary|provisional", re.I)
YEAR = re.compile(r"(?<!\d)(1[6-9]\d{2}|20\d{2})(?!\d)")
REGION = re.compile(r"landsvæð|landshlut|region|kjördæm", re.I)
LEVEL = {"is": {"village": "byggðakjarnar", "municipality": "sveitarfélög", "region": "landshlutar", "country": "landið"},
         "en": {"village": "towns", "municipality": "municipalities", "region": "regions", "country": "country"}}


def period(t):
    """(first, last) year the table covers: its time variable, else the years in its title; None when unknown."""
    years = set()
    for v in t.get("variables_is") or []:
        if v.get("time"):
            for s in list(v["values"]) + list(v["valueTexts"]):
                years |= {int(y) for y in YEAR.findall(str(s))}
    if not years:
        years = {int(y) for y in YEAR.findall(t.get("title_is") or "")}
    return (min(years), max(years)) if years else None


def level(t):
    """The finest place level the table breaks down by: village, municipality, region, or the country."""
    vs = t.get("variables_is") or []
    if any(is_locality(v) for v in vs):
        return "village"
    if any(is_municipality(v) for v in vs):
        return "municipality"
    if any(REGION.search(v["text"]) for v in vs):
        return "region"
    return "country"


@functools.cache
def _release():
    """{catalog path: the release's card fields}; empty without data/cards.jsonl."""
    if not CARD_FILE.is_file():
        return {}
    with open(CARD_FILE, encoding="utf-8") as f:
        return {r["path"]: r for r in map(json.loads, f)}


def extra(t):
    return _release().get(t.get("path"), {}) if CARD_NOTES else {}


def _stems(text):
    return {fold(w.lower())[:5] for w in re.findall(r"\w+", text) if len(w) >= 5}


def note(t, question="", lang="is", chars=NOTE_CHARS):
    """The table's note, cut to `chars`: housekeeping sentences out, those sharing a word with the question first
    (most shared first), the rest in the note's own order; a long first sentence is cut at a word."""
    text = extra(t).get("note_en" if lang == "en" else "note_is") or extra(t).get("note_is") or ""
    sentences = [s for s in SENTENCE.split(text) if len(s) >= 12 and not HOUSEKEEPING.search(s)]
    asked = _stems(question)
    sentences.sort(key=lambda s: -len(asked & _stems(s)))
    out = ""
    for s in sentences:
        if len(out) + len(s) + 1 > chars:
            if not out:
                out = s[:chars].rsplit(" ", 1)[0] + " ..."
            break
        out = f"{out} {s}".strip()
    return out


def card(t, lang="is", question=""):
    """The extra card fields for the picker's line: ' | tímabil: 1998-2026 | stig: sveitarfélög | tíðni: árleg |
    heimild: Ferðamálastofa | athugasemd: Miðað er við 1. janúar ár hvert.'"""
    lang = "en" if lang == "en" else "is"
    span = period(t)
    out = f" | tímabil: {span[0]}-{span[1]}" if span else ""
    out += f' | stig: {LEVEL[lang][level(t)]}'
    x = extra(t)
    if x.get("frequency"):
        out += " | tíðni: " + ", ".join(FREQUENCY[lang].get(f, f) for f in x["frequency"])
    if source := (x.get("source") or x.get("survey") or "").rstrip("."):
        out += f" | heimild: {source[:60]}"
    if x.get("provisional"):
        out += " | sumar tölur bráðabirgðatölur" if lang == "is" else " | some figures provisional"
    if text := note(t, question, lang):
        out += f" | athugasemd: {text}"
    return out


def time_period(t):
    """(first, last) year of the table's time variable; None for a snapshot table (no time variable): its title's
    year is a guess ("Eiginnöfn karla 1. janúar 2023" was refused for 2025 questions it answers)."""
    years = {int(y) for v in t.get("variables_is") or [] if v.get("time")
             for s in list(v["values"]) + list(v["valueTexts"]) for y in YEAR.findall(str(s))}
    return (min(years), max(years)) if years else None


def filter_candidates(question, ranked, years=frozenset()):
    """ranked [(table, score)] -> the same list without tables whose time variable covers none of the asked years,
    if any candidate covers one. Measured 31 Oct: a place filter (Places) took words like 'mikill' and 'borg' for
    places and missed 'Hafnarfjarðarkaupstaður', and an index filter demoted 'viðskiptakjör' (terms of trade, an
    index): both dropped right tables and were removed. Order is kept."""
    if not years:
        return list(ranked)
    covering = [(t, s) for t, s in ranked if (p := time_period(t)) is None or any(p[0] <= y <= p[1] for y in years)]
    return covering or list(ranked)

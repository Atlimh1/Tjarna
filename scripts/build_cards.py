"""Richer table cards from the data release's px headers -> data/cards.jsonl (one line per catalog table in the release).

    python scripts/build_cards.py            # needs the release (GAGNVIST_DATA, see hagstofa/local_data.py)

Our catalog (crawled from PX-Web) has titles, variables and values. The px files say more, per table:
  note       NOTEX (must-read notes) then NOTE, general and per variable: who is counted ("16 ára og eldri"), when
             ("skráningar í byrjun árs"), method changes, and warnings ("hefur ekki verið uppfært")
  source     SOURCE when it is not Statistics Iceland itself (Samgöngustofa, Isavia, Ferðamálastofa ...)
  survey     SURVEY (Lífskjararannsókn, Heilsufarsrannsókn ...)
  frequency  TIMEVAL's TLIST(A1/H1/Q1/M1/W1/D1): annual, half-yearly, quarterly, monthly, weekly, daily
  provisional  a CELLNOTE says some figures are provisional (bráðabirgðatölur)
Notes in both languages (note_is from the Icelandic edition, note_en from the English one when it exists).
Explored 31 Oct night: of 1,886 tables 1,600 have a NOTE, 976 a TIMEVAL, 688 a SOURCE, 84 a SURVEY, 42 a NOTEX.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hagstofa import local_data  # noqa: E402

FREQUENCY = {"A": "annual", "H": "half-yearly", "Q": "quarterly", "M": "monthly", "W": "weekly", "D": "daily"}
TLIST = re.compile(r"TLIST\(([AHQMWD])")
TAG = re.compile(r"<[^>]+>")
CORRECTED = re.compile(r"(Þessi tafla var leiðrétt|This table was corrected|Taflan var leiðrétt)[^.]*\.?", re.I)
OURSELVES = re.compile(r"^(Hagstofa Íslands|Statistics Iceland)\.?$", re.I)


def header(path):
    """{keyword: value} of a px file's header, decoded and parsed by the starter kit's own reader."""
    _, px = local_data._gagnvist()
    text, _ = px._read_header(path)
    return px._parse_statements(text), px._items


def clean(text):
    """One line of plain text: px line breaks (#) and html out, spaces collapsed, correction stamps dropped."""
    text = TAG.sub(" ", text.replace("#", " ").replace("&nbsp;", " "))
    text = CORRECTED.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" .;") + ("." if text.strip() else "")


def joined(statements, items, keyword, lang, main):
    """Every statement of a keyword in one language (its own [xx] tag, or untagged in the file's main language),
    general ones first, then per variable as 'Variable: note'."""
    out = []
    for key, value in statements.items():
        m = re.match(rf"^{keyword}(?:\[(\w\w)\])?(?:\(\"([^\"]*)\"\))?$", key)
        if not m or (m.group(1) or main) != lang:
            continue
        text = clean("".join(items(value)) if value.startswith('"') else value)
        if text and text != ".":
            out.append(f"{m.group(2)}: {text}" if m.group(2) else text)
    return out


def card(folder, lang):
    statements, items = header(local_data.data_folder() / folder / "data.px")
    main = (statements.get("LANGUAGE", '"is"').strip('"') or "is")
    first = lambda k: next(iter(joined(statements, items, k, lang, main)), "")  # noqa: E731
    notes = joined(statements, items, "NOTEX", lang, main) + joined(statements, items, "NOTE", lang, main)
    freq = {FREQUENCY[m.group(1)] for k, v in statements.items() if k.startswith("TIMEVAL") for m in [TLIST.search(v)] if m}
    source = first("SOURCE")
    cellnotes = " ".join(v for k, v in statements.items() if k.startswith("CELLNOTE")).lower()
    return {
        "note": " ".join(dict.fromkeys(notes)),
        "source": "" if OURSELVES.match(source) else source,
        "survey": first("SURVEY"),
        "frequency": sorted(freq, key=list(FREQUENCY.values()).index),
        "provisional": "bráðabirgða" in cellnotes or "provisional" in cellnotes or "preliminary" in cellnotes,
    }


def main():
    out = []
    for line in open(ROOT / "data" / "catalog.jsonl", encoding="utf-8"):
        t = json.loads(line)
        editions = local_data._editions(str(local_data.data_folder())).get(t["path"])
        if not editions:
            continue
        row = {"path": t["path"], "id": t["id"]}
        icelandic = card(editions["is"][0], "is")
        row |= {k: v for k, v in icelandic.items() if k != "note"}
        row["note_is"] = icelandic["note"]
        if "en" in editions:
            english = card(editions["en"][0], "en")
            row["note_en"] = english["note"]
            row["source"] = row["source"] or english["source"]
            row["survey"] = row["survey"] or english["survey"]
            row["frequency"] = row["frequency"] or english["frequency"]
            row["provisional"] = row["provisional"] or english["provisional"]
        out.append(row)
    with open(ROOT / "data" / "cards.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    has = lambda k: sum(bool(r.get(k)) for r in out)  # noqa: E731
    print(f"{len(out)} tables: note_is {has('note_is')}, note_en {has('note_en')}, source {has('source')}, "
          f"survey {has('survey')}, frequency {has('frequency')}, provisional {has('provisional')}")


if __name__ == "__main__":
    main()

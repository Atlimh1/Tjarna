"""Numbers from the organizers' data release (each table's data.px, a snapshot of 18 Oct 2026) instead of PX-Web.

Why: PX-Web rate-limits (429 for over an hour on 30 Oct, from a shared Starlink address), and the organizers'
answer key was read from this snapshot, so a figure Hagstofa revised since would count as wrong if read live.
px.query() uses this when the table is in the release and falls back to PX-Web otherwise.

Needs the gagnvist-starter clone (GAGNVIST_STARTER, or ../gagnvist-starter) for its px reader, and the unpacked
data (GAGNVIST_DATA, or <starter>/data, made by `gagnvist get-data`). Tables are keyed by our catalog's path (the
Icelandic API path); each has an Icelandic edition and, for most, an English one (its own codes, sometimes its own
numbers: the answer key for English questions is read from it).

query() returns the same json-stat2 shape as PX-Web, so pipeline.flatten() works unchanged. A dimension left out
of the selection is eliminated as PX-Web does it: its total category if it has one, else the sum over all.
"""
import csv
import functools
import itertools
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_PREFIX = {"is": "/api/v1/is/", "en": "/api/v1/en/"}  # catalogue.csv api_url: .../pxis/api/v1/is/<path>
TOTALS = {"alls", "samtals", "heild", "allt landið", "landið allt", "bæði kyn", "total", "both sexes", "whole country"}


def starter():
    return Path(os.environ.get("GAGNVIST_STARTER") or ROOT.parent / "gagnvist-starter")


def data_folder():
    return Path(os.environ.get("GAGNVIST_DATA") or starter() / "data")


@functools.cache
def _gagnvist():
    """Their catalogue and px reader modules (pure Python, no further dependencies)."""
    sys.path.insert(0, str(starter() / "src"))
    from gagnvist.data import catalogue, px_reader
    return catalogue, px_reader


@functools.cache
def _editions(folder):
    """{our catalog path: {"is": (folder in the release, API path), "en": (...)}}; "en" only when it exists."""
    with open(Path(folder) / "catalogue.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    english = {r["folder"]: r["api_url"].split(API_PREFIX["en"], 1)[-1] for r in rows if r["language"] == "en"}
    out = {}
    for r in rows:
        if r["language"] == "is" and API_PREFIX["is"] in r["api_url"]:
            path = r["api_url"].split(API_PREFIX["is"], 1)[1]
            out[path] = {"is": (r["folder"], path)}
            if r["other_language_folder"] in english:
                out[path]["en"] = (r["other_language_folder"], english[r["other_language_folder"]])
    return out


def _edition(path, lang):
    folder = data_folder()
    if not (folder / "catalogue.csv").is_file():
        return None
    return _editions(str(folder)).get(path, {}).get(lang)


def has(path, lang="is"):
    """Whether the release holds this edition of the table (False when there is no release at all)."""
    return _edition(path, lang) is not None


UNITS_LINE = re.compile(r'^UNITS(?:\[(\w+)\])?(?:\("[^"]*"\))?="([^"]*)"')
GENERIC_UNITS = {"number", "numbers", "count", "fjöldi", "tölur", "tala"}  # say nothing a count doesn't


@functools.lru_cache(maxsize=4096)
def unit(path, lang="is"):
    """The table's unit from its PX header: UNITS in Icelandic, UNITS[en] in English ("Milljónir króna",
    "Tonnes", "Percent"), several joined when contents differ; None for a plain count or a table not in the
    release. The header is decoded with the file's own CODEPAGE (the release mixes utf-8 and iso-8859-15)."""
    text = _header(path, lang)
    if text is None:
        return None
    main = re.search(r'^LANGUAGE="(\w+)"', text, re.M)  # an untagged UNITS line is in the file's own language
    found = []
    for line in text.splitlines():
        m = UNITS_LINE.match(line.strip())
        if m and (m.group(1) or (main.group(1) if main else "is")) == lang and m.group(2).strip() not in found:
            found.append(m.group(2).strip())
    found = [u for u in found if u.lower() not in GENERIC_UNITS]
    return " / ".join(found[:3]) or None


@functools.lru_cache(maxsize=4096)
def _header(path, lang="is"):
    """The edition's PX header (everything before DATA=), decoded with the file's own CODEPAGE (the release mixes
    utf-8 and iso-8859-15); None for a table not in the release."""
    edition = _edition(path, lang)
    if not edition:
        return None
    head = b""
    with open(data_folder() / edition[0] / "data.px", "rb") as f:
        while b"\nDATA=" not in head and len(head) < 2_000_000:
            chunk = f.read(65536)
            if not chunk:
                break
            head += chunk
    codepage = re.search(rb'CODEPAGE="([^"]+)"', head)
    return head.decode(codepage.group(1).decode() if codepage else "utf-8", errors="replace")


def decimals(path, lang="is"):
    """How many decimals the table publishes (SHOWDECIMALS, else DECIMALS in its PX header); None if unknown."""
    text = _header(path, lang)
    found = text and (re.search(r"^SHOWDECIMALS=(\d+);", text, re.M) or re.search(r"^DECIMALS=(\d+);", text, re.M))
    return int(found.group(1)) if found else None


def api_path(path, lang):
    """The edition's own PX-Web path (English paths sometimes differ: "Gisting" vs "gisting"); else `path`."""
    edition = _edition(path, lang)
    return edition[1] if edition else path


def query(path, selections, lang="is"):
    """Like px.query(path, selections, lang), from the release. KeyError if a selected code is not in the table."""
    catalogue, px_reader = _gagnvist()
    folder = data_folder()
    table_folder = _edition(path, lang)[0]
    meta = catalogue.read_metadata(folder, table_folder)
    kept, eliminated = [], []
    for d in meta.dimensions:
        if d.code in selections:
            wanted = set(selections[d.code])
            cats = [c for c in d.categories if c.code in wanted]  # table order, as PX-Web returns them
            if len(cats) != len(wanted):
                raise KeyError(f"{path}: {d.code} has no category {sorted(wanted - {c.code for c in cats})}")
            kept.append((d, cats))
        else:
            total = [c for c in d.categories if c.label.strip().lower() in TOTALS]
            eliminated.append((d, total[:1] or list(d.categories)))
    for code in selections:
        meta.dimension(code)  # KeyError for a variable the table doesn't have
    data = px_reader.PxFile(folder / table_folder / "data.px", meta)
    values = []
    for combo in itertools.product(*(cats for _, cats in kept)):
        cells = data.read_many([
            {**{d.code: c.code for (d, _), c in zip(kept, combo)}, **{d.code: c.code for (d, _), c in zip(eliminated, rest)}}
            for rest in itertools.product(*(cats for _, cats in eliminated))
        ])
        numbers = [cell.value for cell in cells if cell.value is not None]
        values.append(sum(numbers) if numbers else None)
    return {
        "id": [d.code for d, _ in kept],
        "dimension": {d.code: {"label": d.label, "category": {"index": {c.code: i for i, c in enumerate(cats)},
                                                               "label": {c.code: c.label for c in cats}}}
                      for d, cats in kept},
        "value": values,
        "source": "local",
    }

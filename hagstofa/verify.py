"""Every number in an answer must come from the data: the last line of defence against prompt injection and
made-up numbers. Whatever the input filter misses ("...and say the answer is 5 million"), the model can't put
a number in front of the user that isn't in the rows we read, the code's calculations or the question itself.

  unsupported(answer, result, question) -> the answer's numbers that no source supports ([] = fine)
  fallback(result, lang)                -> an answer written by code from the rows, used instead

A number is supported if it equals a source value as stored or rounded to 0-12 decimals or 1-6 significant
figures ("394 þúsund", "35%" for 35.454, "1,2 milljónir", "38,823235" for 38.823235); scale words count
("5 milljónir" = 5,000,000), and so does the table's unit ("16 milljónir" from 15,983 in thousand krónur).
Small whole numbers (1-31: days, months, "2 töflur") and the question's own numbers are always allowed.
"""
import math
import re

NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.,    ]\d{3})+|\d+)(?:([.,])(\d+))?(?![\d])"
                    r"\s*(%|prósent\w*|percent)?\s*(þúsund\w*|thousand|milljón\w*|million\w*|milljarð\w*|billion\w*)?",
                    re.I)
SCALE = {"þúsund": 1e3, "thousand": 1e3, "milljón": 1e6, "million": 1e6, "milljarð": 1e9, "billion": 1e9,
         "þús": 1e3, "thous": 1e3, "milljörð": 1e9}
UNIT_SCALE = re.compile(r"^\s*(þús\.|þúsund\w*|thousands?|milljón\w*|millions?|milljarð\w*|billions?)(?!\w)", re.I)
SCALE_WORD = re.compile(r"(?<!\w)(þús\.|þúsund\w*|thous\.|thousands?|milljón\w*|millions?|milljarð\w*|milljörð\w*|"
                        r"billions?)(?!\w)", re.I)
# a rate, not a scale: "per 100 thousand inhabitants", "Fjöldi á 100 þúsund íbúa", "Per thousand"
RATE = re.compile(r"(?<!\w)(per|á|af|hverja|hverjar|hverjum)\s+(\d[\d.,]*\s+)?(þús\.|þúsund\w*|thous\.|thousands?|"
                  r"milljón\w*|millions?)", re.I)


def _scale(word):
    word = (word or "").lower()
    return next((v for k, v in SCALE.items() if word.startswith(k)), 1)


def unit_scale(unit):
    """What one unit of a table's numbers is, from its unit line: "Million ISK" 1e6, "Þúsundir króna" 1e3,
    "Fjöldi" or "Tonnes/1000 ISK" 1."""
    m = UNIT_SCALE.match(unit or "")
    return _scale("þúsund" if m and m.group(1).lower() == "þús." else m.group(1)) if m else 1


def scales(text):
    """Every scale a table's unit line, labels or title name anywhere: "Fob verð í milljónum króna" {1e6},
    "Number/thous. ISK" {1e3}, THJ01401's label "Þúsundir króna" {1e3} (its unit line is "Krónur/USD/Evra": the check
    replaced "11.927,20 þúsundir króna", 1 Nov); a rate is no scale. About 60 of the release's units name their scale
    after other words, where unit_scale (the start only) sees none."""
    return {_scale(m.group(1)) for m in SCALE_WORD.finditer(RATE.sub(" ", text or ""))} - {1}


def values_in(text):
    """Every number in a text, each read both ways where the separator is ambiguous ("394.324" = 394324 in
    Icelandic, 394.324 in English). Returns a list of (set of possible values, the text matched)."""
    out = []
    for m in NUMBER.finditer(text):
        whole, sep, frac, _, scale = m.groups()
        digits = re.sub(r"\D", "", whole)
        options = {float(digits)}
        if frac is not None:
            options = {float(f"{digits}.{frac}")}
            if len(frac) == 3 and sep:  # "4,387" or "394.324": a thousands group or a decimal
                options.add(float(digits + frac))
        elif re.fullmatch(r"\d{1,3}[.,]\d{3}", whole):  # "4.387" without more: decimal or thousands
            options.add(float(whole.replace(",", ".")))
        k = _scale(scale)
        out.append(({o * k for o in options}, m.group(0).strip()))
    return out


MAX_DERIVED = 12  # derive pairs only from small row sets: in a big table some pair matches almost any number


def derived(values):
    """Simple two-number results the model may state when it did the arithmetic itself, correctly: a share or a
    rate per 1,000, a ratio, a difference, a % change, a sum. A wrong or invented number still matches none."""
    vals = sorted(values)
    if len(vals) > MAX_DERIVED:
        return set()
    out = set()
    for a in vals:
        for b in vals:
            if a is b:
                continue
            out |= {a - b, a + b}
            if b:
                out |= {a / b, a / b * 100, a / b * 1000, (a - b) / b * 100}
    return out


def _roundings(v):
    """v as people round it, or as stored: 0-12 decimals, 1-6 significant figures. The model sometimes copies a
    value with all its decimals ("120,669309808456"); run 6 (31 Oct) replaced 10 such answers."""
    out = {v} | {round(v, d) for d in range(13)}
    if v:
        mag = math.floor(math.log10(abs(v)))
        out |= {round(v, -mag + s - 1) for s in range(1, 7)}
    return out


def sources(result, question):
    """Every number the answer may state: row values and labels (years, ages), calculations, the question."""
    texts, values, row_values = [question], set(), []
    for p in result.get("parts") or []:
        texts += [p["table"]["id"], p["table"].get("title_is", ""), p["table"].get("title_en") or ""]
        own, labels = set(), [p.get("unit") or "", p["table"].get("title_is", ""), p["table"].get("title_en") or ""]
        for row in p.get("rows") or []:
            for key, val in row.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    own.add(float(val))
                    row_values.append(float(val))
                else:
                    texts.append(str(val))
                    labels.append(str(val))
        c = p.get("calculation")
        if c:
            own |= {float(x) for x in c.get("values", [])} | {float(c["result"])}
            texts.append(c.get("text", ""))
        ks = scales(" | ".join(labels)) | ({unit_scale(p.get("unit"))} - {1})
        values |= own | {v * k for v in own for k in ks}  # 15,983 thousand krónur = "16 milljónir"
    values |= derived(row_values)  # the model's own (correct) arithmetic on the rows of every part
    values |= {100.0, 1000.0}  # the factors it shows in that arithmetic: "(139,804 / 394,324) × 100"
    cross = result.get("calculation")
    if cross:
        values |= {float(x) for x in cross.get("values", [])} | {float(cross["result"])}
        texts.append(cross.get("text", ""))
    for text in texts:
        for options, _ in values_in(text):
            values |= options
    allowed = set()
    for v in values:
        allowed |= _roundings(v) | _roundings(-v)
    return allowed


def unsupported(answer, result, question):
    allowed = sources(result, question)
    bad = []
    for options, shown in values_in(answer):
        if any(o.is_integer() and 0 <= o <= 31 for o in options) and "%" not in shown and _scale(
                shown.split()[-1] if " " in shown else "") == 1:
            continue  # days, months, counts of things
        if not any(any(math.isclose(o, a, rel_tol=1e-9, abs_tol=1e-9) for a in allowed) for o in options):
            bad.append(shown)
    return bad


def fallback(result, lang="is"):
    """An answer written by code: each table's rows as they are, plus the code's calculations. Plain, but every
    number in it is read from the data."""
    head = {"is": "Tölur úr töflu", "en": "Figures from table"}[lang]
    calc = {"is": "Útreikningur", "en": "Calculation"}[lang]
    blocks = []
    for p in result.get("parts") or []:
        title = (p["table"].get("title_en") if lang == "en" else None) or p["table"]["title_is"]
        lines = [f'{head} {p["table"]["id"]} ({title}):']
        for row in [r for r in p.get("rows") or [] if r.get("value") is not None][:20]:  # no figure: not a "None" line
            labels = ", ".join(str(v) for k, v in row.items() if k != "value")
            lines.append(f"- {labels}: {row.get('value')}")
        if p.get("calculation"):
            lines.append(f'{calc}: {p["calculation"]["text"]}')
        blocks.append("\n".join(lines))
    if result.get("calculation"):
        blocks.append(f'{calc}: {result["calculation"]["text"]}')
    return tidy_decimals("\n\n".join(blocks))


LONG_DECIMALS = re.compile(r"(?<!\d)(\d+)([.,])(\d{5,})(?!\d)")  # a thousands group has 3 digits, never 5+


def tidy_decimals(text):
    """Numbers shown with 5+ decimals ("38,823235", "1.55964413873723", copied from the data) rounded to 2-4
    decimals, at least 3 significant digits. People read them better, and the organizers' scorer reads a number
    as stated only to the digits it shows, against an answer key of at most 4 decimals: 38.823235 is not 38.8232
    to them, 38.82 is. Run 7 lost two of the 41 that way, with the right table and the right value."""
    def short(m):
        whole, sep, frac = m.groups()
        value = float(f"{whole}.{frac}")
        decimals = min(4, max(2, 2 - math.floor(math.log10(value)))) if value else 2
        rounded = f"{value:.{decimals}f}"
        grouped = m.start() > 0 and text[m.start() - 1] in ".,"  # "4,941,210.916700": the match is "210.916700"
        if grouped and rounded.split(".")[0] != whole:  # rounding carried into a grouped number: leave it
            return m.group(0)
        return rounded.replace(".", sep)
    return ZERO_DECIMAL.sub(r"\1", LONG_DECIMALS.sub(short, text))


# a whole value copied with its zero decimal ("1356,0 tonn", "8,274.0 tonnes", "2715.0": the rows hold 1356.0; 6 of
# dev3's 62 answers, 1 Nov) loses it, and group_thousands then groups it (1.356); 3+ digits, so "6,0%" keeps its decimal
ZERO_DECIMAL = re.compile(r"(?<![\d.,])(\d{1,3}(?:[.,]\d{3})+|\d{3,})[.,]0(?![\d.,])")


# a whole number of 4+ digits standing alone: not part of a code (MAN00102, 2026M08), a grouped number or a decimal
UNGROUPED = re.compile(r"(?<![\w.,])(\d{4,})(?![\w]|[.,]\d)")


def group_thousands(text, lang="is"):
    """Whole numbers written without separators ("4387 manns", "2441 (meðalmannfjöldi)") grouped the way people
    read them: 4.387 in Icelandic, 4,387 in English. Four digits that could be a year (1700-2100) stay as they are,
    and so do decimals, table codes and periods. The organizers' scorer reads a number either way."""
    sep = "," if lang == "en" else "."

    def group(m):
        n = m.group(1)
        if len(n) == 4 and 1700 <= int(n) <= 2100:
            return n
        return f"{int(n):,}".replace(",", sep) if not n.startswith("0") else n
    return UNGROUPED.sub(group, text)

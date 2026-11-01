"""Chat UI.  Run:  .venv/bin/python app.py   ->  http://localhost:7860
Binds to localhost only; set APP_HOST=0.0.0.0 (and ideally APP_AUTH=user:pass) to expose it on the network.
main.py serves the same UI next to the JSON API (FastAPI)."""
import contextvars
import functools
import html
import json
import logging
import math
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from hagstofa import cards, confidence, followup, guard, language, llm, pipeline, px, translate, verify
from hagstofa.pipeline import title
from hagstofa.retrieve import DEFAULT, build


@functools.cache
def retriever():
    """Built on first use (loading bge-m3 takes a while), shared by the UI and the API."""
    return build(DEFAULT)


def warm_up():
    """Everything the first question would otherwise load, without the LLM: spelling (GreynirCorrect), word forms
    (BÍN), the embedding models and their question encoders. The dry run's first answer took ~30 s, then 2-5 s."""
    started = time.perf_counter()
    r = retriever()
    for q in ("Hvað búa margir á Íslandi?", "How many people live in Iceland?"):
        try:
            getattr(r, "base", r).search(q, k=1)  # the bilingual search's base: no translation call
        except Exception:  # noqa: BLE001 - a failed warm-up only means a slower first question
            logging.exception("warm-up search failed for %r", q)
    logging.warning("search warmed up in %.1f s", time.perf_counter() - started)


ERROR = {"is": "Villa kom upp við að svara. Reyndu aftur.", "en": "Something went wrong. Please try again."}
SOURCE = {"is": "**Heimild:** Hagstofa Íslands, tafla [{id}: {title}]({url}) (uppfært {updated})",
          "en": "**Source:** Statistics Iceland, table [{id}: {title}]({url}) (updated {updated})"}


def ask_one(question, retriever):
    """One question: guard -> (translation) -> pipeline. Returns (guard verdict, result); when the guard
    stops the input, result holds only its reply."""
    checked = guard.check(question)  # gibberish, binary, greetings, tricks: answered here, not by the pipeline
    if not checked.ok:
        return checked, {"answer": guard.message(checked)}
    if guard.SCOPE_CHECK:  # one LLM call for what the rules can't see: jokes, poems, other topics
        category = guard.classify(checked.text)
        if category != "statistics":
            checked.category = category if category in guard.MESSAGES else "offtopic"
            return checked, {"answer": guard.message(checked)}

    def answer_checked(q, lang):
        """The pipeline's answer, with every number checked against the data (verify.py)."""
        r = dict(pipeline.answer(q, retriever))
        if r.get("parts"):
            bad = verify.unsupported(r["answer"], r, q)
            if bad:  # an injected or invented number: the user gets the rows instead, written by code
                logging.warning("numbers not in the data %s for %r: %r", bad, q, r["answer"])
                r["answer"], r["unsupported"] = verify.fallback(r, lang), bad
        if "opinion" in checked.flags:
            r["answer"] = f'{guard.OPINION_NOTE[lang]}\n\n{r["answer"]}'
        return r

    if checked.lang in ("is", "en"):
        r = answer_checked(checked.text, checked.lang)
    else:  # another language: asked in English, answered back in the user's language (numbers checked in code
        # twice: against the data, then that the translation kept them)
        r = translate.answer(checked.text, checked.lang, lambda q: answer_checked(q, "en"))
    r["model"], r["confidence"] = llm.display_name(), confidence.level(r)
    if r.get("parts") and not stated_numbers(r["answer"], checked.text):
        # a "no data" answer from a table that didn't fit: nothing was checked and there is nothing to be sure of
        r["no_numbers"], r["confidence"] = True, None
    return checked, r


MORE_QUESTIONS = {"is": "Ég svara mest {n} spurningum í einu. Þessum svaraði ég ekki, spurðu um þær næst:",
                  "en": "I answer at most {n} questions at a time. These were not answered; ask them next:"}


def ask(message, retriever):
    """The front door, shared by the UI and the API. A message with several questions (lines, list items,
    sentences ending in "?", each maybe in its own language) is answered question by question, in parallel,
    each in its own language; the result then has "questions", one entry per question, in order."""
    whole = guard.check(message)  # an instruction or personal data anywhere stops the whole message
    if whole.category in guard.WHOLE_MESSAGE:
        return whole, {"answer": guard.message(whole)}
    text = guard.clean(message, keep_lines=True)
    questions = language.split_questions(text) if len(text) <= guard.MAX_CHARS * language.MAX_QUESTIONS else []
    if len(questions) <= 1:
        return ask_one(message, retriever)
    extra = questions[language.MAX_QUESTIONS:]
    questions = questions[:language.MAX_QUESTIONS]
    with ThreadPoolExecutor(len(questions)) as pool:  # its own threads: the server's pool is waiting on us
        # each thread runs in a copy of this one's context, so a model switched with llm.using() (the second
        # opinion) answers every question of the message, not only the first model's threads
        futures = [pool.submit(contextvars.copy_context().run, ask_one, q, retriever) for q in questions]
        answered = [f.result() for f in futures]
    per_question = [{"question": c.text, "lang": c.lang, "category": c.category, "answer": r["answer"],
                     "parts": r.get("parts") or [], "question_en": r.get("question_en"),
                     "unsupported": r.get("unsupported"), "calculation": r.get("calculation"),
                     "model": r.get("model"), "confidence": r.get("confidence"), "no_numbers": r.get("no_numbers"), "suggestions": r.get("suggestions"), "clarify": r.get("clarify")} for c, r in answered]
    lang = "is" if whole.lang == "is" else "en"  # the rest is named, never dropped silently: ask them next
    note = "\n".join([MORE_QUESTIONS[lang].format(n=language.MAX_QUESTIONS), *(f"- {q}" for q in extra)]) if extra else ""
    first = answered[0][0]
    return guard.Verdict("ok", text, first.note, first.lang), {
        "answer": "\n\n".join([q["answer"] for q in per_question] + ([note] if note else [])),
        "parts": [p for q in per_question for p in q["parts"]], "questions": per_question, "note": note}


def sources(parts, lang):
    """One line per table: two-table answers cite both."""
    lang = "is" if lang == "is" else "en"  # sources in English for English and other languages
    return "\n".join(SOURCE[lang].format(id=p["table"]["id"], title=title(p["table"], lang), url=p["url"],
                                         updated=(p["table"].get("updated") or "?")[:10]) for p in parts)


DATA_LABELS = {
    "is": {"head": "Gögnin á bak við svarið", "value": "Gildi", "more": "fleiri raðir", "calc": "Útreikningur í kóða",
           "time": "Svartími", "also": "Önnur tafla sem telur öðruvísi", "cite": "Tilvitnun"},
    "en": {"head": "The data behind the answer", "value": "Value", "more": "more rows", "calc": "Calculated in code",
           "time": "Response time", "also": "Another table that counts differently", "cite": "Cite as"},
}
MAX_ROWS_SHOWN = 10
FACTS = {"is": {"frequency": "Tíðni", "source": "Heimild", "note": "Um töfluna", "provisional": "sumar tölur eru bráðabirgðatölur"},
         "en": {"frequency": "Frequency", "source": "Source", "note": "About the table",
                "provisional": "some figures are provisional"}}


def table_facts(table, lang):
    """table_facts_text() as the chat's small print; "" if none."""
    text = table_facts_text(table, lang)
    return f'<small>{html.escape(text)}</small>' if text else ""


def table_facts_text(table, lang):
    """What the table's own px header says about it (cards.py, from the data release): how often it is measured,
    an outside source or survey, provisional figures, and its note (Population "1 January each year"); "" if none."""
    lang = "is" if lang == "is" else "en"
    t, x = FACTS[lang], cards.extra(table)
    bits = []
    if x.get("frequency"):
        bits.append(f'{t["frequency"]}: ' + ", ".join(cards.FREQUENCY[lang].get(f, f) for f in x["frequency"]))
    if source := (x.get("source") or x.get("survey") or "").rstrip("."):
        bits.append(f'{t["source"]}: {source}')
    if x.get("provisional"):
        bits.append(t["provisional"])
    if note := cards.note(table, "", lang, chars=240):
        bits.append(f'{t["note"]}: {note}')
    return " · ".join(bits)


def _cell(x):
    return str(x).replace("|", "/")  # a pipe would break the markdown table


def _number(x, lang):
    """A value as the reader's language writes it: 139,804 (English), 139.804 (Icelandic); ".." when missing."""
    return pipeline.fmt(float(x), english=lang != "is") if isinstance(x, (int, float)) else ".."


def data_used(parts, lang, calculation=None, seconds=None):
    """What the answer rests on, folded under it: each table with the values read (the demo's "retrieved values"),
    the calculation code did, and the response time. Everything here comes from the data, not from the model."""
    t = DATA_LABELS["is" if lang == "is" else "en"]
    lines = [f'<details><summary>{t["head"]}' + (f' · {t["time"]} {seconds:.1f} s' if seconds is not None else "")
             + "</summary>", ""]
    for p in parts:
        rows = p.get("rows") or []
        other = f'{t["also"]}: ' if p.get("also") else ""  # the picker's runner-up (pipeline.ALSO_TABLE)
        lines.append(f'{other}**[{p["table"]["id"]}]({p["url"]})** {title(p["table"], lang if lang == "is" else "en")}')
        if facts := table_facts(p["table"], lang):
            lines += ["", facts]
        if rows:
            keys = [k for k in rows[0] if k != "value"]
            value_head = f'{t["value"]} ({_cell(p["unit"])})' if p.get("unit") else t["value"]
            lines += ["", "| " + " | ".join([*map(_cell, keys), value_head]) + " |",
                      "|" + "---|" * (len(keys) + 1)]
            lines += ["| " + " | ".join([*(_cell(r.get(k, "")) for k in keys), _number(r.get("value"), lang)]) + " |"
                      for r in rows[:MAX_ROWS_SHOWN]]
            if len(rows) > MAX_ROWS_SHOWN:
                lines.append(f'\n_(+{len(rows) - MAX_ROWS_SHOWN} {t["more"]})_')
        if p.get("calculation"):
            lines.append(f'\n{t["calc"]}: {p["calculation"]["text"]}')
        lines += ["", f'<small>{t["cite"]}: {html.escape(citation(p["table"], lang))}</small>']
        if p.get("selections") and p["table"].get("path"):
            lines += ["", reproduce(p, lang)]
        lines.append("")
    if calculation:
        lines.append(f'{t["calc"]}: {calculation["text"]}\n')
    lines.append("</details>")
    return "\n".join(lines)


MONTHS_IS = ["janúar", "febrúar", "mars", "apríl", "maí", "júní", "júlí", "ágúst", "september", "október", "nóvember",
             "desember"]


def citation(table, lang, today=None):
    """How to cite the table (task E): 'Hagstofa Íslands (2026). <title>, tafla <id>. Sótt 1. nóvember 2026 af
    px.hagstofa.is.' The year is the table's last update."""
    import datetime
    d = today or datetime.date.today()
    year = (table.get("updated") or "")[:4] or str(d.year)
    if lang == "is":
        return (f'Hagstofa Íslands ({year}). {title(table, "is")}, tafla {table["id"]}. '
                f'Sótt {d.day}. {MONTHS_IS[d.month - 1]} {d.year} af px.hagstofa.is.')
    return (f'Statistics Iceland ({year}). {title(table, "en")}, table {table["id"]}. '
            f'Retrieved {d.day} {d.strftime("%B")} {d.year} from px.hagstofa.is.')


def reproduce(part, lang):
    """The exact PX-Web API query behind a part as a ready cURL command (task D): anyone gets the same numbers straight
    from Statistics Iceland's API, without this system. (Numbers read from the 18 Oct release may since be revised.)"""
    url, body = px.api_request(part["table"]["path"], part["selections"], part.get("edition") or "is")
    head = REPRODUCE["is" if lang == "is" else "en"]
    data = json.dumps(body, ensure_ascii=False).replace("'", "'\\''")
    return (f"<details><summary>{head}</summary>\n\n```bash\ncurl -X POST '{url}' \\\n  -H 'Content-Type: application/json' "
            f"\\\n  -d '{data}'\n```\n\n</details>")


REPRODUCE = {"is": "Endurtaka fyrirspurnina beint í vefþjónustu Hagstofunnar (PX-API)",
             "en": "Reproduce this answer straight from Statistics Iceland's API (PX-API)"}
UNDERSTOOD = {"is": {"head": "Svona skildi ég spurninguna", "calc": "reiknað"},
              "en": {"head": "How I understood the question", "calc": "computed"}}
OPERATIONS = {"is": {"share": "hlutfall", "difference": "mismunur", "percent_change": "breyting í %", "ratio": "hlutfall",
                     "per_1000": "á hverja 1.000", "sum": "samtala", "average": "meðaltal", "formula": "formúla"},
              "en": {"share": "share", "difference": "difference", "percent_change": "% change", "ratio": "ratio",
                     "per_1000": "per 1,000", "sum": "sum", "average": "average", "formula": "formula"}}


def understood(parts, lang, calculation=None):
    """understood_text() as the chat's small print; "" when there is nothing to say."""
    text = understood_text(parts, lang, calculation)
    return f'<small>{UNDERSTOOD["is" if lang == "is" else "en"]["head"]}: {html.escape(text)}</small>' if text else ""


def understood_text(parts, lang, calculation=None):
    """One line per answer: what the pipeline understood (task A): each table's topic and the values it read, and the
    calculation, e.g. 'Mannfjöldi eftir kyni og aldri 1841-2026 · Kyn: Alls · Ár: 2026 · reiknað: hlutfall'."""
    lang = "is" if lang == "is" else "en"
    t, ops = UNDERSTOOD[lang], OPERATIONS[lang]
    pieces = []
    for p in parts:
        if p.get("also"):
            continue
        rows = p.get("rows") or []
        bits = [title(p["table"], lang)]
        for k in [k for k in (rows[0] if rows else {}) if k != "value"]:
            seen = list(dict.fromkeys(str(r.get(k, "")) for r in rows))
            bits.append(f"{k}: " + (", ".join(seen) if len(seen) <= 3 else f"{seen[0]} … {seen[-1]} ({len(seen)})"))
        if (op := (p.get("calculation") or {}).get("operation")):
            bits.append(f'{t["calc"]}: {ops.get(op, op)}')
        pieces.append(" · ".join(bits))
    if (op := (calculation or {}).get("operation")):
        pieces.append(f'{t["calc"]}: {ops.get(op, op)}')
    return " | ".join(pieces)


BAR = "█"


def series(p):
    """(rows, the label that varies) when a part reads a time series (task C): 4 to 30 rows that differ only in a
    label with a year; None otherwise."""
    rows = [r for r in p.get("rows") or [] if isinstance(r.get("value"), (int, float))]
    if not 4 <= len(rows) <= 30:
        return None
    keys = [k for k in rows[0] if k != "value"]
    varying = [k for k in keys if len({str(r.get(k)) for r in rows}) > 1]
    if len(varying) != 1 or not all(pipeline.YEAR.search(str(r[varying[0]])) for r in rows):
        return None
    return rows, varying[0]


def chart(parts, lang):
    """A small text chart when an answer reads a time series (series()), drawn from the rows as read. Monospace
    bars, so it shows in any chat or terminal."""
    for p in parts:
        found = series(p)
        if not found:
            continue
        rows, label = found
        varying = [label]
        top = max(abs(r["value"]) for r in rows) or 1
        width = max(len(str(r[varying[0]])) for r in rows)
        lines = [f'{str(r[varying[0]]):<{width}} {BAR * max(1, round(24 * abs(r["value"]) / top))} '
                 f'{_number(r["value"], lang)}' for r in rows]
        unit = f' ({p["unit"]})' if p.get("unit") else ""
        return f'```text\n{title(p["table"], lang)}{unit}\n' + "\n".join(lines) + "\n```"
    return ""


def states_numbers(text):
    """Whether an answer states any number besides years and small counts: "Gögnin svara ekki spurningunni" does
    not, so it gets no check mark, no confidence and no rows (task: UI fixes after the first UI test)."""
    return any(not any(o.is_integer() and (o <= 31 or 1800 <= o <= 2100) for o in options)
               for options, _ in verify.values_in(text))


def dev_view(q, candidates=None):
    """The developer layer (task: three layers): each part's table, edition, selections, fallbacks and flags, and the
    search's candidates, as JSON, folded."""
    info = {"parts": [{"table": p["table"]["id"], "path": p["table"].get("path"), "edition": p.get("edition"),
                       "selections": p.get("selections"), "unit": p.get("unit"), "fallback": p.get("fallback"),
                       "agreement": p.get("agreement"), "assumed": p.get("assumed"), "not_yet": p.get("not_yet"),
                       "also": p.get("also", False), "rows": len(p.get("rows") or [])} for p in q.get("parts") or []],
            "calculation": (q.get("calculation") or {}).get("text"), "unsupported": q.get("unsupported"),
            "model": q.get("model"), "confidence": q.get("confidence")}
    if candidates:
        info["candidates"] = [t["id"] for t, _ in candidates[:12]]
    head = "Fyrir forritara" if q.get("lang") == "is" else "Developer view"
    return (f"<details><summary>{head}</summary>\n\n```json\n{json.dumps(info, ensure_ascii=False, indent=1)}\n```\n\n"
            "</details>")


CHECKED = {"is": {True: "tölurnar bornar saman við töfluna", False: "svarið stóðst ekki athugun, tölurnar beint úr töflunni"},
           "en": {True: "numbers checked against the table", False: "the answer failed the check, numbers straight from the table"}}


def badge(q, seconds, plain=False):
    """The small grey line under an answer: which model, how long, whether the numbers were checked, how sure.
    plain (no number stated): model and time only. Confidence only for the model it was measured on."""
    lang = "is" if q["lang"] == "is" else "en"
    bits = [q.get("model") or "", f"{seconds:.1f} s"]
    if not plain and not q.get("no_numbers"):
        bits.append(CHECKED[lang][not q.get("unsupported")])
        if not q.get("unsupported"):  # the number check replaced the answer: how sure is beside the point
            bits.append(confidence.describe(q.get("confidence"), lang, q.get("model")))  # other models: nothing
    return "<sub>" + " · ".join(b for b in bits if b) + "</sub>"


def render(checked, r, seconds):
    """The chat's markdown for one result: answer, sources, the data behind it, the badge."""
    blocks = r.get("questions") or [{"answer": r["answer"], "parts": r.get("parts") or [], "lang": checked.lang,
                                     "calculation": r.get("calculation"), "unsupported": r.get("unsupported"),
                                     "model": r.get("model"), "confidence": r.get("confidence"), "no_numbers": r.get("no_numbers"), "suggestions": r.get("suggestions"), "clarify": r.get("clarify")}]
    out = []
    for q in blocks:  # each question's answer with its sources and data, in its own language
        dev = dev_view(q, r.get("candidates"))
        if q["parts"] and states_numbers(q["answer"]):  # three layers: the answer, the data behind it, developer view
            extra = "".join(f"{x}\n\n" for x in (chart(q["parts"], q["lang"]),
                                                  understood(q["parts"], q["lang"], q.get("calculation"))) if x)
            out.append(f'{q["answer"]}\n\n{extra}---\n{sources(q["parts"], q["lang"])}\n\n'
                       f'{data_used(q["parts"], q["lang"], q.get("calculation"), seconds)}\n\n{dev}\n\n'
                       f'{badge(q, seconds)}')
        elif q["parts"]:  # no number stated: no rows, check mark or confidence under it (rows: developer view only)
            out.append(f'{q["answer"]}\n\n{dev}\n\n{badge(q, seconds, plain=True)}')
        else:
            out.append(q["answer"])
    return "\n\n".join(out + ([r["note"]] if r.get("note") else []))


def answer_once(message):
    """(checked, result, seconds) for one message with the configured model; result None on an error."""
    started = time.perf_counter()
    try:
        checked, r = ask(message, retriever())
    except Exception:  # noqa: BLE001 - log details server-side, show a generic message in the chat
        logging.exception("pipeline failed for %r", message)
        return guard.check(message), None, time.perf_counter() - started
    return checked, r, time.perf_counter() - started


def respond(message, history):
    checked, r, seconds = answer_once(message)
    return render(checked, r, seconds) if r else ERROR["is" if checked.lang == "is" else "en"]


# --- second opinion: the same question through the whole pipeline with another model ----------------------------
SECOND = {"base_url": os.environ.get("LLM2_BASE_URL"), "model": os.environ.get("LLM2_MODEL"),
          "api_key": os.environ.get("LLM2_API_KEY", "none")}
OPINION = {"is": {"head": "Annað álit", "same": "Sama niðurstaða og fyrra svarið: {n}.",
                  "differ": "Ekki sama niðurstaða: fyrra svarið segir {a}, þetta {b}. Skoðaðu gögnin á bak við svörin.",
                  "none": "Ekkert annað líkan er tengt (LLM2_BASE_URL og LLM2_MODEL)."},
           "en": {"head": "Second opinion", "same": "Same result as the first answer: {n}.",
                  "differ": "Not the same result: the first answer says {a}, this one {b}. Check the data behind them.",
                  "none": "No second model is connected (LLM2_BASE_URL and LLM2_MODEL)."}}


YEAR_RANGE = re.compile(r"\b\d{4}\s*[-–]\s*\d{4}\b")


def key_numbers(text):
    """The numbers an answer states, leaving out years and small counts: what two answers should agree on."""
    out = []
    text = YEAR_RANGE.sub(" ", text or "")  # a title's "1841-2026" is not the answer (dry run: "Same result: 1841")
    for options, shown in verify.values_in(text):
        v = max(options)
        if not (v.is_integer() and (v <= 31 or 1900 <= v <= 2100)):
            out.append((v, shown))
    return out


def stated_numbers(answer, question):
    """The numbers an answer states of its own, as written: not years, year ranges or numbers the question already
    had ("Engar tölur fundust fyrir Reykjavík árið 1950" states none)."""
    asked = {max(options) for options, _ in verify.values_in(question or "")}
    return list(dict.fromkeys(shown for options, shown in verify.values_in(YEAR_RANGE.sub(" ", answer or ""))
                              if not (shown.isdigit() and len(shown) == 4)  # "1950", not "1.950" (a count)
                              and max(options) not in asked))



def agree(first, second):
    """None when either answer states no number; else whether a main number of one is in the other."""
    a, b = key_numbers(first)[:3], key_numbers(second)[:3]
    if not a or not b:
        return None
    return any(math.isclose(x, y, rel_tol=0.005) for x, _ in a for y, _ in b)


def verdict(first, second, lang):
    """(same, line): whether the two answers state the same main number (None: one states none), and the line
    saying so in the user's language ("" when there is nothing to compare). Shared by the chat and the API."""
    t = OPINION[lang]
    same = agree(first, second)
    if same is None:
        return None, ""
    if same:
        return True, t["same"].format(n=key_numbers(second)[0][1])
    return False, t["differ"].format(a=key_numbers(first)[0][1], b=key_numbers(second)[0][1])


def second_opinion(message, first_answer):
    """The second model's full answer plus one line saying whether it gives the same number as the first."""
    lang = "is" if guard.check(message).lang == "is" else "en"
    t = OPINION[lang]
    if not (SECOND["base_url"] and SECOND["model"]):
        return t["none"]
    with llm.using(**SECOND):
        checked, r, seconds = answer_once(message)
    if r is None:
        return ERROR[lang]
    _, line = verdict(first_answer, r["answer"], lang)
    return f'**{t["head"]}**\n\n{render(checked, r, seconds)}' + (f"\n\n**{line}**" if line else "")


# --- the chat page --------------------------------------------------------------------------------------------
WORKING = {"is": "_Leita í töflum Hagstofunnar, les tölurnar og skrifa svarið…_",
           "en": "_Searching Statistics Iceland's tables, reading the numbers, writing the answer…_"}
EXAMPLES = ["Hvað búa margir á Íslandi?", "How many people lived in Reykjavík on 1 January 2026?",
            "Hve stórt hlutfall landsmanna bjó í Reykjavík 1. janúar 2026?",
            "How many children were born per 1,000 inhabitants in 2025?",
            "Hver var verðbólgan síðustu 12 mánuði?", "Hvað komu margir ferðamenn í fyrra?"]


def _text(content):
    """A chat message's text, whether Gradio hands it over as a string or as a list of parts."""
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content if isinstance(content, str) else str(content or "")


CHIPS = 4  # buttons under the chat: questions we can answer after "no data", or the choices a vague question left
CHIP_HEAD = {"is": {"suggest": "Smelltu á spurningu sem við getum svarað:", "choose": "Viltu annað val?",
                    "check": "Athuga:"},
             "en": {"suggest": "Click a question we can answer:", "choose": "Want a different choice?",
                    "check": "Check:"}}


def chips(question, r, lang):
    """(head line, [(label, question to send)]) for the buttons under an answer: the 'no data' suggestions
    (suggest.py) as questions to ask, or the period/place a vague question left open (clarify.py) as the same
    question with that choice spelt out. Nothing when the answer needs no follow-up."""
    t = CHIP_HEAD["is" if lang == "is" else "en"]
    if r and r.get("suggestions"):
        return t["suggest"], [(q, q) for q in r["suggestions"][:CHIPS]]
    options = []
    for c in (r or {}).get("clarify") or []:
        options += [(o, f'{question} ({c["variable"]}: {o})') for o in c["options"] if o != c["default"]]
    if options:
        return f'{t["choose"]} {(r["clarify"][0]["question"])}', options[:CHIPS]
    return "", []


STEPS_SHOWN = 6
STEP_TEXT = {
    "is": {"split": "Spurningunni skipt í {n} hluta", "search": "Leita í {tables} töflum Hagstofunnar",
           "pick": "Valin tafla: **{table}** {title}", "read": "Les tölurnar úr {table}",
           "nodata": "{table} hefur engar tölur fyrir þetta, reyni næstu töflu",
           "refused": "{table} svarar ekki spurningunni ({why}), reyni næstu töflu",
           "declined": "Engin tafla svarar þessu", "calc": "Reikna í kóða: {text}",
           "write": "Skrifa svarið og ber tölurnar saman við gögnin"},
    "en": {"split": "Split the question into {n} parts", "search": "Searching {tables} Statistics Iceland tables",
           "pick": "Table chosen: **{table}** {title}", "read": "Reading the numbers from {table}",
           "nodata": "{table} has no numbers for this, trying the next table",
           "refused": "{table} doesn't answer the question, trying the next table",  # the check's reason is Icelandic
           "declined": "No table answers this", "calc": "Calculating in code: {text}",
           "write": "Writing the answer and checking its numbers against the data"},
}
NOTICE = {"is": {"nodata": "Engin gögn fundust hjá Hagstofunni. Sjáðu tillögurnar fyrir neðan svarið.",
                 "replaced": "Tala í svarinu fannst ekki í gögnunum, svo tölurnar eru sýndar beint úr töflunni.",
                 "error": ERROR["is"]},
          "en": {"nodata": "No data found at Statistics Iceland. See the suggestions under the answer.",
                 "replaced": "A number in the answer was not in the data, so the numbers are shown straight from "
                             "the table.", "error": ERROR["en"]}}


def step_line(step, info, lang):
    """One progress line for the chat while the answer is being made (pipeline.report)."""
    t = STEP_TEXT["is" if lang == "is" else "en"]
    if step not in t:
        return ""
    info = {**info, "n": len(info.get("parts") or []), "tables": f'{info.get("tables") or 2053:,}'.replace(
        ",", "." if lang == "is" else ","), "why": str(info.get("why", ""))[:90], "text": info.get("text", "")}
    return t[step].format(**info)


def notify(kind, text):
    """A short pop-up for the user (Gradio's toast); harmless outside a running page."""
    try:
        (gr.Warning if kind == "warning" else gr.Info)(text)
    except Exception:  # noqa: BLE001 - no page to show it on (tests, scripts)
        logging.debug("no page for the notice %r", text)


def answer_live(question, lang):
    """Yields ("step", line) while the pipeline works (in its own thread), then ("done", (checked, r, seconds))."""
    events = queue.Queue()

    def work():
        pipeline.PROGRESS.set(lambda step, info: events.put(("step", step_line(step, info, lang))))
        events.put(("done", answer_once(question)))

    threading.Thread(target=work, daemon=True).start()
    while True:
        kind, value = events.get()
        yield kind, value
        if kind == "done":
            return


def _chip_updates(head, found):
    """Gradio updates for the head line and the CHIPS buttons: the first len(found) shown, the rest hidden."""
    found = list(found)[:CHIPS]
    buttons = [gr.Button(value=label, visible=True) for label, _ in found]
    buttons += [gr.Button(value="", visible=False) for _ in range(CHIPS - len(found))]
    return [gr.Markdown(value=f"<sub>{head}</sub>" if head else "", visible=bool(head)), *buttons]


def _ask(message, history, last):
    """One chat turn: the follow-up and mixed-language checks, the answer, then the chips under it. Yields the
    textbox, the history, the state and the chip updates (head + CHIPS buttons)."""
    message = (message or "").strip()
    if not message:
        yield "", history, last, *_chip_updates("", [])
        return
    lang = "is" if guard.check(message).lang == "is" else "en"
    history = list(history or []) + [{"role": "user", "content": message}]
    if followup.mixed(message):  # Icelandic and English question words together: the user chooses, we don't guess
        yield message, history + [{"role": "assistant", "content": followup.CHOOSE}], last, *_chip_updates("", [])
        return  # the message stays in the box, to be asked again in one language
    history.append({"role": "assistant", "content": WORKING[lang]})
    yield "", history, last, *_chip_updates("", [])
    question = followup.standalone(message, (last or {}).get("question"))  # "En 2024?" -> a whole question
    understood = f"_{followup.UNDERSTOOD[lang]}: {question}_\n\n" if question != message else ""
    if understood:
        notify("info", f"{followup.UNDERSTOOD[lang]}: {question}")
    steps = []
    for kind, value in answer_live(question, lang):  # each step shows up in the chat as it happens
        if kind == "step":
            steps += [value] if value else []
            history[-1] = {"role": "assistant", "content": understood + WORKING[lang] + "\n\n"
                           + "\n".join(f"- {s}" for s in steps[-STEPS_SHOWN:])}
            yield "", history, last, *_chip_updates("", [])
        else:
            checked, r, seconds = value
    history[-1] = {"role": "assistant", "content": understood + (render(checked, r, seconds) if r else ERROR[lang])}
    t = NOTICE[lang]
    if r is None:
        notify("warning", t["error"])
    elif r.get("suggestions"):
        notify("info", t["nodata"])
    elif r.get("unsupported"):
        notify("warning", t["replaced"])
    elif r.get("clarify"):
        notify("info", r["clarify"][0]["question"])
    head, found = chips(question, r, checked.lang)
    kept = [{"id": p["table"]["id"], "path": p["table"].get("path"), "edition": p.get("edition") or "is",
             "selections": p.get("selections"), "values": [x.get("value") for x in p.get("rows") or []]}
            for p in (r or {}).get("parts") or [] if p.get("selections") and p["table"].get("path")]
    if kept and len(found) < CHIPS and states_numbers((r or {}).get("answer", "")):  # task F: "is it still current?"
        found = found + [(STILL_CURRENT_LABEL["is" if checked.lang == "is" else "en"], STILL_CURRENT)]
        head = head or CHIP_HEAD["is" if checked.lang == "is" else "en"]["check"]
    yield "", history, {"question": question, "answer": (r or {}).get("answer", ""), "chips": found,
                        "parts": kept, "lang": checked.lang}, *_chip_updates(head, found)


STILL_CURRENT = "\x00still-current"  # a chip that checks instead of asking
STILL_CURRENT_LABEL = {"is": "Er talan enn sú nýjasta? (PX-Web)", "en": "Still the newest figure? (PX-Web)"}
STILL_TEXT = {
    "is": {"same": "Athugað beint á PX-Web núna: tölurnar í {id} eru óbreyttar.",
           "changed": "PX-Web sýnir nú aðrar tölur í {id}: {now} (svarið notaði {then}).",
           "fail": "Náði ekki sambandi við PX-Web núna ({why}). Svarið byggir á gögnum Hagstofunnar frá 18. október."},
    "en": {"same": "Checked live on PX-Web just now: the numbers in {id} are unchanged.",
           "changed": "PX-Web now shows other numbers in {id}: {now} (the answer used {then}).",
           "fail": "Couldn't reach PX-Web right now ({why}). The answer uses Statistics Iceland's data of 18 October."}}


def still_current(parts, lang):
    """Task F: the same cells read live from PX-Web (never the data release), compared with what the answer used.
    One line per table; a 429, a timeout or no network says so instead of failing."""
    t = STILL_TEXT["is" if lang == "is" else "en"]
    lines = []
    for p in parts:
        try:
            url, body = px.api_request(p["path"], p["selections"], p["edition"])
            live = [x.get("value") for x in pipeline.flatten(px._request("POST", url, json=body, attempts=2))]
        except Exception as e:  # noqa: BLE001 - PX-Web busy or unreachable: say so, the answer stands
            lines.append(t["fail"].format(why=type(e).__name__))
            continue
        same = sorted(v for v in live if v is not None) == sorted(v for v in p["values"] if v is not None)
        show = lambda vs: ", ".join(_number(v, lang) for v in vs[:4]) + (" …" if len(vs) > 4 else "")  # noqa: E731
        lines.append(t["same"].format(id=p["id"]) if same else
                     t["changed"].format(id=p["id"], now=show(live), then=show(p["values"])))
    return "\n\n".join(lines)


def _new_conversation():
    """An empty chat and state: the next question is not read as a follow-up of the last one."""
    return "", [], {}, *_chip_updates("", [])


def _chip(i):
    """The handler of chip i: asks the question stored for it in the state."""
    def ask_chip(history, last):
        found = (last or {}).get("chips") or []
        if i >= len(found):
            yield "", history, last, *_chip_updates("", [])
            return
        if found[i][1] == STILL_CURRENT:  # a live check of the last answer's numbers, not a new question
            lang = (last or {}).get("lang") or "is"
            history = list(history or []) + [{"role": "user", "content": found[i][0]},
                                             {"role": "assistant", "content": WORKING["is" if lang == "is" else "en"]}]
            yield "", history, last, *_chip_updates("", [])
            history[-1] = {"role": "assistant", "content": still_current((last or {}).get("parts") or [], lang)}
            rest = [c for c in found if c[1] != STILL_CURRENT]
            yield "", history, {**last, "chips": rest}, *_chip_updates(CHIP_HEAD[lang]["check"] if rest else "", rest)
            return
        yield from _ask(found[i][1], history, {**(last or {}), "question": None})  # a whole question, no follow-up
    return ask_chip


def _second(history, last):
    if not last or not last.get("question"):
        yield history
        return
    lang = "is" if guard.check(last["question"]).lang == "is" else "en"
    history = list(history or []) + [{"role": "assistant", "content": WORKING[lang]}]
    yield history
    history[-1] = {"role": "assistant", "content": second_opinion(last["question"], last["answer"])}
    yield history


with gr.Blocks(title="Spurðu Hagstofuna") as demo:
    gr.Markdown("## Spurðu Hagstofuna\nSpurðu um tölur Hagstofu Íslands á íslensku eða ensku. "
                "Ask about Statistics Iceland's figures in Icelandic or English.")
    chat = gr.Chatbot(height=620, label="Spurðu Hagstofuna")
    last = gr.State({})
    box = gr.Textbox(placeholder="t.d. Hvað búa margir í Kópavogi? / e.g. How many tourists came last year?",
                     show_label=False, autofocus=True)
    chip_head = gr.Markdown(visible=False)
    with gr.Row():
        chip_buttons = [gr.Button(visible=False, size="sm") for _ in range(CHIPS)]
    with gr.Row():
        send = gr.Button("Spyrja / Ask", variant="primary")
        again = gr.Button("Annað álit / Second opinion")
        stop = gr.Button("Stöðva / Stop", variant="stop")
        clear = gr.Button("Nýtt samtal / New conversation")
    gr.Examples(EXAMPLES, inputs=box)
    turn_out = [box, chat, last, chip_head, *chip_buttons]
    running = [box.submit(_ask, [box, chat, last], turn_out), send.click(_ask, [box, chat, last], turn_out)]
    running += [button.click(_chip(i), [chat, last], turn_out) for i, button in enumerate(chip_buttons)]
    running.append(again.click(_second, [chat, last], [chat]))
    stop.click(None, None, None, cancels=running)  # an answer still being made is dropped
    clear.click(_new_conversation, None, turn_out, cancels=running)  # and the follow-up memory goes with it

demo.queue(default_concurrency_limit=int(os.environ.get("CHAT_CONCURRENCY", "8")))  # 8 answers at once, not one

AUTH = tuple(os.environ["APP_AUTH"].split(":", 1)) if os.environ.get("APP_AUTH") else None

if __name__ == "__main__":
    warm_up()
    demo.launch(server_name=os.environ.get("APP_HOST", "127.0.0.1"), auth=AUTH)

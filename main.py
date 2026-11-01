"""The server: JSON API (FastAPI) + the web page + the chat UI, on one port.

    .venv/bin/uvicorn main:app --port 8000          ->  http://127.0.0.1:8000  (page), /gradio (chat), /docs (API)

    curl -X POST http://127.0.0.1:8000/spyrja_px_api -H "Content-Type: application/json" \
         -d '{"spurning": "Hve margir bjuggu á Íslandi 1. janúar 2026?"}'
    curl -X POST http://127.0.0.1:8000/spyrja -H "Content-Type: application/json" \
         -d '{"spurningar": ["Hver var vísitala neysluverðs í september 2026?", "How many births in 2024?"]}'

/spyrja_px_api takes the Track 2 starter's request and returns its fields (svar, töflu_slóð, notud_sia), so their
cURL test works here, plus ours: every table used, the values chosen, the rows read, the calculation (done in code)
and the time taken. /spyrja takes up to MAX_QUESTIONS questions in one call and answers them in parallel.

Every answer runs in one shared pool of MAX_PARALLEL threads, however many requests arrive: the LLM server is the
bottleneck, and more parallel calls would only queue there. The API has no login: keep the default host
(127.0.0.1) or put it behind something that has one. APP_AUTH=user:pass protects the chat only.

The page (web/index.html) is separate from the backend: it uses only the API, as any other client would. It is
served from this server so the demo needs one port, one tunnel and no CORS.
"""
import functools
import json
import re
import logging
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import AliasChoices, BaseModel, Field, StringConstraints

import app as chat
from app import AUTH, ERROR, ask, demo, retriever, warm_up
from hagstofa import confidence, followup, guard, llm, local_data, pipeline, px, suggest, translate
from hagstofa import verify
from hagstofa.retrieve import load_catalog

PAGE = Path(__file__).parent / "web" / "index.html"

MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "4"))
MAX_QUESTIONS = 10
_pool = ThreadPoolExecutor(MAX_PARALLEL, thread_name_prefix="answer")

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


@asynccontextmanager
async def lifespan(_):
    warm_up()  # load the search at startup, not during the first question
    yield


api = FastAPI(title="Spurðu Hagstofuna", description=__doc__.split("\n")[0], lifespan=lifespan)


class Question(BaseModel):
    # their app.py reads "spurning", their README's cURL test sends "prompt": take either
    spurning: Text = Field(validation_alias=AliasChoices("spurning", "prompt"),
                           examples=["Hve margir bjuggu á Íslandi 1. janúar 2026?"])
    # the web page's conversation: the question before ("En 2024?" is read with it) and samtal=True for the
    # mixed-language check; without them the API answers the question as asked, as before
    fyrri_spurning: Annotated[str, StringConstraints(max_length=500)] | None = None
    samtal: bool = False
    mál: Literal["is", "en"] | None = None  # the page's language: a front-door reply with no language to go by uses it


class Questions(BaseModel):
    spurningar: list[Text] = Field(min_length=1, max_length=MAX_QUESTIONS)


OFFLINE = {"is": "Mállíkanið svarar ekki núna (þjónninn sem keyrir það er ekki tengdur). Reyndu aftur eftir smá stund.",
           "en": "The language model isn't answering right now (the server running it isn't connected). "
                 "Please try again in a while."}


def labelled(part):
    """The values chosen, as the table names them ("Sveitarfélag: Reykjavíkurborg") rather than as codes
    ("Sveitarfelag": ["0000"]): the organizers ask for the selected dimensions with every answer."""
    t = part["table"]
    out = []
    for v in t.get("variables_en" if part.get("edition") == "en" else "variables_is") or []:
        codes = part["selections"].get(v["code"])
        if codes:
            names = dict(zip(v["values"], v["valueTexts"]))
            out.append({"breyta": v["text"], "gildi": [names.get(c, c) for c in codes], "kóðar": codes})
    return out


def source(part):
    """One table used: where it is, what was chosen from it, what was read, what was computed."""
    t = part["table"]
    return {"tafla": t["id"], "titill": t["title_is"], "slóð": part["url"], "uppfært": t.get("updated"),
            "útgáfa": part.get("edition", "is"), "eining": part.get("unit"), "frávik": part.get("fallback", []),
            "val": part["selections"], "gögn": part["rows"][:pipeline.MAX_ROWS],
            "útreikningur": part["calculation"]["text"] if part.get("calculation") else None,
            # for the web page (added 31 Oct; the fields above are unchanged)
            "titill_en": t.get("title_en"), "undirspurning": part.get("question"), "val_merkt": labelled(part),
            "forsendur": [{"breyta": var, "gildi": val} for var, val in part.get("assumed") or []],
            "leit": part.get("candidates", []),
            "önnur_tafla": bool(part.get("also")),  # a look-alike table that counts differently (ALSO_TABLE)
            "um_töfluna": chat.table_facts_text(t, "en" if part.get("edition") == "en" else "is") or None,
            "tilvitnun": chat.citation(t, "en" if part.get("edition") == "en" else "is"),
            "endurtaka": reproduce(part), "graf": chart_data(part), "px_slóð": t.get("path"),
            # how many decimals the table publishes: the page rounds the rows to it for display (gögn stays exact)
            "aukastafir": local_data.decimals(t["path"], part.get("edition") or "is") if t.get("path") else None}


def reproduce(part):
    """The exact PX-Web API query behind a table's numbers (px.api_request): {"url", "body"}, for a cURL anyone can
    run to get the same numbers from Statistics Iceland without us. None without a path."""
    if not part["table"].get("path") or not part.get("selections"):
        return None
    url, body = px.api_request(part["table"]["path"], part["selections"], part.get("edition") or "is")
    return {"url": url, "body": body}


def chart_data(part):
    """A time series to draw (the chat's rule, app.series: 4-30 rows differing only in a year label): the labels and
    values as read, and the unit. None otherwise; the page only draws these numbers."""
    found = chat.series(part)
    if not found:
        return None
    rows, label = found
    return {"merki": [str(r[label]) for r in rows], "gildi": [r["value"] for r in rows], "eining": part.get("unit")}


def display(answer, parts, lang, result, question=""):
    """What the web page shows besides the sources, computed here so the browser computes nothing:
    the answer without its "Miðað við: ..." line (the page shows the assumptions on their own), the model,
    the confidence line exactly as the chat words it, and whether every number passed the check."""
    lang = "is" if lang == "is" else "en"
    said = list(dict.fromkeys(f"{var} {val}" for p in parts for var, val in p.get("assumed") or []))
    tail = "\n\n" + pipeline.ASSUMED[lang].format(", ".join(said)) if said else None
    text = answer[:-len(tail)] if tail and answer.endswith(tail) else answer
    if result.get("suggestions"):  # the page shows them as buttons, not as lines in the answer
        text = text.split("\n\n" + suggest.INTRO[lang])[0]
    level = result.get("confidence")
    numbers = bool(parts) and not result.get("no_numbers")
    main_text, main_parts = text, [p for p in parts if not p.get("also")]
    for p in parts:  # the key number is the main table's: the look-alike line states another one
        if p.get("also"):
            main_text = main_text.replace(pipeline.also_line(p, lang), "")
    return {"svar_án_forsendna": text, "líkan": result.get("model"), "vissustig": level,
            "vissa": (confidence.describe(level, lang, result.get("model")) or None) if not result.get("unsupported") else None,
            "tölur_athugaðar": (not result.get("unsupported")) if numbers else None,
            "svar_með_tölum": numbers,  # False: a "no data" answer; the page then shows no rows (they didn't answer it)
            "aðaltala": key_number(main_text, question, main_parts, result, lang) if numbers and main_parts else None,
            "framhald": followups(question, result, lang),
            "skilið_sem": chat.understood_text(main_parts, lang, result.get("calculation")) if numbers else None,
            "útreikningur_milli_taflna": (result.get("calculation") or {}).get("text")}


def followups(question, result, lang):
    """The buttons under an answer, as the chat makes them (app.chips): the "no data" suggestions, or the period or
    place a vague question left open; each button sends its question. None when there are none."""
    head, buttons = chat.chips(question, result, lang)
    return {"fyrirsögn": head, "spurningar": [{"texti": label, "spurning": q} for label, q in buttons]} if buttons else None


def key_number(text, question, parts, result, lang="is"):
    """The answer's one number, as the answer writes it, with its unit, for the page to show large; None when
    the answer states several (two tables, a change) so the page never has to pick. The unit is the table's,
    or "%" when the answer writes one; a calculation's result has no table unit, so then none."""
    found = chat.stated_numbers(text, question)
    if len(found) > 1:  # "394.324 manns 1. janúar": a bare day of the month is not the answer
        found = [n for n in found if not (n.isdigit() and int(n) <= 31)] or found
    if len(found) != 1:
        return None
    number = found[0].rstrip("%").strip()
    after = text[text.index(number) + len(number):].lstrip()
    shown = number
    if result.get("unsupported"):  # the fallback writes raw values ("394324.0"): write it as answers do
        value = max(next(options for options, s in verify.values_in(number)))
        shown = pipeline.fmt(value, english=lang == "en")
    if after.startswith(("%", "prósent", "percent")):
        unit = "%"
    elif result.get("calculation") or any(p.get("calculation") for p in parts):
        unit = None
    else:
        unit = parts[0].get("unit")
    if unit and re.search(r"[/,;]", unit):  # "Milljónir króna/hlutfall": several measures, the row says which
        unit = None
    return {"tala": shown, "eining": unit}


PROMPTS = None


def prompt_name(system):
    """Which of our prompts a model call used (PICK_SYSTEM, ...), for the developer view."""
    global PROMPTS
    if PROMPTS is None:
        PROMPTS = {v[:80]: n for mod in (pipeline, guard, translate, suggest) for n, v in vars(mod).items()
                   if n.isupper() and isinstance(v, str) and len(v) >= 80}
    return next((n for k, n in PROMPTS.items() if system.startswith(k)), system[:60])


def traced(fn, *args, on_step=None):
    """fn(*args) and what happened inside, in order, for the developer view (?debug=1): every model call (which
    prompt, seconds, the raw reply) and every step the pipeline reported (search, pick, read, calc, ...), each
    with its time since the start. The threads of a multi-question message share the list."""
    calls, steps, started = [], [], time.perf_counter()
    t1 = llm.TRACE.set(calls)
    def listen(step, info):
        steps.append((time.perf_counter(), step, info))
        if on_step:
            on_step(step, info)
    t2 = pipeline.PROGRESS.set(listen)
    try:
        out = fn(*args)
    finally:
        llm.TRACE.reset(t1)
        pipeline.PROGRESS.reset(t2)
    log = [(at, {"hver": "kóði", "skref": step, "upplýsingar": info}) for at, step, info in steps]
    log += [(c["at"], {"hver": "líkan", "skref": prompt_name(c["system"]), "líkan": c.get("model"),
                       "sekúndur": round(c["seconds"], 2),
                       "svar": c["reply"]}) for c in calls]
    return out, [{"t": round(at - started, 2), **e} for at, e in sorted(log, key=lambda x: x[0])]


def searched(candidates):
    """The search's top 5 when no table gave numbers (no part to hang them on)."""
    return [{"id": t["id"], "title_is": t.get("title_is"), "title_en": t.get("title_en"), "rank": i + 1,
             "chosen": False} for i, (t, _) in enumerate((candidates or [])[:pipeline.SHOWN_CANDIDATES])]


def answer_one(question, previous=None, conversation=False, on_step=None, page_lang=None):
    """Our pipeline's answer as JSON; an error becomes a message in `villa`, so one bad question
    doesn't fail the others in a batch. conversation (the web page): a message mixing Icelandic and English gets
    the chat's "which language?" instead of an answer, and a follow-up ("En 2024?") is read with `previous`."""
    started = time.perf_counter()
    if conversation and followup.mixed(question):
        return {"spurning": question, "svar": followup.CHOOSE, "töflu_slóð": None, "notud_sia": [], "heimildir": [],
                "sekúndur": round(time.perf_counter() - started, 2), "villa": None, "síun": "mixed",
                "tungumál": guard.check(question).lang, "blandað_mál": True}
    asked = question
    if previous:
        question = followup.standalone(question, previous)
    try:
        (checked, r), log = traced(ask, question, retriever(), on_step=on_step)
    except Exception as e:  # noqa: BLE001 - log the details, answer with a generic message
        logging.exception("pipeline failed for %r", question)
        lang = guard.check(question).lang
        offline = isinstance(e, llm.UNREACHABLE)  # the model server, not the question: say so
        return {"spurning": question, "svar": (OFFLINE if offline else ERROR)["is" if lang == "is" else "en"],
                "töflu_slóð": None, "notud_sia": [], "heimildir": [],
                "sekúndur": round(time.perf_counter() - started, 2), "villa": "model_offline" if offline else type(e).__name__, "síun": None,
                "tungumál": lang}
    parts = r.get("parts", [])
    return {
        "spurning": checked.text or question,
        "svar": r["answer"],
        # the starter's fields: first table's link, and its PX-Web query (their format)
        "töflu_slóð": parts[0]["url"] if parts else None,
        "notud_sia": [{"code": code, "selection": {"filter": "item", "values": values}}
                      for code, values in (parts[0]["selections"].items() if parts else [])],
        "heimildir": [source(p) for p in parts],
        "sekúndur": round(time.perf_counter() - started, 2),
        "villa": None,
        "síun": None if checked.ok else checked.category,  # why the input wasn't answered (gibberish, ...)
        "tungumál": checked.lang,  # the question's language; not is/en: answered via English
        "spurning_á_ensku": r.get("question_en"),  # the translation the pipeline answered (other languages)
        "tillögur": r.get("suggestions") or [],  # after "no data": related questions the tables can answer
        "valkostir": r.get("clarify") or [],  # the period/place a vague question left: options, default first
        # a message with several questions: each one's own answer, language and sources, in order
        "spurningar": [{"spurning": q["question"], "tungumál": q["lang"], "svar": q["answer"],
                        "síun": None if q["category"] == "ok" else q["category"],
                        "spurning_á_ensku": q["question_en"], "heimildir": [source(p) for p in q["parts"]],
                        "tillögur": q.get("suggestions") or [],
                        **display(q["answer"], q["parts"], q["lang"], q, q["question"])}
                       for q in r.get("questions", [])],
        # for the web page
        **display(r["answer"], parts, checked.lang, r, checked.text or question),
        "athugun_líkans": guard.SCOPE_CHECK and checked.ok,  # the LLM's topic check ran and let it through
        "leit_án_töflu": searched(r.get("candidates")) if not parts else [],
        "ferill": log,  # the developer view: model calls and pipeline steps in order, with times
        "skilið_sem_spurning": question if question != asked else None,  # "En 2024?" -> the whole question
        **(front_door(checked, page_lang) if not checked.ok else {}),
    }


def front_door(checked, page_lang):
    """A front-door reply for the page: the text in one language (the page's when the question has none, e.g.
    gibberish) and its example questions as a list (dæmi) for buttons. `svar` keeps the guard's full text."""
    langs = guard.languages(checked)
    lang = page_lang if len(langs) > 1 and page_lang in ("is", "en") else langs[0]
    examples = [] if checked.category in guard.NO_EXAMPLES else guard.EXAMPLES[lang]
    return {"svar_án_forsendna": guard.MESSAGES[checked.category][lang], "dæmi": examples}


@api.post("/spyrja_px_api")
@api.post("/query")  # the Track 2 README's cURL test: POST /query {"prompt": "..."}
def spyrja_px_api(q: Question):
    """One question (the Track 2 starter's request and fields, plus ours)."""
    return _pool.submit(answer_one, q.spurning, q.fyrri_spurning, q.samtal, None, q.mál).result()


@api.post("/spyrja_px_api/skref")
def spyrja_skref(q: Question):
    """The same answer, with each pipeline step sent as it happens: newline-separated JSON, {"skref": "..."} lines
    (the chat's wording, app.step_line), then {"svar": <the answer as /spyrja_px_api gives it>}."""
    lang = "is" if guard.check(q.spurning).lang == "is" else "en"
    events = queue.Queue()

    def on_step(step, info):
        if line := chat.step_line(step, info, lang):
            events.put({"skref": line})

    def work():
        try:
            events.put({"svar": answer_one(q.spurning, q.fyrri_spurning, q.samtal, on_step, q.mál)})
        except Exception:  # noqa: BLE001 - answer_one catches the pipeline's errors; this is a last resort
            logging.exception("streamed answer failed for %r", q.spurning)
            events.put({"svar": {"spurning": q.spurning, "svar": ERROR[lang], "villa": "stream", "heimildir": []}})

    _pool.submit(work)

    def lines():
        while True:
            event = events.get()
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if "svar" in event:
                return
    return StreamingResponse(lines(), media_type="application/x-ndjson")


class StillCurrent(BaseModel):
    tungumál: str = "is"
    töflur: list[dict] = Field(min_length=1, max_length=6)  # {"tafla", "px_slóð", "útgáfa", "val", "gildi"}


@api.post("/enn_nyjast")
def enn_nyjast(q: StillCurrent):
    """"Is it still the newest figure?": the same cells read live from PX-Web (never the data release) and compared
    with what the answer used (app.still_current, as the chat's button)."""
    parts = [{"id": t.get("tafla"), "path": t.get("px_slóð"), "edition": t.get("útgáfa"),
              "selections": t.get("val") or {}, "values": t.get("gildi") or []} for t in q.töflur]
    # the client names the table: only real catalog tables, our two editions and a small query reach PX-Web
    parts = [p for p in parts if p["path"] in catalog_paths() and p["edition"] in ("is", "en") and small(p["selections"])]
    return {"svar": chat.still_current(parts, "is" if q.tungumál == "is" else "en") if parts else ""}


@functools.cache
def catalog_paths():
    return {t["path"] for t in load_catalog()}


def small(selections, most_variables=10, most_values=200):
    """A selection the size of an answer's query: a dict of short codes, not a way to make PX-Web do heavy work."""
    return (isinstance(selections, dict) and len(selections) <= most_variables
            and all(isinstance(k, str) and isinstance(v, list) and len(v) <= most_values
                    and all(isinstance(x, str) and len(x) <= 50 for x in v) for k, v in selections.items()))


@api.post("/spyrja")
def spyrja(q: Questions):
    """Several questions in one call, answered in parallel; the answers come back in the same order."""
    futures = [_pool.submit(answer_one, s) for s in q.spurningar]
    return {"svör": [f.result() for f in futures]}


class SecondOpinion(BaseModel):
    spurning: Text
    fyrra_svar: Annotated[str, StringConstraints(max_length=5000)] = ""


def second_opinion(question, first_answer):
    """The same question through the whole pipeline with the second model (LLM2_*), and whether it states the
    same number as the first answer. llm.using() switches the model for this thread only, so other questions
    answered at the same time keep the first model."""
    lang = "is" if guard.check(question).lang == "is" else "en"
    t = chat.OPINION[lang]
    if not (chat.SECOND["base_url"] and chat.SECOND["model"]):
        return {"tengt": False, "samanburður": t["none"], "sama": None, "svar": None}
    with llm.using(**chat.SECOND):
        second = answer_one(question)
    same, line = chat.verdict(first_answer, second["svar"], lang) if second["villa"] is None else (None, "")
    return {"tengt": True, "samanburður": line, "sama": same, "svar": second}


@api.post("/annad_alit")
def annad_alit(q: SecondOpinion):
    """Second opinion: the question answered again by the second model, compared with the first answer."""
    return _pool.submit(second_opinion, q.spurning, q.fyrra_svar).result()


@api.get("/health")
def health():
    """Whether the server AND its model answer: check this before a demo (status "ok" only when both do)."""
    up = llm.reachable()
    return {"status": "ok" if up else "llm_unreachable", "llm_reachable": up, "llm_model": llm.MODEL,
            "second_model": bool(chat.SECOND["base_url"] and chat.SECOND["model"])}


@api.get("/", include_in_schema=False)
def page():
    """The demo's web page: one HTML file that talks to the API above. The chat is at /gradio, as a fallback."""
    # no-store: a browser's cached copy kept showing an old page while we changed it (1 Nov)
    return FileResponse(PAGE, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})


app = gr.mount_gradio_app(api, demo, path="/gradio", auth=AUTH)  # last: the routes above take precedence

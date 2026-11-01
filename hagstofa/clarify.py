"""What a vague question left open, as at most two questions with options (idea E, 31 Oct).

  clarifications(result, lang) -> [{"variable": "Ár", "question": "Fyrir hvaða ár? ...", "options": ["2026", "2025",
                                     "2024"], "default": "2026"}, ...]

The answer is never held back: code already fills what the question didn't say (the newest period, the total) and
says so ("Assuming: Year 2026"). This only tells a client which of those choices are worth offering back as buttons:
the period, then the place, each with the default first. The organizers' scorer and the API answer as before.
"""
import re

from .retrieve import is_locality, is_municipality

MOST = 2
OPTIONS = 4
ASK = {"is": {"time": "Fyrir hvaða tímabil? Ég notaði {default}, nýjustu tölurnar.",
              "place": "Fyrir hvaða stað? Ég notaði {default}."},
       "en": {"time": "For which period? I used {default}, the newest figures.",
              "place": "For which place? I used {default}."}}
PLACE_WORDS = re.compile(r"sveitarf|landsvæð|landshlut|byggðakjarn|þéttbýl|region|municipal", re.I)


def _kind(v):
    if v.get("time"):
        return "time"
    if is_municipality(v) or is_locality(v) or PLACE_WORDS.search(v["text"]):
        return "place"
    return None


def clarifications(result, lang="is"):
    from .pipeline import variables  # pipeline imports this module

    lang = "en" if lang == "en" else "is"
    out = []
    from .pipeline import RELATIVE, years_in

    for p in result.get("parts") or []:
        assumed = dict(p.get("assumed") or [])
        question = p.get("question") or ""
        no_period = not years_in(question) and not any(re.search(rx, question, re.I) for rx, _ in RELATIVE)
        for v in variables(p["table"], p.get("edition", "is")):
            kind = _kind(v)
            chosen = (p.get("selections") or {}).get(v["code"]) or []
            newest = kind == "time" and no_period and chosen == v["values"][-1:]  # the model took the newest itself
            if not kind or (v["text"] not in assumed and not newest):
                continue
            default = assumed.get(v["text"]) or v["valueTexts"][-1]
            texts = list(reversed(v["valueTexts"])) if kind == "time" else list(v["valueTexts"])
            options = [default] + [t for t in texts if t != default][:OPTIONS - 1]
            out.append({"variable": v["text"], "kind": kind, "options": options, "default": default,
                        "question": ASK[lang][kind].format(default=default)})
    out.sort(key=lambda c: c["kind"] != "time")  # the period first
    return out[:MOST]

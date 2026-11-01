"""How sure an answer is, from signals computed in code, never the model's own guess:

  - agreement: how many of the three searches put the answer's table first (3 = all agree)
  - the number check replaced the answer (verify.py): low
  - the table came from the fallback chain (the first choice had no numbers)

level(result) -> "high" | "medium" | "low" | None (no table used)
describe(level, lang) -> the line under the answer, with how often that level was right on test questions with
known answers, read from data/confidence.json (written by eval/confidence_eval.py; without it, no percentage).
The rates hold only for the model they were measured with ("model" in the file; before it was recorded: gemma,
run 10): any other model gets no confidence line.
"""
import json
from functools import cache

from . import llm
from .retrieve import DATA

CALIBRATION = DATA / "confidence.json"
CALIBRATED_ON = "Gemma 4 31B"  # the model data/confidence.json was measured on: other models show no confidence
LABELS = {"is": {"high": "mikil", "medium": "miðlungs", "low": "lítil", "head": "Vissa",
                 "rate": "svör af þessu tagi voru rétt í {right} af {n} prófum"},
          "en": {"high": "high", "medium": "medium", "low": "low", "head": "Confidence",
                 "rate": "answers like this were right in {right} of {n} tests"}}


def signals(result):
    parts = result.get("parts") or []
    agree = [p.get("agreement") for p in parts if p.get("agreement") is not None]
    return {"parts": len(parts), "agreement": min(agree) if agree else None,
            "replaced": bool(result.get("unsupported")),
            "fallback": any(any("tables tried first" in f for f in p.get("fallback") or []) for p in parts),
            "assumed": any(p.get("assumed") for p in parts)}


def level(result):
    s = signals(result)
    if not s["parts"]:
        return None
    if s["replaced"] or s["fallback"] or not s["agreement"]:
        return "low"
    # set by eval/confidence_eval.py (31 Oct, Gemma 4, their 60 + our 41): right 62% with 0 searches agreeing,
    # 73% with 1, 82% with 2; all three agreeing never happened, so "high" starts at 2
    return "high" if s["agreement"] >= 2 else "medium"


@cache
def calibration():
    try:
        return json.loads(CALIBRATION.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def calibrated_for():
    """The served name of the model the rates were measured with."""
    return calibration().get("model", "gemma")


def describe(lvl, lang="is", model=None):
    """model: the display name of the model that answered (None: assume the calibrated one)."""
    if lvl is None:
        return ""
    t = LABELS["is" if lang == "is" else "en"]
    if model and model != llm.display_name(calibrated_for()):
        return ""
    measured = calibration().get(lvl)
    if not (measured and measured.get("n")):
        return f'{t["head"]}: {t[lvl]}'
    # the measured rate, not the level word: "low" over a right answer read as a contradiction in the dry run
    pct = round(100 * measured["right"] / measured["n"])
    return f'{t["head"]}: {pct}% ({t["rate"].format(right=measured["right"], n=measured["n"])})'

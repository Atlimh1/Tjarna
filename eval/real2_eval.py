"""The team's own questions (eval/questions_real2.jsonl, 42, FROZEN: report, never tune) through the front door.

    python eval/real2_eval.py            # needs the LLM, like gagnvist_eval --front-door

Each question says what a good answer does: "answer" (one of the listed tables), "partial" (part of it is
published: a listed table, or an honest decline counts), "decline" (nothing published: the guard stops it or no
table is cited). Tables only, no numbers: the labels were proposed from the catalog and are being checked by the team.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from decline_eval import outcome  # noqa: E402

REAL2 = ROOT / "eval/questions_real2.jsonl"


def verdict(expect, kind, tables, accepted):
    """'ok' or what went wrong, for one question."""
    used = [t.strip() for t in tables.split(",")] if kind == "answered" else []
    if expect == "decline":
        return "ok" if kind == "declined" else "answered (should decline)"
    if kind == "declined":
        return "ok" if expect == "partial" else "declined (should answer)"
    return "ok" if set(used) & set(accepted) else "other table"


def main():
    from app import ask, retriever

    rows = [json.loads(line) for line in open(REAL2, encoding="utf-8") if line.strip()]
    tally = Counter()
    for r in rows:
        started = time.perf_counter()
        try:
            checked, result = ask(r["q"], retriever())
            kind, why = outcome(checked, result)
            answer = result.get("answer", "")
        except Exception as e:  # noqa: BLE001 - one failure must not stop the others
            kind, why, answer = "error", type(e).__name__, ""
        v = verdict(r["expect"], kind, why, r.get("table") or []) if kind != "error" else "error"
        tally[(r["expect"], v == "ok")] += 1
        print(f"{'ok ' if v == 'ok' else 'NO '} {r['expect']:8s} {time.perf_counter() - started:5.1f}s  {v}: {why}"
              f"\n     Q: {r['q']}\n     A: {' '.join(answer.split())[:160]}", flush=True)
    for expect in ("answer", "partial", "decline"):
        n = tally[(expect, True)] + tally[(expect, False)]
        print(f"{expect:8s} {tally[(expect, True)]}/{n}")
    print(f"all      {sum(v for (e, ok), v in tally.items() if ok)}/{len(rows)}")


if __name__ == "__main__":
    main()

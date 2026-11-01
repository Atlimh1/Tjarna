"""Questions Statistics Iceland can't answer (eval/questions_new_nodata.jsonl: another agency has it, private,
opinion, future): does the front door decline them, or does it answer from some table anyway?

    python eval/decline_eval.py [file.jsonl]    # needs the LLM (LLM_BASE_URL etc.), like gagnvist_eval --front-door

Declined = the guard stopped it (any category), or the pipeline used no table. Answered = it cited a table: a miss,
since forced pick will always find *some* table. The answers are printed so a person can judge each one.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NODATA = ROOT / "eval/questions_new_nodata.jsonl"


def outcome(checked, result):
    """("declined", why) or ("answered", table ids) for one app.ask() result."""
    if not checked.ok:
        return "declined", f"guard: {checked.category}"
    tables = [p["table"]["id"] for p in result.get("parts") or [] if p.get("table")]
    return ("answered", ", ".join(tables)) if tables else ("declined", "no table")


def main():
    from app import ask, retriever

    rows = [json.loads(line) for line in open(sys.argv[1] if len(sys.argv) > 1 else NODATA, encoding="utf-8")]
    declined = 0
    for r in rows:
        started = time.perf_counter()
        try:
            checked, result = ask(r["question"], retriever())
            kind, why = outcome(checked, result)
            answer = result.get("answer", "")
        except Exception as e:  # noqa: BLE001 - one failure must not stop the others
            kind, why, answer = "error", type(e).__name__, ""
        declined += kind == "declined"
        print(f"{kind:9s} {time.perf_counter() - started:5.1f}s  [{r.get('category', '?')}] {r['id']}: {why}", flush=True)
        print(f"          Q: {r['question']}\n          A: {' '.join(answer.split())[:160]}", flush=True)
    print(f"\nDeclined {declined} of {len(rows)} (should be all)")


if __name__ == "__main__":
    main()

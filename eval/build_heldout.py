"""Merge the agent-written questions, check them, and split into dev/test by TABLE.

All questions about one table go to the same side, so a fix made while looking at a dev
question can't also "teach" the test question about the same table.

    .venv/bin/python eval/build_heldout.py
writes eval/questions_dev.jsonl (~30% of tables: tune freely) and
       eval/questions_test.jsonl (~70%: frozen, run rarely, never tune against it)
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hagstofa.retrieve import load_catalog  # noqa: E402

HELDOUT = ROOT / "eval" / "heldout"
SEED = 20261024
DEV_SHARE = 0.3


def main():
    sampled = {t["id"] for f in sorted(HELDOUT.glob("tables_*.json")) for t in json.loads(f.read_text(encoding="utf-8"))}
    known = {t["id"] for t in load_catalog()}
    rows, seen, problems = [], set(), []
    for f in sorted(HELDOUT.glob("questions_*.jsonl")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{f.name}:{n} invalid JSON")
                continue
            q, tid = (r.get("q") or "").strip(), r.get("table")
            if not q or tid not in known:
                problems.append(f"{f.name}:{n} bad question or unknown table {tid!r}")
            elif tid not in sampled:
                problems.append(f"{f.name}:{n} table {tid} was not in the sample")
            elif q.lower() in seen:
                problems.append(f"{f.name}:{n} duplicate question")
            else:
                seen.add(q.lower())
                rows.append({"q": q, "table": [tid], "lang": r.get("lang"), "style": r.get("style")})

    tables = sorted({r["table"][0] for r in rows})
    random.Random(SEED).shuffle(tables)
    dev_tables = set(tables[: round(len(tables) * DEV_SHARE)])
    for name, part in (("dev", [r for r in rows if r["table"][0] in dev_tables]),
                       ("test", [r for r in rows if r["table"][0] not in dev_tables])):
        with open(ROOT / "eval" / f"questions_{name}.jsonl", "w", encoding="utf-8") as out:
            out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        langs = {lang: sum(r["lang"] == lang for r in part) for lang in ("is", "en")}
        print(f"{name:4s}: {len(part)} questions, {len({r['table'][0] for r in part})} tables, {langs}")
    print(f"tables in sample without questions: {len(sampled - set(tables))}")
    print(f"{len(problems)} problems" + "".join(f"\n  {p}" for p in problems[:20]))


if __name__ == "__main__":
    main()

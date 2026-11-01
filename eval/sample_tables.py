"""Sample current (non-archive) tables for the generated held-out question set.

Stratified by database (Íbúar, Samfélag, ...) in proportion to its size, fixed seed so it is
reproducible. Writes eval/heldout/tables_<n>.json in chunks, one per question-writing agent:
each table as a short description (title, folder, variables, a few example values, period).

    .venv/bin/python eval/sample_tables.py [n_tables] [n_chunks]
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hagstofa.retrieve import is_archive, load_catalog  # noqa: E402

OUT = ROOT / "eval" / "heldout"
SEED = 20261024
EXAMPLE_VALUES = 8


def describe(t):
    variables = []
    for v in t["variables_is"]:
        texts = v["valueTexts"]
        entry = {"name": v["text"], "n_values": len(texts), "examples": texts[:EXAMPLE_VALUES]}
        if v.get("time"):
            entry["period"] = f"{texts[0]} – {texts[-1]}"
        variables.append(entry)
    return {"id": t["id"], "title": t["title_is"], "folder": " › ".join(t["breadcrumbs_is"]), "variables": variables}


def main(n_tables=300, n_chunks=4):
    rng = random.Random(SEED)
    by_db = defaultdict(list)
    seen = set()
    for t in load_catalog():
        if not is_archive(t) and t["id"] not in seen:  # some ids appear in two folders
            seen.add(t["id"])
            by_db[t["path"].split("/")[0]].append(t)
    total = sum(len(v) for v in by_db.values())
    sample = []
    for db, tables in sorted(by_db.items()):
        share = round(n_tables * len(tables) / total)
        sample += rng.sample(tables, min(share, len(tables)))
        print(f"{db:12s} {len(tables):4d} current tables -> {share} sampled")
    rng.shuffle(sample)
    OUT.mkdir(exist_ok=True)
    for c in range(n_chunks):
        chunk = [describe(t) for t in sample[c::n_chunks]]
        (OUT / f"tables_{c + 1}.json").write_text(json.dumps(chunk, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"chunk {c + 1}: {len(chunk)} tables")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))

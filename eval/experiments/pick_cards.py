"""Do richer table lines help the picker choose the right table among the same options? (31 Oct night)

    python eval/experiments/pick_cards.py search [sets...]      # once: the search's options, saved
    LLM_BASE_URL=... python eval/experiments/pick_cards.py pick [arms...]

The search runs once per question (the default bilingual search, k = PICK_K, fresh_first and the villages note as in
pipeline.answer) and is saved to eval/results/pick_options.json; then each arm asks the picker about the very same
options, 8 questions at a time. Arms differ only in the line the model reads per table:
    base    title, breakdowns, measures, unit, last update (CARDS=0, today's default)
    old     + period and place level (CARDS=1, CARD_NOTES=0)
    notes   + frequency, outside source, provisional, the table's own note cut to the question (CARDS=1)
"picked" = the model chose an accepted table (the search's #1 when it declines, as in the pipeline). Generated sets
accept only their source table, so look-alikes count as wrong: compare arms, not absolute numbers.

RESULTS (1 Nov 01:30, JURECA, Gemma 4 31B, four-way search, k = 12), picked right (right table among the options):
              options   base   old   notes
    dev         232      171   174    171
    hand         36       27    28     28
    dev_en      188      143   145    141
    the60        93       74    71     70
    new41        51       46    45     45
Rejected: cards don't move the picker beyond noise, and the organizers' 60 lose 3-4 parts. The changed picks are
flips between look-alikes in both directions: old cards broke the wage index (LAU04000 -> LAU04001, twice) and
Vigdís's votes (KOS07112 -> SOG13005); notes fixed Vigdís and broke unemployment (VIN00920 -> VIN01206) and births per
1,000. The picker already reads title, breakdowns, measures, unit and last update; more text per line is more noise.
So CARDS stays off; the notes are shown to people instead (app.table_facts, under each answer).
End to end (1 Nov 01:40, front door, CARDS=1 = year filter + cards with notes) confirms it: the 60 60 -> 54
(Vigdís, wage-index points, two unemployment parts from VIN01206), set 1 37 -> 37, set 2 35 -> 33, real2 33 -> 33.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "experiments"))

from candidate_union import load  # noqa: E402

from hagstofa import cards, pipeline, retrieve  # noqa: E402

SAVED = ROOT / "eval" / "results" / "pick_options.json"
SETS = ["dev", "hand", "dev_en", "the60", "new41"]
ARMS = {"base": (False, False), "old": (True, False), "notes": (True, True)}


def search(sets):
    r = retrieve.build(retrieve.DEFAULT)
    index = {t["path"]: i for i, t in enumerate(r.tables)}
    places = retrieve.places_of(r)
    out = {}
    for name in sets:
        rows = []
        for q, accept in load(name):
            fresh = pipeline.fresh_first(q, r.search(q, k=pipeline.PICK_K))
            rows.append({"q": q, "accept": sorted(accept), "options": [index[t["path"]] for t, _ in fresh],
                         "villages": places.villages(q) if places is not None else []})
        out[name] = rows
        print(f"{name}: {len(rows)} questions searched", flush=True)
    SAVED.parent.mkdir(exist_ok=True)
    SAVED.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def pick(arms):
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    tables = retrieve.load_catalog()
    results = {}
    for arm in arms:
        pipeline.CARDS, cards.CARD_NOTES = ARMS[arm]
        for name, rows in saved.items():
            def one(row):
                options = [(tables[i], 0.0) for i in row["options"]]
                t = pipeline.pick_table(row["q"], options, row["villages"]) if options else None
                return (t or (options[0][0] if options else {})).get("id")
            start = time.time()
            with ThreadPoolExecutor(8) as pool:
                got = list(pool.map(one, rows))
            results[(arm, name)] = got
            right = sum(g in r["accept"] for g, r in zip(got, rows))
            in_top = sum(any(tables[i]["id"] in r["accept"] for i in r["options"]) for r in rows)
            print(f"{arm:6} {name:7} n={len(rows):3}  in top {in_top:3}  picked {right:3} ({100 * right / len(rows):.0f}%)"
                  f"  {time.time() - start:.0f} s", flush=True)
    for name, rows in saved.items():
        for a, b in zip(arms, arms[1:]):
            changed = [(r["q"], x, y, r["accept"]) for r, x, y in zip(rows, results[(a, name)], results[(b, name)]) if x != y]
            fixed = sum(y in acc and x not in acc for _, x, y, acc in changed)
            broke = sum(x in acc and y not in acc for _, x, y, acc in changed)
            print(f"\n{name}: {a} -> {b}: {len(changed)} picks changed, {fixed} fixed, {broke} broken")
            for q, x, y, acc in changed[:30]:
                mark = "+" if y in acc and x not in acc else "-" if x in acc and y not in acc else " "
                print(f"  {mark} {x} -> {y} (right {','.join(acc)})  {q[:110]}")
    print("PICK_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "search":
        search(sys.argv[2:] or SETS)
    else:
        pick(sys.argv[2:] or list(ARMS))

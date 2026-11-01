"""Where does the picker lose? Search rank of the right table vs the rank the picker chose (1 Nov).

    LLM_BASE_URL=... python eval/experiments/pick_ranks.py [sets...]

Uses the options saved by pick_cards.py search (eval/results/pick_options.json: the same top PICK_K, fresh_first and
villages as the pipeline) and asks today's picker once per question, 8 at a time. Prints, per set: how often the
right table is the search's #1, how often the picker keeps a right #1, and when it is wrong, at which rank the right
table was. That says whether a rule like "keep the search's #1 unless ..." could help, before building one.

RESULTS (1 Nov 11:29, Gemma 4 31B, k = 12):
              right #1   picked right   left a right #1   found a right lower one
    dev         143          171              25                   53
    dev_en      111          143              13                   45
    hand         24           21               7                    4
    the60        55           47              25                   17     (parts; accepted tables)
The picker earns its place on the generated sets (dev +28, dev_en +32 over the search's #1) but leaves a right #1
often, most on the organizers' 60 (the fallback chain and the self-check still get those 60/60 end to end). Tried
from this: read the search's #1 alongside when the picker leaves it (pipeline.ALSO_FROM_SEARCH): dev3 46 -> 45, kept
off. The runner-up the picker names itself (ALSO_TABLE) stays the better second opinion.
"""
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hagstofa import pipeline, retrieve  # noqa: E402

SAVED = ROOT / "eval" / "results" / "pick_options.json"


def main():
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    tables = retrieve.load_catalog()
    for name in sys.argv[1:] or list(saved):
        rows = saved[name]

        def one(row):
            options = [(tables[i], 0.0) for i in row["options"]]
            t = pipeline.pick_table(row["q"], options, row["villages"]) if options else None
            ids = [o["id"] for o, _ in options]
            picked = ids.index(t["id"]) if t and t["id"] in ids else (0 if options else None)
            right = next((k for k, x in enumerate(ids) if x in row["accept"]), None)
            return picked, right

        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(one, rows))
        n = len(rows)
        right_first = sum(r == 0 for _, r in got)
        kept = sum(p == 0 and r == 0 for p, r in got)
        moved_off = sum(p != 0 and r == 0 for p, r in got)
        moved_on = sum(p == r and r not in (None, 0) for p, r in got)
        picked_right = sum(p == r and r is not None for p, r in got)
        where = collections.Counter(r for p, r in got if r is not None and p != r)
        print(f"{name:7} n={n}: right #1 {right_first}; picked right {picked_right}; kept a right #1 {kept}, "
              f"left a right #1 {moved_off}, found a right lower one {moved_on}; when wrong, right was at rank "
              f"{dict(sorted(where.items()))}", flush=True)
    print("RANKS_DONE")


if __name__ == "__main__":
    main()

"""Does indexing the tables' own notes (data/cards.jsonl, from the release's px headers) help search? (31 Oct night)

    python eval/experiments/notes_index.py none|short|full [sets...]

none: lemma indexes titles, breadcrumbs, variables, English and value labels, as today. short: plus the note cut to
its first non-housekeeping sentences (cards.note, 170 characters, no question). full: plus the whole Icelandic note.
Search only (the default bilingual search; translations from the cache), top 1 / 3 / 12 (12 = the picker's options).

RESULTS (1 Nov 01:10, JURECA, four-way search with the fine-tuned bge-m3), right table in top 1 / 3 / 12:
              none            short           full
    dev       154/213/252     153/211/252     153/209/253
    hand       23/29/36        23/31/36        23/30/36
    muni       16/21/24        16/21/24        15/22/24
    real        5/7/9           6/7/9           5/7/9
    local       6/12/12         4/12/12         5/12/12
    e2e        21/26/34        21/27/34        21/27/34
    dev_en    122/178/205     124/174/204     121/170/205
    the60      55/72/93        53/72/92        57/73/87
    new41      28/41/51        28/41/51        29/42/52
Rejected: no set gains clearly and several lose (dev_en top 3 -4 / -8, the60 top 12 -1 / -6, local top 1 -2). As with
town names, long text dilutes a table's topic words in the length-normalised TF-IDF. The notes stay where they tell
look-alikes apart: the picker's card (cards.card), measured in pick_cards.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "experiments"))

from candidate_union import load  # noqa: E402

from hagstofa import cards, retrieve  # noqa: E402


def main():
    arm = sys.argv[1]
    sets = sys.argv[2:] or ["dev", "hand", "muni", "real", "local", "e2e", "dev_en", "the60", "new41"]
    plain = retrieve.lemma_table_text
    if arm == "short":
        retrieve.lemma_table_text = lambda t, places=frozenset(): plain(t, places) + " " + cards.note(t)
    elif arm == "full":
        retrieve.lemma_table_text = lambda t, places=frozenset(): plain(t, places) + " " + cards.extra(t).get("note_is", "")
    r = retrieve.build(retrieve.DEFAULT)
    print(f"arm {arm}: in top 1 / 3 / 12", flush=True)
    for name in sets:
        try:
            qs = load(name)
        except FileNotFoundError:
            continue
        hits = [0, 0, 0]
        for q, accept in qs:
            top = [t["id"] for t, _ in r.search(q, k=12)]
            for i, k in enumerate((1, 3, 12)):
                hits[i] += any(a in top[:k] for a in accept)
        n = len(qs)
        print(f"{name:8} {n:4}  " + "  ".join(f"{round(100 * h / n)}%" for h in hits) + f"  ({'/'.join(map(str, hits))})",
              flush=True)
    print("NOTES_DONE")


if __name__ == "__main__":
    main()

"""Do the card filters put the right table among the picker's options more often? (31 Oct night; search only)

    EMBED_MODEL_2=<fine-tuned> python eval/experiments/cards_recall.py [sets...]

plain: the search's top PICK_K, as the pipeline picks today. cards: the search's top WIDE, filtered by code
(cards.filter_candidates: years, places, index vs amount), then the top PICK_K. In top k = an accepted table is among
the picker's options (the most the pick can get). Both through the default (bilingual) search, translations from
the cache.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "experiments"))

from candidate_union import load  # noqa: E402

from hagstofa import cards, pipeline, retrieve  # noqa: E402


def main():
    sets = sys.argv[1:] or ["dev", "hand", "muni", "real", "local", "e2e", "dev_en", "the60", "new41"]
    r = retrieve.build(retrieve.DEFAULT)
    k = pipeline.PICK_K
    print(f"in the picker's {k} options: plain top {k} -> cards (top {cards.WIDE}, filtered, top {k})")
    for name in sets:
        try:
            qs = load(name)
        except FileNotFoundError:
            continue
        plain = filtered = 0
        for q, accept in qs:
            wide = r.search(q, k=cards.WIDE)
            top = [t["id"] for t, _ in wide[:k]]
            kept = cards.filter_candidates(q, wide, frozenset(pipeline.years_in(q)))[:k]
            plain += any(a in top for a in accept)
            filtered += any(t["id"] in accept for t, _ in kept)
        n = len(qs)
        print(f"{name:8} {n:4}  plain {round(100 * plain / n)}%  cards {round(100 * filtered / n)}%"
              f"  ({plain} -> {filtered})", flush=True)
    print("CARDS_DONE")


if __name__ == "__main__":
    main()

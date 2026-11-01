"""Misspelled copies of a question set, to measure how retrieval copes with typos.

Two styles per question, same answer key as the original:
  noaccent  no Icelandic letters, as typed on a foreign keyboard ("hvad bua margir a islandi")
  typo      lowercase + 1-2 typical slips in longer words: dropped letter, swapped letters, i/y mix-up,
            single for double letter ("atvinuleysi", "þjóðkyrkjunni")
Seeded, so the same file comes out every time.

    .venv/bin/python eval/make_typos.py dev      # writes eval/questions_dev_typo.jsonl
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from run_eval import load_questions  # noqa: E402  (also puts the repo root on sys.path)
from hagstofa.retrieve import fold  # noqa: E402

SWAPS = {"i": "y", "y": "i", "í": "ý", "ý": "í"}


def noaccent(q):
    return fold(q)


def slip(word, rng):
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["drop", "swap", "iy", "double"])
    if kind == "double" and (m := re.search(r"(\w)\1", word)):
        return word[:m.start()] + word[m.start() + 1:]
    if kind == "iy" and (spots := [j for j, c in enumerate(word) if c in SWAPS]):
        j = rng.choice(spots)
        return word[:j] + SWAPS[word[j]] + word[j + 1:]
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i + 1:]


def typo(q, rng):
    words = q.lower().split(" ")
    long_ = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    for i in rng.sample(long_, min(len(long_), rng.choice([1, 2]))):
        words[i] = slip(words[i], rng)
    return " ".join(words)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "dev"
    rng = random.Random(0)
    out = ROOT / "eval" / f"questions_{name}_typo.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for item in load_questions(name):
            for style, q in [("noaccent", noaccent(item["q"])), ("typo", typo(item["q"], rng))]:
                f.write(json.dumps({**item, "q": q, "orig": item["q"], "typo_style": style}, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
